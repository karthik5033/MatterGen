
import os
import sys
import random
import logging
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# Suppress warnings to prevent console spam (e.g. from pymatgen)
warnings.filterwarnings("ignore")

print("Initializing Environment...")


# Dependency Hack: Remove sklearn/matplotlib requirement
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt

def simple_train_test_split(data, test_size=0.2, random_state=42):
    random.seed(random_state)
    indices = list(data)
    random.shuffle(indices)
    split = int(len(indices) * (1 - test_size))
    return indices[:split], indices[split:]

class SimpleScaler:
    def fit(self, data):
        self.mean_ = np.mean(data, axis=0)
        self.scale_ = np.std(data, axis=0)
        # Avoid zero div
        self.scale_[self.scale_ == 0] = 1.0

# --- 1. Path Setup (Robust for Terminal Execution) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# If running from scripts/, project root is ../../
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    # print(f"Adding project root to sys.path: {project_root}")
    sys.path.append(project_root)

# Import Custom Modules
try:
    from training.datasets.loader import MaterialDataLoader
    from training.datasets.graph_builder import CrystalGraphBuilder
    from training.datasets.graph_dataset import CrystalGraphDataset, collate_batch
    from training.models.cgcnn import CGCNN
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# --- 2. Configuration ---
class Config:
    # Point to the robust CLEAN dataset
    DATA_PATH = os.path.join(project_root, "data", "datasets", "mp20_clean.json")
    SAVE_DIR = os.path.join(project_root, "models")
    MODEL_NAME = "cgcnn_best.pt"
    
    # Hyperparameters
    EPOCHS = 100
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    PATIENCE = 20
    SEED = 42

    # Model Params
    NODE_INPUT_DIM = 4      # Z, Group, Period, X
    EDGE_INPUT_DIM = 41     # Gaussian params
    NODE_HIDDEN_DIM = 64
    N_CONV_LAYERS = 3
    N_TARGETS = 3

    NUM_WORKERS = 0 

args = Config()

# Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    print("=== CGCNN Training Pipeline (Fast Mode) ===")
    set_seed(args.SEED)
    
    # 1. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    # 2. Load Data
    if not os.path.exists(args.DATA_PATH):
        print(f"ERROR: Dataset file not found at {args.DATA_PATH}")
        return

    print("Loading Data... (This involves parsing 45k structures, please wait ~1-2 mins)")
    loader = MaterialDataLoader(args.DATA_PATH)
    raw_data = loader.load()
    
    if not raw_data:
        print("ERROR: Loader returned 0 samples.")
        return
        
    print(f"Successfully loaded {len(raw_data)} samples.")
    
    # 3. Build Graphs
    builder = CrystalGraphBuilder(radius=8.0, dStep=0.2)
    full_dataset = CrystalGraphDataset(raw_data, builder, cache_graphs=True)
    
    # 4. Split using Custom Splitter
    indices = list(range(len(full_dataset)))
    train_idx, test_idx = simple_train_test_split(indices, test_size=0.2, random_state=args.SEED)
    train_idx, val_idx = simple_train_test_split(train_idx, test_size=0.1, random_state=args.SEED)
    
    train_set = Subset(full_dataset, train_idx)
    val_set = Subset(full_dataset, val_idx)
    test_set = Subset(full_dataset, test_idx)
    
    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    
    # 5. Normalize Targets
    train_targets = []
    for i in train_idx:
        train_targets.append(full_dataset[i]['target'].numpy())
    train_targets = np.array(train_targets)
    
    scaler = SimpleScaler()
    scaler.fit(train_targets)
    
    target_mean = torch.tensor(scaler.mean_, device=device, dtype=torch.float32)
    target_scale = torch.tensor(scaler.scale_, device=device, dtype=torch.float32)
    
    print(f"Target Mean: {target_mean.cpu().numpy()}")
    print(f"Target Std:  {target_scale.cpu().numpy()}")
    
    # 6. DataLoaders
    # Collator imported from dataset
    train_loader = DataLoader(train_set, batch_size=args.BATCH_SIZE, shuffle=True, 
                              collate_fn=collate_batch, num_workers=args.NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=args.BATCH_SIZE, shuffle=False, 
                            collate_fn=collate_batch, num_workers=args.NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=args.BATCH_SIZE, shuffle=False, 
                             collate_fn=collate_batch, num_workers=args.NUM_WORKERS)
    
    # 7. Model
    model = CGCNN(
        node_input_dim=args.NODE_INPUT_DIM,
        edge_input_dim=args.EDGE_INPUT_DIM,
        node_hidden_dim=args.NODE_HIDDEN_DIM,
        n_conv_layers=args.N_CONV_LAYERS,
        n_targets=args.N_TARGETS
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.L1Loss()
    
    # 8. Loop
    print("\nStarting Training...")
    best_val_mae = float('inf')
    early_stop_counter = 0
    os.makedirs(args.SAVE_DIR, exist_ok=True)
    save_path = os.path.join(args.SAVE_DIR, args.MODEL_NAME)
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(args.EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            atom_fea = batch['atom_fea'].to(device)
            nbr_fea = batch['nbr_fea'].to(device)
            nbr_idx = batch['nbr_idx'].to(device)
            batch_map = batch['batch'].to(device)
            targets = batch['target'].to(device)
            
            # Normalize Targets
            targets_norm = (targets - target_mean) / target_scale
            
            optimizer.zero_grad()
            preds_norm = model(atom_fea, nbr_fea, nbr_idx, batch_map)
            loss = criterion(preds_norm, targets_norm)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * targets.size(0)
            
        train_loss /= len(train_set)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mae_sum = torch.zeros(3, device=device)
        total_val = 0
        
        with torch.no_grad():
            for batch in val_loader:
                atom_fea = batch['atom_fea'].to(device)
                nbr_fea = batch['nbr_fea'].to(device)
                nbr_idx = batch['nbr_idx'].to(device)
                batch_map = batch['batch'].to(device)
                targets = batch['target'].to(device)
                
                preds_norm = model(atom_fea, nbr_fea, nbr_idx, batch_map)
                targets_norm = (targets - target_mean) / target_scale
                loss = criterion(preds_norm, targets_norm)
                
                val_loss += loss.item() * targets.size(0)
                
                preds = preds_norm * target_scale + target_mean
                val_mae_sum += torch.sum(torch.abs(preds - targets), dim=0)
                total_val += targets.size(0)
        
        val_loss /= len(val_set)
        val_maes = val_mae_sum / total_val
        avg_val_mae = torch.mean(val_maes).item()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{args.EPOCHS} | Train Loss: {train_loss:.4f} | Val Avg MAE: {avg_val_mae:.4f}")
            print(f"   >> Detail MAE: Ef={val_maes[0]:.4f}, Bg={val_maes[1]:.4f}, Dens={val_maes[2]:.4f}")
            
        # Checkpoint
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            early_stop_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler_mean': target_mean,
                'scaler_scale': target_scale,
                'config': vars(args)
            }, save_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.PATIENCE:
                print(f"Early stop at epoch {epoch+1}")
                break
                
    print(f"Training Complete. Best Val MAE: {best_val_mae:.4f}")
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
