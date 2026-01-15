import os
import argparse
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from training.datasets.loader import MaterialDataLoader
from training.datasets.graph_builder import CrystalGraphBuilder
from training.datasets.graph_dataset import CrystalGraphDataset, collate_batch
from training.models.cgcnn import CGCNN

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """Ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load Data
    logger.info("Loading and processing data...")
    loader = MaterialDataLoader(args.data_path)
    raw_data = loader.load()
    
    if not raw_data:
        raise ValueError("No data loaded.")
        
    # 2. Initialize Graph Builder & Dataset
    # Using defaults: radius=8.0, max_neighbors=12
    # Important: dStep=0.2 means (8.0/0.2) + 1 = 41 edge features
    builder = CrystalGraphBuilder(radius=8.0, dStep=0.2)
    
    # Pre-compute graphs (cached)
    full_dataset = CrystalGraphDataset(raw_data, builder, cache_graphs=True)
    
    # 3. Split Indices
    indices = list(range(len(full_dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=42) # 0.1 of remaining 0.8 -> 0.08 total
    
    train_set = Subset(full_dataset, train_idx)
    val_set = Subset(full_dataset, val_idx)
    test_set = Subset(full_dataset, test_idx)
    
    logger.info(f"Train size: {len(train_set)}, Val size: {len(val_set)}, Test size: {len(test_set)}")

    # 4. Target Normalization (Important for regression stability)
    # We collect all training targets to fit the scaler
    train_targets = []
    for i in train_idx:
        train_targets.append(full_dataset[i]['target'].numpy())
    train_targets = np.array(train_targets)
    
    scaler = StandardScaler()
    scaler.fit(train_targets)
    
    # We need a wrapper or normalization inside the loop. 
    # Ideally, dataset returns raw, loop normalizes y, and un-normalizes pred for metrics.
    # Convert scaler mean/std to tensors for GPU
    target_mean = torch.tensor(scaler.mean_, device=device, dtype=torch.float32)
    target_scale = torch.tensor(scaler.scale_, device=device, dtype=torch.float32)

    # 5. DataLoaders
    # num_workers=0 for simple debugging, increase in prod
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    # 6. Initialize Model
    # Node dim is 4 (Z, Group, Period, X)
    # Edge dim is 41 (from GaussianDistance defaults above)
    model = CGCNN(
        node_input_dim=4, 
        edge_input_dim=41, 
        node_hidden_dim=64, 
        n_conv_layers=3, 
        n_targets=3
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.L1Loss() # MAE Loss as requested (or SmoothL1Loss)

    # 7. Training Loop
    best_val_mae = float('inf')
    early_stop_counter = 0
    save_path = os.path.join(args.save_dir, args.model_name)
    os.makedirs(args.save_dir, exist_ok=True)
    
    logger.info(f"Starting training... (Saving to {save_path})")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            # Move batch to device
            atom_fea = batch['atom_fea'].to(device)
            nbr_fea = batch['nbr_fea'].to(device)
            nbr_idx = batch['nbr_idx'].to(device)
            batch_map = batch['batch'].to(device)
            targets = batch['target'].to(device)
            
            # Normalize targets for training stability
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
        val_loss = 0.0 # Loss on normalized
        val_mae_sum = torch.zeros(3, device=device) # Real MAE per property
        total_val_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                atom_fea = batch['atom_fea'].to(device)
                nbr_fea = batch['nbr_fea'].to(device)
                nbr_idx = batch['nbr_idx'].to(device)
                batch_map = batch['batch'].to(device)
                targets = batch['target'].to(device)
                
                # Predict
                preds_norm = model(atom_fea, nbr_fea, nbr_idx, batch_map)
                
                # Compute Loss (Normalized)
                targets_norm = (targets - target_mean) / target_scale
                loss = criterion(preds_norm, targets_norm)
                val_loss += loss.item() * targets.size(0)
                
                # Compute Real Metrics (Un-normalized)
                preds = preds_norm * target_scale + target_mean
                abs_err = torch.abs(preds - targets)
                val_mae_sum += torch.sum(abs_err, dim=0)
                total_val_samples += targets.size(0)

        val_loss /= len(val_set)
        val_maes = val_mae_sum / total_val_samples # (3,)
        avg_val_mae = torch.mean(val_maes).item()

        # Update Scheduler
        scheduler.step(val_loss)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        logger.info(f"   >> Val MAE per target: E_f={val_maes[0]:.3f}, Bg={val_maes[1]:.3f}, Dens={val_maes[2]:.3f}")

        # Checkpointing
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            early_stop_counter = 0
            # Save model and scaler stats
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler_mean': target_mean,
                'scaler_scale': target_scale
            }, save_path)
            # logger.info("   >> Model saved.")
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.patience:
                logger.info("Early stopping triggered.")
                break

    # 8. Final Test
    logger.info("Loading best model for testing...")
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    target_mean = checkpoint['scaler_mean']
    target_scale = checkpoint['scaler_scale']
    
    model.eval()
    test_mae_sum = torch.zeros(3, device=device)
    total_test = 0
    
    with torch.no_grad():
        for batch in test_loader:
            atom_fea = batch['atom_fea'].to(device)
            nbr_fea = batch['nbr_fea'].to(device)
            nbr_idx = batch['nbr_idx'].to(device)
            batch_map = batch['batch'].to(device)
            targets = batch['target'].to(device)
            
            preds_norm = model(atom_fea, nbr_fea, nbr_idx, batch_map)
            preds = preds_norm * target_scale + target_mean
            
            abs_err = torch.abs(preds - targets)
            test_mae_sum += torch.sum(abs_err, dim=0)
            total_test += targets.size(0)
            
    test_maes = test_mae_sum / total_test
    
    print("\n" + "="*40)
    print("CGCNN TRAINING COMPLETE RESULTS")
    print(f"Formation Energy MAE: {test_maes[0]:.4f} eV/atom")
    print(f"Band Gap MAE:       {test_maes[1]:.4f} eV")
    print(f"Density MAE:        {test_maes[2]:.4f} g/cm^3")
    print("="*40 + "\n")
    print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CGCNN Model")
    parser.add_argument("--data_path", type=str, default="../../data/mock_materials.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-3) # Higher LR for GNNs often ok
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="../../models")
    parser.add_argument("--model_name", type=str, default="cgcnn_best.pt", help="Filename for saved model")
    
    args = parser.parse_args()
    
    # Path handling
    base_dir = os.path.dirname(__file__)
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(base_dir, args.data_path)
    if not os.path.isabs(args.save_dir):
        args.save_dir = os.path.join(base_dir, args.save_dir)
        
    main(args)
