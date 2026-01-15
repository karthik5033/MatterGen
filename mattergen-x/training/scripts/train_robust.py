import os
import argparse
import random
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from training.datasets.loader import MaterialDataLoader
from training.datasets.featurizer import ChemicalFeaturizer
from training.models.deep_regressor import DeepMaterialRegressor

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

def prepare_data(data_path: str, test_size: float = 0.2, val_size: float = 0.1):
    """Load, featurize, and split data."""
    logger.info("Loading and processing data...")
    loader = MaterialDataLoader(data_path)
    raw_data = loader.load()
    
    if not raw_data:
        raise ValueError("No data loaded.")

    featurizer = ChemicalFeaturizer()
    X, y = [], []
    
    skipped = 0
    for sample in raw_data:
        formula = sample.get("formula")
        # Targets: Formation Energy, Band Gap, Density
        targets = [
            sample.get("label_formation_energy"),
            sample.get("label_band_gap"),
            sample.get("label_density")
        ]
        
        if formula and all(t is not None for t in targets):
            features = featurizer.featurize_formula(formula)
            X.append(features)
            y.append(targets)
        else:
            skipped += 1
            
    logger.info(f"Loaded {len(X)} samples (Skipped {skipped}). Feature Dim: {len(X[0])}")
    
    # Split: Train+Val / Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        np.array(X), np.array(y), test_size=test_size, random_state=42
    )
    
    # Split: Train / Val
    # Adjust val_size to be relative to the original full dataset size approximately
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val_size, random_state=42
    )
    
    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return (
        torch.FloatTensor(X_train), torch.FloatTensor(y_train),
        torch.FloatTensor(X_val), torch.FloatTensor(y_val),
        torch.FloatTensor(X_test), torch.FloatTensor(y_test),
        scaler
    )

def train_model(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Prepare Data
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = prepare_data(args.data_path)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size)
    
    # 2. Initialize Model
    model = DeepMaterialRegressor(input_dim=X_train.shape[1], output_dim=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.MSELoss()

    # 3. Training Loop with Early Stopping
    best_val_loss = float('inf')
    early_stop_counter = 0
    save_path = os.path.join(args.save_dir, "best_deep_regressor.pt")
    os.makedirs(args.save_dir, exist_ok=True)

    logger.info("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.patience:
                logger.info("Early stopping triggered.")
                break

    logger.info(f"Training complete. Best Val Loss: {best_val_loss:.4f}")
    
    # 4. Final Evaluation
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    with torch.no_grad():
        test_preds = model(X_test.to(device)).cpu().numpy()
        y_test_np = y_test.numpy()
        
    targets = ["Formation Energy", "Band Gap", "Density"]
    results = []
    
    for i, target in enumerate(targets):
        mae = mean_absolute_error(y_test_np[:, i], test_preds[:, i])
        rmse = np.sqrt(mean_squared_error(y_test_np[:, i], test_preds[:, i]))
        results.append({"Property": target, "MAE": mae, "RMSE": rmse})
        
    print("\n--- Final Evaluation on Test Set ---")
    print(pd.DataFrame(results).to_string(index=False))
    print("-" * 40)
    print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust Training Script for MatterGen X")
    parser.add_argument("--data_path", type=str, default="../../data/mock_materials.json", help="Path to properties dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="../../models", help="Directory to save models")
    
    args = parser.parse_args()
    
    # Ensure paths are absolute or correct relative to script execution
    if not os.path.isabs(args.data_path):
        base_dir = os.path.dirname(__file__)
        args.data_path = os.path.join(base_dir, args.data_path)
    
    if not os.path.isabs(args.save_dir):
        base_dir = os.path.dirname(__file__)
        args.save_dir = os.path.join(base_dir, "../../models") # Correct relative path for models

    train_model(args)
