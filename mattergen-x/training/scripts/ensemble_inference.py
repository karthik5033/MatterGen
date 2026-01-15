import os
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from training.datasets.loader import MaterialDataLoader
from training.datasets.graph_builder import CrystalGraphBuilder
from training.datasets.graph_dataset import CrystalGraphDataset, collate_batch
from training.models.cgcnn import CGCNN
from training.scripts.train_robust import set_seed

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGETS = ["Formation Energy", "Band Gap", "Density"]

def predict_ensemble(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    logger.info("Loading Data...")
    loader = MaterialDataLoader(args.data_path)
    raw_data = loader.load()
    
    # Reuse split logic to get test set
    builder = CrystalGraphBuilder(radius=8.0, dStep=0.2)
    full_dataset = CrystalGraphDataset(raw_data, builder, cache_graphs=False) # No caching needed for inference pass
    
    indices = list(range(len(full_dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    test_set = Subset(full_dataset, test_idx)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, collate_fn=collate_batch, shuffle=False)
    
    logger.info(f"Test Set Size: {len(test_set)}")

    # 2. Load Models
    model_files = [f for f in os.listdir(args.model_dir) if f.startswith("cgcnn_seed_") and f.endswith(".pt")]
    model_files.sort()
    
    if not model_files:
        logger.error(f"No ensemble models found in {args.model_dir}")
        return

    logger.info(f"Found {len(model_files)} models in ensemble.")
    
    models = []
    scalers = [] # Each model has its own scaler params
    
    for mf in model_files:
        path = os.path.join(args.model_dir, mf)
        checkpoint = torch.load(path, map_location=device)
        
        model = CGCNN(node_input_dim=4, edge_input_dim=41, node_hidden_dim=64, n_conv_layers=3, n_targets=3).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        models.append(model)
        scalers.append({
            "mean": checkpoint['scaler_mean'],
            "scale": checkpoint['scaler_scale']
        })

    # 3. Predict Feature
    all_means = []
    all_stds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            atom_fea = batch['atom_fea'].to(device)
            nbr_fea = batch['nbr_fea'].to(device)
            nbr_idx = batch['nbr_idx'].to(device)
            batch_map = batch['batch'].to(device)
            targets = batch['target']
            
            # Predict with ALL models
            batch_preds = []
            for model, scaler in zip(models, scalers):
                preds_norm = model(atom_fea, nbr_fea, nbr_idx, batch_map)
                # Un-normalize using THIS model's scaler
                preds = preds_norm * scaler["scale"] + scaler["mean"]
                batch_preds.append(preds.cpu().numpy())
            
            # Shape: (N_models, Batch, 3)
            batch_preds = np.array(batch_preds)
            
            # Aggregate
            mean_pred = np.mean(batch_preds, axis=0) # (Batch, 3)
            std_pred = np.std(batch_preds, axis=0)   # (Batch, 3)
            
            all_means.append(mean_pred)
            all_stds.append(std_pred)
            all_targets.append(targets.numpy())

    all_means = np.concatenate(all_means, axis=0)
    all_stds = np.concatenate(all_stds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 4. Report
    print("\n" + "="*50)
    print("ENSEMBLE UNCERTAINTY REPORT")
    print("="*50)
    
    results = []
    for i, target in enumerate(TARGETS):
        avg_std = np.mean(all_stds[:, i])
        max_std = np.max(all_stds[:, i])
        mae = np.mean(np.abs(all_means[:, i] - all_targets[:, i]))
        
        print(f"Property: {target}")
        print(f"  > MAE (Ensemble Mean): {mae:.4f}")
        print(f"  > Avg Uncertainty (Std): {avg_std:.4f}")
        print(f"  > Max Uncertainty:       {max_std:.4f}")
        print("-" * 30)
        
        results.append({
            "Property": target,
            "Ensemble MAE": mae,
            "Avg Uncertainty": avg_std
        })
        
    # Save raw predictions for deeper analysis?
    # np.save(...) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CGCNN Ensemble")
    parser.add_argument("--data_path", type=str, default="../../data/mock_materials.json")
    parser.add_argument("--model_dir", type=str, default="../../models/ensemble")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Path handling
    base_dir = os.path.dirname(__file__)
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(base_dir, args.data_path)
    if not os.path.isabs(args.model_dir):
        args.model_dir = os.path.join(base_dir, args.model_dir)

    predict_ensemble(args)
