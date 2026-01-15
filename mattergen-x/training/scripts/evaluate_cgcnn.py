import os
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pymatgen.core.structure import Structure
from torch.utils.data import DataLoader

# Add project root to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from training.datasets.loader import MaterialDataLoader
from training.datasets.graph_dataset import CrystalGraphDataset, collate_batch
from training.datasets.graph_builder import CrystalGraphBuilder
from training.models.cgcnn import CGCNN

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Target Labels
TARGETS = ["Formation Energy", "Band Gap", "Density"]
UNITS = ["eV/atom", "eV", "g/cm^3"]

def plot_scatter(y_true, y_pred, target_name, unit, save_path):
    """Generate True vs Predicted Scatter Plot."""
    plt.figure(figsize=(8, 6))
    
    # Calculate limits for identity line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    buffer = (max_val - min_val) * 0.05
    
    plt.plot([min_val-buffer, max_val+buffer], [min_val-buffer, max_val+buffer], 'k--', alpha=0.5, label='Ideal')
    plt.scatter(y_true, y_pred, alpha=0.6, c='royalblue', edgecolors='w', s=50)
    
    plt.xlabel(f"True {target_name} ({unit})", fontsize=12, fontweight='bold')
    plt.ylabel(f"Predicted {target_name} ({unit})", fontsize=12, fontweight='bold')
    plt.title(f"{target_name}: True vs Predicted", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    
    # Add metrics annotation
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    metrics_text = f"MAE: {mae:.3f}\n$R^2$: {r2:.3f}"
    plt.annotate(metrics_text, xy=(0.05, 0.9), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_error_dist(y_true, y_pred, target_name, unit, save_path):
    """Generate Error Distribution Histogram."""
    errors = y_pred - y_true
    
    plt.figure(figsize=(8, 6))
    sns.histplot(errors, kde=True, color='crimson', bins=30, alpha=0.6)
    
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    plt.xlabel(f"Prediction Error ({unit})", fontsize=12, fontweight='bold')
    plt.ylabel("Count", fontsize=12, fontweight='bold')
    plt.title(f"{target_name}: Error Distribution", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data (for Test Split)
    logger.info("Loading Data...")
    loader = MaterialDataLoader(args.data_path)
    df = loader.load_data()
    
    # Simple deterministic split (last 20% as per training script)
    test_size = int(len(df) * 0.2)
    test_df = df.iloc[-test_size:].reset_index(drop=True)
    
    logger.info(f"Evaluating on {len(test_df)} test samples.")

    # 2. Build Dataset
    logger.info("Building Test Graph Dataset...")
    # Targets: E_f, Bg, Density
    targets = test_df[["formation_energy_per_atom", "band_gap", "density"]].values
    structures = test_df["structure"].tolist()
    
    dataset = CrystalGraphDataset(
        structures, targets, 
        atom_fea_len=64, n_nbr=12, 
        data_dir=os.path.dirname(args.data_path) # Utilize cache if available
    )
    
    test_loader = DataLoader(
        dataset, batch_size=args.batch_size, 
        shuffle=False, collate_fn=collate_batch
    )
    
    # 3. Load Model
    # Need to load scaler stats too
    checkpoint_path = os.path.join(args.model_dir, "cgcnn_best.pt")
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        return

    logger.info(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize Model Structure
    model = CGCNN(
        atom_fea_len=64, 
        nbr_fea_len=40, # Default Gaussian distance expansion
        n_conv=3, 
        h_fea_len=64, 
        n_h=1, 
        n_out=3 # 3 Targets
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    scaler_mean = checkpoint['scaler_mean'].to(device)
    scaler_scale = checkpoint['scaler_scale'].to(device)
    model.eval()
    
    # 4. Predict
    all_preds = []
    all_targets = []
    
    logger.info("Running Inference...")
    with torch.no_grad():
        for batch_idx, (atom_fea, nbr_fea, nbr_idx, batch_map, target) in enumerate(test_loader):
            atom_fea = atom_fea.to(device)
            nbr_fea = nbr_fea.to(device)
            nbr_idx = nbr_idx.to(device)
            batch_map = batch_map.to(device)
            target = target.to(device)
            
            # Forward
            preds_norm = model(atom_fea, nbr_fea, nbr_idx, batch_map)
            
            # Denormalize
            preds = preds_norm * scaler_scale + scaler_mean
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 5. Analyze & Plot
    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    print("\n" + "="*50)
    print("FINAL CGCNN EVALUATION REPORT (50k Dataset)")
    print("="*50 + "\n")

    for i, (name, unit) in enumerate(zip(TARGETS, UNITS)):
        y_t = all_targets[:, i]
        y_p = all_preds[:, i]
        
        # Metrics
        mae = mean_absolute_error(y_t, y_p)
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        r2 = r2_score(y_t, y_p)
        
        results.append({
            "Property": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })
        
        # Plots
        sanitized_name = name.lower().replace(" ", "_")
        scatter_path = os.path.join(args.output_dir, f"cgcnn_scatter_{sanitized_name}.png")
        error_path = os.path.join(args.output_dir, f"cgcnn_error_{sanitized_name}.png")
        
        plot_scatter(y_t, y_p, name, unit, scatter_path)
        plot_error_dist(y_t, y_p, name, unit, error_path)
        
        logger.info(f"Saved plots for {name} to {args.output_dir}")

    # Print Table
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    print("\n" + "-"*50)
    print(f"Evaluation complete. Screenshots saved to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CGCNN Model")
    parser.add_argument("--data_path", type=str, default="../../data/synthetic_materials.json")
    parser.add_argument("--model_dir", type=str, default="../../models")
    parser.add_argument("--output_dir", type=str, default="../../docs/screenshots")
    parser.add_argument("--batch_size", type=int, default=128)
    
    args = parser.parse_args()
    
    # Path handling
    base_dir = os.path.dirname(__file__)
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(base_dir, args.data_path)
    if not os.path.isabs(args.model_dir):
        args.model_dir = os.path.join(base_dir, args.model_dir)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(base_dir, args.output_dir)
        
    evaluate(args)
