import os
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from training.scripts.train_robust import prepare_data, set_seed
from training.models.deep_regressor import DeepMaterialRegressor

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
    plt.scatter(y_true, y_pred, alpha=0.6, c='royalblue', edgecolors='w', s=70)
    
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
    sns.histplot(errors, kde=True, color='crimson', bins=20, alpha=0.6)
    
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    plt.xlabel(f"Prediction Error ({unit})", fontsize=12, fontweight='bold')
    plt.ylabel("Count", fontsize=12, fontweight='bold')
    plt.title(f"{target_name}: Error Distribution", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def evaluate(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Prepare Data
    logger.info("Loading Test Data...")
    _, _, _, _, X_test, y_test, _ = prepare_data(args.data_path)
    
    # 2. Load Model
    model = DeepMaterialRegressor(input_dim=X_test.shape[1], output_dim=3).to(device)
    model_path = os.path.join(args.model_dir, "best_deep_regressor.pt")
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info("Model loaded.")

    # 3. Predict
    with torch.no_grad():
        inputs = X_test.to(device)
        preds = model(inputs).cpu().numpy()
        targets = y_test.numpy()

    # 4. Analyze & Plot
    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    print("\n" + "="*50)
    print("FINAL MODEL EVALUATION REPORT")
    print("="*50 + "\n")

    for i, (name, unit) in enumerate(zip(TARGETS, UNITS)):
        y_t = targets[:, i]
        y_p = preds[:, i]
        
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
        scatter_path = os.path.join(args.output_dir, f"scatter_{sanitized_name}.png")
        error_path = os.path.join(args.output_dir, f"error_dist_{sanitized_name}.png")
        
        plot_scatter(y_t, y_p, name, unit, scatter_path)
        plot_error_dist(y_t, y_p, name, unit, error_path)
        
        logger.info(f"Saved plots for {name} to {args.output_dir}")

    # Print Table
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("\n" + "-"*50)
    print(f"Evaluation complete. Screenshots saved to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Material Predictor")
    parser.add_argument("--data_path", type=str, default="../../data/mock_materials.json")
    parser.add_argument("--model_dir", type=str, default="../../models")
    parser.add_argument("--output_dir", type=str, default="../../docs/screenshots")
    parser.add_argument("--seed", type=int, default=42)
    
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
