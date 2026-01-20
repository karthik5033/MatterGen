"""
CGCNN Model Metrics Report Generator
=====================================
Generates comprehensive evaluation metrics for the CGCNN model including:
- Regression metrics (MAE, RMSE, R², MAPE)
- Binned confusion matrix for band gap classification
- Per-property analysis
- Saves results to JSON and generates visualizations
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from training.datasets.loader import MaterialDataLoader
from training.datasets.graph_dataset import CrystalGraphDataset, collate_batch
from training.datasets.graph_builder import CrystalGraphBuilder
from training.models.cgcnn import CGCNN

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TARGETS = ["Formation Energy", "Band Gap", "Density"]
UNITS = ["eV/atom", "eV", "g/cm³"]

# Band gap classification bins
BAND_GAP_BINS = {
    "Metal": (0.0, 0.1),
    "Narrow Gap Semi": (0.1, 1.0),
    "Wide Gap Semi": (1.0, 3.0),
    "Insulator": (3.0, float('inf'))
}


def classify_band_gap(value: float) -> str:
    """Classify band gap value into material category."""
    for name, (low, high) in BAND_GAP_BINS.items():
        if low <= value < high:
            return name
    return "Insulator"  # Fallback


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute comprehensive regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error (handle zeros)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float('nan')
    
    # Max error
    max_error = np.max(np.abs(y_true - y_pred))
    
    # Median Absolute Error
    median_ae = np.median(np.abs(y_true - y_pred))
    
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE": float(mape),
        "Max_Error": float(max_error),
        "Median_AE": float(median_ae),
        "N_Samples": int(len(y_true))
    }


def generate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_dir: str) -> dict:
    """
    Generate confusion matrix for band gap classification.
    Converts continuous band gap predictions to categorical classes.
    """
    # Convert to categories
    true_classes = [classify_band_gap(v) for v in y_true]
    pred_classes = [classify_band_gap(v) for v in y_pred]
    
    class_names = list(BAND_GAP_BINS.keys())
    
    # Compute confusion matrix
    cm = confusion_matrix(true_classes, pred_classes, labels=class_names)
    
    # Compute classification metrics
    accuracy = accuracy_score(true_classes, pred_classes)
    
    # Per-class metrics
    precision = precision_score(true_classes, pred_classes, labels=class_names, average=None, zero_division=0)
    recall = recall_score(true_classes, pred_classes, labels=class_names, average=None, zero_division=0)
    f1 = f1_score(true_classes, pred_classes, labels=class_names, average=None, zero_division=0)
    
    # Macro averages
    macro_precision = precision_score(true_classes, pred_classes, labels=class_names, average='macro', zero_division=0)
    macro_recall = recall_score(true_classes, pred_classes, labels=class_names, average='macro', zero_division=0)
    macro_f1 = f1_score(true_classes, pred_classes, labels=class_names, average='macro', zero_division=0)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
    plt.ylabel('True Class', fontsize=12, fontweight='bold')
    plt.title('Band Gap Classification Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, "confusion_matrix_band_gap.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix to {cm_path}")
    
    # Normalized confusion matrix
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2%', 
        cmap='Greens',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage'}
    )
    plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
    plt.ylabel('True Class', fontsize=12, fontweight='bold')
    plt.title('Band Gap Classification Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    cm_norm_path = os.path.join(output_dir, "confusion_matrix_band_gap_normalized.png")
    plt.savefig(cm_norm_path, dpi=300)
    plt.close()
    
    return {
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "accuracy": float(accuracy),
        "per_class_precision": {n: float(p) for n, p in zip(class_names, precision)},
        "per_class_recall": {n: float(r) for n, r in zip(class_names, recall)},
        "per_class_f1": {n: float(f) for n, f in zip(class_names, f1)},
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1)
    }


def plot_parity(y_true: np.ndarray, y_pred: np.ndarray, name: str, unit: str, output_dir: str):
    """Generate parity plot (True vs Predicted)."""
    plt.figure(figsize=(8, 8))
    
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    buffer = (max_val - min_val) * 0.05
    
    plt.plot([min_val - buffer, max_val + buffer], 
             [min_val - buffer, max_val + buffer], 
             'k--', alpha=0.5, linewidth=2, label='Ideal (y=x)')
    
    plt.scatter(y_true, y_pred, alpha=0.5, c='royalblue', edgecolors='w', s=30)
    
    plt.xlabel(f"True {name} ({unit})", fontsize=12, fontweight='bold')
    plt.ylabel(f"Predicted {name} ({unit})", fontsize=12, fontweight='bold')
    plt.title(f"{name}: Parity Plot", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='upper left')
    
    # Add metrics annotation
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    metrics_text = f"MAE: {mae:.4f}\nR²: {r2:.4f}"
    plt.annotate(metrics_text, xy=(0.95, 0.05), xycoords='axes fraction', 
                 ha='right', va='bottom',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
                 fontsize=10)
    
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    path = os.path.join(output_dir, f"parity_{safe_name}.png")
    plt.savefig(path, dpi=300)
    plt.close()
    logger.info(f"Saved parity plot for {name}")


def run_evaluation(args):
    """Main evaluation function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Data
    logger.info(f"Loading data from {args.data_path}...")
    loader = MaterialDataLoader(args.data_path)
    data = loader.load()
    
    if not data:
        logger.error("No data loaded. Check data path.")
        return None
    
    logger.info(f"Loaded {len(data)} samples")
    
    # Test split (last 20%)
    test_size = int(len(data) * 0.2)
    test_data = data[-test_size:]
    logger.info(f"Test set size: {len(test_data)} samples")
    
    # 2. Build Dataset using correct API
    logger.info("Building graph dataset...")
    builder = CrystalGraphBuilder(radius=8.0, max_neighbors=12, dStep=0.2)
    
    dataset = CrystalGraphDataset(
        data_list=test_data,
        builder=builder,
        targets=["label_formation_energy", "label_band_gap", "label_density"],
        cache_graphs=True
    )
    
    test_loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_batch
    )
    
    # 3. Load Model
    checkpoint_path = os.path.join(args.model_dir, "cgcnn_best.pt")
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    logger.info(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model with correct architecture
    model = CGCNN(
        node_input_dim=4,
        edge_input_dim=41,  # Gaussian expansion: (8.0 / 0.2) + 1 = 41
        node_hidden_dim=64,
        n_conv_layers=3,
        n_targets=3
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    scaler_mean = checkpoint['scaler_mean'].to(device)
    scaler_scale = checkpoint['scaler_scale'].to(device)
    model.eval()
    
    # 4. Run Inference
    logger.info("Running inference...")
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            atom_fea = batch["atom_fea"].to(device)
            nbr_fea = batch["nbr_fea"].to(device)
            nbr_idx = batch["nbr_idx"].to(device)
            batch_map = batch["batch"].to(device)
            target = batch["target"]
            
            preds_norm = model(atom_fea, nbr_fea, nbr_idx, batch_map)
            preds = preds_norm * scaler_scale + scaler_mean
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(target.numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    logger.info(f"Predictions shape: {all_preds.shape}, Targets shape: {all_targets.shape}")
    
    # 5. Compute Metrics
    logger.info("Computing metrics...")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_checkpoint": checkpoint_path,
        "test_samples": len(test_data),
        "device": str(device),
        "regression_metrics": {},
        "classification_metrics": None
    }
    
    for i, (name, unit) in enumerate(zip(TARGETS, UNITS)):
        y_true = all_targets[:, i]
        y_pred = all_preds[:, i]
        
        # Regression metrics
        metrics = compute_regression_metrics(y_true, y_pred)
        metrics["unit"] = unit
        results["regression_metrics"][name] = metrics
        
        # Parity plot
        plot_parity(y_true, y_pred, name, unit, args.output_dir)
        
        logger.info(f"{name}: MAE={metrics['MAE']:.4f}, R²={metrics['R2']:.4f}")
    
    # 6. Band Gap Confusion Matrix
    logger.info("Generating band gap confusion matrix...")
    bg_true = all_targets[:, 1]  # Band gap is index 1
    bg_pred = all_preds[:, 1]
    
    classification_results = generate_confusion_matrix(bg_true, bg_pred, args.output_dir)
    results["classification_metrics"] = classification_results
    
    # 7. Save Results
    output_file = os.path.join(args.output_dir, "cgcnn_metrics_report.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved metrics report to {output_file}")
    
    # 8. Print Summary
    print("\n" + "=" * 60)
    print("CGCNN MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"Timestamp: {results['timestamp']}")
    print(f"Test Samples: {results['test_samples']}")
    print("-" * 60)
    
    print("\n📊 REGRESSION METRICS:")
    print("-" * 60)
    for prop, metrics in results["regression_metrics"].items():
        print(f"\n  {prop} ({metrics['unit']}):")
        print(f"    MAE:       {metrics['MAE']:.4f}")
        print(f"    RMSE:      {metrics['RMSE']:.4f}")
        print(f"    R²:        {metrics['R2']:.4f}")
        print(f"    Median AE: {metrics['Median_AE']:.4f}")
    
    print("\n\n📋 BAND GAP CLASSIFICATION METRICS:")
    print("-" * 60)
    cm = results["classification_metrics"]
    print(f"  Overall Accuracy:  {cm['accuracy']:.4f} ({cm['accuracy']*100:.1f}%)")
    print(f"  Macro F1-Score:    {cm['macro_f1']:.4f}")
    print(f"  Macro Precision:   {cm['macro_precision']:.4f}")
    print(f"  Macro Recall:      {cm['macro_recall']:.4f}")
    
    print("\n  Per-Class F1 Scores:")
    for cls_name, f1_val in cm["per_class_f1"].items():
        print(f"    {cls_name:20s}: {f1_val:.4f}")
    
    print("\n" + "=" * 60)
    print(f"Full report saved to: {output_file}")
    print("=" * 60 + "\n")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate CGCNN Metrics Report")
    parser.add_argument("--data_path", type=str, default="../../data/synthetic_materials.json")
    parser.add_argument("--model_dir", type=str, default="../../models")
    parser.add_argument("--output_dir", type=str, default="../../docs/metrics")
    parser.add_argument("--batch_size", type=int, default=128)
    
    args = parser.parse_args()
    
    # Resolve paths
    base_dir = os.path.dirname(__file__)
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(base_dir, args.data_path)
    if not os.path.isabs(args.model_dir):
        args.model_dir = os.path.join(base_dir, args.model_dir)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(base_dir, args.output_dir)
    
    run_evaluation(args)
