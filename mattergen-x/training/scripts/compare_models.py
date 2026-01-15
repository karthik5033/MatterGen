import os
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from training.datasets.loader import MaterialDataLoader
from training.datasets.featurizer import ChemicalFeaturizer
from training.datasets.graph_builder import CrystalGraphBuilder
from training.datasets.graph_dataset import CrystalGraphDataset, collate_batch
from torch.utils.data import DataLoader

from training.models.deep_regressor import DeepMaterialRegressor
from training.models.cgcnn import CGCNN
from training.scripts.train_robust import set_seed

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGETS = ["Formation Energy", "Band Gap", "Density"]

def evaluate_mlp(model, X_test, y_test, scaler, device):
    """Evaluate MLP model."""
    model.eval()
    
    # Scale Input
    X_scaled = scaler.transform(X_test)
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()
        
    # y_test is already raw values
    mae_per_target = []
    for i in range(3):
        mae = mean_absolute_error(y_test[:, i], preds[:, i])
        mae_per_target.append(mae)
        
    return np.array(mae_per_target)

def evaluate_cgcnn(model, raw_data, indices, device, scaler_mean, scaler_scale):
    """Evaluate CGCNN model."""
    model.eval()
    
    builder = CrystalGraphBuilder(radius=8.0, dStep=0.2)
    # Subset of data
    test_data = [raw_data[i] for i in indices]
    dataset = CrystalGraphDataset(test_data, builder, cache_graphs=False)
    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_batch, shuffle=False)
    
    preds_all = []
    targets_all = []
    
    with torch.no_grad():
        for batch in loader:
            atom_fea = batch['atom_fea'].to(device)
            nbr_fea = batch['nbr_fea'].to(device)
            nbr_idx = batch['nbr_idx'].to(device)
            batch_map = batch['batch'].to(device)
            targets = batch['target'] # Keep CPU for metrics
            
            # Predict
            preds_norm = model(atom_fea, nbr_fea, nbr_idx, batch_map)
            
            # Un-normalize
            preds = preds_norm * scaler_scale + scaler_mean
            
            preds_all.append(preds.cpu())
            targets_all.append(targets)
            
    preds_all = torch.cat(preds_all, dim=0).numpy()
    targets_all = torch.cat(targets_all, dim=0).numpy()
    
    mae_per_target = []
    for i in range(3):
        mae = mean_absolute_error(targets_all[:, i], preds_all[:, i])
        mae_per_target.append(mae)
        
    return np.array(mae_per_target)

def compare(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Common Data
    logger.info("Loading Data...")
    loader = MaterialDataLoader(args.data_path)
    raw_data = loader.load()
    
    # Filter for intersection of validity (Structure AND Formula)
    valid_indices = []
    featurizer = ChemicalFeaturizer()
    
    X_mlp = []
    y_raw = []
    
    for i, sample in enumerate(raw_data):
        formula = sample.get("formula")
        struct = sample.get("structure_obj")
        targets = [
            sample.get("label_formation_energy"),
            sample.get("label_band_gap"),
            sample.get("label_density")
        ]
        
        # Must have formula, structure, chemicals
        if formula and struct and all(t is not None for t in targets):
            try:
                # Check featurizer works
                feat = featurizer.featurize_formula(formula)
                X_mlp.append(feat)
                y_raw.append(targets)
                valid_indices.append(i)
            except:
                continue
                
    X_mlp = np.array(X_mlp)
    y_raw = np.array(y_raw)
    valid_indices = np.array(valid_indices)
    
    logger.info(f"Valid overlapping samples: {len(valid_indices)}")
    
    # 2. Split (Align with training split logic)
    # We split the *indices* of the valid subset to ensure we evaluate on a holdout set
    # Note: This might theoretically slightly differ from training sets if filtering was different,
    # but provides a fair side-by-side comparison on THIS set.
    train_idx_loc, test_idx_loc = train_test_split(np.arange(len(valid_indices)), test_size=0.2, random_state=42)
    
    # MLP Data
    X_train = X_mlp[train_idx_loc]
    X_test = X_mlp[test_idx_loc]
    y_test = y_raw[test_idx_loc]
    
    # CGCNN Data (Indices into raw_data)
    test_raw_indices = valid_indices[test_idx_loc]
    
    # 3. Load MLP
    logger.info("Evaluating MLP...")
    mlp_scaler = StandardScaler()
    mlp_scaler.fit(X_train) # Fit on test-run train split
    
    mlp_model = DeepMaterialRegressor(input_dim=20, output_dim=3).to(device)
    mlp_path = os.path.join(args.model_dir, "best_deep_regressor.pt")
    if os.path.exists(mlp_path):
        mlp_model.load_state_dict(torch.load(mlp_path, map_location=device))
        mlp_mae = evaluate_mlp(mlp_model, X_test, y_test, mlp_scaler, device)
    else:
        logger.warning("MLP model not found. Filling with NaNs.")
        mlp_mae = np.array([np.nan]*3)

    # 4. Load CGCNN
    logger.info("Evaluating CGCNN...")
    cgcnn_model = CGCNN(node_input_dim=4, edge_input_dim=41, node_hidden_dim=64, n_conv_layers=3, n_targets=3).to(device)
    cgcnn_path = os.path.join(args.model_dir, "cgcnn_best.pt")
    
    if os.path.exists(cgcnn_path):
        checkpoint = torch.load(cgcnn_path, map_location=device)
        cgcnn_model.load_state_dict(checkpoint['model_state_dict'])
        scaler_mean = checkpoint['scaler_mean'] # Tensor
        scaler_scale = checkpoint['scaler_scale'] # Tensor
        
        cgcnn_mae = evaluate_cgcnn(cgcnn_model, raw_data, test_raw_indices, device, scaler_mean, scaler_scale)
    else:
        logger.warning("CGCNN model not found. Filling with NaNs.")
        cgcnn_mae = np.array([np.nan]*3)

    # 5. Report
    print("\n" + "="*60)
    print(f"{'Property':<20} | {'MLP MAE':<10} | {'CGCNN MAE':<10} | {'Improvement':<10}")
    print("-" * 60)
    
    for i, target in enumerate(TARGETS):
        m_mae = mlp_mae[i]
        c_mae = cgcnn_mae[i]
        
        if not np.isnan(m_mae) and not np.isnan(c_mae):
            imp = ((m_mae - c_mae) / m_mae) * 100
            print(f"{target:<20} | {m_mae:.4f}     | {c_mae:.4f}       | {imp:+.1f}%")
        else:
            print(f"{target:<20} | {m_mae:.4f}     | {c_mae:.4f}       | N/A")
            
    print("=" * 60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare MLP and CGCNN Models")
    parser.add_argument("--data_path", type=str, default="../../data/mock_materials.json")
    parser.add_argument("--model_dir", type=str, default="../../models")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Path handling
    base_dir = os.path.dirname(__file__)
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(base_dir, args.data_path)
    if not os.path.isabs(args.model_dir):
        args.model_dir = os.path.join(base_dir, args.model_dir)

    compare(args)
