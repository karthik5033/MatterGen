import os
import sys
import argparse
import logging
import warnings
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Suppress warnings
warnings.filterwarnings("ignore")

# Add project root to sys.path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from training.datasets.loader import MaterialDataLoader
from training.datasets.graph_builder import CrystalGraphBuilder
from training.datasets.graph_dataset import CrystalGraphDataset, collate_batch
from training.models.cgcnn import CGCNN

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def generate_embeddings(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Device: {device}")
    
    # 1. Load Data
    logger.info(f"Loading Data from {args.data_path}...")
    loader = MaterialDataLoader(args.data_path)
    raw_data = loader.load()
    
    if not raw_data:
        logger.error("No data loaded!")
        return

    # 2. Build Dataset
    logger.info("Building Graph Dataset...")
    builder = CrystalGraphBuilder(radius=8.0, dStep=0.2)
    dataset = CrystalGraphDataset(raw_data, builder, cache_graphs=True)
    
    graph_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_batch, shuffle=False)
    
    # 3. Load Model
    model = CGCNN(node_input_dim=4, edge_input_dim=41, node_hidden_dim=64, n_conv_layers=3, n_targets=3).to(device)
    model_path = os.path.join(args.model_dir, "cgcnn_best.pt")
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return

    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 4. Extract Embeddings
    all_embeddings = []
    metadata = []
    
    logger.info("Extracting embeddings...")
    with torch.no_grad():
        for batch in graph_loader:
            atom_fea = batch['atom_fea'].to(device)
            nbr_fea = batch['nbr_fea'].to(device)
            nbr_idx = batch['nbr_idx'].to(device)
            batch_map = batch['batch'].to(device)
            batch_ids = batch['ids']
            
            # Extract
            n_graphs = len(batch_ids)
            crys_emb = model.get_crystal_embedding(atom_fea, nbr_fea, nbr_idx, batch_map, n_graphs)
            all_embeddings.append(crys_emb.cpu().numpy())
            
            for mid in batch_ids:
                metadata.append({"id": mid})

    # Concatenate
    if len(all_embeddings) == 0:
        logger.error("No embeddings generated.")
        return

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    # Enhance Metadata
    logger.info("Attaching metadata...")
    for i, meta in enumerate(metadata):
        entry = dataset.data_list[i]
        meta["formula"] = entry.get("formula")
        meta["targets"] = {
            "formation_energy": entry.get("label_formation_energy"),
            "band_gap": entry.get("label_band_gap"),
            "density": entry.get("label_density")
        }

    logger.info(f"Extracted {len(all_embeddings)} embeddings (Dim: {all_embeddings.shape[1]})")

    # 5. Save
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save as JSON (Frontend friendly)
    output_json_path = os.path.join(args.output_dir, "material_embeddings.json")
    
    # We want a list of objects: {id, formula, embedding: [x, y, ...], targets: {...}}
    final_output = []
    for i, meta in enumerate(metadata):
        final_output.append({
            "id": meta["id"],
            "formula": meta["formula"],
            "embedding": all_embeddings[i].tolist(), # Convert numpy array to list for JSON
            "properties": meta["targets"]
        })
        
    with open(output_json_path, "w") as f:
        json.dump(final_output, f, indent=None) # Compact JSON to save space
        
    logger.info(f"Saved material map to {output_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CGCNN Embeddings")
    parser.add_argument("--data_path", type=str, default="../../data/datasets/mp20_clean.json")
    parser.add_argument("--model_dir", type=str, default="../../models")
    parser.add_argument("--output_dir", type=str, default="../../data") # Save directly to data so frontend/backend can reach it
    parser.add_argument("--batch_size", type=int, default=64)
    
    args = parser.parse_args()
    
    # Path handling
    base_dir = os.path.dirname(__file__)
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(base_dir, args.data_path)
    if not os.path.isabs(args.model_dir):
        args.model_dir = os.path.join(base_dir, args.model_dir)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(base_dir, args.output_dir)
        
    generate_embeddings(args)
