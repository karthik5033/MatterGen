import os
import argparse
import logging
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from training.datasets.loader import MaterialDataLoader
from training.datasets.graph_builder import CrystalGraphBuilder
from training.datasets.graph_dataset import CrystalGraphDataset, collate_batch
from training.models.cgcnn import CGCNN
from training.scripts.train_robust import set_seed

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_embeddings(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    logger.info("Loading Data...")
    loader = MaterialDataLoader(args.data_path)
    raw_data = loader.load()
    
    # 2. Build Dataset (No splitting here, we might want embeddings for EVERYTHING or let user filter)
    # For this script, we'll extract for the ENTIRE dataset to enable a full material map.
    builder = CrystalGraphBuilder(radius=8.0, dStep=0.2)
    # cache_graphs=False to save memory if dataset is huge, or True if small. 
    # With mock data, True is fine.
    dataset = CrystalGraphDataset(raw_data, builder, cache_graphs=True)
    
    graph_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_batch, shuffle=False)
    
    # 3. Load Model
    model = CGCNN(node_input_dim=4, edge_input_dim=41, node_hidden_dim=64, n_conv_layers=3, n_targets=3).to(device)
    model_path = os.path.join(args.model_dir, "cgcnn_best.pt")
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("Model loaded.")

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
            
            # IDs and Formulas
            batch_ids = batch['ids']
            # We can't access 'formula' directly from batch dict if we didn't put it in collate?
            # collate returns 'ids', but not formula explicitly in the return dict unless we added it?
            # Let's check graph_dataset.py.
            # ... collate_batch returns "ids" list. Dataset __getitem__ returns "formula".
            # Oops, collate_batch logic in graph_dataset.py calculates 'ids' but doesn't explicitly pass through 'formula'.
            # However, we can recover formula from raw_data using the ID if needed, OR we can rely on ID being enough.
            # Actually, let's just assume we need IDs for now.
            
            # Extract
            # n_graphs is batch_ids length
            n_graphs = len(batch_ids)
            
            crys_emb = model.get_crystal_embedding(atom_fea, nbr_fea, nbr_idx, batch_map, n_graphs)
            all_embeddings.append(crys_emb.cpu().numpy())
            
            for i, mid in enumerate(batch_ids):
                # Find the original entry to get formula/targets for metadata
                # This is O(N) lookup/scan if not careful. 
                # Ideally dataset[i] corresponds to raw_data[i] if shuffle=False.
                # Since shuffle=False in DataLoader, we can just use the index tracker?
                # No, batches. 
                # Let's just store the ID.
                metadata.append({"id": mid})

    # Concatenate
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    # Enhance Metadata with formulas/targets (Post-process)
    # Since we iterated sequentially with shuffle=False, metadata indices align with dataset indices
    # We can just iterate dataset again or use raw_data if 1:1
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
    
    np.save(os.path.join(args.output_dir, "cgcnn_embeddings.npy"), all_embeddings)
    
    with open(os.path.join(args.output_dir, "cgcnn_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    logger.info(f"Saved artifacts to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CGCNN Embeddings")
    parser.add_argument("--data_path", type=str, default="../../data/mock_materials.json")
    parser.add_argument("--model_dir", type=str, default="../../models")
    parser.add_argument("--output_dir", type=str, default="../../models/embeddings")
    parser.add_argument("--batch_size", type=int, default=64)
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
        
    generate_embeddings(args)
