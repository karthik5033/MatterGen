import os
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
try:
    import umap
except ImportError:
    umap = None

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MaterialMapGenerator:
    """
    Generates a 2D map of the material latent space.
    
    Uses UMAP (Uniform Manifold Approximation and Projection) to project 
    high-dimensional CGCNN embeddings (e.g., 64-dim) into 2D coordinates.
    Also computes nearest neighbors for each point to enabling browsing in UI.
    
    If UMAP is not installed, falls back to t-SNE.
    """
    
    def __init__(self, embeddings_path: str, metadata_path: str):
        self.embeddings_path = embeddings_path
        self.metadata_path = metadata_path
        
        # Load Data
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Embeddings not found at {embeddings_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
            
        self.embeddings = np.load(embeddings_path)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        if len(self.embeddings) != len(self.metadata):
            logger.warning(f"Mismatch: {len(self.embeddings)} embeddings vs {len(self.metadata)} metadata entries.")
            # Truncate to minimum
            min_len = min(len(self.embeddings), len(self.metadata))
            self.embeddings = self.embeddings[:min_len]
            self.metadata = self.metadata[:min_len]

    def generate_map(self, n_neighbors_umap: int = 15, min_dist: float = 0.1, n_neighbors_search: int = 5) -> Dict:
        """
        Run dimensionality reduction and neighbor search.
        
        Returns:
            Dict containing 'points' list with x, y, id, formula, and 'neighbors'.
        """
        logger.info(f"Generating Map for {len(self.embeddings)} materials...")
        
        # 1. Dimensionality Reduction (UMAP / t-SNE)
        if umap:
            logger.info("Using UMAP for projection...")
            reducer = umap.UMAP(n_neighbors=n_neighbors_umap, min_dist=min_dist, n_components=2, random_state=42)
            coords = reducer.fit_transform(self.embeddings)
        else:
            logger.info("UMAP not found. Using t-SNE fallback...")
            # Use PCA init for stability
            tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
            coords = tsne.fit_transform(self.embeddings)
            
        # 2. Nearest Neighbors Search (in High-Dim space)
        # We calculate neighbors in the ORIGINAL 64-dim space for physical similarity,
        # even though we visualize in 2D.
        logger.info("Computing Nearest Neighbors...")
        nbrs = NearestNeighbors(n_neighbors=n_neighbors_search + 1, algorithm='ball_tree').fit(self.embeddings)
        distances, indices = nbrs.kneighbors(self.embeddings)
        
        # 3. Construct Result
        map_data = []
        for i in range(len(self.embeddings)):
            # Get Neighbors (exclude self, which is index 0)
            neighbor_indices = indices[i][1:]
            neighbor_ids = [self.metadata[idx]["id"] for idx in neighbor_indices]
            
            entry = {
                "id": self.metadata[i]["id"],
                "formula": self.metadata[i].get("formula", "Unknown"),
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "neighbors": neighbor_ids,
                "targets": self.metadata[i].get("targets", {})
            }
            map_data.append(entry)
            
        return {"points": map_data}

    def save_map(self, output_path: str, data: Dict):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved Material Map to {output_path}")

if __name__ == "__main__":
    # Test Block
    print("Testing Material Map Generator...")
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dir", default="../../models/embeddings")
    parser.add_argument("--out_path", default="../../models/embeddings/material_map.json")
    args = parser.parse_args()
    
    # Resolving Paths
    base_dir = os.path.dirname(__file__)
    if not os.path.isabs(args.emb_dir):
        args.emb_dir = os.path.join(base_dir, args.emb_dir)
    if not os.path.isabs(args.out_path):
        args.out_path = os.path.join(base_dir, args.out_path)
        
    emb_file = os.path.join(args.emb_dir, "cgcnn_embeddings.npy")
    meta_file = os.path.join(args.emb_dir, "cgcnn_metadata.json")
    
    if os.path.exists(emb_file) and os.path.exists(meta_file):
        generator = MaterialMapGenerator(emb_file, meta_file)
        result = generator.generate_map()
        generator.save_map(args.out_path, result)
        print(f"Done. Map has {len(result['points'])} points.")
    else:
        print(f"Skipping: Embeddings not found at {args.emb_dir}")
