import numpy as np
import torch
from pymatgen.core import Structure
from typing import List, Dict, Tuple

class GaussianDistance:
    def __init__(self, dmin: float = 0.0, dmax: float = 8.0, step: float = 0.2, var: float = None):
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances: np.ndarray) -> np.ndarray:
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 / self.var**2)

class CrystalGraphBuilder:
    def __init__(self, radius: float = 8.0, max_neighbors: int = 12, dStep: float = 0.2):
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.gdf = GaussianDistance(dmin=0, dmax=radius, step=dStep)
        
    def get_node_features(self, structure: Structure) -> torch.Tensor:
        features = []
        for site in structure:
            el = site.specie
            X = el.X if el.X else 0.0 
            features.append([
                float(el.Z),
                float(el.group),
                float(el.row),
                float(X)
            ])
        return torch.tensor(features, dtype=torch.float32)

    def get_graph(self, structure: Structure) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        atom_feas = self.get_node_features(structure)
        all_neighbors = structure.get_all_neighbors(self.radius, include_index=True)
        
        all_nbrs = []
        for k, neighbors in enumerate(all_neighbors):
            neighbors = sorted(neighbors, key=lambda x: x[1])
            if len(neighbors) > self.max_neighbors:
                neighbors = neighbors[:self.max_neighbors]
            for nbr in neighbors:
                distance = nbr[1]
                nbr_index = nbr[2]
                all_nbrs.append([k, nbr_index, distance])

        if len(all_nbrs) == 0:
            nbr_feas = torch.zeros(0, len(self.gdf.filter))
            nbr_idx = torch.zeros(0, 2).long()
            return atom_feas, nbr_feas, nbr_idx

        all_nbrs = np.array(all_nbrs)
        distances = all_nbrs[:, 2]
        nbr_feas_np = self.gdf.expand(distances)
        nbr_feas = torch.tensor(nbr_feas_np, dtype=torch.float32)
        nbr_idx = torch.LongTensor(all_nbrs[:, :2])
        
        return atom_feas, nbr_feas, nbr_idx
