import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .layers import CGCNNConv
from typing import Tuple

class CGCNN(nn.Module):
    """
    Crystal Graph Convolutional Neural Network (CGCNN).
    """
    
    def __init__(self, 
                 node_input_dim: int = 4, 
                 edge_input_dim: int = 41,
                 node_hidden_dim: int = 64,
                 n_conv_layers: int = 3,
                 n_targets: int = 3):
        super(CGCNN, self).__init__()
        
        # 1. Embedding
        self.embedding = nn.Linear(node_input_dim, node_hidden_dim)
        
        # 2. Convolutions
        self.convs = nn.ModuleList([
            CGCNNConv(node_hidden_dim, edge_input_dim) 
            for _ in range(n_conv_layers)
        ])
        
        # 3. Output Head
        self.fc_out = nn.Sequential(
            nn.Linear(node_hidden_dim, node_hidden_dim),
            nn.BatchNorm1d(node_hidden_dim),
            nn.Softplus(),
            nn.Linear(node_hidden_dim, n_targets)
        )
        
        self.pooling = 'mean'
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def pooling_layer(self, atom_features: torch.Tensor, batch_indices: torch.Tensor, n_graphs: int) -> torch.Tensor:
        device = atom_features.device
        crystal_features = torch.zeros(n_graphs, atom_features.size(1), device=device)
        
        # Sum pooling
        crystal_features.index_add_(0, batch_indices, atom_features)
        
        if self.pooling == 'mean':
            counts = torch.zeros(n_graphs, device=device)
            ones = torch.ones(atom_features.size(0), device=device)
            counts.index_add_(0, batch_indices, ones)
            counts = torch.clamp(counts, min=1.0)
            crystal_features = crystal_features / counts.unsqueeze(1)
            
        return crystal_features

    def get_crystal_embedding(self, atom_fea, nbr_fea, nbr_idx, batch_mapping, n_graphs) -> torch.Tensor:
        x = self.embedding(atom_fea)
        for conv in self.convs:
            x = conv(x, nbr_fea, nbr_idx)
        crys_fea = self.pooling_layer(x, batch_mapping, n_graphs)
        return crys_fea

    def forward(self, atom_fea: torch.Tensor, nbr_fea: torch.Tensor, nbr_idx: torch.Tensor, batch_mapping: torch.Tensor, **kwargs) -> torch.Tensor:
        n_graphs = batch_mapping.max().item() + 1
        crys_fea = self.get_crystal_embedding(atom_fea, nbr_fea, nbr_idx, batch_mapping, n_graphs)
        out = self.fc_out(crys_fea)
        return out
