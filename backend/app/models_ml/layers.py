import torch
import torch.nn as nn
import torch.nn.functional as F

class CGCNNConv(nn.Module):
    """
    Crystal Graph Convolutional Neural Network (CGCNN) Layer.
    """

    def __init__(self, node_dim: int, edge_dim: int):
        super(CGCNNConv, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        # The input to the linear layers is the concatenation of:
        # Atom_i (node_dim) + Atom_j (node_dim) + Bond_ij (edge_dim)
        full_dim = 2 * node_dim + edge_dim
        
        # Soft Gating Filter (Sigmoid)
        self.fc_filter = nn.Linear(full_dim, node_dim)
        self.bn_filter = nn.BatchNorm1d(node_dim)
        
        # Core Signal
        self.fc_core = nn.Linear(full_dim, node_dim)
        self.bn_core = nn.BatchNorm1d(node_dim)

    def forward(self, atom_features: torch.Tensor, edge_features: torch.Tensor, edge_indices: torch.Tensor) -> torch.Tensor:
        # 1. Gather Features
        src_idx = edge_indices[:, 0]
        dst_idx = edge_indices[:, 1]
        
        atom_i = atom_features[dst_idx] # Target
        atom_j = atom_features[src_idx] # Source
        
        # 2. Concatenate [vi, vj, u_ij]
        z_ij = torch.cat([atom_i, atom_j, edge_features], dim=1)
        
        # 3. Compute Signal and Filter
        signal = self.fc_core(z_ij)
        signal = self.bn_core(signal)
        signal = F.softplus(signal) # Non-linearity g(z)
        
        # Filter (Gate)
        gate = self.fc_filter(z_ij)
        gate = self.bn_filter(gate)
        gate = torch.sigmoid(gate) # Sigmoid sigma(z)
        
        # 4. Message = Gate * Signal
        message = gate * signal
        
        # 5. Aggregate Messages (Sum pooling)
        nbr_sumed = torch.zeros_like(atom_features)
        nbr_sumed.index_add_(0, dst_idx, message)
        
        # 6. Update Step (Residual Connection)
        out = atom_features + nbr_sumed
        
        return out
