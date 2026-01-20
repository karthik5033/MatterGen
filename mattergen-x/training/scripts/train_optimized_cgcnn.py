"""
CGCNN Optimized Training Script
================================
Uses real Materials Project data from parquet files to achieve:
- Formation Energy R² > 0.90
- Band Gap R² > 0.70
- Band Gap MAE < 0.5 eV
- Classification Accuracy > 70%

Key Optimizations:
1. Real MP dataset (540k samples)
2. Larger architecture (128 hidden dim, 5 conv layers)
3. Learning rate scheduling with warmup
4. Early stopping with patience
5. Proper data normalization
6. Gradient clipping
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from tqdm import tqdm
from datetime import datetime
from pymatgen.core import Structure, Lattice

# Add project root
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from training.datasets.graph_builder import CrystalGraphBuilder
from training.models.cgcnn import CGCNN

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MPDataset(Dataset):
    """
    Dataset loader for Materials Project parquet files.
    Builds crystal graphs from atomic positions and lattice parameters.
    """
    
    def __init__(self, parquet_path: str, builder: CrystalGraphBuilder, 
                 max_samples: int = None, cache: bool = True):
        """
        Args:
            parquet_path: Path to .parquet file
            builder: CrystalGraphBuilder instance
            max_samples: Limit samples for faster debugging
            cache: Pre-compute all graphs
        """
        logger.info(f"Loading data from {parquet_path}...")
        self.df = pd.read_parquet(parquet_path)
        
        if max_samples:
            self.df = self.df.head(max_samples)
            
        logger.info(f"Loaded {len(self.df)} samples")
        
        self.builder = builder
        self.graphs = []
        self.targets = []
        self.target_masks = []
        self.valid_indices = []
        
        # Target columns
        # energy_above_hull is our stability proxy
        # dft_band_gap is the band gap
        # ml_bulk_modulus as third target (or we can compute density)
        
        if cache:
            self._build_all_graphs()
    
    def _build_structure(self, row) -> Structure:
        """Build pymatgen Structure from parquet row."""
        try:
            # Extract data - positions and cell are arrays of arrays in parquet
            positions = row['positions']
            cell = row['cell']
            atomic_numbers = row['atomic_numbers']
            
            # Handle positions - reshape from array of 1D arrays to 2D array
            if isinstance(positions, np.ndarray) and len(positions.shape) == 1:
                positions = np.vstack(positions)
            else:
                positions = np.array(positions)
            
            # Handle cell - reshape from array of 1D arrays to 3x3 matrix
            if isinstance(cell, np.ndarray) and len(cell.shape) == 1:
                cell = np.vstack(cell)
            else:
                cell = np.array(cell)
            
            # Convert atomic numbers to species
            from pymatgen.core.periodic_table import Element
            species = [Element.from_Z(int(z)).symbol for z in atomic_numbers]
            
            # Create lattice from cell vectors
            lattice = Lattice(cell)
            
            # Create structure (positions are in Cartesian)
            structure = Structure(
                lattice, 
                species, 
                positions, 
                coords_are_cartesian=True
            )
            return structure
        except Exception as e:
            return None
    
    def _build_all_graphs(self):
        """Pre-compute all graphs with progress bar."""
        logger.info("Building crystal graphs...")
        
        for idx in tqdm(range(len(self.df)), desc="Processing structures"):
            row = self.df.iloc[idx]
            
            # Build structure
            structure = self._build_structure(row)
            if structure is None:
                continue
                
            # Build graph
            try:
                graph = self.builder.get_graph(structure)
                if graph[0].shape[0] == 0:  # Empty graph
                    continue
            except Exception as e:
                continue
            
            # Extract targets
            # 1. Energy above hull (stability)
            e_above_hull = row.get('energy_above_hull', 0.0)
            e_mask = 1.0
            if pd.isna(e_above_hull):
                e_above_hull = 0.0
                e_mask = 0.0 # Should not happen based on inspection
                
            # 2. Band gap
            band_gap = row.get('dft_band_gap', 0.0)
            bg_mask = 1.0
            if pd.isna(band_gap):
                band_gap = 0.0
                bg_mask = 0.0 # CRITICAL: Ignore these 500k samples during BG training
                
            # 3. Bulk modulus
            bulk_mod = row.get('ml_bulk_modulus', 50.0)
            bm_mask = 1.0
            if pd.isna(bulk_mod):
                bulk_mod = 50.0
                bm_mask = 0.0
            
            self.graphs.append(graph)
            self.targets.append([e_above_hull, band_gap, bulk_mod])
            self.target_masks.append([e_mask, bg_mask, bm_mask])
            self.valid_indices.append(idx)
        
        self.targets = np.array(self.targets, dtype=np.float32)
        self.target_masks = np.array(self.target_masks, dtype=np.float32)
        logger.info(f"Successfully built {len(self.graphs)} graphs")
        
        # Compute target statistics (ONLY using masked valid data)
        # Avoid zero division
        masked_targets = np.ma.masked_array(self.targets, mask=(1 - self.target_masks))
        self.target_mean = masked_targets.mean(axis=0).data
        self.target_std = masked_targets.std(axis=0).data
        self.target_std = np.where(self.target_std < 1e-6, 1.0, self.target_std)
        
        logger.info(f"Target means (valid only): {self.target_mean}")
        logger.info(f"Target stds (valid only): {self.target_std}")
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        atom_fea, nbr_fea, nbr_idx = self.graphs[idx]
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        mask = torch.tensor(self.target_masks[idx], dtype=torch.float32)
        
        return {
            "atom_fea": atom_fea,
            "nbr_fea": nbr_fea,
            "nbr_idx": nbr_idx,
            "target": target,
            "mask": mask,
            "id": str(self.valid_indices[idx])
        }


def collate_fn(batch):
    """Collate graphs into batched tensors."""
    batch_atom_fea = []
    batch_nbr_fea = []
    batch_nbr_idx = []
    batch_target = []
    batch_mask = []
    batch_mapping = []
    
    base_idx = 0
    for i, sample in enumerate(batch):
        n_atoms = sample["atom_fea"].shape[0]
        
        batch_atom_fea.append(sample["atom_fea"])
        batch_nbr_fea.append(sample["nbr_fea"])
        batch_nbr_idx.append(sample["nbr_idx"] + base_idx)
        batch_target.append(sample["target"])
        batch_mask.append(sample["mask"])
        batch_mapping.append(torch.full((n_atoms,), i, dtype=torch.long))
        
        base_idx += n_atoms
    
    return {
        "atom_fea": torch.cat(batch_atom_fea, dim=0),
        "nbr_fea": torch.cat(batch_nbr_fea, dim=0),
        "nbr_idx": torch.cat(batch_nbr_idx, dim=0),
        "target": torch.stack(batch_target, dim=0),
        "mask": torch.stack(batch_mask, dim=0),
        "batch": torch.cat(batch_mapping, dim=0)
    }


class ImprovedCGCNN(nn.Module):
    """
    Enhanced CGCNN with:
    - Deeper architecture
    - Residual connections
    - Dropout for regularization
    - Layer normalization
    """
    
    def __init__(self, 
                 node_input_dim: int = 4,
                 edge_input_dim: int = 41,
                 hidden_dim: int = 128,
                 n_conv_layers: int = 5,
                 n_targets: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.embedding = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # Edge embedding
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU()
        )
        
        # Convolution layers with residual
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for _ in range(n_conv_layers):
            self.convs.append(
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim * 2),
                    nn.SiLU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.SiLU()
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # Output head
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_targets)
        )
        
        self.hidden_dim = hidden_dim
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, atom_fea, nbr_fea, nbr_idx, batch_mapping):
        # Embed nodes
        x = self.embedding(atom_fea)
        
        # Embed edges
        edge_feat = self.edge_embedding(nbr_fea)
        
        # Message passing
        for conv, ln in zip(self.convs, self.layer_norms):
            # Gather neighbor features
            src_idx = nbr_idx[:, 0]
            dst_idx = nbr_idx[:, 1]
            
            # Message: concat source node with edge feature
            messages = torch.cat([x[src_idx], edge_feat], dim=1)
            messages = conv(messages)
            
            # Aggregate messages to destination nodes
            aggr = torch.zeros_like(x)
            aggr.index_add_(0, dst_idx, messages)
            
            # Count edges per node for mean aggregation
            edge_count = torch.zeros(x.size(0), device=x.device)
            edge_count.index_add_(0, dst_idx, torch.ones(dst_idx.size(0), device=x.device))
            edge_count = edge_count.clamp(min=1.0)
            aggr = aggr / edge_count.unsqueeze(1)
            
            # Residual + LayerNorm
            x = ln(x + self.dropout(aggr))
        
        # Global pooling
        n_graphs = batch_mapping.max().item() + 1
        crystal_fea = torch.zeros(n_graphs, self.hidden_dim, device=x.device)
        crystal_fea.index_add_(0, batch_mapping, x)
        
        # Mean pool
        counts = torch.zeros(n_graphs, device=x.device)
        counts.index_add_(0, batch_mapping, torch.ones(x.size(0), device=x.device))
        counts = counts.clamp(min=1.0)
        crystal_fea = crystal_fea / counts.unsqueeze(1)
        
        # Predict
        return self.output(crystal_fea)

    def get_crystal_embedding(self, atom_fea, nbr_fea, nbr_idx, batch_mapping):
        """
        Extract crystal-level feature vector.
        """
        # Embed nodes
        x = self.embedding(atom_fea)
        
        # Embed edges
        edge_feat = self.edge_embedding(nbr_fea)
        
        # Message passing
        for conv, ln in zip(self.convs, self.layer_norms):
            # Gather neighbor features
            src_idx = nbr_idx[:, 0]
            dst_idx = nbr_idx[:, 1]
            
            # Message: concat source node with edge feature
            messages = torch.cat([x[src_idx], edge_feat], dim=1)
            messages = conv(messages)
            
            # Aggregate messages to destination nodes
            aggr = torch.zeros_like(x)
            aggr.index_add_(0, dst_idx, messages)
            
            # Count edges per node for mean aggregation
            edge_count = torch.zeros(x.size(0), device=x.device)
            edge_count.index_add_(0, dst_idx, torch.ones(dst_idx.size(0), device=x.device))
            edge_count = edge_count.clamp(min=1.0)
            aggr = aggr / edge_count.unsqueeze(1)
            
            # Residual + LayerNorm
            x = ln(x + self.dropout(aggr))
        
        # Global pooling
        n_graphs = batch_mapping.max().item() + 1
        crystal_fea = torch.zeros(n_graphs, self.hidden_dim, device=x.device)
        crystal_fea.index_add_(0, batch_mapping, x)
        
        # Mean pool
        counts = torch.zeros(n_graphs, device=x.device)
        counts.index_add_(0, batch_mapping, torch.ones(x.size(0), device=x.device))
        counts = counts.clamp(min=1.0)
        crystal_fea = crystal_fea / counts.unsqueeze(1)
        
        return crystal_fea


def masked_mse_loss(preds, targets, masks):
    """
    Compute MSE loss only for valid (masked) elements.
    Args:
        preds: (B, T)
        targets: (B, T)
        masks: (B, T), 1.0 if valid, 0.0 if missing
    """
    diff = preds - targets
    squared_diff = diff ** 2
    
    # Apply mask
    masked_diff = squared_diff * masks
    
    # Mean over valid elements only
    # Add epsilon to denominator to avoid div by zero if a batch has NO valid samples for a target
    loss = masked_diff.sum() / (masks.sum() + 1e-8)
    return loss


def train_epoch(model, loader, optimizer, scaler_mean, scaler_std, device, grad_clip=1.0):
    """Train for one epoch using Masked Loss."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        
        atom_fea = batch["atom_fea"].to(device)
        nbr_fea = batch["nbr_fea"].to(device)
        nbr_idx = batch["nbr_idx"].to(device)
        batch_map = batch["batch"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        
        # Normalize targets (valid ones will be correct, invalid ones will be ignored anyway)
        target_norm = (target - scaler_mean) / scaler_std
        
        # Forward
        pred = model(atom_fea, nbr_fea, nbr_idx, batch_map)
        
        # Masked Loss
        loss = masked_mse_loss(pred, target_norm, mask)
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def evaluate(model, loader, scaler_mean, scaler_std, device):
    """Evaluate model and return masked metrics."""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_masks = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            atom_fea = batch["atom_fea"].to(device)
            nbr_fea = batch["nbr_fea"].to(device)
            nbr_idx = batch["nbr_idx"].to(device)
            batch_map = batch["batch"].to(device)
            target = batch["target"]
            mask = batch["mask"]
            
            pred_norm = model(atom_fea, nbr_fea, nbr_idx, batch_map)
            pred = pred_norm * scaler_std + scaler_mean
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.numpy())
            all_masks.append(mask.numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    
    # Compute metrics per target (only for valid samples)
    from sklearn.metrics import mean_absolute_error, r2_score
    
    metrics = {}
    target_names = ["Energy Above Hull", "Band Gap", "Bulk Modulus"]
    
    for i, name in enumerate(target_names):
        # Extract valid samples for this target
        valid_mask = all_masks[:, i] > 0.5
        y_true = all_targets[valid_mask, i]
        y_pred = all_preds[valid_mask, i]
        
        if len(y_true) > 0:
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            count = len(y_true)
        else:
            mae = 0.0
            r2 = 0.0
            count = 0
            
        metrics[name] = {"MAE": mae, "R2": r2, "Count": count}
    
    return metrics, all_preds, all_targets


def main(args):
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Setup paths
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build graph builder
    builder = CrystalGraphBuilder(radius=8.0, max_neighbors=12, dStep=0.2)
    
    # Load datasets
    logger.info("Loading training data...")
    train_dataset = MPDataset(
        args.train_path, 
        builder, 
        max_samples=args.max_train_samples,
        cache=True
    )
    
    logger.info("Loading validation data...")
    val_dataset = MPDataset(
        args.val_path, 
        builder, 
        max_samples=args.max_val_samples,
        cache=True
    )
    
    # Use training set statistics for normalization
    scaler_mean = torch.tensor(train_dataset.target_mean, dtype=torch.float32).to(device)
    scaler_std = torch.tensor(train_dataset.target_std, dtype=torch.float32).to(device)
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Model
    model = ImprovedCGCNN(
        node_input_dim=4,
        edge_input_dim=41,
        hidden_dim=args.hidden_dim,
        n_conv_layers=args.n_layers,
        n_targets=3,
        dropout=args.dropout
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5
    )
    
    # Training loop
    start_epoch = 1
    best_val_loss = float('inf')
    best_metrics = None
    patience_counter = 0

    # Resume from checkpoint if provided
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"Loading checkpoint from '{args.resume}'")
            checkpoint = torch.load(args.resume)
            
            # Load model and optimizer
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restore training state
            start_epoch = checkpoint['epoch'] + 1
            best_metrics = checkpoint['val_metrics']
            
            # Calculate best validation loss from stored metrics
            best_val_loss = np.mean([
                m["MAE"] for m in best_metrics.values() if m["Count"] > 0
            ])
            
            # Restore scheduler state if available (for future compatibility)
            if 'scheduler_state_dict' in checkpoint:
                 scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            logger.info(f"Resumed from epoch {checkpoint['epoch']} with Best Val Loss: {best_val_loss:.4f}")
        else:
            logger.warning(f"Checkpoint file '{args.resume}' not found. Starting from scratch.")
    
    logger.info("Starting Masked Training...")
    
    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer,
            scaler_mean, scaler_std, device, args.grad_clip
        )
        
        # Validate
        val_metrics, _, _ = evaluate(model, val_loader, scaler_mean, scaler_std, device)
        
        # Compute validation loss (weighted mean of MAEs?)
        # Or just simple mean. 
        # Note: We should probably weight BG higher now to ensure it learning.
        val_loss = np.mean([m["MAE"] for m in val_metrics.values() if m["Count"] > 0])
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Logging
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Metrics:")
        for name, m in val_metrics.items():
            logger.info(f"    {name} (N={m['Count']}): MAE={m['MAE']:.4f}, R²={m['R2']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_mean': scaler_mean.cpu(),
                'scaler_scale': scaler_std.cpu(),
                'val_metrics': val_metrics,
                # Save architecture args too for inference
                'arch_args': {
                    'node_input_dim': 4,
                    'edge_input_dim': 41,
                    'hidden_dim': args.hidden_dim,
                    'n_conv_layers': args.n_layers,
                    'n_targets': 3,
                    'dropout': args.dropout
                }
            }
            
            save_path = os.path.join(args.output_dir, "cgcnn_best.pt")
            torch.save(checkpoint, save_path)
            logger.info(f"  ✓ New best model saved!")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # Final report
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best Validation Metrics:")
    for name, m in best_metrics.items():
        logger.info(f"  {name}: MAE={m['MAE']:.4f}, R²={m['R2']:.4f}")
    
    # Save final metrics
    metrics_path = os.path.join(args.output_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({
            "best_metrics": best_metrics,
            "config": vars(args),
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Optimized CGCNN")
    
    # Data paths
    parser.add_argument("--train_path", type=str, 
                        default="D:/coding_files/Projects/matterGen/material dataset/train.parquet")
    parser.add_argument("--val_path", type=str,
                        default="D:/coding_files/Projects/matterGen/material dataset/val.parquet")
    parser.add_argument("--output_dir", type=str, default="../../models")
    
    # Data limits (None = Full Dataset)
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Max training samples (None for all)")
    parser.add_argument("--max_val_samples", type=int, default=20000,
                        help="Max validation samples")
    
    # Model architecture (Scaled Up)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Training hyperparameters  
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=128) # Increased batch size for speed
    parser.add_argument("--lr", type=float, default=5e-4) # Slightly lower LR for larger model
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=20)
    
    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    args = parser.parse_args()
    
    # Resolve output path
    base_dir = os.path.dirname(__file__)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(base_dir, args.output_dir)
    
    main(args)
