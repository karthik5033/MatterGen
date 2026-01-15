import torch
from torch.utils.data import DataLoader
from training.datasets.loader import MaterialDataLoader
from training.datasets.graph_builder import CrystalGraphBuilder
from training.datasets.graph_dataset import CrystalGraphDataset, collate_batch
import os

def test_graph_dataset():
    print("--- Testing Crystal Graph Dataset ---")
    
    # 1. Load Mock Data
    data_path = "../../data/mock_materials.json"
    loader = MaterialDataLoader(data_path)
    try:
        raw_data = loader.load()
    except:
        # Fallback if no file, create dummy data
        print("Mock file not found, creating dummy structure data...")
        from pymatgen.core import Structure, Lattice
        l = Lattice.cubic(4.0)
        s1 = Structure(l, ["Cs", "Cl"], [[0,0,0], [0.5,0.5,0.5]])
        s2 = Structure(l, ["Si", "Si"], [[0,0,0], [0.25,0.25,0.25]])
        raw_data = [
            {"structure_obj": s1, "label_formation_energy": -1.0, "label_band_gap": 2.0, "label_density": 3.0, "formula": "CsCl"},
            {"structure_obj": s2, "label_formation_energy": -0.5, "label_band_gap": 1.1, "label_density": 2.3, "formula": "Si2"}
        ]

    # 2. Init Dataset
    builder = CrystalGraphBuilder(radius=5.0)
    dataset = CrystalGraphDataset(raw_data, builder, cache_graphs=True)
    
    print(f"Dataset Size: {len(dataset)}")
    
    # 3. Test Individual Item
    sample = dataset[0]
    print(f"Sample 0 Atom Fea Shape: {sample['atom_fea'].shape}")
    print(f"Sample 0 Edge Fea Shape: {sample['nbr_fea'].shape}")
    
    # 4. Test Batching
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_batch)
    batch = next(iter(loader))
    
    print(f"\nBatch (Size 2) Keys: {batch.keys()}")
    print(f"Batched Atom Fea: {batch['atom_fea'].shape}")
    print(f"Batched Edge Idx: {batch['nbr_idx'].shape}")
    print(f"Batched Targets: {batch['target'].shape}")
    print(f"Batch Vector: {batch['batch']}")
    
    # Validation checks
    # S1 has 2 atoms, S2 has 2 atoms -> Total 4 atoms
    assert batch['atom_fea'].shape[0] == 4, f"Expected 4 atoms in batch, got {batch['atom_fea'].shape[0]}"
    # Targets should be (2, 3)
    assert batch['target'].shape == (2, 3)
    
    print("\n✅ CrystalGraphDataset Verified.")

if __name__ == "__main__":
    test_graph_dataset()
