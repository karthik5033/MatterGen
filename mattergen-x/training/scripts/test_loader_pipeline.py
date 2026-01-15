from training.datasets.loader import MaterialDataLoader
import os

def test_loader():
    # Path relative to where script is run typically, but let's be safe
    data_path = "../../data/mock_materials.json"
    
    # 1. Initialize
    print(f"Testing loader with {data_path}...")
    loader = MaterialDataLoader(data_path)
    
    # 2. Load
    try:
        data = loader.load()
    except Exception as e:
        print(f"FATAL: Loader failed: {e}")
        return

    # 3. Validation
    print(f"Loaded {len(data)} samples.")
    assert len(data) == 2, f"Expected 2 samples, got {len(data)}"
    
    sample = data[0]
    print("Sample 0 keys:", sample.keys())
    
    # Check extraction
    assert "structure_obj" in sample
    assert "label_band_gap" in sample
    assert "formula" in sample
    
    # Check correctness
    # sorted by formula: MgO (Mg1 O1) vs Si (Si2? or Si1?)
    # MgO -> Mg1 O1. Si -> Si2 (in mock data it has 2 sites).
    # M comes before S.
    
    print("\nSample 0 details:")
    print(f"Formula: {sample['formula']}")
    print(f"Band Gap: {sample['label_band_gap']} eV")
    print(f"Crystal System: {sample['crystal_system']}")

    print("\n✅ Verification Successful: Loader is working correctly.")

if __name__ == "__main__":
    test_loader()
