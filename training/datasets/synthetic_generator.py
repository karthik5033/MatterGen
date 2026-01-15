import numpy as np
import pandas as pd
import os

def generate_synthetic_materials(n_samples=1000):
    """
    Generate synthetic material property data for development.
    """
    data = {
        'id': [f'MAT_{i:04d}' for i in range(n_samples)],
        'formula': np.random.choice(['Fe2O3', 'SrTiO3', 'CsPbI3', 'SiC', 'GaN'], n_samples),
        'formation_energy': np.random.uniform(-4.0, 0.5, n_samples),
        'band_gap': np.random.uniform(0.0, 5.0, n_samples),
        'stability': np.random.uniform(0.0, 1.0, n_samples),
        'is_metal': np.random.choice([0, 1], n_samples)
    }
    
    df = pd.DataFrame(data)
    os.makedirs('training/datasets', exist_ok=True)
    df.to_csv('training/datasets/synthetic_materials.csv', index=False)
    print(f"Generated {n_samples} samples in training/datasets/synthetic_materials.csv")

if __name__ == "__main__":
    generate_synthetic_materials()
