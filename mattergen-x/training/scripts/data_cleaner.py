
import os
import json
import zipfile
import pandas as pd
import glob
import numpy as np
from pymatgen.core import Structure, Lattice
from tqdm import tqdm

# --- Configuration ---
SOURCE_DIR = r"d:\coding_files\Projects\matterGen\material dataset"
OUTPUT_FILE = r"d:\coding_files\Projects\matterGen\mattergen-x\data\final_cleaned_dataset.json"

def parse_structure_from_row(row):
    """
    Handle various structure formats:
    1. 'structure' string (CIF/JSON)
    2. 'structure' dict
    3. 'positions', 'cell', 'atomic_numbers' (MPtrj format)
    """
    try:
        # Case 1: MPtrj / Atomistic components
        if 'atomic_numbers' in row and 'positions' in row and 'cell' in row:
            # Check if values are not None
            if row['atomic_numbers'] is None: return None
            
            species = row['atomic_numbers']
            coords = row['positions']
            lattice = row['cell']
            
            # If stored as numpy string representation or something odd, might need eval
            # Assuming loaded as lists/arrays from Parquet/JSON
            return Structure(lattice, species, coords, coords_are_cartesian=True)

        # Case 2: 'structure' key
        if 'structure' in row:
            s_data = row['structure']
            if isinstance(s_data, str):
                # Try JSON dict
                try: 
                    d = json.loads(s_data.replace("'", '"')) 
                    return Structure.from_dict(d)
                except: 
                    # Try CIF
                    try: return Structure.from_str(s_data, fmt="cif")
                    except: return None
            elif isinstance(s_data, dict):
                return Structure.from_dict(s_data)
        
        # Case 3: 'cif' key
        if 'cif' in row and isinstance(row['cif'], str):
             return Structure.from_str(row['cif'], fmt="cif")

    except Exception:
        return None
    return None

def process_file(filepath):
    print(f"Processing {filepath}...")
    data = []
    
    try:
        if filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
            data = df.to_dict(orient='records')
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            data = df.to_dict(orient='records')
        elif filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                content = json.load(f)
                if isinstance(content, list):
                    data = content
                elif isinstance(content, dict) and 'data' in content:
                    data = content['data']
        elif filepath.endswith('.zip'):
             with zipfile.ZipFile(filepath, 'r') as z:
                for filename in z.namelist():
                    if filename.endswith('.json'):
                        with z.open(filename) as f:
                            c = json.load(f)
                            if isinstance(c, list):
                                data.extend(c)
                            elif isinstance(c, dict):
                                # Check if it's a single entry or dict of entries
                                if "structure" not in c and len(c) > 0:
                                     # Maybe dict of id -> entry?
                                     data.extend(c.values())
                                else:
                                     data.append(c)

    except Exception as e:
        print(f"Failed to read {filepath}: {e}")
        return []

    processed = []
    for entry in tqdm(data, desc=os.path.basename(filepath), leave=False):
        try:
            struct = parse_structure_from_row(entry)
            if not struct: continue

            # Extract Properties
            # Energy
            e_form = entry.get('formation_energy_per_atom')
            if e_form is None: e_form = entry.get('energy_per_atom') # MPtrj fallback
            if e_form is None and 'energy' in entry and struct.num_sites > 0:
                 e_form = entry['energy'] / struct.num_sites
            
            if e_form is None: continue # Skip if no energy
            
            # Band Gap (Default 0.0)
            bg = entry.get('band_gap') or entry.get('gap pbe') or 0.0
            
            # Density
            density = entry.get('density') or struct.density

            processed.append({
                "structure": struct.as_dict(),
                "formation_energy_per_atom": float(e_form),
                "band_gap": float(bg),
                "density": float(density),
                "formula": struct.composition.reduced_formula,
                "nsites": struct.num_sites
            })
        except Exception:
            continue
            
    print(f"  -> Extracted {len(processed)} valid entries.")
    return processed

def main():
    all_clean_data = []

    # 1. Walk through SOURCE_DIR
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            path = os.path.join(root, file)
            if file.endswith(('.json', '.csv', '.parquet', '.zip')):
                all_clean_data.extend(process_file(path))
    
    print(f"Total entries before deduplication: {len(all_clean_data)}")
    
    # 2. Deduplicate
    # Deduplicate based on Formula + Energy (rounded) to avoid storing identical snapshots
    seen = set()
    unique_data = []
    for d in all_clean_data:
        # Key: Formula + Energy (2 dec places)
        key = (d['formula'], round(d['formation_energy_per_atom'], 3))
        if key not in seen:
            seen.add(key)
            unique_data.append(d)
            
    print(f"Total unique entries: {len(unique_data)}")

    # 3. Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(unique_data, f, indent=None) # Compact JSON
    
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
