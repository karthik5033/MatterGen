
import json
import zipfile
import os
from pymatgen.core import Structure, Lattice

# Paths
ZIP_PATH = r"d:\coding_files\Projects\matterGen\material dataset\mp_20.json.zip"
OUTPUT_DIR = r"d:\coding_files\Projects\matterGen\mattergen-x\data\datasets"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "mp20_clean.json")

def process():
    print(f"Processing {ZIP_PATH}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    clean_data = []
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        for filename in z.namelist():
            if not filename.endswith('.json'): continue
            
            with z.open(filename) as f:
                content = json.load(f)
                if isinstance(content, dict) and 'data' in content: content = content['data']
                if isinstance(content, dict): content = [content] 

                print(f"  Entries in {filename}: {len(content)}")
                
                for i, entry in enumerate(content):
                    try:
                        # 1. Structure Parsing (Manual from 'atoms' dict)
                        atoms = entry.get('atoms')
                        if not atoms: continue
                        
                        # Parse Lattice
                        # Found key: 'lattice_mat'
                        lat_raw = atoms.get('lattice_mat') or atoms.get('lattice')
                        if not lat_raw: continue
                        
                        if isinstance(lat_raw, dict) and 'matrix' in lat_raw:
                            lattice = Lattice(lat_raw['matrix'])
                        else:
                            lattice = Lattice(lat_raw)
                            
                        # Parse Sites
                        # Found keys: 'coords', 'elements', 'cartesian'
                        coords = atoms.get('coords') or atoms.get('positions')
                        species = atoms.get('elements') or atoms.get('species')
                        
                        calc_cartesian = atoms.get('cartesian', False)
                        
                        if not coords or not species: continue

                        # Build Structure
                        # Note: Pymatgen 'coords_are_cartesian' defaults to False (Fractional)
                        # If 'cartesian' is False in dict, it means Fractional.
                        struct = Structure(lattice, species, coords, coords_are_cartesian=calc_cartesian)

                        # 2. Properties
                        e_form = entry.get('formation_energy_per_atom')
                        bg = entry.get('band_gap')
                        dens = entry.get('density') or struct.density
                        
                        if e_form is None: continue

                        clean_data.append({
                            "structure": struct.as_dict(),
                            "formation_energy_per_atom": float(e_form),
                            "band_gap": float(bg if bg else 0.0),
                            "density": float(dens),
                            "formula": struct.composition.reduced_formula,
                            "nsites": struct.num_sites,
                            "material_id": entry.get('material_id', f"mp-{i}")
                        })

                    except Exception as e:
                        # if i < 10: print(f"Skipping {i}: {e}")
                        continue

    print(f"Total clean samples: {len(clean_data)}")
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(clean_data, f, indent=None)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process()
