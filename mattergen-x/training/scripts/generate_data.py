import json
import random
import os

def generate_synthetic_data(num_samples=50000):
    # Comprehensive Element Data with ionic radii (approx) and mass
    elements_data = {
        "Li": {"mass": 6.94, "radius": 1.45, "type": "metal", "group": 1},
        "Na": {"mass": 22.99, "radius": 1.80, "type": "metal", "group": 1},
        "K": {"mass": 39.10, "radius": 2.20, "type": "metal", "group": 1},
        "Mg": {"mass": 24.31, "radius": 1.50, "type": "metal", "group": 2},
        "Ca": {"mass": 40.08, "radius": 1.80, "type": "metal", "group": 2},
        "Sr": {"mass": 87.62, "radius": 2.15, "type": "metal", "group": 2},
        "Ba": {"mass": 137.33, "radius": 2.15, "type": "metal", "group": 2},
        "Ti": {"mass": 47.87, "radius": 1.40, "type": "transition_metal", "group": 4},
        "Fe": {"mass": 55.85, "radius": 1.40, "type": "transition_metal", "group": 8},
        "Co": {"mass": 58.93, "radius": 1.35, "type": "transition_metal", "group": 9},
        "Ni": {"mass": 58.69, "radius": 1.35, "type": "transition_metal", "group": 10},
        "Cu": {"mass": 63.55, "radius": 1.35, "type": "transition_metal", "group": 11},
        "Zn": {"mass": 65.38, "radius": 1.35, "type": "transition_metal", "group": 12},
        "Al": {"mass": 26.98, "radius": 1.25, "type": "metal", "group": 13},
        "Ga": {"mass": 69.72, "radius": 1.30, "type": "metal", "group": 13},
        "Si": {"mass": 28.09, "radius": 1.10, "type": "metalloid", "group": 14},
        "O": {"mass": 15.99, "radius": 0.60, "type": "non_metal", "group": 16},
        "S": {"mass": 32.06, "radius": 1.00, "type": "non_metal", "group": 16},
        "Se": {"mass": 78.96, "radius": 1.15, "type": "non_metal", "group": 16},
        "F": {"mass": 19.00, "radius": 0.50, "type": "halogen", "group": 17},
        "Cl": {"mass": 35.45, "radius": 1.00, "type": "halogen", "group": 17},
        "Br": {"mass": 79.90, "radius": 1.15, "type": "halogen", "group": 17},
        "I": {"mass": 126.90, "radius": 1.40, "type": "halogen", "group": 17},
        "Mn": {"mass": 54.94, "radius": 1.35, "type": "transition_metal", "group": 7},
        "Cr": {"mass": 52.00, "radius": 1.40, "type": "transition_metal", "group": 6},
        "Cd": {"mass": 112.41, "radius": 1.50, "type": "transition_metal", "group": 12},
        "Pb": {"mass": 207.2, "radius": 1.75, "type": "metal", "group": 14},
        "Sn": {"mass": 118.71, "radius": 1.60, "type": "metal", "group": 14},
        "P": {"mass": 30.97, "radius": 1.10, "type": "non_metal", "group": 15},
        "N": {"mass": 14.01, "radius": 0.75, "type": "non_metal", "group": 15},
    }
    
    element_keys = list(elements_data.keys())
    
    data = []

    for _ in range(num_samples):
        # PROTOTYPE SELECTION
        prototype = random.choices(
            ["Perovskite", "RockSalt", "ZincBlende", "Fluorite", "Spinel", "Rutile", "Wurtzite", "Random"], 
            weights=[0.15, 0.15, 0.1, 0.1, 0.1, 0.15, 0.15, 0.1]
        )[0]
        
        sites = []
        lattice = None
        
        if prototype == "Perovskite": 
            # ABX3
            A = random.choice([e for e in element_keys if elements_data[e]['group'] in [1, 2]])
            B = random.choice([e for e in element_keys if elements_data[e]['type'] == 'transition_metal'])
            X = random.choice(["O", "F", "Cl", "Br", "S"])
            r_A = elements_data[A]['radius']; r_B = elements_data[B]['radius']; r_X = elements_data[X]['radius']
            
            t_factor = (r_A + r_X) / (1.414 * (r_B + r_X))
            a = 2 * (r_B + r_X) * random.uniform(0.95, 1.05)
            vol = a**3
            
            sites.append({"species": [{"element": A, "occu": 1.0}], "abc": [0,0,0], "xyz": [0,0,0], "label": A})
            sites.append({"species": [{"element": B, "occu": 1.0}], "abc": [0.5,0.5,0.5], "xyz": [0.5*a, 0.5*a, 0.5*a], "label": B})
            sites.append({"species": [{"element": X, "occu": 1.0}], "abc": [0.5,0.5,0], "xyz": [0.5*a, 0.5*a, 0], "label": X})
            sites.append({"species": [{"element": X, "occu": 1.0}], "abc": [0.5,0,0.5], "xyz": [0.5*a, 0, 0.5*a], "label": X})
            sites.append({"species": [{"element": X, "occu": 1.0}], "abc": [0,0.5,0.5], "xyz": [0, 0.5*a, 0.5*a], "label": X})
            lattice = {"matrix": [[a,0,0],[0,a,0],[0,0,a]], "a": a, "b": a, "c": a, "alpha": 90, "beta": 90, "gamma": 90, "volume": vol}
            
            stability_penalty = abs(1.0 - t_factor) * 2.0
            formation_energy = -3.0 + stability_penalty + random.uniform(-0.5, 0.5)
            band_gap = max(0.0, 3.0 + random.uniform(-0.5, 0.5))
            if X in ["S", "Se"]: band_gap -= 1.0
            if "Fe" in [A, B] or "Co" in [A, B]: band_gap = min(band_gap, 1.8)

        elif prototype == "RockSalt":
            # NaCl
            Cation = random.choice([e for e in element_keys if elements_data[e]['group'] in [1, 2]])
            Anion = random.choice(["O", "S", "Se", "F", "Cl", "Br", "I"])
            r_C = elements_data[Cation]['radius']; r_A = elements_data[Anion]['radius']
            a = 2 * (r_C + r_A) * random.uniform(0.98, 1.02)
            vol = a**3
            
            fcc_pos = [[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]]
            offset_pos = [[0.5,0.5,0.5], [0,0,0.5], [0,0.5,0], [0.5,0,0]]
            for p in fcc_pos: sites.append({"species": [{"element": Cation, "occu": 1.0}], "abc": p, "xyz": [p[0]*a, p[1]*a, p[2]*a], "label": Cation})
            for p in offset_pos: sites.append({"species": [{"element": Anion, "occu": 1.0}], "abc": p, "xyz": [p[0]*a, p[1]*a, p[2]*a], "label": Anion})
            lattice = {"matrix": [[a,0,0],[0,a,0],[0,0,a]], "a": a, "b": a, "c": a, "alpha": 90, "beta": 90, "gamma": 90, "volume": vol}
            formation_energy = -2.5 + random.uniform(-0.5, 0.5)
            band_gap = 4.0 + random.uniform(-1.0, 1.0)
            
        elif prototype == "Fluorite":
            # AX2
            A = random.choice([e for e in element_keys if elements_data[e]['group'] in [2, 4]]) 
            X = random.choice(["O", "F", "Cl"])
            r_A = elements_data[A]['radius']; r_X = elements_data[X]['radius']
            a = (4/1.732) * (r_A + r_X) * random.uniform(0.95, 1.05)
            vol = a**3
            
            fcc_pos = [[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]]
            for p in fcc_pos: sites.append({"species": [{"element": A, "occu": 1.0}], "abc": p, "xyz": [p[0]*a, p[1]*a, p[2]*a], "label": A})
            shifts = [[0.25,0.25,0.25], [0.75,0.75,0.75], [0.75,0.25,0.25], [0.25,0.75,0.25], [0.25,0.25,0.75], [0.75,0.75,0.25], [0.75,0.25,0.75], [0.25,0.75,0.75]]
            for p in shifts: sites.append({"species": [{"element": X, "occu": 1.0}], "abc": p, "xyz": [p[0]*a, p[1]*a, p[2]*a], "label": X})
            lattice = {"matrix": [[a,0,0],[0,a,0],[0,0,a]], "a": a, "b": a, "c": a, "alpha": 90, "beta": 90, "gamma": 90, "volume": vol}
            formation_energy = -3.5 + random.uniform(-0.5, 0.5) 
            band_gap = 5.0 + random.uniform(-1.0, 1.0) 

        elif prototype == "Rutile":
            # AX2 Tetragonal (TiO2)
            A = random.choice(["Ti", "Sn", "Mn", "Pb", "Si"])
            X = random.choice(["O", "F"])
            a = 4.6; c = 2.9 # Approx TiO2
            a *= random.uniform(0.9, 1.1); c *= random.uniform(0.9, 1.1)
            vol = a*a*c
            
            # Simple P42/mnm approx (2 formula units)
            # A at 0,0,0 and 0.5,0.5,0.5
            sites.append({"species": [{"element": A, "occu": 1.0}], "abc": [0,0,0], "xyz": [0,0,0], "label": A})
            sites.append({"species": [{"element": A, "occu": 1.0}], "abc": [0.5,0.5,0.5], "xyz": [0.5*a, 0.5*a, 0.5*c], "label": A})
            # X at u,u,0 ... usually u=0.3
            u = 0.3
            x_pos = [[u,u,0], [(1-u),(1-u),0], [0.5+u, 0.5-u, 0.5], [0.5-u, 0.5+u, 0.5]]
            for p in x_pos: sites.append({"species": [{"element": X, "occu": 1.0}], "abc": p, "xyz": [p[0]*a, p[1]*a, p[2]*c], "label": X})
            
            lattice = {"matrix": [[a,0,0],[0,a,0],[0,0,c]], "a": a, "b": a, "c": c, "alpha": 90, "beta": 90, "gamma": 90, "volume": vol}
            formation_energy = -3.0 + random.uniform(-0.2, 0.2)
            band_gap = 3.0 # TiO2 is 3.0-3.2
            
        elif prototype == "Wurtzite":
            # AX Hexagonal (ZnO, GaN)
            A = random.choice(["Zn", "Cd", "Ga", "Al"])
            X = random.choice(["O", "S", "Se", "N", "P"])
            a = 3.25; c = 5.2 # Approx ZnO
            a *= random.uniform(0.9, 1.1); c *= random.uniform(0.9, 1.1)
            # Hexagonal Lattice Matrix (gamma = 120)
            # a1 = (a, 0, 0)
            # a2 = (-a/2, a*sqrt(3)/2, 0)
            # a3 = (0, 0, c)
            # For simplicity in cubic-like viewer we might approximate or output full hex matrix
            import math
            matrix = [[a, 0, 0], [-a/2, a*math.sqrt(3)/2, 0], [0, 0, c]]
            vol = a * a * c * math.sin(math.radians(120))
            
            # 2 formula units
            # A at 1/3, 2/3, 0 and 2/3, 1/3, 1/2
            sites.append({"species": [{"element": A, "occu": 1.0}], "abc": [1/3, 2/3, 0], "xyz": None, "label": A}) # Let pymatgen handle xyz if we omit or calc manually
            sites.append({"species": [{"element": A, "occu": 1.0}], "abc": [2/3, 1/3, 0.5], "xyz": None, "label": A})
            
            # X at 1/3, 2/3, u and 2/3, 1/3, 0.5+u (u ~ 0.375)
            u = 0.375
            sites.append({"species": [{"element": X, "occu": 1.0}], "abc": [1/3, 2/3, u], "xyz": None, "label": X})
            sites.append({"species": [{"element": X, "occu": 1.0}], "abc": [2/3, 1/3, 0.5+u], "xyz": None, "label": X})
            
            lattice = {"matrix": matrix, "a": a, "b": a, "c": c, "alpha": 90, "beta": 90, "gamma": 120, "volume": vol}
            formation_energy = -1.5 + random.uniform(-0.2, 0.2)
            band_gap = 3.3 if "Zn" in A else 2.5 # GaN/ZnO wide gap
            
        elif prototype == "Spinel":
            # AB2X4 (MgAl2O4). Complex cubic. 
            # Simplified version: Normal Spinel.
            # 8 A atoms (Td), 16 B atoms (Oh), 32 X atoms (FCC packed)
            # This is too big for our simple generator to hardcode easily without errors.
            # We will approximate a "Spinel-like" unit cell with correct stoichiometry but smaller cell
            # A2B4X8? Or just AB2X4 in a minimal P1 box.
            
            A = "Mg"; B = "Al"; X = "O" # defaults
            A = random.choice(["Mg", "Zn", "Fe", "Mn"])
            B = random.choice(["Al", "Fe", "Cr", "Co"])
            X = "O"
            
            # Use a ~8.0 Angstrom cubic cell
            a = 8.0 * random.uniform(0.95, 1.05)
            vol = a**3
            
            # Sparse placement for 7 atoms? No, needs to be stochiometric.
            # Let's place 1 A, 2 B, 4 X in a small box (a/2) to mimic local env
            a = a / 2.0; vol = a**3
            
            sites.append({"species": [{"element": A, "occu": 1.0}], "abc": [0,0,0], "xyz": [0,0,0], "label": A})
            sites.append({"species": [{"element": B, "occu": 1.0}], "abc": [0.5,0.5,0], "xyz": [0.5*a, 0.5*a, 0], "label": B})
            sites.append({"species": [{"element": B, "occu": 1.0}], "abc": [0.5,0,0.5], "xyz": [0.5*a, 0, 0.5*a], "label": B})
            # X at tetrahedral interstices
            x_pos = [[0.25,0.25,0.25], [0.75,0.75,0.75], [0.25,0.75,0.75], [0.75,0.25,0.25]]
            for p in x_pos: sites.append({"species": [{"element": X, "occu": 1.0}], "abc": p, "xyz": [p[0]*a, p[1]*a, p[2]*a], "label": X})
            
            lattice = {"matrix": [[a,0,0],[0,a,0],[0,0,a]], "a": a, "b": a, "c": a, "alpha": 90, "beta": 90, "gamma": 90, "volume": vol}
            formation_energy = -2.8 + random.uniform(-0.2, 0.2)
            band_gap = 3.0 + random.uniform(-1.0, 1.0)
            
        else:
            # Random / ZincBlende etc. 
            elements_subset = random.sample(element_keys, 2)
            a = random.uniform(4.0, 6.0)
            vol = a**3 
            
            # Zinc Blende positions
            zp_A = [[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]]
            zp_B = [[0.25,0.25,0.25], [0.75,0.75,0.75], [0.75,0.25,0.25], [0.25,0.75,0.25]] # shifted by 1/4 1/4 1/4
            
            for p in zp_A: sites.append({"species": [{"element": elements_subset[0], "occu": 1.0}], "abc": p, "xyz": [p[0]*a, p[1]*a, p[2]*a], "label": elements_subset[0]})
            for p in zp_B: sites.append({"species": [{"element": elements_subset[1], "occu": 1.0}], "abc": p, "xyz": [p[0]*a, p[1]*a, p[2]*a], "label": elements_subset[1]})
                
            lattice = {"matrix": [[a,0,0],[0,a,0],[0,0,a]], "a": a, "b": a, "c": a, "alpha": 90, "beta": 90, "gamma": 90, "volume": vol}
            formation_energy = -1.0 + random.uniform(-0.5, 0.5)
            band_gap = 1.0 + random.uniform(-0.5, 0.5)

        # Finalize structure
        structure = {
            "@module": "pymatgen.core.structure", 
            "@class": "Structure", 
            "charge": None, 
            "lattice": lattice, 
            "sites": sites
        }
        
        # Calculate Density (Total Mass / Volume) * 1.66
        total_mass = sum([elements_data[s["species"][0]["element"]]["mass"] for s in sites])
        density = (total_mass / lattice["volume"]) * 1.66
        
        entry = {
            "structure": structure,
            "formation_energy_per_atom": round(formation_energy, 3),
            "band_gap": round(band_gap, 3),
            "density": round(density, 3),
            "formula": "Gen-" + "".join(sorted(set([s["species"][0]["element"] for s in sites])))
        }
        data.append(entry)
        
    return data

if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), "../../data/synthetic_materials.json")
    data = generate_synthetic_data(50000)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Generated {len(data)} samples to {output_path}")
