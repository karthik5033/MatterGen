import random
from pymatgen.core import Structure
from typing import List, Dict, Any, Optional

class StructureGenerator:
    """
    Generates synthetic crystal structures based on chemical rules and prototypes.
    Used to simulate a generative AI model (like DiffCSP or CDVAE).
    """

    def __init__(self):
        # Comprehensive Element Data
        self.elements_data = {
            "Li": {"mass": 6.94, "radius": 1.45, "type": "alkali", "group": 1},
            "Na": {"mass": 22.99, "radius": 1.80, "type": "alkali", "group": 1},
            "K": {"mass": 39.10, "radius": 2.20, "type": "alkali", "group": 1},
            "Mg": {"mass": 24.31, "radius": 1.50, "type": "alkaline_earth", "group": 2},
            "Ca": {"mass": 40.08, "radius": 1.80, "type": "alkaline_earth", "group": 2},
            "Sr": {"mass": 87.62, "radius": 2.15, "type": "alkaline_earth", "group": 2},
            "Ba": {"mass": 137.33, "radius": 2.15, "type": "alkaline_earth", "group": 2},
            "Ti": {"mass": 47.87, "radius": 1.40, "type": "transition_metal", "group": 4},
            "Fe": {"mass": 55.85, "radius": 1.40, "type": "transition_metal", "group": 8},
            "Co": {"mass": 58.93, "radius": 1.35, "type": "transition_metal", "group": 9},
            "Ni": {"mass": 58.69, "radius": 1.35, "type": "transition_metal", "group": 10},
            "Cu": {"mass": 63.55, "radius": 1.35, "type": "transition_metal", "group": 11},
            "Zn": {"mass": 65.38, "radius": 1.35, "type": "transition_metal", "group": 12},
            "Al": {"mass": 26.98, "radius": 1.25, "type": "metal", "group": 13},
            "Si": {"mass": 28.09, "radius": 1.10, "type": "metalloid", "group": 14},
            "O": {"mass": 15.99, "radius": 0.60, "type": "chalcogen", "group": 16},
            "S": {"mass": 32.06, "radius": 1.00, "type": "chalcogen", "group": 16},
            "Se": {"mass": 78.96, "radius": 1.15, "type": "chalcogen", "group": 16},
            "F": {"mass": 19.00, "radius": 0.50, "type": "halogen", "group": 17},
            "Cl": {"mass": 35.45, "radius": 1.00, "type": "halogen", "group": 17},
            "Br": {"mass": 79.90, "radius": 1.15, "type": "halogen", "group": 17},
            "I": {"mass": 126.90, "radius": 1.40, "type": "halogen", "group": 17},
        }
        self.element_keys = list(self.elements_data.keys())

    def generate_batch(self, n_samples: int = 3, context_keywords: List[str] = []) -> List[Dict[str, Any]]:
        """
        Generate a batch of structures, optionally biased by context keywords.
        """
        results = []
        for _ in range(n_samples):
            results.append(self._generate_single(context_keywords))
        return results

    def _generate_single(self, context_keywords: List[str]) -> Dict[str, Any]:
        """
        Generate a single structure.
        """
        # 1. Bias Selection based on Context
        keywords_str = " ".join(context_keywords).lower()
        
        forced_elements = []
        preferred_prototype = None
        
        if "battery" in keywords_str or "cathode" in keywords_str or "anode" in keywords_str:
            forced_elements = ["Li", "Na", "Mg"]
        elif "solar" in keywords_str or "photovoltaic" in keywords_str:
            preferred_prototype = "Perovskite"
        elif "magnet" in keywords_str:
            forced_elements = ["Fe", "Co", "Ni"]
        
        # 2. Select Prototype
        if preferred_prototype:
            prototype = preferred_prototype
        else:
            prototype = random.choices(
                ["Perovskite", "RockSalt", "ZincBlende", "Random"], 
                weights=[0.35, 0.35, 0.1, 0.2]
            )[0]

        # 3. Build Structure
        struct = None
        formula = ""
        
        try:
            if prototype == "Perovskite": 
                # ABX3
                # Try to use forced elements if valid
                valid_A = [e for e in self.element_keys if self.elements_data[e]['group'] in [1, 2]]
                valid_B = [e for e in self.element_keys if self.elements_data[e]['type'] == 'transition_metal']
                
                # Filter by forced
                A_cands = [e for e in valid_A if e in forced_elements] if forced_elements else valid_A
                B_cands = [e for e in valid_B if e in forced_elements] if forced_elements else valid_B
                
                # Fallback if intersection empty
                A = random.choice(A_cands) if A_cands else random.choice(valid_A)
                B = random.choice(B_cands) if B_cands else random.choice(valid_B)
                X = random.choice(["O", "F", "Cl", "Br", "S", "I"])
                
                r_B = self.elements_data[B]['radius']
                r_X = self.elements_data[X]['radius']
                
                # Lattice
                t_distort = random.uniform(0.95, 1.05)
                a = 2 * (r_B + r_X) * t_distort
                
                species = [A, B, X, X, X]
                coords = [
                    [0,0,0],         # A
                    [0.5,0.5,0.5],   # B
                    [0.5,0.5,0],     # X
                    [0.5,0,0.5],     # X
                    [0,0.5,0.5]      # X
                ]
                
                lattice = [[a,0,0],[0,a,0],[0,0,a]]
                struct = Structure(lattice, species, coords)
                
            elif prototype == "RockSalt":
                # AB
                valid_C = [e for e in self.element_keys if self.elements_data[e]['group'] in [1, 2, 13, 4, 8, 9, 10, 11, 12]]
                valid_A = ["O", "S", "Se", "F", "Cl", "Br", "I", "N"]
                
                C_cands = [e for e in valid_C if e in forced_elements] if forced_elements else valid_C
                
                C = random.choice(C_cands) if C_cands else random.choice(valid_C)
                A = random.choice(valid_A)
                
                r_C = self.elements_data[C]['radius']
                r_A = self.elements_data[A]['radius']
                
                a = 2 * (r_C + r_A) * random.uniform(0.98, 1.02)
                
                species = [C]*4 + [A]*4
                # FCC Basis
                coords = [
                    [0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5], # Cation
                    [0.5,0.5,0.5], [0,0,0.5], [0,0.5,0], [0.5,0,0]  # Anion
                ]
                
                struct = Structure(lattice, species, coords)
                
            elif prototype == "Rutile":
                # AX2 Tetragonal (TiO2)
                A = random.choice(["Ti", "Sn", "Mn", "Pb", "Si"])
                X = random.choice(["O", "F"])
                
                if forced_elements:
                     # Check if forced elements match
                     pass
                
                a = 4.6; c = 2.9
                a *= random.uniform(0.9, 1.1); c *= random.uniform(0.9, 1.1)
                
                species = [A, A, X, X, X, X]
                u = 0.3
                coords = [
                    [0,0,0], [0.5,0.5,0.5],
                    [u,u,0], [(1-u),(1-u),0], [0.5+u, 0.5-u, 0.5], [0.5-u, 0.5+u, 0.5]
                ]
                lattice = [[a,0,0],[0,a,0],[0,0,c]]
                struct = Structure(lattice, species, coords)
                
            elif prototype == "Wurtzite":
                # AX Hexagonal
                A = random.choice(["Zn", "Cd", "Ga", "Al"])
                X = random.choice(["O", "S", "Se", "N", "P"])
                
                a = 3.25; c = 5.2 
                a *= random.uniform(0.9, 1.1); c *= random.uniform(0.9, 1.1)

                import math
                matrix = [[a, 0, 0], [-a/2, a*math.sqrt(3)/2, 0], [0, 0, c]]
                
                species = [A, A, X, X]
                u = 0.375
                coords = [
                    [1/3, 2/3, 0], [2/3, 1/3, 0.5],
                    [1/3, 2/3, u], [2/3, 1/3, 0.5+u]
                ]
                struct = Structure(matrix, species, coords)
                
            else:
                # Random
                n_species = random.randint(2, 3)
                chosen = random.sample(self.element_keys, n_species)
                if forced_elements:
                    chosen[0] = random.choice(forced_elements)
                
                a = random.uniform(3.0, 8.0)
                lattice = [[a,0,0],[0,a,0],[0,0,a]]
                
                species = []
                coords = []
                for _ in range(random.randint(4, 8)):
                    species.append(random.choice(chosen))
                    coords.append([random.random(), random.random(), random.random()])
                    
                struct = Structure(lattice, species, coords)

            formula = struct.composition.reduced_formula
            
        except Exception:
            # Fallback if Generation Fails
            return self._generate_fallback()

        return {
            "formula": formula,
            "structure": struct,
            "cif": struct.to(fmt="cif")
        }

    def _generate_fallback(self):
        # Fallback to simple MgO
        lattice = [[4.2,0,0],[0,4.2,0],[0,0,4.2]]
        struct = Structure(lattice, ["Mg", "O"], [[0,0,0], [0.5,0.5,0.5]])
        return {
            "formula": "MgO",
            "structure": struct,
            "cif": struct.to(fmt="cif")
        }

structure_generator = StructureGenerator()
