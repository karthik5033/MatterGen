import random
import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompositionSearchSpace:
    """
    Defines the search space for material discovery.
    
    The "Combinatorial Explosion" Problem:
    ------------------------------------
    Exploring the full chemical space is intractable (118^N combinations).
    By constraining the search to a specific subset of elements and stoichiometry rules,
    we reduce the search space from ~10^30 to ~10^6, making it navigable by AI.
    
    Functionality:
    - Allowed Elements: Restrict search to specific elements (e.g., Li-ion battery materials).
    - Constraints: Enforce stoichiometry logic (e.g., sum to 100% or integer ratios).
    - Sampling: Generate random valid formulas.
    - Mutation: Perturb existing formulas for evolutionary optimization.
    """
    
    def __init__(self, allowed_elements: List[str], max_elements: int = 4):
        """
        Args:
            allowed_elements: List of atomic symbols (e.g., ["Li", "Fe", "Mn", "O", "P"]).
            max_elements: Maximum number of distinct elements in a formula (default 4 -> Quaternary).
        """
        self.allowed_elements = allowed_elements
        self.max_elements = max_elements
        
        # Valid stoichiometry ratios (Integers for simplicity in crystal building)
        # e.g., A1B1, A1B2, A2B3, A1B1C3
        self.common_ratios = [1, 2, 3, 4, 5, 6, 7] 

    def sample(self) -> str:
        """
        Sample a random valid composition from the search space.
        
        Returns:
            Formula string (e.g., "Li1Fe1P1O4").
        """
        # 1. Choose N elements (2 to max_elements)
        n_elements = random.randint(2, self.max_elements)
        elements = random.sample(self.allowed_elements, n_elements)
        
        # 2. Assign stoichiometry
        # Heuristic: Ensure reasonable ratios (avoid Li100Fe1)
        amounts = [random.choice(self.common_ratios) for _ in elements]
        
        # 3. Construct Formula
        formula = ""
        for el, amt in zip(elements, amounts):
            formula += f"{el}{amt}"
            
        return formula

    def mutate(self, formula: str, mutation_rate: float = 0.5) -> str:
        """
        Mutate an existing composition.
        
        Strategies:
        - Element Swap: Replace 'Fe' with 'Mn'.
        - Stoichiometry Perturbation: Change amount '2' to '3'.
        
        Args:
            formula: Input formula string (e.g., "Li1Fe1O4").
            mutation_rate: Probability of mutation type.
            
        Returns:
            Mutated formula string.
        """
        # Parse (Naive parsing, assumes Element+Int format standard from sample())
        # Ideally use pymatgen.core.Composition, but let's keep it self-contained for speed.
        import re
        parsed = re.findall(r'([A-Z][a-z]*)([0-9]+)', formula)
        if not parsed:
            return self.sample() # Fallback
            
        elements = [p[0] for p in parsed]
        amounts = [int(p[1]) for p in parsed]
        
        # Strategy 1: Element Substitution
        if random.random() < 0.5:
            idx_to_swap = random.randint(0, len(elements) - 1)
            original_el = elements[idx_to_swap]
            
            # Pick distinct new element
            new_el = original_el
            while new_el == original_el or new_el in elements:
                new_el = random.choice(self.allowed_elements)
                
            elements[idx_to_swap] = new_el
            
        # Strategy 2: Stoichiometry Adjustment
        else:
            idx_to_change = random.randint(0, len(amounts) - 1)
            change = random.choice([-1, 1])
            new_amount = max(1, amounts[idx_to_change] + change) # Ensure > 0
            amounts[idx_to_change] = new_amount
            
        # Reconstruct
        new_formula = ""
        for el, amt in zip(elements, amounts):
            new_formula += f"{el}{amt}"
            
        return new_formula

if __name__ == "__main__":
    # Test
    print("--- Testing Composition Search Space ---")
    
    # Define a Battery Search Space
    battery_elements = ["Li", "Na", "K", "Fe", "Mn", "Co", "Ni", "P", "S", "O", "F", "Cl"]
    space = CompositionSearchSpace(allowed_elements=battery_elements)
    
    print(f"Allowed Elements: {space.allowed_elements}")
    
    # 1. Random Sampling
    print("\n[Sampling]")
    for _ in range(5):
        print(f"  > {space.sample()}")
        
    # 2. Mutation
    start = "Li1Fe1P1O4"
    print(f"\n[Mutation] Start: {start}")
    for _ in range(5):
        mutated = space.mutate(start)
        print(f"  > {mutated}")
