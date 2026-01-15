import random
import os
import torch
from typing import List, Dict
from app.schemas.generation import MaterialCandidate, GenerateResponse
from pymatgen.core import Lattice, Structure
from training.inference import CGCNNPredictor

class MaterialService:
    """
    Service for handling material discovery and property prediction logic.
    In a research-grade application, this layer manages the interaction between
    API endpoints and the underlying AI models (PyTorch/TensorFlow).
    """

    # Mock formulas for generation
    MOCK_FORMULAS = ["Fe2O3", "SrTiO3", "CsPbI3", "GaN", "SiC", "LiFePO4", "BaTiO3", "MoS2"]

    def __init__(self):
        # Initialize Predictor
        # Ideally path should be config-driven
        # We assume the model is at mattergen-x/models/cgcnn_best.pt
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        model_path = os.path.join(base_dir, "models", "cgcnn_best.pt")
        
        self.predictor = None
        if os.path.exists(model_path):
            try:
                self.predictor = CGCNNPredictor(model_path=model_path, device="cpu") # Force CPU for API stability
            except Exception as e:
                print(f"Warning: Failed to load CGCNN model: {e}")
        else:
             print(f"Warning: CGCNN model not found at {model_path}")
        
        # Initialize Explainer
        from training.explainability import CGCNNExplainer
        self.explainer = None
        if self.predictor:
            self.explainer = CGCNNExplainer(self.predictor)

    async def explain_structure(self, structure_cif: str = None, formula: str = None, target_property: str = "band_gap") -> Dict:
        """
        Explain the model's prediction for a given structure.
        """
        if not self.explainer:
            raise Exception("Explainer model not initialized")
            
        target_map = {"formation_energy": 0, "band_gap": 1, "density": 2}
        idx = target_map.get(target_property, 1)
        
        # Get Structure
        if structure_cif:
            from pymatgen.core import Structure
            struct = Structure.from_str(structure_cif, fmt="cif")
        elif formula:
            struct = self._create_mock_structure(formula)
        else:
            raise ValueError("Must provide either structure_cif or formula")
            
        # Run Explanation
        result = self.explainer.explain(struct, target_index=idx)
        
        # Heuristic Mutation
        influential_atom = result['most_influential_element']
        # Simple Logic: Suggest replacing influential atom with a neighbor in the periodic table
        # In a real app, use the CompositionSearchSpace logic
        replacements = {"Li": "Na", "Fe": "Mn", "O": "S", "Si": "Ge", "C": "Si", "Ti": "Zr", "Pb": "Sn"}
        new_atom = replacements.get(influential_atom, "X")
        suggestion = f"Try replacing influential {influential_atom} (at index {result['most_influential_atom_index']}) with {new_atom} to tune {target_property}."
        
        return {
            "atomic_importance": result['atomic_importance'],
            "elements": result['elements'],
            "most_influential_atom": result['most_influential_element'],
            "most_influential_index": result['most_influential_atom_index'],
            "mutation_suggestion": suggestion
        }

    @staticmethod
    def _create_mock_structure(formula: str) -> Structure:
        """
        Create a dummy structure for the formula to satisfy the graph builder.
        In a real app, this would come from a Generative Model (VAE/Diffusion).
        """
        # Create a simple cubic lattice for testing
        l = Lattice.cubic(4.0)
        
        # Simple parser for mock atoms (very naive)
        # Assumes formula like "Fe2O3" -> just take atoms as if 1 each for graph structure?
        # Or just hardcode a mix for now.
        # Let's map formula string to a valid pymatgen composition or just dummy atoms
        # For graph builder, we just need valid atomic numbers.
        
        # Naive "Parsing" for demo
        import re
        elements = re.findall(r'([A-Z][a-z]*)', formula)
        # Mock positions
        coords = []
        for i in range(len(elements)):
            coords.append([i * 0.1, i * 0.1, i * 0.1])
            
        return Structure(l, elements, coords)

    async def generate_candidates(self, prompt: str, weights: Dict[str, float], n_candidates: int = 3) -> List[MaterialCandidate]:
        """
        Generate candidate materials based on natural language prompt and property weights.
        
        Args:
            prompt: Text describing desired material properties.
            weights: Dictionary of properties and their importance.
            n_candidates: Number of candidates to generate.
            
        Returns:
            A list of MaterialCandidate objects with predicted properties.
        """
        candidates = []
        for _ in range(n_candidates):
            formula = random.choice(self.MOCK_FORMULAS)
            struct = self._create_mock_structure(formula)
            
            # Predict
            if self.predictor:
                try:
                    preds = self.predictor.predict(struct)
                    predicted_props = {
                        "formation_energy": preds["formation_energy"],
                        "band_gap": preds["band_gap"],
                        "density": preds["density"]
                    }
                    embedding = preds["embedding"]
                except Exception as e:
                    print(f"Prediction failed for {formula}: {e}")
                    # Fallback
                    predicted_props = {"formation_energy": 0.0, "band_gap": 0.0, "density": 0.0}
                    embedding = []
            else:
                # Fallback if no model loaded
                predicted_props = {prop: round(random.uniform(0.1, 5.0), 3) for prop in weights.keys()}
                embedding = []
            
            # Add stability if missing (derived mock)
            if "stability" not in predicted_props:
                predicted_props["stability"] = 0.95 if predicted_props.get("formation_energy", 0) < 0 else 0.5
            
            candidates.append(MaterialCandidate(
                formula=formula,
                predicted_properties=predicted_props,
                crystal_structure_cif=f"# CIF data for {formula}\n# Latent Vector: {embedding[:5]}...",
                structure_embedding=embedding
            ))
            
        return candidates

    async def predict_properties(self, crystal_structure: str) -> Dict[str, float | bool]:
        """
        Predict physical properties for a given crystal structure.
        """
        # Placeholder
        return {
            "formation_energy": -2.45,
            "is_metal": False,
            "band_gap": 1.5,
            "magnetic_moment": 0.0
        }

# Singleton instance for dependency injection/usage across the app
material_service = MaterialService()
