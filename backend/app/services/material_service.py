import os
import torch
import logging
import numpy as np
from typing import List, Dict, Union
from app.schemas.material import MaterialResponse
from app.models_ml.cgcnn import CGCNN
from app.services.graph_builder import CrystalGraphBuilder
from pymatgen.core import Structure

logger = logging.getLogger(__name__)

class MaterialService:
    """
    Service for handling material discovery and property prediction logic.
    Integrates with trained CGCNN model.
    """

    def __init__(self):
        self.model = None
        self.graph_builder = None
        self.scaler_mean = None
        self.scaler_scale = None
        self._load_model()

    def _load_model(self):
        try:
            # Path to the trained model artifact
            # Assuming running from backend/
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../mattergen-x/models/cgcnn_best.pt"))
            
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found at {model_path}. Using mock logic.")
                return

            logger.info(f"Loading model from {model_path}...")
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            
            # Initialize Model Structure
            # Node dim=4, Edge dim=41 (defaults matching training script)
            self.model = CGCNN(node_input_dim=4, edge_input_dim=41, node_hidden_dim=64, n_targets=3)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.scaler_mean = checkpoint['scaler_mean'].cpu()
            self.scaler_scale = checkpoint['scaler_scale'].cpu()
            
            self.graph_builder = CrystalGraphBuilder(radius=8.0, dStep=0.2)
            logger.info("CGCNN Model loaded successfully.")
            
        except Exception as e:
            logger.error(f"Failed to load AI model: {e}")
            self.model = None

    async def generate_candidates(self, prompt: str, weights: Dict[str, float], n_candidates: int = 3) -> List[MaterialResponse]:
        """
        Generate a list of candidate materials dynamicall based on natural language prompt.
        Uses StructureGenerator (GenAI Proxy) -> CGCNN (Predictor).
        """
        from app.services.structure_generator import structure_generator
        
        # 1. Extract context keywords from prompt
        keywords = prompt.split()
        
        # 2. Generate Structures
        generated_data = structure_generator.generate_batch(n_candidates, context_keywords=keywords)
        
        candidates = []
        for i, data in enumerate(generated_data):
            cif = data["cif"]
            formula = data["formula"]
            struct_obj = data["structure"] # Get the object directly
            
            # 3. Predict Properties using trained AI (Pass object to avoid re-parsing)
            props = await self.predict_properties(struct_obj)
            
            # 4. Construct Response
            # ID generation
            mat_id = f"gen_{formula}_{i}"
            
            # Stability Heuristic string for UI
            stability_label = "High"
            # Normalize energy for display (heuristic range -4 to 0)
            e_val = props.get("formation_energy", 0)
            if e_val > -0.5:
                stability_label = "Low"
            elif e_val > -1.5:
                stability_label = "Medium"
                
            # Convert properties to UI format
            ui_props = {
                "band_gap": props.get("band_gap"),
                "stability": stability_label, # UI expects string label often or we handle it
                "conductivity": 1.0 if props.get("is_metal") else 0.0,
                "density": props.get("density")
            }

            candidates.append(MaterialResponse(
                id=mat_id,
                formula=formula,
                properties=ui_props,
                crystal_structure=cif # Frontend still needs CIF string
            ))
            
        return candidates

    async def predict_properties(self, crystal_data: Union[str, Structure]) -> Dict[str, Union[float, bool]]:
        """
        Predict properties for a given crystal structure (CIF string or Pymatgen Structure).
        """
        if not self.model:
            # Fallback to mock if model failed to load
            return {
                "formation_energy": -2.45,
                "is_metal": False,
                "band_gap": 1.5,
                "magnetic_moment": 0.0,
                "density": 5.0
            }

        try:
            # 1. Parse Structure if needed
            if isinstance(crystal_data, str):
                try:
                    struct = Structure.from_str(crystal_data, fmt="cif")
                except:
                    struct = Structure.from_str(crystal_data, fmt="poscar")
            else:
                struct = crystal_data

            # 2. Build Graph 
            # (Note: get_graph is CPU bound, could offload to threadpool if blocking heavily)
            atom_fea, nbr_fea, nbr_idx = self.graph_builder.get_graph(struct)
            
            # Batch mapping (single graph, all atoms 0)
            n_atoms = atom_fea.shape[0]
            batch_map = torch.zeros(n_atoms, dtype=torch.long)
            
            # 3. Inference
            with torch.no_grad():
                preds_norm = self.model(atom_fea, nbr_fea, nbr_idx, batch_map)
                
                # un-scale: preds = norm * scale + mean
                preds = preds_norm * self.scaler_scale + self.scaler_mean
                preds = preds.squeeze().numpy()
                
                # Map outputs (Order in training: E_f, Bg, Density)
                formation_energy = float(preds[0])
                band_gap = float(preds[1])
                density = float(preds[2])

            return {
                "formation_energy": round(formation_energy, 3),
                "band_gap": round(band_gap, 3),
                "density": round(density, 3),
                "is_metal": band_gap < 0.1, 
                "magnetic_moment": 0.0 
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "formation_energy": -1.0,
                "band_gap": 0.0,
                "density": 0.0,
                "error": str(e)
            }

material_service = MaterialService()
