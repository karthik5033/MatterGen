import os
import json
import logging
import pandas as pd
from typing import List, Dict, Optional, Any
from pymatgen.core import Structure
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class MaterialDataLoader:
    """
    Robust dataset loader for material science data using pymatgen.
    Supports Materials Project-style JSON/CSV formats.
    """

    REQUIRED_COLUMNS = ["structure", "formation_energy_per_atom", "band_gap", "density"]

    def __init__(self, data_path: str):
        """
        Initialize the loader.
        
        Args:
            data_path: Path to the dataset file (JSON or CSV).
        """
        self.data_path = data_path
        self.raw_data: List[Dict[str, Any]] = []
        self.cleaned_data: List[Dict[str, Any]] = []

    def load(self) -> List[Dict[str, Any]]:
        """
        Load, parse, and validate the dataset.
        
        Returns:
            A list of validated material samples.
        """
        logger.info(f"Loading data from {self.data_path}...")
        
        if not os.path.exists(self.data_path):
            logger.error(f"File not found: {self.data_path}")
            raise FileNotFoundError(f"File not found: {self.data_path}")

        try:
            if self.data_path.endswith(".json"):
                with open(self.data_path, "r") as f:
                    self.raw_data = json.load(f)
            elif self.data_path.endswith(".csv"):
                df = pd.read_csv(self.data_path)
                self.raw_data = df.to_dict(orient="records")
            else:
                raise ValueError("Unsupported file format. Use .json or .csv")
            
            logger.info(f"Loaded {len(self.raw_data)} raw entries.")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

        self._process_data()
        return self.cleaned_data

    def _process_data(self):
        """
        Process raw data: parse structures, extract properties, and validate.
        Ensures determinism by sorting the output.
        """
        valid_samples = []
        
        for idx, entry in enumerate(self.raw_data):
            try:
                # 1. Parse Structure
                struct = self._parse_structure(entry)
                if not struct:
                    logging.warning(f"Skipping entry {idx}: Invalid structure data.")
                    continue

                # 2. Extract Properties
                properties = self._extract_properties(entry)
                if not properties:
                    logging.warning(f"Skipping entry {idx}: Missing target properties.")
                    continue

                # 3. Extract Computed Features (Lattice, Composition)
                features = self._compute_features(struct)

                # 4. Combine
                sample = {
                    "structure_obj": struct,  # Keep pymatgen object for advanced usage
                    **features,
                    **properties
                }
                valid_samples.append(sample)

            except Exception as e:
                logger.warning(f"Error processing entry {idx}: {e}")

        # ensure determinism
        # sort by formula to ensure reproducible order
        valid_samples.sort(key=lambda x: x["formula"])
        
        self.cleaned_data = valid_samples
        logger.info(f"Successfully processed {len(self.cleaned_data)} valid samples.")

    def _parse_structure(self, entry: Dict[str, Any]) -> Optional[Structure]:
        """Parse structure from dict or string."""
        try:
            struct_data = entry.get("structure") or entry.get("cif") or entry.get("poscar")
            
            if isinstance(struct_data, dict):
                return Structure.from_dict(struct_data)
            elif isinstance(struct_data, str):
                return Structure.from_str(struct_data, fmt="cif" if "cif" in str(entry.keys()).lower() else "poscar")
            
            return None
        except Exception as e:
            logger.debug(f"Structure parsing error: {e}")
            return None

    def _extract_properties(self, entry: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Extract target labels."""
        try:
            # Map common property names if needed
            e_form = entry.get("formation_energy_per_atom")
            bg = entry.get("band_gap")
            density = entry.get("density")
            
            # Simple validation: we need at least these for our specific model requirement
            if e_form is None or bg is None: 
                return None
                
            return {
                "label_formation_energy": float(e_form),
                "label_band_gap": float(bg),
                "label_density": float(density) if density else 0.0
            }
        except ValueError:
            return None

    def _compute_features(self, struct: Structure) -> Dict[str, Any]:
        """Compute basic features from pymatgen Structure."""
        sga = SpacegroupAnalyzer(struct)
        return {
            "formula": struct.composition.reduced_formula,
            "nsites": struct.num_sites,
            "volume": struct.volume,
            "density_computed": struct.density,
            "lattice_a": struct.lattice.a,
            "lattice_b": struct.lattice.b,
            "lattice_c": struct.lattice.c,
            "lattice_alpha": struct.lattice.alpha,
            "lattice_beta": struct.lattice.beta,
            "lattice_gamma": struct.lattice.gamma,
            "crystal_system": sga.get_crystal_system(),
            "spacegroup_number": sga.get_space_group_number()
        }

if __name__ == "__main__":
    # Test block
    print("This module is a library. Import MaterialDataLoader to use.")
