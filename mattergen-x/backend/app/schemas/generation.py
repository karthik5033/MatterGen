from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import uuid

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Natural language description of desired material properties")
    weights: Dict[str, float] = Field(..., description="Dictionary mapping property names to their relative importance (0.0 to 1.0)")
    n_candidates: int = Field(default=3, ge=1, le=10, description="Number of candidate materials to generate")

class MaterialCandidate(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the candidate")
    formula: str = Field(..., description="Chemical formula of the material")
    predicted_properties: Dict[str, float | bool] = Field(..., description="Predicted properties for the generated material")
    crystal_structure_cif: Optional[str] = Field(None, description="Optional CIF string representation of the crystal structure")

class GenerateResponse(BaseModel):
    candidates: List[MaterialCandidate] = Field(..., description="List of generated material candidates")
    model_version: str = Field(default="v0.1-dummy", description="Version of the generator model used")

class ExplainRequest(BaseModel):
    structure_cif: Optional[str] = Field(None, description="CIF string of the structure to explain")
    formula: Optional[str] = Field(None, description="Formula to generate a mock structure for if CIF is missing")
    target_property: str = Field("band_gap", description="Property to explain (formation_energy, band_gap, density)")

class ExplainResponse(BaseModel):
    atomic_importance: List[float] = Field(..., description="Normalized importance score for each atom")
    elements: List[str] = Field(..., description="Element symbol for each atom corresponding to scores")
    most_influential_atom: str = Field(..., description="Symbol of the atom with the highest score")
    most_influential_index: int = Field(..., description="Index of the most influential atom")
    mutation_suggestion: str = Field(..., description="Heuristic suggestion for material improvement")
