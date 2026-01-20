
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import uuid

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Natural language description of desired material properties")
    weights: Dict[str, float] = Field(..., description="Dictionary mapping property names to their relative importance (0.0 to 1.0)")
    n_candidates: int = Field(default=3, ge=1, le=100, description="Number of candidate materials to generate")

class MaterialCandidate(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the candidate")
    formula: str = Field(..., description="Chemical formula of the material")
    predicted_properties: Dict[str, float | bool] = Field(..., description="Predicted properties for the generated material")
    crystal_structure_cif: Optional[str] = Field(None, description="Optional CIF string representation of the crystal structure")
    structure_embedding: Optional[List[float]] = Field(None, description="Vector embedding of the structure")
    score: float = Field(default=0.0, description="Match confidence score (0.0 to 1.0)")

class GenerateResponse(BaseModel):
    candidates: List[MaterialCandidate] = Field(..., description="List of generated material candidates")
    model_version: str = Field(default="Optimized-CGCNN-v2", description="Version of the generator model used")

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

class AnalysisRequest(BaseModel):
    formula: str
    properties: Dict[str, float | bool]

class Application(BaseModel):
    title: str
    description: str
    performance_metric: str

class Ratings(BaseModel):
    commercial_viability: int
    sustainability_index: int
    manufacturing_complexity: int

class AnalysisResponse(BaseModel):
    executive_summary: str
    scientific_deep_dive: str
    industrial_applications: List[Application]
    future_tech_lore: str
    ratings: Ratings
    
    # New Extended Content
    synthesis_guide: Dict[str, Any] = Field(..., description="Rich synthesis details")
    risk_profile: Dict[str, Any] = Field(..., description="Comprehensive risk analysis")
    economic_outlook: Dict[str, Any] = Field(..., description="Market and cost analysis")
    # Backwards compatibility fields effectively deprecated but kept if needed or can be removed if frontend updated
class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    suggested_prompt: Optional[str] = None
    suggested_weights: Optional[Dict[str, float]] = None
