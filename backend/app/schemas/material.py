from pydantic import BaseModel
from typing import Dict, Optional

class MaterialRequest(BaseModel):
    prompt: str
    weights: Dict[str, float]
    n_candidates: Optional[int] = 3

class MaterialResponse(BaseModel):
    id: str
    formula: str
    properties: Dict[str, float | str | bool]
    crystal_structure: str

class PredictionRequest(BaseModel):
    crystal_structure: str

class PredictionResponse(BaseModel):
    predicted_properties: Dict[str, float | bool]
