from fastapi import APIRouter
from app.schemas.material import PredictionRequest, PredictionResponse
from app.services.material_service import material_service

router = APIRouter()

@router.post("/", response_model=PredictionResponse)
async def predict_properties(request: PredictionRequest):
    """
    Predict properties for a given material structure.
    """
    properties = await material_service.predict_properties(request.crystal_structure)
    return PredictionResponse(predicted_properties=properties)
