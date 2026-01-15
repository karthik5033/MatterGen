from fastapi import APIRouter
from typing import List
from app.schemas.material import MaterialRequest, MaterialResponse
from app.services.material_service import material_service

router = APIRouter()

@router.post("/", response_model=List[MaterialResponse])
async def generate_material(request: MaterialRequest):
    """
    """
    return await material_service.generate_candidates(request.prompt, request.weights, request.n_candidates)
