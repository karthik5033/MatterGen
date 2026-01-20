
from fastapi import APIRouter, HTTPException
from app.schemas.generation import AnalysisRequest, AnalysisResponse
from app.services.gemini_service import gemini_service

router = APIRouter()

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_material(request: AnalysisRequest):
    """
    Generate a detailed scientific and sci-fi analysis of a material using Gemini AI.
    """
    try:
        # Call Gemini Service
        result = await gemini_service.analyze_material(request.formula, request.properties)
        
        # Ensure we handle potential missing keys gracefully or let validation kick in
        # For simplicity, assuming service returns correct structure due to Prompt
        
        # Helper to safely get nested dict
        ratings_data = result.get("ratings", {})
        
        return AnalysisResponse(
            executive_summary=result.get("executive_summary", "Summary unavailable."),
            scientific_deep_dive=result.get("scientific_deep_dive", "Analysis incomplete."),
            industrial_applications=result.get("industrial_applications", []),
            future_tech_lore=result.get("future_tech_lore", "Classified."),
            ratings={
                "commercial_viability": ratings_data.get("commercial_viability", 50),
                "sustainability_index": ratings_data.get("sustainability_index", 50),
                "manufacturing_complexity": ratings_data.get("manufacturing_complexity", 50)
            },
            synthesis_guide=result.get("synthesis_guide", {"step_1": "N/A", "step_2": "N/A", "conditions": "N/A"}),
            risk_profile=result.get("risk_profile", {"environmental_impact": "Unknown", "safety_hazards": "Unknown"}),
            economic_outlook=result.get("economic_outlook", {"market_readiness": "Unknown", "cost_estimation": "Unknown"})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
