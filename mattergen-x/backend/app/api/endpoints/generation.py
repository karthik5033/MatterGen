
from fastapi import APIRouter, HTTPException
from typing import List, Dict
from app.schemas.generation import GenerateRequest, GenerateResponse, ExplainRequest, ExplainResponse, ChatRequest, ChatResponse
from app.services.material_service import material_service

router = APIRouter()

@router.post("/generate", response_model=GenerateResponse)
async def generate_materials(request: GenerateRequest):
    """
    Propose new material candidates based on project specifications.
    Utilizes the Generative Discovery Engine via MaterialService.
    """
    try:
        candidates = await material_service.generate_candidates(
            prompt=request.prompt,
            weights=request.weights,
            n_candidates=request.n_candidates
        )
        return GenerateResponse(candidates=candidates)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Discovery engine error: {str(e)}")

@router.post("/plots", response_model=Dict[str, str])
async def generate_plots(request: GenerateRequest):
    """
    Generate scientific plots based on mock training data (demo purpose).
    In a real scenario, this would visualize the specific job's results.
    """
    from training.visualization.plotting import PlotGenerator
    import numpy as np
    
    # Initialize Generator (outputs to app/static/plots)
    # We need to map relative path correctly. 
    # Current CWD is backend/. So app/static/plots is correct respective to CWD.
    plotter = PlotGenerator(output_dir="app/static/plots")
    
    # Mock Data for Demo
    true_vals = np.random.rand(50) * 5
    pred_vals = true_vals + np.random.normal(0, 0.2, 50)
    scores = np.cumsum(np.random.normal(0.1, 0.05, 20))
    
    # Generate Plots
    parity_url = plotter.plot_predicted_vs_true(true_vals, pred_vals, "Band Gap")
    traj_url = plotter.plot_optimization_trajectory(scores.tolist())
    
    return {
        "parity_plot": parity_url,
        "trajectory_plot": traj_url
    }

@router.post("/explain", response_model=ExplainResponse)
async def explain_prediction(request: ExplainRequest):
    """
    Explain the model's prediction for a given material structure.
    Returns atomic importance scores and a heuristic mutation suggestion.
    """
    from app.schemas.generation import ExplainRequest, ExplainResponse # Ensure import is available if not at top
    # Note: Imports at top are better. I will rely on existing imports or update them.
    # checking imports: 'from app.schemas.generation import GenerateRequest, GenerateResponse'
    # I should update the import line first or doing it inside if needed, but cleaner to update top.
    
    try:
        explanation = await material_service.explain_structure(
            structure_cif=request.structure_cif,
            formula=request.formula,
            target_property=request.target_property
        )
        return ExplainResponse(**explanation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

@router.post("/chat/refine", response_model=ChatResponse)
async def refine_specification(request: ChatRequest):
    """
    Refine a raw material specification using Aether Assistant.
    Translates chat into ML-ready prompts and weights.
    """
    from app.services.gemini_service import gemini_service
    result = await gemini_service.chat_with_assistant(
        message=request.message,
        context=request.context
    )
    return ChatResponse(**result)

@router.get("/map")
async def get_material_map():
    """
    Get 2D material map data for visualization.
    Uses cached MapService to return points efficiently.
    """
    from app.services.map_service import map_service
    points = await map_service.get_map_points()
    return {"points": points}

@router.get("/stats/elements")
async def get_element_statistics():
    """
    Get aggregated elemental statistics (frequency, stability) from the dataset.
    Used for the Periodic Table Heatmap.
    """
    from app.services.map_service import map_service
    stats = await map_service.get_elemental_stats()
    return stats
