from fastapi import APIRouter, HTTPException
from typing import List, Dict
from app.schemas.generation import GenerateRequest, GenerateResponse, ExplainRequest, ExplainResponse
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

@router.get("/map")
async def get_material_map():
    """
    Get 2D material map data for visualization.
    Serves the pre-computed 'material_map.json' or generates a mock one.
    """
    import os
    import json
    
    # Path to pre-computed map
    # Assuming models/embeddings/material_map.json relative to project root
    # We need to find project root relative to this file
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    map_path = os.path.join(base_dir, "models", "embeddings", "material_map.json")
    
    if os.path.exists(map_path):
        with open(map_path, 'r') as f:
            data = json.load(f)
        return data
    else:
        # Mock Data if file missing
        import random
        points = []
        formulas = ["LiFePO4", "SrTiO3", "GaN", "SiC", "BaTiO3", "MoS2", "CsPbI3", "Fe2O3"]
        for i in range(50):
            points.append({
                "id": f"mock-{i}",
                "formula": random.choice(formulas),
                "x": random.uniform(-10, 10),
                "y": random.uniform(-10, 10),
                "neighbors": [],
                "targets": {"band_gap": random.uniform(0, 4)}
            })
        return {"points": points}
