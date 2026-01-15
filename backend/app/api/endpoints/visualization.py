from fastapi import APIRouter
import random

router = APIRouter()

@router.get("/map")
async def get_embedding_map():
    """
    Returns 2D embedding coordinates for material visualization.
    Mock data for now.
    """
    points = []
    # Generate 100 mock points
    for i in range(100):
        # Create 3 distinct clusters
        cluster = i % 3
        center_x = [0.2, 0.5, 0.8][cluster]
        center_y = [0.2, 0.8, 0.5][cluster]
        
        # Add random noise
        x = max(0, min(1, center_x + random.gauss(0, 0.1)))
        y = max(0, min(1, center_y + random.gauss(0, 0.1)))
        
        points.append({
            "id": f"mat-{i}",
            "x": x,
            "y": y,
            "formula": f"Material-{i}",
            "band_gap": round(random.uniform(0.5, 4.0), 2),
            "cluster": cluster
        })
        
    return {"points": points}
