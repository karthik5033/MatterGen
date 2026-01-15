from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import generation

app = FastAPI(
    title="MATTERGEN X - API",
    description="Backend API for AI-driven material discovery and visualization.",
    version="0.1.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
import os

# Mount static files
# Ensure directory exists
os.makedirs("app/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include the generation router
app.include_router(generation.router, prefix="/api", tags=["Generation"])

@app.get("/")
async def root():
    return {
        "project": "MATTERGEN X",
        "status": "Operational",
        "message": "Welcome to the research-grade material discovery API."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
