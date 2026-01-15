from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import generation, prediction, visualization

app = FastAPI(
    title="MATTERGEN X API",
    description="Research-grade AI Material Discovery Platform API",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
# Include routers
app.include_router(generation.router, prefix="/api/v1/generate", tags=["generation"])
app.include_router(prediction.router, prefix="/api/v1/predict", tags=["prediction"])
app.include_router(visualization.router, prefix="/api", tags=["visualization"])

@app.get("/")
async def root():
    return {"message": "Welcome to MATTERGEN X API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
