from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api.endpoints import generation, analysis

app = FastAPI(
    title="MATTERGEN X - API",
    description="Backend API for AI-driven material discovery and visualization.",
    version="0.1.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    print(f"DEBUG: Request {request.method} {request.url.path}")
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    print(f"DEBUG: Response {response.status_code} (took {process_time:.2f}ms)")
    return response

from fastapi.staticfiles import StaticFiles
# Mount static files
os.makedirs("app/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include the generation router
app.include_router(generation.router, prefix="/api", tags=["Generation"])
app.include_router(analysis.router, prefix="/api", tags=["Analysis"])

@app.get("/")
async def root():
    return {
        "project": "MATTERGEN X",
        "status": "Operational",
        "message": "Welcome to the research-grade material discovery API."
    }

if __name__ == "__main__":
    import uvicorn
    # Defaulting to 8002 to match frontend expectations
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
