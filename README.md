# MATTERGEN X

Research-grade AI Material Discovery and Visualization Platform.

## Architecture

- **`frontend/`**: Next.js (App Router, TypeScript, Tailwind CSS)
- **`backend/`**: FastAPI (Python, Pydantic)
- **`training/`**: PyTorch scripts for model development and synthetic data generation.

## Getting Started

### Backend
1. `cd backend`
2. `pip install -r requirements.txt`
3. `python main.py`

### Frontend
1. `cd frontend`
2. `npm install`
3. `npm run dev`

### Training
1. `cd training`
2. `pip install -r requirements.txt`
3. `python datasets/synthetic_generator.py`
4. `python scripts/train_property_predictor.py`

## Features

- **Generative Design**: Input natural language constraints and property weights.
- **Real-time Prediction**: (Simulated) AI-driven property prediction for crystal structures.
- **Sleek UI**: Industrial-grade research dashboard.
