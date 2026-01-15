# Training Framework - MATTERGEN X

This directory contains the machine learning pipeline for material property prediction and discovery.

## Structure
- `/models`: PyTorch model architectures (e.g., GNNs, MLPs).
- `/scripts`: Training loops, evaluation, and data preparation scripts.
- `/datasets`: Placeholder for dataset loaders and synthetic data generators.

## Getting Started
1. **Prepare Data**: Place CIF/POSCAR files or property CSVs in the root `/data` directory.
2. **Train Model**:
   ```bash
   python scripts/train.py
   ```
3. **Export**: Trained weights are saved to the root `/models` directory for backend inference.

## Models
- `PropertyPredictor`: Standard MLP for scalar property prediction from latent embeddings.
- *Future*: Graph Neural Networks (GNNs) for direct crystal structure processing.
