import os
import argparse
import numpy as np
import torch
import logging
from training.scripts.train_robust import prepare_data, set_seed
from training.models.deep_regressor import DeepMaterialRegressor

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_embeddings(args):
    """
    Generate and save material embeddings for the test set.
    """
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("Loading Data...")
    # Reuse valid splitting logic from train_robust
    # prepare_data returns: X_train, y_train, X_val, y_val, X_test, y_test, scaler
    _, _, _, _, X_test, y_test, scaler = prepare_data(args.data_path)
    
    logger.info(f"Test Set Size: {len(X_test)}")
    
    # Load Model
    model = DeepMaterialRegressor(input_dim=X_test.shape[1], output_dim=3).to(device)
    model_path = os.path.join(args.model_dir, "best_deep_regressor.pt")
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info("Model loaded.")

    # Generate Embeddings
    with torch.no_grad():
        inputs = X_test.to(device)
        embeddings = model.extract_embeddings(inputs).cpu().numpy()
        targets = y_test.numpy()

    logger.info(f"Generated Embeddings Shape: {embeddings.shape}")
    
    # Save Results
    os.makedirs(args.output_dir, exist_ok=True)
    
    np.save(os.path.join(args.output_dir, "test_embeddings.npy"), embeddings)
    np.save(os.path.join(args.output_dir, "test_targets.npy"), targets)
    
    logger.info(f"Embeddings saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Material Embeddings")
    parser.add_argument("--data_path", type=str, default="../../data/mock_materials.json")
    parser.add_argument("--model_dir", type=str, default="../../models")
    parser.add_argument("--output_dir", type=str, default="../../models/embeddings")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Path handling
    base_dir = os.path.dirname(__file__)
    if not os.path.isabs(args.data_path):
        args.data_path = os.path.join(base_dir, args.data_path)
    if not os.path.isabs(args.model_dir):
        args.model_dir = os.path.join(base_dir, args.model_dir)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(base_dir, args.output_dir)
        
    generate_embeddings(args)
