import torch
import numpy as np
import logging
from typing import Tuple, List, Dict
from training.models.deep_regressor import DeepMaterialRegressor

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MaterialPredictor:
    """
    Interface for making predictions with the DeepMaterialRegressor.
    Includes Monte Carlo Dropout for uncertainty estimation.
    """
    
    def __init__(self, model_path: str, input_dim: int = 20, output_dim: int = 3, device: str = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the saved model weights (.pt file).
            input_dim: Dimension of input features.
            output_dim: Dimension of output targets.
            device: 'cpu' or 'cuda'.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepMaterialRegressor(input_dim=input_dim, output_dim=output_dim).to(self.device)
        
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval() # Default to eval mode
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _enable_dropout(self, model):
        """
        Enable dropout layers during test time (for MC Dropout).
        Keep BatchNorm layers in eval mode to use running stats.
        """
        for m in model.modules():
            if list(m.children()) == []: # Leaf node
                if isinstance(m, torch.nn.Dropout):
                    m.train()

    def predict_with_uncertainty(self, features: np.ndarray, n_samples: int = 50) -> Dict[str, np.ndarray]:
        """
        Perform Monte Carlo Dropout Prediction.
        
        Why Uncertainty Matters in Scientific Discovery:
        ----------------------------------------------
        In materials discovery, we often explore unknown chemical spaces.
        1. Reliability: Identify when the model is "guessing" (high uncertainty) vs "knowing" (low uncertainty).
        2. Active Learning: Prioritize experiments on materials where the model is uncertain, maximizing information gain.
        3. Risk Mitigation: Avoid committing expensive synthesis resources to candidates with high predicted performance but high uncertainty.
        
        Args:
            features: Input feature vector(s). Shape (batch_size, input_dim) or (input_dim,).
            n_samples: Number of forward passes to perform.
            
        Returns:
            Dictionary containing:
            - 'mean': The average prediction (expected value).
            - 'std': The standard deviation (epistemic uncertainty).
        """
        # Ensure input is a batch
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        inputs = torch.FloatTensor(features).to(self.device)
        
        # 1. Enable Dropout
        self._enable_dropout(self.model)
        
        # 2. Multiple Forward Passes
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                outputs = self.model(inputs)
                predictions.append(outputs.cpu().numpy())
        
        # Stack predictions: Shape (n_samples, batch_size, output_dim)
        predictions = np.stack(predictions)
        
        # 3. Compute Statistics
        mean_preds = np.mean(predictions, axis=0)
        std_preds = np.std(predictions, axis=0)
        
        # Restore model to full eval mode (disable dropout) just in case
        self.model.eval()
        
        return {
            "mean": mean_preds,
            "std": std_preds,
            "raw_samples": predictions # Optional: useful for distribution plotting
        }


from training.models.cgcnn import CGCNN
from training.datasets.graph_builder import CrystalGraphBuilder
from pymatgen.core import Structure

class CGCNNPredictor:
    """
    Inference wrapper for the CGCNN model.
    Handles graph construction, batching (dim 0), and target un-scaling.
    """
    
    def __init__(self, model_path: str, device: str = None):
        """
        Args:
            model_path: Path to .pt file containing model_state_dict and scaler stats.
            device: 'cpu' or 'cuda'.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.builder = CrystalGraphBuilder(radius=8.0, dStep=0.2)
        
        # Load Checkpoint first to get scaler stats
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model_state = checkpoint['model_state_dict']
            self.scaler_mean = checkpoint['scaler_mean'].to(self.device)
            self.scaler_scale = checkpoint['scaler_scale'].to(self.device)
            logger.info(f"Loaded checkpoint from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

        # Initialize Model
        self.model = CGCNN(
            node_input_dim=4, 
            edge_input_dim=41, 
            node_hidden_dim=64, 
            n_conv_layers=3, 
            n_targets=3
        ).to(self.device)
        
        self.model.load_state_dict(self.model_state)
        self.model.eval()
        
    def predict(self, structure: Structure) -> Dict[str, float]:
        """
        Predict properties for a single structure.
        
        Returns:
            Dict with keys: formation_energy, band_gap, density, embedding
        """
        # 1. Build Graph
        atom_fea, nbr_fea, nbr_idx = self.builder.get_graph(structure)
        
        # 2. Batching (Batch size = 1)
        # We need to add batch dimension? 
        # Actually CGCNN expects 2D tensors for features, but batch_mapping handles mapping.
        # For 1 graph, batch_mapping is just zeros(n_atoms).
        n_atoms = atom_fea.shape[0]
        batch_mapping = torch.zeros(n_atoms, dtype=torch.long)
        
        # Move to device
        atom_fea = atom_fea.to(self.device)
        nbr_fea = nbr_fea.to(self.device)
        nbr_idx = nbr_idx.to(self.device)
        batch_mapping = batch_mapping.to(self.device)
        
        with torch.no_grad():
            # Get Embedding
            # n_graphs = 1
            embedding = self.model.get_crystal_embedding(atom_fea, nbr_fea, nbr_idx, batch_mapping, 1)
            
            # Get Prediction
            preds_norm = self.model(atom_fea, nbr_fea, nbr_idx, batch_mapping)
            
            # Un-normalize
            preds = preds_norm * self.scaler_scale + self.scaler_mean
            
        # Convert to python types
        preds = preds.cpu().numpy()[0]
        embedding = embedding.cpu().numpy()[0]
        
        return {
            "formation_energy": float(preds[0]),
            "band_gap": float(preds[1]),
            "density": float(preds[2]),
            "embedding": embedding.tolist()
        }

if __name__ == "__main__":
    # Test Block
    print("Testing Predictors...")
    
    # Mock model creation for test
    import os
    dummy_mlp_path = "temp_dummy_mlp.pt"
    dummy_cgcnn_path = "temp_dummy_cgcnn.pt"
    
    # 1. MLP Test
    model = DeepMaterialRegressor()
    torch.save(model.state_dict(), dummy_mlp_path)
    try:
        predictor = MaterialPredictor(model_path=dummy_mlp_path)
        print("MLP Predictor loaded.")
    except:
        pass
        
    # 2. CGCNN Test
    c_model = CGCNN()
    # Save with scaler stats
    torch.save({
        'model_state_dict': c_model.state_dict(),
        'scaler_mean': torch.tensor([0.0, 0.0, 0.0]),
        'scaler_scale': torch.tensor([1.0, 1.0, 1.0])
    }, dummy_cgcnn_path)
    
    try:
        cg_predictor = CGCNNPredictor(model_path=dummy_cgcnn_path)
        
        # Dummy Struct
        from pymatgen.core import Lattice
        s = Structure(Lattice.cubic(4.0), ["Cs", "Cl"], [[0,0,0], [0.5,0.5,0.5]])
        
        res = cg_predictor.predict(s)
        print("\nCGCNN Prediction for CsCl:")
        print(f"Formation Energy: {res['formation_energy']:.3f} eV/atom")
        print(f"Embedding Dim: {len(res['embedding'])}")
    except Exception as e:
        print(f"CGCNN Test Failed: {e}")

    # Clean up
    if os.path.exists(dummy_mlp_path): os.remove(dummy_mlp_path)
    if os.path.exists(dummy_cgcnn_path): os.remove(dummy_cgcnn_path)
