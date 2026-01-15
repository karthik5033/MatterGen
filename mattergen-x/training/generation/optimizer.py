import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor

from training.generation.search_space import CompositionSearchSpace
from training.datasets.featurizer import ChemicalFeaturizer
from backend.app.services.material_service import material_service # Use existing singleton wrapper or predictor directly?
from training.inference import CGCNNPredictor
# Ideally we should accept predictor instance to decouple from backend service, 
# but for now we can rely on passing it in or init inside.

logger = logging.getLogger(__name__)

class BayesianMaterialOptimizer:
    """
    Bayesian/Evolutionary Optimizer for Material Discovery.
    
    Workflow:
    1. Initial Sampling: Generate random candidates.
    2. Evaluation (Oracle): Run CGCNN on candidates.
    3. Update Surrogate: Train Random Forest on (Features -> Score).
    4. Acquisition: 
       - Generate mutated candidates (Exploration).
       - Predict scores using Surrogate (Exploitation/Filtering).
       - Select top candidates to run Oracle on.
    5. Repeat.
    """
    
    def __init__(self, 
                 predictor: CGCNNPredictor,
                 search_space: CompositionSearchSpace,
                 target_weights: Dict[str, float]):
        """
        Args:
            predictor: The expensive "Oracle" (CGCNN).
            search_space: Defines valid compositions.
            target_weights: e.g., {"formation_energy": -1.0, "band_gap": 0.5} 
                            (Negative weight means minimize).
        """
        self.predictor = predictor
        self.search_space = search_space
        self.weights = target_weights
        
        self.featurizer = ChemicalFeaturizer()
        self.surrogate = RandomForestRegressor(n_estimators=50, random_state=42)
        
        # History
        self.formulas: List[str] = []
        self.scores: List[float] = []
        self.features: List[np.ndarray] = []
        
    def _calculate_score(self, preds: Dict[str, float]) -> float:
        """Compute scalar objective from predictions."""
        score = 0.0
        for prop, weight in self.weights.items():
            # For band_gap, usually maximize or target specific? Assumed linear for now.
            # Usually E_f should be minimized (negative weight).
            val = preds.get(prop, 0.0)
            score += val * weight
        return score

    def _update_surrogate(self):
        """Train surrogate model on observed data."""
        if len(self.features) < 10:
            return # Not enough data yet
            
        X = np.array(self.features)
        y = np.array(self.scores)
        self.surrogate.fit(X, y)

    def optimize(self, n_iterations: int = 5, n_candidates_per_iter: int = 10):
        """
        Run the optimization loop.
        
        Returns:
            trajectory: List of best scores per iteration.
            best_candidates: List of best material per iteration.
        """
        trajectory = []
        best_candidates_history = []
        
        # 1. Initialization (Random Search)
        current_formulas = [self.search_space.sample() for _ in range(n_candidates_per_iter)]
        
        for iteration in range(n_iterations):
            logger.info(f"--- Iteration {iteration+1}/{n_iterations} ---")
            
            # A. Evaluate Candidates (Oracle)
            results = []
            for formula in current_formulas:
                # Mock structure generation (In real app, use generator)
                from backend.app.services.material_service import MaterialService
                struct = MaterialService._create_mock_structure(formula)
                
                try:
                    preds = self.predictor.predict(struct)
                    score = self._calculate_score(preds)
                    
                    feat = self.featurizer.featurize_formula(formula)
                    
                    # Store observed
                    self.formulas.append(formula)
                    self.scores.append(score)
                    self.features.append(feat)
                    
                    results.append((formula, score, preds))
                except Exception as e:
                    logger.warning(f"Failed to evaluate {formula}: {e}")

            # Update Surrogate
            self._update_surrogate()
            
            # Track Best
            if results:
                best_in_iter = max(results, key=lambda x: x[1])
                trajectory.append(best_in_iter[1])
                best_candidates_history.append(best_in_iter)
                logger.info(f"Best: {best_in_iter[0]} (Score: {best_in_iter[1]:.4f})")
            
            # B. Propose Next Batch (Acquisition)
            # Strategy: Generate M mutations, Rank with Surrogate, Select Top N
            m_pool_size = n_candidates_per_iter * 5
            pool = []
            
            # Seed pool with best formulas found so far (Evolutionary)
            top_indices = np.argsort(self.scores)[-5:] # Top 5 overall
            parents = [self.formulas[i] for i in top_indices]
            if not parents: parents = [self.search_space.sample()]
            
            for _ in range(m_pool_size):
                parent = random.choice(parents)
                child = self.search_space.mutate(parent)
                pool.append(child)
                
            # Filter pool with surrogate
            if len(self.features) >= 10:
                pool_feats = [self.featurizer.featurize_formula(f) for f in pool]
                pred_scores = self.surrogate.predict(pool_feats)
                
                # UCB-like Acquisition? Or just Greedy for now.
                # Greedy on surrogate:
                # combined = list(zip(pool, pred_scores))
                # combined.sort(key=lambda x: x[1], reverse=True)
                # current_formulas = [x[0] for x in combined[:n_candidates_per_iter]]
                
                # Let's add some exploration (epsilon-greedy or retain some randoms)
                top_k = n_candidates_per_iter - 2
                best_indices = np.argsort(pred_scores)[-top_k:]
                
                next_batch = [pool[i] for i in best_indices]
                # Add 2 randoms for exploration
                next_batch.extend([self.search_space.sample() for _ in range(2)])
                
                current_formulas = next_batch
            else:
                # Random if not enough data
                current_formulas = [self.search_space.sample() for _ in range(n_candidates_per_iter)]

        return trajectory, best_candidates_history

if __name__ == "__main__":
    # Test Block
    print("Testing Optimizer...")
    
    # 1. Mock Predictor Setup
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(base_dir, "models", "cgcnn_best.pt")
    
    # Needs a real model file to work, or we mock
    if os.path.exists(model_path):
        predictor = CGCNNPredictor(model_path)
        space = CompositionSearchSpace(allowed_elements=["Li", "Fe", "O", "P"])
        opt = BayesianMaterialOptimizer(predictor, space, {"formation_energy": -1.0})
        
        traj, bests = opt.optimize(n_iterations=2, n_candidates_per_iter=3)
        print("\nOptimization Complete.")
        print(f"Trajectory: {traj}")
    else:
        print("Skipping run (No model found).")
