import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import uuid
import logging
from typing import List, Dict, Optional

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for publication quality
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

class PlotGenerator:
    """
    Generates scientific plots for material analysis.
    Output: Saves PNG files and returns their paths/URLs.
    """
    
    def __init__(self, output_dir: str = "backend/static/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _save_fig(self, fig, filename_prefix: str) -> str:
        """Helper to save figure and return relative path."""
        unique_id = uuid.uuid4().hex[:8]
        filename = f"{filename_prefix}_{unique_id}.png"
        filepath = os.path.join(self.output_dir, filename)
        
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Return URL path assuming /static mount
        # If output_dir is backend/static/plots, url is /static/plots/...
        # This assumes standard FastAPI static mount at /static
        relative_path = os.path.relpath(filepath, start="backend")
        return f"/{relative_path.replace(os.sep, '/')}"

    def plot_predicted_vs_true(self, y_true: np.ndarray, y_pred: np.ndarray, property_name: str) -> str:
        """
        Generate Predicted vs True Scatter Plot.
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        
        ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='w', s=60)
        
        # Diagonal line
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
        
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        ax.set_xlabel(f"True {property_name}")
        ax.set_ylabel(f"Predicted {property_name}")
        ax.set_title(f"Parity Plot: {property_name}")
        
        mae = np.mean(np.abs(y_true - y_pred))
        ax.text(0.05, 0.95, f"MAE: {mae:.3f}", transform=ax.transAxes, 
                verticalalignment='top', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
        return self._save_fig(fig, f"parity_{property_name}")

    def plot_optimization_trajectory(self, scores: List[float], label: str = "Optimization Score") -> str:
        """
        Generate Optimization Trajectory Plot.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        iterations = range(1, len(scores) + 1)
        ax.plot(iterations, scores, marker='o', linestyle='-', linewidth=2, markersize=8, color='#2ecc71')
        
        # Compute cumulative max (best so far)
        cum_max = np.maximum.accumulate(scores)
        ax.plot(iterations, cum_max, linestyle='--', color='gray', alpha=0.7, label='Best So Far')

        ax.set_xlabel("Iteration")
        ax.set_ylabel(label)
        ax.set_title("Optimization Trajectory")
        ax.legend()
        
        return self._save_fig(fig, "trajectory")

    def plot_property_distribution(self, properties: Dict[str, List[float]]) -> Dict[str, str]:
        """
        Generate histograms for multiple properties.
        Returns dict mapping property name to image URL.
        """
        urls = {}
        for prop, values in properties.items():
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.histplot(values, kde=True, ax=ax, color='#3498db')
            ax.set_title(f"Distribution: {prop}")
            urls[prop] = self._save_fig(fig, f"dist_{prop}")
        return urls

if __name__ == "__main__":
    # Test
    print("Testing Plot Generator...")
    gen = PlotGenerator(output_dir="../../backend/app/static/plots") # Local test path adjustment
    
    # Parity
    true = np.random.rand(50) * 5
    pred = true + np.random.normal(0, 0.2, 50)
    url = gen.plot_predicted_vs_true(true, pred, "Band Gap (eV)")
    print(f"Parity Plot saved: {url}")
    
    # Trajectory
    scores = [0.5, 0.6, 0.55, 0.7, 0.8, 0.82, 0.85]
    url = gen.plot_optimization_trajectory(scores)
    print(f"Trajectory Plot saved: {url}")
