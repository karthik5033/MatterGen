import os
import json
import numpy as np
import logging
import random
import re
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

class MapService:
    def __init__(self):
        self._cached_points = None
        self._is_loading = False
        self._cached_element_stats = None
        self._element_pattern = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)")
        
    async def get_map_points(self):
        if self._cached_points is not None:
            return self._cached_points
            
        if self._is_loading:
            return []

        self._is_loading = True
        try:
            from app.services.material_service import material_service
            material_service._lazy_init()
            raw_data = material_service._dataset
            
            if not raw_data:
                self._cached_points = []
            else:
                self._cached_points = self._process_embeddings(raw_data)
        except Exception as e:
            logger.error(f"Failed to load map data: {e}")
            self._cached_points = []
        finally:
            self._is_loading = False
            
        return self._cached_points

    def _process_embeddings(self, raw_data):
        valid_entries = [e for e in raw_data if "embedding" in e and len(e["embedding"]) > 0]
        if not valid_entries: return []
            
        # Optimization: use a smaller subset if extremely large
        MAX_POINTS = 800 
        if len(valid_entries) > MAX_POINTS:
            valid_entries = random.sample(valid_entries, MAX_POINTS)

        embeddings = np.array([e["embedding"] for e in valid_entries])
        embeddings_centered = embeddings - embeddings.mean(axis=0)
        u, s, vh = np.linalg.svd(embeddings_centered, full_matrices=False)
        coords = (u[:, :2] * s[:2])
        
        coords_min = coords.min(axis=0)
        coords_max = coords.max(axis=0)
        coords_norm = 20 * (coords - coords_min) / (coords_max - coords_min + 1e-6) - 10
        
        points = []
        for i, entry in enumerate(valid_entries):
            points.append({
                "id": str(entry.get("id", f"mat-{i}")),
                "formula": entry.get("formula", "Unknown"),
                "x": float(coords_norm[i, 0]),
                "y": float(coords_norm[i, 1]),
                "targets": entry.get("properties", {})
            })
        return points

    async def get_elemental_stats(self):
        if self._cached_element_stats is not None:
            return self._cached_element_stats

        from app.services.material_service import material_service
        material_service._lazy_init()
        raw_data = material_service._dataset
        
        if not raw_data:
            return {}
            
        # Use defaultdict for faster aggregation
        stats_agg = defaultdict(lambda: {"count": 0, "total_eh": 0.0, "total_bm": 0.0})
        
        print(f"DEBUG: Processing stats for {len(raw_data)} materials...")
        
        for entry in raw_data:
            formula = entry.get('formula', '')
            props = entry.get('properties', {})
            # Use 'energy_above_hull' for stability (lower is better, 0 is stable)
            eh = props.get('energy_above_hull', 0.0) 
            # Use 'bulk_modulus' for mechanical strength
            bm = props.get('bulk_modulus', 0.0)
            
            # Find all elements in formula
            elements = self._element_pattern.findall(formula)
            for el, amt in elements:
                s = stats_agg[el]
                s["count"] += 1
                s["total_eh"] += float(eh)
                s["total_bm"] += float(bm)
                
        # Final formatting
        self._cached_element_stats = {
            el: {
                "count": d["count"],
                "avg_energy_above_hull": d["total_eh"] / d["count"],
                "avg_bulk_modulus": d["total_bm"] / d["count"]
            }
            for el, d in stats_agg.items()
        }
        print("DEBUG: Elemental stats aggregation complete.")
        return self._cached_element_stats

map_service = MapService()
