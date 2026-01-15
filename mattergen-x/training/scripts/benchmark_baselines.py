import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from training.datasets.loader import MaterialDataLoader
from training.datasets.featurizer import ChemicalFeaturizer

def evaluate_model(model, X_test, y_test, name):
    """Evaluate a model and return metrics."""
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return {"Model": name, "MAE": mae, "RMSE": rmse}

def benchmark(data_path: str, target_property: str = "label_band_gap"):
    print(f"--- Starting Baseline Benchmark ---")
    print(f"Dataset: {data_path}")
    print(f"Target: {target_property}\n")

    # 1. Load Data
    loader = MaterialDataLoader(data_path)
    try:
        data = loader.load()
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    if not data:
        print("No data found.")
        return

    # 2. Featurize
    print("Featurizing data...")
    featurizer = ChemicalFeaturizer()
    
    X = []
    y = []
    
    for sample in data:
        formula = sample.get("formula")
        target = sample.get(target_property)
        
        if formula and target is not None:
            features = featurizer.featurize_formula(formula)
            X.append(features)
            y.append(target)
            
    X = np.array(X)
    y = np.array(y)
    
    print(f"Samples: {len(X)}, Features: {X.shape[1]}")

    if len(X) < 10:
        print("Not enough data to split for benchmarking.")
        return

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []

    # 4. Train & Evaluate Baselines

    # Mean Predictor
    mean_model = DummyRegressor(strategy="mean")
    mean_model.fit(X_train, y_train)
    results.append(evaluate_model(mean_model, X_test, y_test, "Mean Baseline"))

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    results.append(evaluate_model(lr, X_test, y_test, "Linear Regression"))

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    results.append(evaluate_model(rf, X_test, y_test, "Random Forest"))

    # 5. Report
    results_df = pd.DataFrame(results)
    print("\n--- Benchmark Results ---")
    print(results_df.to_string(index=False))
    print("\n" + "-"*30)

if __name__ == "__main__":
    import os
    # Default to mock data if no args
    default_path = os.path.join(os.path.dirname(__file__), "../../data/mock_materials.json")
    benchmark(default_path)
