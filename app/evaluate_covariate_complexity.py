import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import entropy
from typing import Dict, List, Tuple, Union
import shap
import os
from pathlib import Path

# ---------- Utility Functions ----------

def load_data(csv_path: str, target_col: str = "target") -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    print("🔹 First 5 columns:", df.columns.tolist()[:5])
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def ensure_2d_shap(shap_output: Union[np.ndarray, shap.Explanation], class_idx: int = 1) -> np.ndarray:
    if isinstance(shap_output, shap.Explanation):
        values = shap_output.values
    else:
        values = shap_output

    if len(values.shape) == 3:  # (samples, features, classes)
        return values[:, :, class_idx]
    return values

# ---------- Covariate Complexity Components ----------

def calculate_feature_entropy(shap_matrix: np.ndarray) -> float:
    abs_shap = np.abs(shap_matrix)
    total_importance = np.sum(abs_shap, axis=0)
    if np.sum(total_importance) == 0:
        return float("nan")
    normalized = total_importance / np.sum(total_importance)
    return entropy(normalized, base=2)

def compute_mean_homogeneity(shap_matrix: np.ndarray) -> float:
    abs_shap = np.abs(shap_matrix)
    corr_matrix = np.corrcoef(abs_shap.T)
    mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
    mean_corr = np.nanmean(corr_matrix[mask])
    return mean_corr

def compute_regularity(shap_mean_values: np.ndarray, top_k: int = 10) -> float:
    top_indices = np.argsort(shap_mean_values)[-top_k:]
    top_values = shap_mean_values[top_indices]
    if np.sum(top_values) == 0:
        return float("nan")
    norm = top_values / np.sum(top_values)
    top_entropy = entropy(norm, base=2)
    return 1 - (top_entropy / np.log2(top_k))

# ---------- Main Evaluation Pipeline ----------

def evaluate_covariate_complexity(csv_path: str = "synthetic_data100_with_target.csv") -> Dict[str, float]:
    possible_paths = [
        Path(csv_path),
        Path("data") / csv_path,
        Path(__file__).parent / csv_path,
        Path(__file__).parent / "data" / csv_path
    ]

    found_path = next((p for p in possible_paths if p.exists()), None)
    if not found_path:
        raise FileNotFoundError(f"File '{csv_path}' not found in expected locations.")

    X, y = load_data(found_path)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = ensure_2d_shap(explainer.shap_values(X))

    mean_shap_values = np.mean(np.abs(shap_values), axis=0)
    entropy_score = calculate_feature_entropy(shap_values)
    homogeneity_score = compute_mean_homogeneity(shap_values)
    regularity_score = compute_regularity(mean_shap_values)

    return {
        "entropy": entropy_score,
        "homogeneity": homogeneity_score,
        "regularity": regularity_score,
        "covariate_complexity_score": np.nanmean([entropy_score, homogeneity_score, regularity_score])
    }

