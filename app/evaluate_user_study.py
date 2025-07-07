import os
import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Union


def load_csv_data(csv_path: str, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")

    print("\n[Data Loading]")
    print(f"Initial shape: {df.shape}")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    return X, y


def mark_untrustworthy_features(X: pd.DataFrame, n_untrustworthy: int = 5, seed: int = 42) -> List[str]:
    np.random.seed(seed)
    return list(np.random.choice(X.columns, size=n_untrustworthy, replace=False))


def add_spurious_correlations(X: pd.DataFrame, y: pd.Series, untrustworthy_features: List[str]) -> pd.DataFrame:
    X_new = X.copy()
    for f in untrustworthy_features:
        noise = np.random.normal(0.5, 0.1, size=len(y))
        X_new[f] = X_new[f] + (y * noise)
    return X_new


def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


def get_shap_explanations(model: RandomForestClassifier, X_sample: pd.DataFrame) -> np.ndarray:
    

    explainer = shap.TreeExplainer(model)
    shap_output = explainer(X_sample)

    if len(shap_output.shape) == 3:
        return shap_output.values[:, :, 1]
    return shap_output.values


def explanation_uses_untrustworthy(shap_values: np.ndarray, feature_names: List[str], untrustworthy_features: List[str], top_k: int = 5) -> List[Tuple[int, List[str]]]:
    used_untrustworthy = []
    for i in range(shap_values.shape[0]):
        abs_vals = np.abs(shap_values[i])
        top_features = np.argsort(abs_vals)[-top_k:]
        top_feature_names = [feature_names[j] for j in top_features]
        used = [f for f in top_feature_names if f in untrustworthy_features]
        used_untrustworthy.append((i, used))
    return used_untrustworthy


def mask_untrustworthy_features(X: pd.DataFrame, untrustworthy_features: List[str]) -> pd.DataFrame:
    X_masked = X.copy()
    for f in untrustworthy_features:
        if f in X_masked.columns:
            X_masked[f] = 0
    return X_masked


def evaluate_trustworthiness(model: RandomForestClassifier, X: pd.DataFrame, y: pd.Series, untrustworthy_features: List[str]) -> float:
    print("\n[Experiment 1: Trust Evaluation]")
    X_sample = X.sample(n=min(20, len(X)), random_state=42)

    shap_values = get_shap_explanations(model, X_sample)

    used_untrustworthy = explanation_uses_untrustworthy(
        shap_values,
        X.columns.tolist(),
        untrustworthy_features,
        top_k=5
    )

    for i, used in used_untrustworthy:
        if used:
            print(f"Sample {i} uses untrustworthy features: {used}")

    X_masked = mask_untrustworthy_features(X_sample, untrustworthy_features)
    y_pred_orig = model.predict(X_sample)
    y_pred_masked = model.predict(X_masked)

    changed = (y_pred_orig != y_pred_masked).sum()
    change_ratio = changed / len(X_sample)
    print(f"Predictions changed after masking: {changed}/{len(X_sample)} ({change_ratio:.1%})")

    return change_ratio


def evaluate_model_selection(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, untrustworthy_features: List[str]) -> str:
    print("\n[Experiment 2: Model Selection]")

    model_a = train_model(X_train, y_train)
    acc_a = model_a.score(X_test, y_test)

    X_train_spurious = add_spurious_correlations(X_train, y_train, untrustworthy_features)
    X_test_spurious = add_spurious_correlations(X_test, y_test, untrustworthy_features)
    model_b = train_model(X_train_spurious, y_train)
    acc_b = model_b.score(X_test_spurious, y_test)

    print(f"\nModel A (clean) accuracy: {acc_a:.3f}")
    print(f"Model B (spurious) accuracy: {acc_b:.3f}")

    X_eval = X_test.sample(n=min(20, len(X_test)), random_state=42)
    shap_a = get_shap_explanations(model_a, X_eval)
    shap_b = get_shap_explanations(model_b, X_eval)

    used_untrustworthy_a = explanation_uses_untrustworthy(shap_a, X_train.columns.tolist(), untrustworthy_features)
    used_untrustworthy_b = explanation_uses_untrustworthy(shap_b, X_train.columns.tolist(), untrustworthy_features)

    count_a = sum(1 for _, u in used_untrustworthy_a if u)
    count_b = sum(1 for _, u in used_untrustworthy_b if u)

    print(f"\nModel A used untrustworthy features in {count_a}/20 explanations")
    print(f"Model B used untrustworthy features in {count_b}/20 explanations")

    better_model = "A" if count_a < count_b else "B"
    print(f"Conclusion: Model {better_model} is more trustworthy")

    return better_model


def run_user_study(csv_path: str, target_column: str, n_untrustworthy: int = 5) -> Dict[str, Union[List[str], float, str]]:
    try:
        print("\n=== Starting User Study ===")

        X, y = load_csv_data(csv_path, target_column)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        untrustworthy = mark_untrustworthy_features(X_train, n_untrustworthy)
        print("\nUntrustworthy features:")
        for feat in untrustworthy:
            print(f" - {feat}")

        model = train_model(X_train, y_train)
        trust_score = evaluate_trustworthiness(model, X_test, y_test, untrustworthy)
        better_model = evaluate_model_selection(X_train, y_train, X_test, y_test, untrustworthy)

        return {
            "untrustworthy_features": untrustworthy,
            "trustworthiness_score": trust_score,
            "better_model_by_explanation": better_model
        }

    except Exception as e:
        print(f"\nError in user study: {str(e)}")
        return {
            "error": str(e),
            "untrustworthy_features": [],
            "trustworthiness_score": -1,
            "better_model_by_explanation": "Error"
        }
