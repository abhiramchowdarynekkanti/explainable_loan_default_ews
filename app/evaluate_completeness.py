import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# --------------------
# Preservation Check
# --------------------
def preservation_check(X: pd.DataFrame, shap_df: pd.DataFrame, model, top_k=5, strategy="zero"):
    X_preserved = X.copy()
    original_preds = model.predict(X)

    for i in range(len(X)):
        shap_row = shap_df.iloc[i]
        top_features = shap_row.abs().sort_values(ascending=False).head(top_k).index.tolist()

        for feature in X.columns:
            if feature not in top_features:
                if strategy == 'zero':
                    X_preserved.iat[i, X.columns.get_loc(feature)] = 0
                #elif strategy == 'mean':
                #   X_preserved.iat[i, X.columns.get_loc(feature)] = X[feature].mean()

    preserved_preds = model.predict(X_preserved)
    acc = accuracy_score(original_preds, preserved_preds)
    print(f" Preservation Accuracy (top-{top_k} only): {acc:.4f}")
    return acc

def deletion_check(X: pd.DataFrame, shap_df: pd.DataFrame, model, top_k=5):
    X_deleted = X.copy()
    original_preds = (model.predict(X) > 0.5).astype(int)

    for i in range(len(X)):
        shap_row = shap_df.iloc[i]
        top_features = shap_row.abs().sort_values(ascending=False).head(top_k).index.tolist()

        for feature in top_features:
            X_deleted.loc[i, feature] = 0

    deleted_preds = (model.predict(X_deleted) > 0.5).astype(int)

    print("\n🔍 Prediction Changes During Deletion:")
    for i in range(len(original_preds)):
        if original_preds[i] != deleted_preds[i]:
            print(f"❗ Row {i}: {original_preds[i]} → {deleted_preds[i]}")

    acc = accuracy_score(original_preds, deleted_preds)
    print(f"✅ Deletion Accuracy (after removing top-{top_k}): {acc:.4f}")
    return acc

# Cosine Similarity Fidelity Check
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def evaluate_fidelity_by_prediction(predicted_class: int, llm_responses: dict, threshold=0.5):
    """
    Use semantic similarity to check if each LLM explanation aligns with predicted class (0 or 1).
    """
    print("\n🔍 Fidelity Evaluation (Does LLM explanation reflect predicted class?)")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    class_templates = {
        1: "The applicant is at high risk of defaulting on the loan.",
        0: "The applicant is likely to repay the loan successfully."
    }

    reference_embedding = model.encode([class_templates[predicted_class]])[0]
    scores = {}

    for model_name, explanation in llm_responses.items():
        try:
            explanation_embedding = model.encode([explanation])[0]
            similarity = cosine_similarity([reference_embedding], [explanation_embedding])[0][0]
            match = int(similarity >= threshold)
            status = " Match" if match else " No Match"
            print(f" {model_name}: {status} (sim={similarity:.2f}, Expected class: {predicted_class})")
            scores[model_name] = similarity
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            scores[model_name] = 0

    return scores


# ----------------------
# Unified Completeness Evaluation
# ----------------------
def evaluate_output_completeness(X_raw, model, explainer, feature_names=None, predicted_class=None, llm_explanations=None):

    print("\n🚀 Evaluating Output Completeness")

    # Step 1: Infer feature names
    if hasattr(X_raw, 'columns'):
        feature_names = list(X_raw.columns)
        X_df = X_raw.copy()
    else:
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_raw.shape[1])]
        X_df = pd.DataFrame(X_raw, columns=feature_names)

    # Step 2: SHAP values
    shap_values = explainer(X_raw)

    # Handle SHAP output (Explanation or np.ndarray)
    if isinstance(shap_values, (list, tuple)):
        shap_matrix = shap_values[0]
    else:
        shap_matrix = shap_values.values if hasattr(shap_values, 'values') else shap_values

    if len(shap_matrix.shape) == 1:
        shap_matrix = shap_matrix.reshape(1, -1)

    shap_df = pd.DataFrame(shap_matrix, columns=feature_names)

    # Step 3: Checks
    preservation_score = preservation_check(X_df.copy(), shap_df, model)
    deletion_score = deletion_check(X_df.copy(), shap_df, model)

    # Step 4: Final Score (completeness only)
    final_score = round(((preservation_score * 0.5) + (( deletion_score) * 0.5)) * 10, 2)
    print(f"\n📊 Output Completeness Score (out of 10): {final_score}/10")

    # Step 5: Fidelity (if explanations provided)
    fidelity_scores = None
    if llm_explanations is not None:
        predicted_class = int(model.predict(X_df)[0])  # assume one sample
        fidelity_scores = evaluate_fidelity_by_prediction(predicted_class, llm_explanations)


    return {
        "preservation_accuracy": preservation_score,
        "deletion_accuracy": deletion_score,
        "completeness_score_10pt": final_score,
        "fidelity_scores": fidelity_scores
    }
