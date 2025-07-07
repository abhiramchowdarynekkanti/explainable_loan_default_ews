import numpy as np
from typing import Dict, Any, Union
from app.explainer_lime_tabular import explain_with_lime
import shap
from app.schemas import FeatureImpact


def get_top_positive_shap_features(shap_values, original_values, feature_names, threshold=0.8):
    shap_values = shap_values.flatten()  # 🛠 Ensure 1D vector


    # Proceed with assertions if lengths match
    assert len(shap_values) == len(feature_names), \
        f"SHAP values ({len(shap_values)}) and features ({len(feature_names)}) mismatch"
    assert len(original_values) == len(feature_names), \
        f"Original values ({len(original_values)}) and features ({len(feature_names)}) mismatch"
    impacts = [
        FeatureImpact(feature=feature_names[i], value=original_values[i], shap_impact=float(shap_values[i]))
        for i in range(len(shap_values)) if float(shap_values[i]) > 0
    ]
    impacts.sort(key=lambda x: abs(x.shap_impact), reverse=True)
    total = sum(x.shap_impact for x in impacts)
    top_feats = []
    cumulative = 0.0
    for feat in impacts:
        top_feats.append(feat)
        cumulative += feat.shap_impact
        if total > 0 and cumulative / total >= threshold:
            break
    return top_feats


def get_top_lime_features(lime_features, threshold=0.4):
    impacts = sorted(lime_features, key=lambda x: abs(x['impact']), reverse=True)
    total = sum(abs(x['impact']) for x in impacts)
    top_feats = []
    cumulative = 0.0
    for feat in impacts:
        top_feats.append(feat)
        cumulative += abs(feat['impact'])
        if total > 0 and cumulative / total >= threshold:
            break
    return top_feats


def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0.0


def evaluate_single_instance_consistency(
    ordered_values: list,
    X_processed: np.ndarray,
    xgb_model: Any,
    xgb_explainer: shap.Explainer,
    logreg_model: Any,
    logreg_explainer: shap.Explainer,
    feature_order: list
) -> Dict[str, Union[float, bool, list]]:
    """
    Compare SHAP and LIME explanations for XGBoost and LogisticRegression on the same input.
    """

    # Prediction consistency
    pred_xgb = int(xgb_model.predict(X_processed)[0])
    pred_logreg = int(logreg_model.predict(X_processed)[0])
    prediction_match = pred_xgb == pred_logreg

    # SHAP values for XGBoost
    shap_vals_xgb = xgb_explainer(X_processed)[0].values
    top_shap_feats_xgb = get_top_positive_shap_features(shap_vals_xgb, ordered_values, feature_order)
    top_shap_xgb_set = {f.feature for f in top_shap_feats_xgb}

    # SHAP values for LogisticRegression
    shap_output_logreg = logreg_explainer.shap_values(X_processed)
    shap_vals_logreg = shap_output_logreg[0] if isinstance(shap_output_logreg, list) else shap_output_logreg
    # Fix: Only use SHAP values corresponding to class 1
    logreg_class1_shap_vals = shap_vals_logreg[0] if isinstance(shap_vals_logreg, list) else shap_vals_logreg
    logreg_class1_shap_vals = logreg_class1_shap_vals.flatten()  # just in case
    if logreg_class1_shap_vals.shape[0] == 2 * len(feature_order):
        logreg_class1_shap_vals = logreg_class1_shap_vals[len(feature_order):]  # only keep class-1 SHAP values

    top_shap_feats_logreg = get_top_positive_shap_features(
        logreg_class1_shap_vals,
    ordered_values,
    feature_order
    )

    top_shap_logreg_set = {f.feature for f in top_shap_feats_logreg}

    shap_similarity = jaccard_similarity(top_shap_xgb_set, top_shap_logreg_set)

    # Temporarily override the global model for LIME explanations
    import app.encoder as encoder

    # LIME for XGBoost
    encoder.model = xgb_model
    lime_result_xgb = explain_with_lime(X_processed, ordered_values)
    top_lime_feats_xgb = get_top_lime_features(lime_result_xgb["top_features"])
    top_lime_xgb_set = {f['feature'] for f in top_lime_feats_xgb}

    # LIME for LogisticRegression
    encoder.model = logreg_model
    lime_result_logreg = explain_with_lime(X_processed, ordered_values)
    top_lime_feats_logreg = get_top_lime_features(lime_result_logreg["top_features"])
    top_lime_logreg_set = {f['feature'] for f in top_lime_feats_logreg}

    lime_similarity = jaccard_similarity(top_lime_xgb_set, top_lime_logreg_set)

    # Restore model if needed (optional cleanup)
    encoder.model = xgb_model

    return {
        "prediction_match": prediction_match,
        "shap_jaccard": shap_similarity,
        "lime_jaccard": lime_similarity,
        "top_shap_xgb": list(top_shap_xgb_set),
        "top_shap_logreg": list(top_shap_logreg_set),
        "top_lime_xgb": list(top_lime_xgb_set),
        "top_lime_logreg": list(top_lime_logreg_set)
    }
