import numpy as np
import joblib
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from app.encoder import load_model, load_scaler, load_features

# Load assets
model = load_model()
scaler = load_scaler()
feature_names = load_features()

# Prepare LIME explainer (we’ll reuse a generic training-like distribution)
explainer = LimeTabularExplainer(
    training_data=np.zeros((1, len(feature_names))), 
    feature_names=feature_names,
    class_names=["Not Default", "Default"],
    mode="classification"
)

def get_top_lime_features(lime_features, threshold=0.6):
    """
    Filters LIME features that together contribute >= `threshold` proportion of total absolute impact.
    Returns a list of dicts: {feature, value, impact}
    """
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

def explain_with_lime(X_scaled, original_values, threshold=0.6):
    """
    Parameters:
        - X_scaled: Scaled NumPy array of shape (1, n_features)
        - original_values: Ordered raw input values (not scaled)
        - threshold: Cumulative absolute impact threshold (default 0.6)
    Returns:
        - Dictionary of top features (covering threshold) and narrative
    """
    explanation = explainer.explain_instance(
        data_row=X_scaled[0],
        predict_fn=model.predict_proba,
        num_features=len(feature_names),
        num_samples=1000
    )

    lime_raw = explanation.as_list()
    lime_features = []
    for feature, weight in lime_raw:
        # Extract raw feature name and index from LIME format (e.g., 'DAYS_EMPLOYED <= -5000.0')
        for name in feature_names:
            if name in feature:
                idx = feature_names.index(name)
                lime_features.append({
                    "feature": name,
                    "value": original_values[idx],
                    "impact": round(weight, 4)
                })
                break

    # Get only the top features covering the threshold
    top_lime_feats = get_top_lime_features(lime_features, threshold=threshold)

    narrative = "LIME identified the following key influences: " + \
        ", ".join([f"{r['feature']} ({r['value']}) with impact {r['impact']:+.2f}" for r in top_lime_feats]) + "."

    return {
        "top_features": top_lime_feats,
        "narrative": narrative
    }
