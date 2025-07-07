import joblib
import pandas as pd
import shap
from app.schemas import FeatureImpact
from typing import List


def load_model():
    return joblib.load("models/xgboost_model175.pkl")

def load_scaler():
    return joblib.load("models/feature_scaler175.pkl")

def load_features():
    df = pd.read_csv("models/selected_features_175.csv", header=None)
    return df.iloc[:, 0].tolist()

def load_explainer(model):
    return shap.TreeExplainer(model)


def get_top_positive_shap_features(shap_values, original_values, feature_names, threshold=0.6) -> List[FeatureImpact]:
    impacts = [
        FeatureImpact(feature=feature_names[i], value=original_values[i], shap_impact=float(shap_values[i]))
        for i in range(len(shap_values)) if shap_values[i] > 0
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

def get_top_lime_features(lime_features, threshold=0.6):
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
