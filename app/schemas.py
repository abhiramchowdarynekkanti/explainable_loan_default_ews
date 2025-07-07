from typing import Dict, List
from pydantic import BaseModel, Field
import pandas as pd
from pathlib import Path
from typing import Union, Dict
# ----- load feature list once -----
FEATURES_CSV = Path(__file__).parent.parent / "models" / "selected_features_175.csv"
SELECTED_FEATURES: List[str] = pd.read_csv(FEATURES_CSV, header=None)[0].tolist()

# build an example with zeros
_example_body = {f: 0.0 for f in SELECTED_FEATURES}
NumberOrString = Union[float, int, str, bool]


class FeatureImpact(BaseModel):
    feature: str
    value: float
    shap_impact: float

class PredictionRequest(BaseModel):
    data: Dict[str, NumberOrString] = Field(
        ...,
        example=_example_body      
    )
class LimeFeature(BaseModel):
    feature: str
    value: float
    impact: float

'''class PredictionResponse(BaseModel):
    prediction: int
    probability_of_default: float
    top_feature_impacts: List[FeatureImpact]
    natural_explanation: str
    #natural_explanation_fingpt: str
    model: str = "XGBoost"
    explanation_method: str = "SHAP + GPT-2"
    template_explanation: str'''
class PredictionResponse(BaseModel):
    prediction: int
    probability_of_default: float
    top_feature_impacts: List[FeatureImpact]
    top_feature_impacts_lime: List[LimeFeature]
    natural_explanation_groq: str
    natural_explanation_qwen : str
    natural_explanation_chatglm: str 
    natural_explanation_mistral: str 
    natural_explanation_mixtral: str 
    natural_explanation_gemma: str
    natural_explanation_gpt2: str
    natural_explanation_lime: str
    template_explanation: str
    llm_scores: Dict[str, float]

    model: str = "XGBoost"
    explanation_method: str = "SHAP + GPTs"