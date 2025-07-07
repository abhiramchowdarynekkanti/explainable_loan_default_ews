from openai import OpenAI, RateLimitError
import os
from app.schemas import FeatureImpact  # LIME features are dicts, not LimeFeature objects!
from app.feature_aliases import FEATURE_ALIASES
from app.explainer_gpt_groq import retriever
import torch

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@torch.inference_mode()
def explain_with_openai(top_shap_feats: list[FeatureImpact], top_lime_feats: list[dict]):
    if not top_shap_feats:
        return {
            "top_features": [],
            "narrative": "No significant risk factors identified."
        }

    prompt = "Generate a concise explanation for a loan decision.\n\n"
    seen_facts = set()

    # SHAP features section
    prompt += "Key risk factors identified by SHAP analysis:\n"
    for feat in top_shap_feats:
        readable = FEATURE_ALIASES.get(feat.feature, feat.feature)
        direction = "increased" if feat.shap_impact > 0 else "decreased"
        fact_list = retriever.retrieve(readable, top_k=3)
        used_facts = []
        for fact in fact_list:
            if fact not in seen_facts:
                used_facts.append(fact)
                seen_facts.add(fact)
        if used_facts:
            bullet_points = "\n  - " + "\n  - ".join(used_facts)
            fact_line = f"Supporting facts:{bullet_points}"
        else:
            fact_line = "This feature typically impacts risk based on model behavior."
        prompt += f"- {readable} = {feat.value:.2f} ({direction} risk). {fact_line}\n"

    # LIME features section (dictionary access!)
    if top_lime_feats:
        prompt += "\nAdditional factors highlighted by LIME analysis:\n"
        for feat in top_lime_feats:
            direction = "increased" if feat["impact"] > 0 else "decreased"
            prompt += f"- {feat['feature']} ({feat['value']:.2f} {direction} risk)\n"

    prompt += (
        "\nWrite a 3-4 sentence explanation in a formal, decision-oriented tone. "
        "Start with 'Your application indicates...' or 'The model predicts...'. "
        "Focus on risk factors and justification. Avoid emotional language."
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial analyst generating explanations for loan approvals or rejections."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=120,
            top_p=0.9,
        )
        narrative = response.choices[0].message.content.strip()
    except RateLimitError:
        narrative = "[OpenAI Error: You have exceeded your quota. Please upgrade your OpenAI plan or add credits.]"
    except Exception as e:
        narrative = f"[OpenAI Error: {str(e)}]"

    return {
        "top_features": top_shap_feats,
        "narrative": narrative
    }
