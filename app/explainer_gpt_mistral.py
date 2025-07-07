import os
from groq import Groq
from dotenv import load_dotenv
from app.schemas import FeatureImpact  # LIME features are dicts, not LimeFeature objects!
from app.feature_aliases import FEATURE_ALIASES
from app.explainer_gpt_groq import retriever

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
LLM_MODEL = "mistral-saba-24b"

def call_llm(prompt: str, max_tokens: int = 120) -> str:
    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a financial analyst generating loan decision explanations."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()

def explain_with_mistral(top_shap_feats: list[FeatureImpact], top_lime_feats: list[dict]):
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
        "Write a 3–4 sentence explanation in a formal, decision-oriented tone. Start with 'The Bank predicts...' or 'Your application indicates...'. Use the retrieved facts in conjunction with the applicant's feature values to justify the prediction. Focus on risk factors and factual reasoning; avoid emotional or subjective language."
    )

    narrative = call_llm(prompt)
    if narrative is None or not isinstance(narrative, str) or not narrative.strip():
        narrative = "Explanation temporarily unavailable due to Mistral API error."

    return {
        "top_features": top_shap_feats,
        "narrative": narrative
    }
