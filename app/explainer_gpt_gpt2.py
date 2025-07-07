import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from app.schemas import FeatureImpact, LimeFeature
from app.feature_aliases import FEATURE_ALIASES
from app.explainer_gpt_groq import retriever

# Load GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

def generate_gpt2_text(prompt: str, max_new_tokens: int = 100) -> str:
    # Truncate long prompts to fit GPT-2 limits
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]

    print(f"🔢 Truncated prompt token length: {input_ids.shape[1]}")

    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
        model.to("cuda")
    else:
        model.to("cpu")

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

def explain_with_gpt2(top_shap_feats: list[FeatureImpact], top_lime_feats: list[LimeFeature]):
    if not top_shap_feats:
        return {
            "top_features": [],
            "narrative": "No significant risk factors identified."
        }

    prompt = "Generate a concise explanation for a loan decision:\n\n"
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

    # LIME features section
    if top_lime_feats:
        prompt += "\nAdditional factors highlighted by LIME analysis:\n"
        for feat in top_lime_feats:
            direction = "increased" if feat['impact'] > 0 else "decreased"
            prompt += f"- {feat['feature']} ({feat['value']:.2f} {direction} risk)\n"

    prompt += (
        "Write a 3–4 sentence explanation in a formal, decision-oriented tone. "
        "Start with 'The Bank predicts...' or 'Your application indicates...'. "
        "Use the retrieved facts in conjunction with the applicant's feature values to justify the prediction. "
        "Focus on risk factors and factual reasoning; avoid emotional or subjective language."
    )

    narrative = generate_gpt2_text(prompt, max_new_tokens=120)
    if narrative is None or not isinstance(narrative, str) or not narrative.strip():
        narrative = "Explanation temporarily unavailable due to GPT-2 error."

    return {
        'top_features': top_shap_feats,
        'narrative': narrative
    }
