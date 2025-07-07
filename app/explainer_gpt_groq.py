import os
import json
import numpy as np
import faiss
import torch
from groq import Groq
from sentence_transformers import SentenceTransformer
from app.schemas import FeatureImpact  # LIME features are dicts, not LimeFeature objects!
from app.feature_aliases import FEATURE_ALIASES
from dotenv import load_dotenv

load_dotenv()

# Setup Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
LLM_MODEL = "llama3-8b-8192"

# RAG setup
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class RAGRetriever:
    def __init__(self, index_path, jsonl_path, model_name="BAAI/bge-small-en"):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.docs = [json.loads(line)["text"] for line in f]

    def retrieve(self, query: str, top_k: int = 1):
        vec = self.model.encode([query])
        _, indices = self.index.search(np.array(vec).astype("float32"), top_k)
        return [self.docs[i] for i in indices[0]]

retriever = RAGRetriever(
    index_path=os.path.join(ROOT, "data", "index.faiss"),
    jsonl_path=os.path.join(ROOT, "data", "index.jsonl")
)

def call_llm(prompt: str, max_tokens: int = 120) -> str:
    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a financial analyst generating decision-based model explanations for loan applications. Use a professional tone. Avoid personal expressions. Focus on justifying model decisions like approval or rejection, clearly referencing feature impacts."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=max_tokens,
        top_p=0.9,
    )
    return response.choices[0].message.content.strip()

@torch.inference_mode()
@torch.inference_mode()
def explain_with_gpt(top_shap_feats: list[FeatureImpact], top_lime_feats: list[dict]):
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

        # Remove already seen facts
        fact_list = [fact for fact in fact_list if fact not in seen_facts]
        seen_facts.update(fact_list)

        # ✅ Console print of retrieved facts
        print(f"\n🔍 RAG Facts for Feature: {readable} (Value: {feat.value}, Impact: {feat.shap_impact:+.2f})")
        if fact_list:
            for fact in fact_list:
                print(f"• {fact}")
        else:
            print("• No supporting facts retrieved.")

        # Add facts to prompt
        if fact_list:
            bullet_points = "\n  - " + "\n  - ".join(fact_list)
            fact_line = f"Supporting facts:{bullet_points}"
        else:
            fact_line = "This feature typically impacts risk based on model behavior."

        prompt += f"- {readable} = {feat.value:.2f} ({direction} risk). {fact_line}\n"

    # LIME features section
    if top_lime_feats:
        prompt += "\nAdditional factors highlighted by LIME analysis:\n"
        for feat in top_lime_feats:
            direction = "increased" if feat["impact"] > 0 else "decreased"
            prompt += f"- {feat['feature']} ({feat['value']:.2f} {direction} risk)\n"

    prompt += (
        "\nWrite a 3–4 sentence explanation in a formal, decision-oriented tone. "
        "Start with 'The Bank predicts...' or 'Your application indicates...'. "
        "Use the retrieved facts in conjunction with the applicant's feature values to justify the prediction. "
        "Focus on risk factors and factual reasoning; avoid emotional or subjective language."
    )

    # Print prompt token length
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print("🔢 Prompt token length:", len(tokenizer(prompt)["input_ids"]))

    # Call LLM
    narrative = call_llm(prompt)
    if narrative is None or not isinstance(narrative, str) or not narrative.strip():
        narrative = "Explanation temporarily unavailable due to Groq API error."

    return {
        "top_features": top_shap_feats,
        "narrative": narrative
    }


def template_explanation(probability, top_feats):
    parts = [f"The model predicts a default probability of {round(probability*100, 2)}%."]
    for feat in top_feats:
        readable_name = FEATURE_ALIASES.get(feat.feature, feat.feature)
        # For SHAP features, use .shap_impact and .value
        direction = "increased" if feat.shap_impact > 0 else "decreased"
        parts.append(f"{readable_name} ({feat.value}) {direction} the risk by {feat.shap_impact:+.2f}.")
    return " ".join(parts)
