import re
from typing import List, Dict
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
from app.feature_aliases import FEATURE_ALIASES
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

model = SentenceTransformer("all-MiniLM-L6-v2")
rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
smoothie = SmoothingFunction().method4

def split_into_sentences(text: str):
    return re.split(r'[.?!]\s+', text.strip())

def compute_bleu_rouge_scores(reference: str, hypothesis: str) -> dict:
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    bleu_score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
    rouge_scores = rouge.score(reference, hypothesis)

    return {
        "bleu": round(bleu_score, 4),
        "rouge1": round(rouge_scores["rouge1"].fmeasure, 4),
        "rougeL": round(rouge_scores["rougeL"].fmeasure, 4),
    }

def compute_alignment_with_domain_knowledge(
    llm_explanations: Dict[str, str],
    top_features: List[str],
    threshold_fuzzy: float = 0.7,
    threshold_semantic: float = 0.6
) -> Dict[str, Dict[str, float]]:
    """
    Alignment with Domain Knowledge:
    Combines feature mention check with BLEU and ROUGE metrics.
    """
    def is_feature_mentioned(feature: str, explanation_sentences: List[str]) -> bool:
        candidates = set()
        candidates.add(feature.lower().replace('_', ' '))
        alias = FEATURE_ALIASES.get(feature)
        if isinstance(alias, str):
            candidates.add(alias.lower())
        elif isinstance(alias, list):
            candidates.update(a.lower() for a in alias)

        for phrase in candidates:
            for sent in explanation_sentences:
                sent = sent.lower()

                # Fuzzy match
                if phrase in sent or SequenceMatcher(None, phrase, sent).ratio() > threshold_fuzzy:
                    return True

                # Semantic similarity
                emb_phrase = model.encode(phrase, convert_to_tensor=True)
                emb_sent = model.encode(sent, convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(emb_phrase, emb_sent).item()
                if similarity > threshold_semantic:
                    return True
        return False

    scores = {}
    reference_summary = " ".join([f.lower().replace('_', ' ') for f in top_features])

    for model_name, explanation in llm_explanations.items():
        if not explanation.strip():
            scores[model_name] = {
                "feature_coverage": 0.0,
                "bleu": 0.0,
                "rouge1": 0.0,
                "rougeL": 0.0,
            }
            continue

        explanation_sentences = split_into_sentences(explanation)
        matches = sum(is_feature_mentioned(feat, explanation_sentences) for feat in top_features)
        feature_coverage = round(matches / len(top_features), 2) if top_features else 0.0

        bleu_rouge = compute_bleu_rouge_scores(reference_summary, explanation)

        scores[model_name] = {
            "feature_coverage": feature_coverage,
            **bleu_rouge
        }

    return scores

def evaluate_xai_methods_agreement(
    shap_features: List[str],
    lime_features: List[str]
) -> Dict[str, float | str | List[str]]:
    """
    XAI Methods Agreement: Compare SHAP and LIME features using Jaccard similarity.
    """
    shap_set = set(shap_features)
    lime_set = set(lime_features)

    intersection = shap_set & lime_set
    union = shap_set | lime_set

    jaccard = round(len(intersection) / len(union), 2) if union else 0.0

    if jaccard >= 0.6:
        rating = "High Agreement "
    elif jaccard >= 0.3:
        rating = "Moderate Agreement "
    else:
        rating = "Low Agreement "

    return {
        "jaccard_score": jaccard,
        "rating": rating,
        "shared_features": list(intersection),
        "shap_only": list(shap_set - lime_set),
        "lime_only": list(lime_set - shap_set),
    }
