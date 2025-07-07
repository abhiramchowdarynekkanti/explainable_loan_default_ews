import re
from typing import List, Dict
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
from app.feature_aliases import FEATURE_ALIASES

model = SentenceTransformer("all-MiniLM-L6-v2")

def split_into_sentences(text: str):
    return re.split(r'[.?!]\s+', text.strip())

def compute_confidence_accuracy(
    llm_explanations: Dict[str, str],
    top_features: List[str],
    threshold_fuzzy: float = 0.7,
    threshold_semantic: float = 0.6
) -> Dict[str, float]:
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
    for model_name, explanation in llm_explanations.items():
        if not explanation.strip():
            scores[model_name] = 0.0
            continue

        explanation_sentences = split_into_sentences(explanation)
        matches = sum(is_feature_mentioned(feat, explanation_sentences) for feat in top_features)
        scores[model_name] = round(matches / len(top_features), 2) if top_features else 0.0

    return scores
