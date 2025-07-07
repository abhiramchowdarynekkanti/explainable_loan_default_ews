import pandas as pd
import math
from collections import Counter

def calculate_entropy(text):
    words = text.lower().split()
    if not words:
        return 0.0
    freq = Counter(words)
    total = len(words)
    probs = [count / total for count in freq.values()]
    return -sum(p * math.log2(p) for p in probs)

def calculate_size_score(text, baseline=50):
    word_count = len(text.split())
    score = max(0, 5 - (word_count / baseline))
    return round(score, 2)

def calculate_redundancy_score(text):
    entropy = calculate_entropy(text)
    return round(min(5, entropy), 2)

def evaluate_compactness_scores(df: pd.DataFrame) -> pd.DataFrame:
    df["size_score"] = df["explanation"].fillna("").apply(calculate_size_score)
    df["redundancy_score"] = df["explanation"].fillna("").apply(calculate_redundancy_score)
    df["compactness_score"] = df["size_score"] + df["redundancy_score"]
    return df[["sample_id", "model_name", "size_score", "redundancy_score", "compactness_score"]]
