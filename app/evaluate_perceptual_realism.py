# evaluate_perceptual_realism.py
import pandas as pd
from collections import defaultdict

def evaluate_perceptual_realism():
    df = pd.read_csv("llm_explanations.csv")
    results = []

    print(" Human-Only Perceptual Realism Evaluation\n")

    for _, row in df.iterrows():
        explanation = row['explanation']
        sample_id = row['sample_id']
        model_name = row['model_name'].strip().lower()

        if not explanation or "Error" in explanation or explanation.strip() == "":
            continue

        print(f"\n Sample ID: {sample_id}, Model: {model_name}")
        print(f" Explanation:\n{explanation}\n")

        # Human input
        while True:
            try:
                human_score = float(input(" Enter your human realism rating (1–5): "))
                if 1 <= human_score <= 5:
                    break
                else:
                    print(" Please enter a number between 1 and 5.")
            except ValueError:
                print(" Invalid input. Please enter a number.")

        results.append({
            "sample_id": sample_id,
            "model_name": model_name,
            "human_score": human_score
        })

    # Calculate average realism score per model
    model_scores = defaultdict(list)
    for r in results:
        model_scores[r['model_name']].append(r['human_score'])

    avg_scores = {model: round(sum(scores) / len(scores), 2) for model, scores in model_scores.items()}

    print("\n Final Human Realism Scores")
    print("{:<12} {:<14}".format("Model", "Avg Human Score"))
    print("-" * 30)
    for model, score in avg_scores.items():
        print("{:<12} {:<14}".format(model, score))

    return avg_scores
