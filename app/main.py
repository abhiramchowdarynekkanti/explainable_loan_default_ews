from typing import Dict
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
load_dotenv()
from app.explainer_lime_tabular import explain_with_lime
import matplotlib
matplotlib.use('Agg')  
import numpy as np 
from tabulate import tabulate
import os
from datetime import datetime
import csv
import datetime
SAMPLE_COUNTER_FILE = "sample_id.txt"
EXPLANATION_LOG_FILE = "llm_explanations.csv"
from app.evaluate_compactness import evaluate_compactness_scores
from app.evaluate_perceptual_realism import evaluate_perceptual_realism
from app.evaluate_completeness import evaluate_output_completeness
from app.evaluate_confidence_accuracy import compute_confidence_accuracy
from app.evaluate_correctness import (
    evaluate_single_deletion,
    evaluate_incremental_deletion,
    evaluate_explanation_randomization,
    evaluate_model_parameter_randomization,
    run_controlled_synthetic_check
)
import matplotlib.pyplot as plt
import seaborn as sns
def log_llm_scores(sample_id: int, prediction: int, probability: float, llm_scores: dict, csv_path="llm_score_log.csv"):
    log_row = {
        "sample_id": sample_id,
        "prediction": prediction,
        "probability": round(probability, 4)
    }
    for llm, score in llm_scores.items():
        log_row[llm] = round(score, 4)

    df = pd.DataFrame([log_row])
    df.to_csv(csv_path, mode="a", index=False, header=not os.path.exists(csv_path))
def generate_llm_score_boxplot(csv_path="llm_score_log.csv", output_path="frontend/public/llm_score_boxplot.png"):
    # Load the logged LLM scores
    df = pd.read_csv(csv_path)

    llm_columns = ['groq', 'chatglm', 'qwen', 'mistral', 'mixtral', 'gemma', 'gpt2']
    missing = [col for col in llm_columns if col not in df.columns]
    if df.empty or missing:
        print("[⚠️] CSV is empty or missing required LLM columns:", missing)
        return

    # Convert from wide to long format
    long_df = df.melt(id_vars=["sample_id"], 
                      value_vars=llm_columns,
                      var_name="llm_model", 
                      value_name="llm_score")

    # Plot boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=long_df, x="llm_model", y="llm_score", palette="Set3")

    plt.title("LLM Score Distribution per Model")
    plt.xlabel("LLM Model")
    plt.ylabel("Score (Aggregate from all metrics)")
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[✅] LLM score boxplot saved to: {output_path}")
def generate_boxplot(input_csv="prediction_log.csv", output_png="frontend/public/prediction_boxplot.png"):
    if not os.path.exists(input_csv):
        return

    df = pd.read_csv(input_csv)
    if "prediction" not in df.columns or "probability" not in df.columns:
        return

    plt.figure(figsize=(8, 6))
    sns.boxplot(x="prediction", y="probability", data=df)
    plt.title("Predicted Probability by Class")
    plt.xlabel("Predicted Class (0 or 1)")
    plt.ylabel("Probability of Default")
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()

def log_prediction_result(input_data: dict, probability: float, prediction: int, file_path="prediction_log.csv"):
    log_entry = input_data.copy()
    log_entry["probability"] = probability
    log_entry["prediction"] = prediction

    file_exists = os.path.isfile(file_path)
    df = pd.DataFrame([log_entry])

    if file_exists:
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, mode='w', header=True, index=False)

def print_continuity_results(results: Dict):
    """Pretty print continuity analysis results"""
    print("\n🔍 Explanation Continuity Report")
    print(f"\n📌 Analyzed Features: {list(results['feature_stability'].keys())}")
    
    print("\n📊 Stability Scores (Spearman Correlation):")
    for feature, metrics in results['feature_stability'].items():
        print(f"\n⭐ {feature}:")
        print(f"  Average: {metrics['mean_stability']:.3f}")
        print(f"  Range: {metrics['min_stability']:.3f} - {metrics['max_stability']:.3f}")
        print(f"  - Fidelity Pearson: {metrics['fidelity_pearson']:.4f}")
        print(f"  - Fidelity Spearman: {metrics['fidelity_spearman']:.4f}\n")
    
    print(f"\n📈 Visualizations saved to: {results['plots'][0].rsplit('/', 1)[0]}")
from app.evaluate_coherence import (
    evaluate_xai_methods_agreement,
    compute_alignment_with_domain_knowledge
)
from app.schemas import PredictionRequest, PredictionResponse
from app.encoder import (
    load_model,
    load_scaler,
    load_features,
    load_explainer,
)
from app.preprocess import preprocess_input, get_ordered_values
from app.encoder import FeatureImpact
from app.explainer_gpt_groq import explain_with_gpt, template_explanation
from app.explainer_gpt_chatglm import explain_with_chatglm
from app.explainer_gpt_qwen import explain_with_qwen
from app.explainer_gpt_mistral import explain_with_mistral
from app.explainer_gpt_mixtral import explain_with_mixtral
from app.explainer_gpt_gemma import explain_with_gemma
from app.explainer_gpt_gpt2 import explain_with_gpt2  
import joblib
import shap
import pandas as pd
from app.evaluate_consistency import evaluate_single_instance_consistency
feature_order = load_features()
logreg_model = joblib.load("models/logistic_regression_model.pkl")
logreg_background = pd.read_csv("data/train_sample.csv")[feature_order].sample(
    n=min(100, sum(1 for _ in open("data/train_sample.csv")) - 1), replace=False
)
def print_continuity_results(results):
    print("\n📌 Feature Stability Summary:")
    for feat, metrics in results['feature_stability'].items():
        print(f"\n🧠 Feature: {feat}")
        print(f"  - Mean Spearman: {metrics['mean_spearman']:.4f}")
        print(f"  - Mean Cosine: {metrics['mean_cosine']:.4f}")
        print(f"  - Mean Top-k Intersection: {metrics['mean_top_k']:.4f}")
        # print(f"  - Mean SSIM: {metrics['mean_ssim']:.4f}")  # Only if you're using SSIM

    print("\n📊 Prediction Consistency Summary:")
    for feat, metrics in results['prediction_consistency'].items():
        print(f"\n🧪 Feature: {feat}")
        print(f"  - Mean MSE: {metrics['mean_mse']:.4f}")
        print(f"  - Mean Pearson Correlation: {metrics['mean_pearson']:.4f}")
        print(f"  - Mean Agreement: {metrics['mean_agreement']:.4f}")

logreg_explainer = shap.KernelExplainer(logreg_model.predict_proba, logreg_background)

app = FastAPI(title="Loan Default Risk Explanation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model()
scaler = load_scaler()
explainer = load_explainer(model)
REQUIRED_FEATURES = {
    "AGE",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "DAYS_EMPLOYED",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
}

def check_required_fields(data: Dict):
    missing = REQUIRED_FEATURES - set(data.keys())
    if missing:
        raise ValueError(f"Insufficient data: missing fields: {', '.join(missing)}")

def get_top_positive_shap_features(shap_values, original_values, feature_names, threshold=0.7):
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

def get_top_lime_features(lime_features, threshold=0.4):
    # Filter only features with NEGATIVE impact (decrease risk)
    negative_impacts = [f for f in lime_features if f['impact'] < 0]
    
    # Sort by absolute impact (most significant mitigators first)
    negative_impacts_sorted = sorted(negative_impacts, key=lambda x: abs(x['impact']), reverse=True)
    
    # Calculate total negative impact (sum of absolute values)
    total_negative_impact = sum(abs(f['impact']) for f in negative_impacts_sorted)
    
    # Select top features until cumulative impact reaches threshold (60%)
    top_negative_feats = []
    cumulative_impact = 0.0
    
    for feat in negative_impacts_sorted:
        top_negative_feats.append(feat)
        cumulative_impact += abs(feat['impact'])
        if total_negative_impact > 0 and cumulative_impact / total_negative_impact >= threshold:
            break
    
    return top_negative_feats

def get_next_sample_id():
    if not os.path.exists(SAMPLE_COUNTER_FILE):
        with open(SAMPLE_COUNTER_FILE, "w") as f:
            f.write("1")
        return 1
    with open(SAMPLE_COUNTER_FILE, "r+") as f:
        current = int(f.read().strip())
        new = current + 1
        f.seek(0)
        f.write(str(new))
        f.truncate()
    return new

def log_llm_explanations(sample_id: int, responses: dict, file_path: str = "llm_explanations.csv"):
    fieldnames = ["timestamp", "sample_id", "model_name", "explanation"]
    timestamp = datetime.datetime.now().isoformat()

    rows = []
    for key, explanation in responses.items():
        model_name = key.replace("natural_explanation_", "")
        rows.append({
            "timestamp": timestamp,
            "sample_id": sample_id,
            "model_name": model_name,
            "explanation": explanation
        })

    with open(file_path, mode="w", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Return probability, SHAP-GPT narratives and template explanation."""
    try:
        check_required_fields(request.data)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    ordered_values = get_ordered_values(request.data, feature_order)
    X_processed = preprocess_input(ordered_values, scaler)

    probability = float(model.predict_proba(X_processed)[0][1])
    prediction = int(probability > 0.5)

    
    lime_result = explain_with_lime(X_processed, ordered_values)
    shap_vals = explainer(X_processed)[0].values

    # Filter top features for LLM explainers
    top_shap_feats = get_top_positive_shap_features(shap_vals, ordered_values, feature_order, threshold=0.6)
    top_lime_feats = get_top_lime_features(lime_result["top_features"], threshold=0.6)

    if prediction == 0:
        # No explanations if not default
        # Log prediction and generate updated boxplot
        log_prediction_result(request.data, probability, prediction)
        generate_boxplot()

        return PredictionResponse(
            prediction=prediction,
            probability_of_default=probability,
            top_feature_impacts=[],
            top_feature_impacts_lime=[],
            natural_explanation_groq="Approved",
            natural_explanation_chatglm="Approved",
            natural_explanation_qwen="Approved",
            natural_explanation_mistral="Approved",
            natural_explanation_mixtral="Approved",
            natural_explanation_gemma="Approved",
            natural_explanation_gpt2="Approved",
            natural_explanation_lime="Approved",
            template_explanation="Approved",
            llm_scores={
                 "groq": 0.1,
            "chatglm": 0.0,
            "qwen": 0.0,
            "mistral": 0.0,
            "mixtral": 0.0,
            "gemma": 0.0,
            "gpt2": 0.0,
            }
        )
    else:
        # Call LLM explainers only if prediction == 1

        llm_scores = {
            "groq": 0.0,
            "chatglm": 0.0,
            "qwen": 0.0,
            "mistral": 0.0,
            "mixtral": 0.0,
            "gemma": 0.0,
            "gpt2": 0.0,
        }
        exp_groq = explain_with_gpt(top_shap_feats, top_lime_feats)
        exp_chatglm = explain_with_chatglm(top_shap_feats, top_lime_feats)
        exp_qwen = explain_with_qwen(top_shap_feats, top_lime_feats)
        exp_mistral = explain_with_mistral(top_shap_feats, top_lime_feats)
        exp_mixtral = explain_with_mixtral(top_shap_feats, top_lime_feats)
        exp_gemma = explain_with_gemma(top_shap_feats, top_lime_feats)
        exp_gpt2 = explain_with_gpt2(top_shap_feats, top_lime_feats)
        template_text = template_explanation(probability, exp_groq["top_features"])
        sample_id = get_next_sample_id()
        print("Logging explanations now...")

        log_llm_explanations(sample_id=sample_id, responses={
            "natural_explanation_groq": exp_groq["narrative"],
            "natural_explanation_chatglm": exp_chatglm["narrative"],
            "natural_explanation_qwen": exp_qwen["narrative"],
            "natural_explanation_mistral": exp_mistral["narrative"],
            "natural_explanation_mixtral": exp_mixtral["narrative"],
            "natural_explanation_gemma": exp_gemma["narrative"],
            "natural_explanation_gpt2": exp_gpt2["narrative"]
        })
        df = pd.read_csv("llm_explanations.csv") 


        print("\n================= Compactness =================") 
        result = evaluate_compactness_scores(df)
        print(result)
        model_name_mapping = {
            "groq": "groq",
            "chatglm": "chatglm",
            "qwen": "qwen",
            "mistral": "mistral",
            "mixtral": "mixtral",
            "gemma": "gemma",
            "gpt2": "gpt2",
            # If CSV had variants like "mistral-7b" or "GPT-2-small", map them:
            # "mistral-7b": "mistral",
            # "GPT-2-small": "gpt2"
        }
        for _, row in result.iterrows():
            raw_name = row["model_name"].strip().lower()
            model_key = model_name_mapping.get(raw_name)
            if model_key and model_key in llm_scores:
                llm_scores[model_key] = round(row["compactness_score"], 2)

       

        predicted_class = int(model.predict(X_processed)[0])



        print("\n================= Completeness =================")
        completeness_scores = evaluate_output_completeness(
            X_processed.copy(),
            model,
            explainer,
            feature_names=feature_order,
            predicted_class=predicted_class, 
            llm_explanations={
                "groq": exp_groq["narrative"],
                "chatglm": exp_chatglm["narrative"],
                "qwen": exp_qwen["narrative"],
                "mistral": exp_mistral["narrative"],
                "mixtral": exp_mixtral["narrative"],
                "gemma": exp_gemma["narrative"],
                "gpt2": exp_gpt2["narrative"]
            }
            
        )


        if completeness_scores.get("fidelity_scores"):
            fidelity_data = []
            for model_name, score in completeness_scores["fidelity_scores"].items():
                
                fidelity_data.append([model_name, f"{score:.2f}"])
            
            print(tabulate(fidelity_data, headers=["LLM", "Similarity", "Match"], tablefmt="fancy_grid"))
        if completeness_scores.get("fidelity_scores"):
            for raw_name, fidelity_score in completeness_scores["fidelity_scores"].items():
                model_key = model_name_mapping.get(raw_name.strip().lower())
                if model_key and model_key in llm_scores:
                    llm_scores[model_key] += round(fidelity_score, 2)





        from app.evaluate_continuity import ContinuityEvaluator

        print("\n================= Continuity =================")
        print("\n Running Explanation Continuity Analysis")
        evaluator = ContinuityEvaluator("models/xgboost_model175.pkl")
        results = evaluator.run_full_analysis()
        print_continuity_results(results)
        





        print("=================== User Study Results/Context =================")
        from pathlib import Path
        from app.evaluate_user_study import run_user_study

        # Define path to CSV inside data folder
        data_path = Path("data") / "synthetic_data100_with_target.csv"

        # Verify the file exists before running
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at: {data_path}")

        # Run the user study and store results
        study_results = run_user_study(
            csv_path=str(data_path),  
            target_column="target",
            n_untrustworthy=5
        )
        for key, value in study_results.items():
            print(f"{key.replace('_', ' ').title()}: {value}")


        from app.evaluate_covariate_complexity import evaluate_covariate_complexity
        print("=============== Covariate Comeplxity   ===============\n")
        result = evaluate_covariate_complexity()
        print("\n📈 Covariate Complexity Evaluation")
        for k, v in result.items():
            print(f"{k}: {v:.4f}")



        print("\n=============== Confidence  ===============")
        from app.evaluate_confidence_accuracy import compute_confidence_accuracy

        shap_output = explainer(X_processed)
        if isinstance(shap_output, (list, tuple)):
            shap_matrix = shap_output[0].values if hasattr(shap_output[0], 'values') else shap_output[0]
        else:
            shap_matrix = shap_output.values if hasattr(shap_output, 'values') else shap_output
        confidence_scores = compute_confidence_accuracy(
            llm_explanations={
                "groq": exp_groq["narrative"],
                "chatglm": exp_chatglm["narrative"],
                "qwen": exp_qwen["narrative"],
                "mistral": exp_mistral["narrative"],
                "mixtral": exp_mixtral["narrative"],
                "gemma": exp_gemma["narrative"],
                "gpt2": exp_gpt2["narrative"]
            },
            top_features=[feat.feature for feat in top_shap_feats]
        )
        df_conf = pd.DataFrame(list(confidence_scores.items()), columns=["LLM Model", "Confidence Score"])
        print(df_conf.to_markdown(index=False))

      
        
        print("\n=============== Consistency  ===============")
        # Create feature string (SHAP + LIME combined top features)
        feature_str = "\n".join([
            f"{f.feature} = {f.value:.2f} (impact: {f.shap_impact:.2f})" for f in exp_groq["top_features"]
        ] + [
            f"{f['feature']} = {f['value']} (impact: {f['impact']:.2f})" for f in top_lime_feats
        ])
        consistency_result = evaluate_single_instance_consistency(
            ordered_values=ordered_values,
            X_processed=X_processed,
            feature_order=feature_order,
            xgb_model=model,
            logreg_model=logreg_model,
            xgb_explainer=explainer,
            logreg_explainer=logreg_explainer
        )
        print(f"Prediction Match: {consistency_result['prediction_match']}")
        print(f"SHAP Jaccard:    {consistency_result['shap_jaccard']:.2f}") 
        print(f"LIME Jaccard:    {consistency_result['lime_jaccard']:.2f}")

        X_processed_df = pd.DataFrame(X_processed, columns=feature_order)
        shap_array = shap_vals if isinstance(shap_vals, np.ndarray) else shap_vals.values

        
        print("\n=============== Correctness  ===============")
        evaluate_single_deletion(X_processed_df, model, shap_array, feature_order)
        evaluate_incremental_deletion(X_processed_df, model, shap_array, feature_order)
        evaluate_explanation_randomization(X_processed_df, model, shap_array, feature_order)
        evaluate_model_parameter_randomization(X_processed_df, model, feature_order, shap_array)
        controlled_check_result = run_controlled_synthetic_check(verbose=True)
        print(f" Controlled Synthetic Check Result: {controlled_check_result}")
        explanations_dict = {
            "groq": exp_groq["narrative"],
            "chatglm": exp_chatglm["narrative"],
            "qwen": exp_qwen["narrative"],
            "mistral": exp_mistral["narrative"],
            "mixtral": exp_mixtral["narrative"],
            "gemma": exp_gemma["narrative"],
            "gpt2": exp_gpt2["narrative"]
        }






        print("\n=============== Coherence  ===============")
        alignment_scores = compute_alignment_with_domain_knowledge(
            llm_explanations=explanations_dict,
            top_features=[f.feature for f in top_shap_feats]
        )
        print("\n🧠 Alignment with Domain Knowledge (SHAP-based):")
        print(alignment_scores)

        agreement_result = evaluate_xai_methods_agreement(
        shap_features=[f.feature for f in top_shap_feats],
        lime_features=[f["feature"] for f in top_lime_feats]
        )
        print(f"\n🔁 SHAP vs LIME Agreement:")
        print(f"Jaccard = {agreement_result['jaccard_score']:.2f}")
        print(f"Rating = {agreement_result['rating']}")
        print(f"Shared Features: {agreement_result['shared_features']}")
        print(f"SHAP only: {agreement_result['shap_only']}")
        print(f"LIME only: {agreement_result['lime_only']}")





        print("\n=============== Composition  ===============\n")
        human_realism_scores = evaluate_perceptual_realism()

        # ➕ Add realism score to llm_scores
        for raw_name, realism_score in human_realism_scores.items():
            model_key = model_name_mapping.get(raw_name.strip().lower())
            if model_key and model_key in llm_scores:
                llm_scores[model_key] += realism_score



        # Log prediction and generate updated boxplot
        log_prediction_result(request.data, probability, prediction)
        generate_boxplot()


         # ✅ Print the updated LLM scores
        print("\n📊 Final LLM Scores (Compactness):")
        for llm, score in llm_scores.items():
            print(f"{llm}: {score:.2f}")

        log_llm_scores(
            sample_id=sample_id,
            prediction=prediction,
            probability=probability,
            llm_scores=llm_scores
        )
        generate_llm_score_boxplot()


        return PredictionResponse(
            prediction=prediction,
            probability_of_default=probability,
            top_feature_impacts=exp_groq["top_features"],
            top_feature_impacts_lime=top_lime_feats,
            natural_explanation_groq=exp_groq["narrative"],
            natural_explanation_chatglm=exp_chatglm["narrative"],
            natural_explanation_qwen=exp_qwen["narrative"],
            natural_explanation_mistral=exp_mistral["narrative"],
            natural_explanation_mixtral=exp_mixtral["narrative"],
            natural_explanation_gemma=exp_gemma["narrative"],
            natural_explanation_gpt2=exp_gpt2["narrative"],  
            natural_explanation_lime=lime_result["narrative"],
            template_explanation=template_text,
            llm_scores=llm_scores
        )
