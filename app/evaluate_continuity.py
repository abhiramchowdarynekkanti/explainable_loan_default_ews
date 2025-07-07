
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
import os
from app.encoder import load_features

from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error
import shap
from typing import Dict, List, Tuple
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim
class ContinuityEvaluator:
    def __init__(self, model_path: str, data_path: str = "data/synthetic_data100_with_target.csv"):
        """Initialize the continuity evaluator with model and data paths"""
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = "continuity_analysis"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load data and model
        self.df = pd.read_csv(data_path)
        self.X = self.df.drop(columns=['target'])
        self.y = self.df['target']
        self.feature_order = load_features()
        self.X = self.X[self.feature_order]  # Ensure base DataFrame matches model training order

        self.model = self._load_model(model_path)
        
        # Initialize explainer
        self.explainer = self._initialize_explainer()
        self.original_shap = self._get_shap_values(self.X)
        
        # Generate perturbations
        self.perturbations = self._generate_perturbations(n_variations=10)
        
    def _load_model(self, model_path: str):
        """Load saved model from pickle file"""
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def _initialize_explainer(self):
        """Initialize appropriate SHAP explainer based on model type"""
        if hasattr(self.model, 'predict_proba'):
            return shap.Explainer(self.model, self.X)
        return shap.TreeExplainer(self.model)
    
    def _generate_perturbations(self, n_variations: int = 10) -> Dict[str, np.ndarray]:
        """Generate slightly perturbed versions of each sample"""
        perturbations = {}
        for col in self.X.columns:
            if pd.api.types.is_numeric_dtype(self.X[col]):
                # Add 5% Gaussian noise for numerical features
                std = self.X[col].std()
                perturbations[col] = self.X[col].values[:, None] + np.random.normal(0, std*0.05, (len(self.X), n_variations))
            else:
                # Random category sampling for categorical features
                unique_vals = self.X[col].unique()
                perturbations[col] = np.random.choice(unique_vals, (len(self.X), n_variations))
        return perturbations
    
    def _get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Get SHAP values for input data"""
        shap_values = self.explainer.shap_values(X)
        return shap_values[1] if isinstance(shap_values, list) else shap_values
    
    def _check_prediction_consistency(self, original_pred: np.ndarray, perturbed_pred: np.ndarray) -> Dict[str, float]:
        """Check if predictions remain consistent after perturbation"""
        metrics = {
            'mse': mean_squared_error(original_pred, perturbed_pred),
            'pearson_r': pearsonr(original_pred, perturbed_pred)[0],
            'classification_agreement': np.mean((original_pred > 0.5) == (perturbed_pred > 0.5))
        }
        return metrics
    
    def _calculate_explanation_similarity(self, original_expl: np.ndarray, perturbed_expl: np.ndarray) -> Dict[str, float]:
        """Calculate various similarity metrics between explanations"""
        # Flatten explanations if they're not already 1D
        orig_flat = original_expl.flatten()
        pert_flat = perturbed_expl.flatten()
        
        metrics = {
            'spearman_r': spearmanr(orig_flat, pert_flat)[0],
            'pearson_r': pearsonr(orig_flat, pert_flat)[0],
            'cosine_sim': np.dot(orig_flat, pert_flat) / (np.linalg.norm(orig_flat) * np.linalg.norm(pert_flat)),
            #'ssim': ssim(orig_flat.reshape(1, -1), pert_flat.reshape(1, -1), 
            'top_k_intersection': len(set(np.argsort(-orig_flat)[:5]) & set(np.argsort(-pert_flat)[:5])) / 5
        }
        return metrics
    
    def _plot_stability_comparison(self, feature: str, original_data: np.ndarray, 
                                 perturbed_data: np.ndarray, data_type: str = "SHAP") -> str:
        """Generate comparison plots for original vs perturbed values"""
        plt.figure(figsize=(12, 6))
        
        # Prepare data for boxplot
        plot_data = []
        for i in range(perturbed_data.shape[1]):
            plot_data.append(pd.DataFrame({
                'Value': perturbed_data[:, i],
                'Type': f'Perturbed {i+1}'
            }))
        
        plot_data.append(pd.DataFrame({
            'Value': original_data,
            'Type': 'Original'
        }))
        
        df_plot = pd.concat(plot_data)
        
        # Create boxplot
        sns.boxplot(x='Type', y='Value', data=df_plot)
        plt.title(f'{data_type} Value Stability for {feature}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plot_name = f"{feature}_{data_type.lower()}_stability.png".replace(" ", "_")
        plot_path = os.path.join(self.output_dir, plot_name)
        plt.savefig(plot_path)
        plt.close()
        return plot_path
    
    def evaluate_feature_continuity(self, n_top_features: int = 5) -> Dict:
        """Evaluate continuity for top features"""
        # Get important features
        feature_importance = np.abs(self.original_shap).mean(axis=0)
        important_features = self.X.columns[np.argsort(feature_importance)[-n_top_features:]][::-1]
        
        results = {
            'feature_stability': defaultdict(dict),
            'prediction_consistency': defaultdict(dict),
            'plots': []
        }

        # Get original prediction once
        original_pred = self.model.predict_proba(self.X)[:, 1] if hasattr(self.model, 'predict_proba') else self.model.predict(self.X)

        for feature in important_features:
            feature_idx = self.X.columns.get_loc(feature)
            original_shap = self.original_shap[:, feature_idx]
            
            # Get perturbed data and explanations
            perturbed_shap_values = []
            perturbed_predictions = []
            
            for i in range(10):  # 10 variations
                X_temp = self.X.copy()
                X_temp[feature] = self.perturbations[feature][:, i]
                X_temp = X_temp[self.feature_order]  # Ensure correct column order before prediction

                # Get perturbed predictions
                preds = self.model.predict_proba(X_temp)[:, 1] if hasattr(self.model, 'predict_proba') else self.model.predict(X_temp)
                perturbed_predictions.append(preds)

                # Get perturbed SHAP values
                perturbed_shap = self._get_shap_values(X_temp)
                perturbed_shap_values.append(perturbed_shap[:, feature_idx])

            perturbed_shap_matrix = np.column_stack(perturbed_shap_values)
            perturbed_pred_matrix = np.column_stack(perturbed_predictions)

            # Calculate stability metrics
            stability_metrics = []
            for i in range(10):
                metrics = self._calculate_explanation_similarity(original_shap, perturbed_shap_matrix[:, i])
                stability_metrics.append(metrics)

            # Fidelity for Slight Variations
            delta_preds = [mean_squared_error(original_pred, perturbed_pred_matrix[:, i]) for i in range(10)]
            delta_shaps = [mean_squared_error(original_shap, perturbed_shap_matrix[:, i]) for i in range(10)]
            fidelity_score_pearson = pearsonr(delta_preds, delta_shaps)[0]
            fidelity_score_spearman = spearmanr(delta_preds, delta_shaps)[0]

            # Calculate prediction consistency
            pred_consistency = []
            for i in range(10):
                pred_consistency.append(self._check_prediction_consistency(original_pred, perturbed_pred_matrix[:, i]))

            # Generate plots
            shap_plot = self._plot_stability_comparison(feature, original_shap, perturbed_shap_matrix, "SHAP")
            pred_plot = self._plot_stability_comparison(feature, original_pred, perturbed_pred_matrix, "Prediction")

            # Store results
            results['feature_stability'][feature] = {
                'mean_spearman': np.mean([m['spearman_r'] for m in stability_metrics]),
                'mean_cosine': np.mean([m['cosine_sim'] for m in stability_metrics]),
                'mean_top_k': np.mean([m['top_k_intersection'] for m in stability_metrics]),
                'fidelity_pearson': fidelity_score_pearson,
                'fidelity_spearman': fidelity_score_spearman,
                'all_metrics': stability_metrics
            }

            results['prediction_consistency'][feature] = {
                'mean_mse': np.mean([m['mse'] for m in pred_consistency]),
                'mean_pearson': np.mean([m['pearson_r'] for m in pred_consistency]),
                'mean_agreement': np.mean([m['classification_agreement'] for m in pred_consistency]),
                'all_metrics': pred_consistency
            }

            results['plots'].extend([shap_plot, pred_plot])

        # Save results
        self._save_results(results)
        return results

    
    def _save_results(self, results: Dict):
        """Save evaluation results to files"""
        # Save feature stability results
        stability_df = pd.DataFrame(results['feature_stability']).T
        stability_path = os.path.join(self.output_dir, "feature_stability_results.csv")
        stability_df.to_csv(stability_path)
        
        # Save prediction consistency results
        pred_df = pd.DataFrame(results['prediction_consistency']).T
        pred_path = os.path.join(self.output_dir, "prediction_consistency_results.csv")
        pred_df.to_csv(pred_path)
        
        print(f"📝 Saved stability results to {stability_path}")
        print(f"📝 Saved prediction consistency results to {pred_path}")
    
    def run_full_analysis(self):
        """Run complete continuity analysis pipeline"""
        print("🔍 Starting continuity evaluation...")
        results = self.evaluate_feature_continuity()
        print("✅ Continuity evaluation completed!")
        return results


# Example usage
if __name__ == "__main__":
    evaluator = ContinuityEvaluator(model_path="models/credit_model.pkl")
    results = evaluator.run_full_analysis()

"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
import os
from scipy.stats import spearmanr
import shap
from typing import Dict, List

def load_model(model_path: str):
   
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def generate_perturbations(X: pd.DataFrame, n_variations: int = 10) -> Dict[str, np.ndarray]:
    perturbations = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            # Add 5% Gaussian noise for numerical features
            std = X[col].std()
            perturbations[col] = X[col].values[:, None] + np.random.normal(0, std*0.05, (len(X), n_variations))
        else:
            # Random category sampling for categorical features
            unique_vals = X[col].unique()
            perturbations[col] = np.random.choice(unique_vals, (len(X), n_variations))
    return perturbations

def plot_stability_boxplots(feature: str, original_shap: np.ndarray, perturbed_shap: np.ndarray, output_dir: str):
    plt.figure(figsize=(12, 6))
    
    # Prepare data for boxplot
    plot_data = []
    for i in range(perturbed_shap.shape[1]):
        plot_data.append(pd.DataFrame({
            'SHAP Value': perturbed_shap[:, i],
            'Type': f'Perturbed {i+1}'
        }))
    
    plot_data.append(pd.DataFrame({
        'SHAP Value': original_shap,
        'Type': 'Original'
    }))
    
    df_plot = pd.concat(plot_data)
    
    # Create boxplot
    sns.boxplot(x='Type', y='SHAP Value', data=df_plot)
    plt.title(f'SHAP Value Stability for {feature}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f"{feature}_stability_boxplot.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def evaluate_continuity(model_path: str, data_path: str = "data/synthetic_data100_with_target.csv"):
    # Load data
    df = pd.read_csv(data_path)
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Load model
    model = load_model(model_path)
    
    # Initialize explainer
    explainer = shap.Explainer(model, X) if hasattr(model, 'predict_proba') else shap.TreeExplainer(model)
    
    # Create output directory
    output_dir = "stability_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate perturbations
    perturbations = generate_perturbations(X)
    
    results = {
        'feature_stability': {},
        'plots': []
    }
    
    # Analyze top 5 most important features
    shap_values = explainer.shap_values(X)[1] if isinstance(explainer.shap_values(X), list) else explainer.shap_values(X)
    important_features = X.columns[np.argsort(np.abs(shap_values).mean(axis=0))[-5:]][::-1]
    
    for feature in important_features:
        # Get original SHAP values
        original_shap = shap_values[:, X.columns.get_loc(feature)]
        
        # Get perturbed SHAP values
        perturbed_data = X.copy()
        perturbed_shap_values = []
        
        for i in range(10):  # 10 variations
            perturbed_data[feature] = perturbations[feature][:, i]
            perturbed_shap = explainer.shap_values(perturbed_data)[1] if isinstance(explainer.shap_values(perturbed_data), list) else explainer.shap_values(perturbed_data)
            perturbed_shap_values.append(perturbed_shap[:, X.columns.get_loc(feature)])
        
        perturbed_shap_matrix = np.column_stack(perturbed_shap_values)
        
        # Calculate stability metrics
        stability_scores = []
        for i in range(10):
            stability_scores.append(spearmanr(original_shap, perturbed_shap_matrix[:, i])[0])
        
        # Generate boxplot
        plot_path = plot_stability_boxplots(feature, original_shap, perturbed_shap_matrix, output_dir)
        
        # Store results
        results['feature_stability'][feature] = {
            'mean_stability': np.mean(stability_scores),
            'min_stability': np.min(stability_scores),
            'max_stability': np.max(stability_scores),
            'stability_scores': stability_scores
        }
        results['plots'].append(plot_path)
        
        print(f"📊 Generated stability boxplot for {feature} at {plot_path}")
    
    # Save summary results
    results_path = os.path.join(output_dir, "stability_results.csv")
    pd.DataFrame(results['feature_stability']).T.to_csv(results_path)
    print(f"\n📝 Saved stability results at {results_path}")
    
    return results
def generate_prediction_plots(model, X: pd.DataFrame, feature: str, output_dir: str):
    plt.figure(figsize=(10, 6))
    
    # Original predictions
    original_pred = model.predict_proba(X)[:, 1]
    
    # Generate perturbations
    perturbed_data = generate_perturbations(X[[feature]])  # Single feature perturbation
    
    # Get predictions for all perturbations
    preds = []
    for i in range(10):
        X_temp = X.copy()
        X_temp[feature] = perturbed_data[feature][:, i]
        preds.append(model.predict_proba(X_temp)[:, 1])
    
    # Create combined plot
    plt.boxplot([original_pred] + preds)
    plt.xticks([1] + list(range(2, 12)), 
              ['Original'] + [f'Perturb {i}' for i in range(1, 11)])
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.3)
    plt.title(f'Prediction Stability for {feature}\n(0=Reject, 1=Approve)')
    plt.ylabel('Prediction Probability (Class 1)')
    
    # Save plot
    plot_path = os.path.join(output_dir, f"{feature}_pred_stability.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path"""