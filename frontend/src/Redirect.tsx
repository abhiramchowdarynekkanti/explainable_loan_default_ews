import React, { useEffect, useState } from "react";
import featureAliases from "./feature_aliases.json";

type FeatureImpact = {
  feature: string;
  value: number | string;
  shap_impact: number;
};

type LimeImpact = {
  feature: string;
  value: number | string;
  impact: number;
};

type PredictionResponse = {
  prediction: number;
  probability_of_default: number;
  top_feature_impacts: FeatureImpact[];
  top_feature_impacts_lime: LimeImpact[];
  natural_explanation_groq: string;
  natural_explanation_openai: string;
  natural_explanation_chatglm: string;
  natural_explanation_qwen: string;
  natural_explanation_mistral: string;
  natural_explanation_mixtral: string;
  natural_explanation_gemma: string;
  natural_explanation_gpt2: string;
  natural_explanation_lime: string;
  template_explanation: string;
};

const Redirect: React.FC = () => {
  const [response, setResponse] = useState<PredictionResponse | null>(null);

  useEffect(() => {
    const saved = localStorage.getItem("latestResponse");
    if (saved) {
      try {
        const parsed = JSON.parse(saved) as PredictionResponse;
        setResponse(parsed);
      } catch (err) {
        console.error("Error parsing prediction response:", err);
      }
    }
  }, []);

  const getReadableFeatureName = (key: string) => {
    return (featureAliases as { [key: string]: string })[key] || key;
  };

  if (!response) {
    return <div style={{ padding: "20px" }}>No prediction result found.</div>;
  }

  return (
    <div className="container">
      <h2>🧭 Explanation Redirect Page</h2>

      <h3>
        Prediction:{" "}
        <span style={{ color: response.prediction ? "red" : "green" }}>
          {response.prediction === 1 ? "Loan Default" : "Loan Approved"}
        </span>
      </h3>

      <p>
        Probability of Default:{" "}
        <strong>{(response.probability_of_default * 100).toFixed(2)}%</strong>
      </p>

      <h4>Top SHAP Features</h4>
      <ul>
        {response.top_feature_impacts.map((feat, i) => (
          <li key={i}>
            <strong>{getReadableFeatureName(feat.feature)}</strong>: {feat.value} (Impact:{" "}
            {feat.shap_impact.toFixed(4)})
          </li>
        ))}
      </ul>

      <h4>Top LIME Features</h4>
      <ul>
        {response.top_feature_impacts_lime.map((feat, i) => (
          <li key={i}>
            <strong>{getReadableFeatureName(feat.feature)}</strong>: {feat.value} (Impact:{" "}
            {feat.impact.toFixed(4)})
          </li>
        ))}
      </ul>

      <h4>LLM Narrative Explanations</h4>
      <ul>
        <li><strong>Groq:</strong> {response.natural_explanation_groq}</li>
        <li><strong>OpenAI:</strong> {response.natural_explanation_openai}</li>
        <li><strong>ChatGLM:</strong> {response.natural_explanation_chatglm}</li>
        <li><strong>Qwen:</strong> {response.natural_explanation_qwen}</li>
        <li><strong>Mistral:</strong> {response.natural_explanation_mistral}</li>
        <li><strong>Mixtral:</strong> {response.natural_explanation_mixtral}</li>
        <li><strong>Gemma:</strong> {response.natural_explanation_gemma}</li>
        <li><strong>GPT-2:</strong> {response.natural_explanation_gpt2}</li>
        <li><strong>LIME Explanation:</strong> {response.natural_explanation_lime}</li>
      </ul>

      <h4>📋 Template Explanation</h4>
      <p>{response.template_explanation}</p>
      <h4>📊 Evaluation Visualizations</h4>
        <div style={{ display: "flex", justifyContent: "center", gap: "20px", flexWrap: "wrap" }}>
          <div>
            <img
              src="/llm_score_boxplot.png"
              alt="LLM Score Boxplot"
              style={{ width: "400px", border: "1px solid #ccc", borderRadius: "8px" }}
            />
            <p style={{ textAlign: "center" }}>LLM Score Boxplot</p>
          </div>
          <div>
            <img
              src="/prediction_boxplot.png"
              alt="Prediction Boxplot"
              style={{ width: "400px", border: "1px solid #ccc", borderRadius: "8px" }}
            />
            <p style={{ textAlign: "center" }}>Prediction Boxplot</p>
          </div>
        </div>

    </div>
  );
};

export default Redirect;
