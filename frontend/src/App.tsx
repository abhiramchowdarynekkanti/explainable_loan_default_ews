import React, { useEffect, useState } from "react";
import axios from "axios";
import { BrowserRouter, Routes, Route, useNavigate } from "react-router-dom";
import featureList from "./features.json";
import featureAliases from "./feature_aliases.json";
import "./App.css";
import a from "./a.jpeg";

type FormData = { [key: string]: string };

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
  llm_scores: { [key: string]: number };
};

const REQUIRED_FEATURES = [
  "AGE",
  "CNT_CHILDREN",
  "CNT_FAM_MEMBERS",
  "DAYS_EMPLOYED",
  "AMT_INCOME_TOTAL",
  "AMT_CREDIT",
  "AMT_ANNUITY",
  "AMT_GOODS_PRICE",
];
const REQUIRED_FEATURES_SET = new Set(REQUIRED_FEATURES);

const getReadableFeatureName = (key: string) =>
  (featureAliases as { [key: string]: string })[key] || key;

const PredictionForm: React.FC = () => {
  const [formData, setFormData] = useState<FormData>({});
  const [response, setResponse] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showSecondBest, setShowSecondBest] = useState(false); // ✅ Top-level useState
  const navigate = useNavigate();

  useEffect(() => {
    const initial: FormData = {};
    featureList.forEach((key) => (initial[key] = ""));
    setFormData(initial);
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async () => {
    setError(null);

    const missingFields = Array.from(REQUIRED_FEATURES_SET).filter(
      (key) => !formData[key] || formData[key].trim() === ""
    );

    if (missingFields.length > 0) {
      alert(
        `Please fill all required fields:\n\n${missingFields
          .map((key) => `• ${getReadableFeatureName(key)}`)
          .join("\n")}`
      );
      return;
    }

    try {
      const numericData: { [key: string]: number } = {};
      Object.entries(formData).forEach(([key, value]) => {
        numericData[key] = parseFloat(value) || 0;
      });

      const res = await axios.post<PredictionResponse>(
        "http://localhost:8000/predict",
        { data: numericData }
      );
      setResponse(res.data);
      localStorage.setItem("latestResponse", JSON.stringify(res.data));
      setShowSecondBest(false); // ✅ Reset on new prediction
    } catch (err: any) {
      console.error(err);
      setError(err?.response?.data?.error || "Prediction failed.");
      setResponse(null);
    }
  };

  return (
    <div className="container">
      <div className="header">
        <img src={a} alt="IDRBT Logo" className="logo-img" />
        <h1>IDRBT Loan Default Predictor</h1>
      </div>

      <form className="form-box">
        {Object.entries(formData).map(([key, value]) => (
          <div key={key} className="form-row">
            <label>
              {getReadableFeatureName(key)}
              {REQUIRED_FEATURES_SET.has(key) && (
                <span style={{ color: "red", marginLeft: "4px" }}>*</span>
              )}
            </label>
            <input
              type="text"
              name={key}
              value={value}
              onChange={handleChange}
            />
          </div>
        ))}
      </form>

      <div className="button-container">
        <button onClick={handleSubmit} className="predict-button">
          Predict
        </button>
        <button
          onClick={() => navigate("/redirect")}
          className="predict-button"
          style={{ marginLeft: "10px" }}
        >
          Bank Official
        </button>
      </div>

      {error && <p style={{ color: "red" }}>{error}</p>}

      {response && (
        <div className="result-section">
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

          {response.llm_scores && (() => {
            const entries = Object.entries(response.llm_scores || {});
            if (entries.length === 0) return <p>No LLM scores available.</p>;

            const sorted = [...entries].sort((a, b) => b[1] - a[1]);
            const [bestLLM, secondBestLLM] = sorted;

            const explanationMap: { [key: string]: string } = {
              groq: response.natural_explanation_groq,
              chatglm: response.natural_explanation_chatglm,
              qwen: response.natural_explanation_qwen,
              mistral: response.natural_explanation_mistral,
              mixtral: response.natural_explanation_mixtral,
              gemma: response.natural_explanation_gemma,
              gpt2: response.natural_explanation_gpt2,
            };

            return (
              <>
                <h4>🧠 Explanation</h4>
                <p>
                  <strong>{bestLLM[0].toUpperCase()}</strong>:{" "}
                  {explanationMap[bestLLM[0]] || "No explanation found."}
                </p>

                <div style={{ marginTop: "10px" }}>
                  <button
                    onClick={() => alert("Thanks for your feedback!")}
                    style={{ marginRight: "10px", cursor: "pointer" }}
                  >
                    👍 Like
                  </button>
                  <button
                    onClick={() => setShowSecondBest(true)}
                    style={{ cursor: "pointer" }}
                  >
                    👎 Dislike
                  </button>
                </div>

                {showSecondBest && secondBestLLM && (
                  <div style={{ marginTop: "15px" }}>
                    <p>
                      <strong>{secondBestLLM[0].toUpperCase()}</strong>:{" "}
                      {explanationMap[secondBestLLM[0]] || "No explanation found."}
                    </p>
                  </div>
                )}
              </>
            );
          })()}
        </div>
      )}
    </div>
  );
};

const RedirectPage: React.FC = () => {
  const [response, setResponse] = useState<PredictionResponse | null>(null);

  useEffect(() => {
    const data = localStorage.getItem("latestResponse");
    if (data) setResponse(JSON.parse(data));
  }, []);

  if (!response) return <p style={{ padding: "20px" }}>No prediction available.</p>;

  const getReadableFeatureName = (key: string) =>
    (featureAliases as { [key: string]: string })[key] || key;

  return (
    <div className="container">
      <h2>📍 Bank official Authenticated page</h2>
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

const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<PredictionForm />} />
        <Route path="/redirect" element={<RedirectPage />} />
      </Routes>
    </BrowserRouter>
  );
};

export default App;
