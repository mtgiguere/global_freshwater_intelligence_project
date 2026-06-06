/**
 * Panel 5 — ML Futures
 *
 * Calls GET /api/v1/predict/{iso3} to surface the Phase 4 model scores for the
 * currently selected country. Displays per-component scores (scarcity, instability,
 * migration) and the combined Compound Risk Score.
 *
 * When is_trained=false the API is returning synthetic CI fallback data — a banner
 * is shown so users know the numbers are illustrative, not real forecasts.
 *
 * TDD note (per TDD contract §"Where strict RED/GREEN TDD is not fully applicable"):
 * UI rendering is excluded from strict RED/GREEN cycles. The useEffect data-fetch
 * pattern and the is_trained warning banner are covered by RTL tests in
 * __tests__/MLFutures.test.tsx. WebGL-dependent rendering is not tested.
 */

import { useEffect, useState } from "react";
import type { CountryPrediction } from "../api/client";
import { api } from "../api/client";

const MODELS: Array<{
  key: keyof CountryPrediction;
  label: string;
  method: string;
  predicts: string;
  color: string;
}> = [
  {
    key: "scarcity_score",
    label: "Model 1 — Water Scarcity Forecaster",
    method: "Gradient Boosting Regression",
    predicts: "log(freshwater per capita) 5 years ahead",
    color: "#1565c0",
  },
  {
    key: "instability_probability",
    label: "Model 2 — Instability Risk Predictor",
    method: "XGBoost Binary Classifier",
    predicts: "P(FSI jump >5 pts OR new conflict within 3 years)",
    color: "#b71c1c",
  },
  {
    key: "migration_score",
    label: "Model 3 — Migration Pressure Estimator",
    method: "Random Forest Regression",
    predicts: "log(refugee outflow + 1)",
    color: "#e65100",
  },
];

function ScoreBar({ value, color }: { value: number; color: string }) {
  return (
    <div style={{ background: "#eee", borderRadius: 4, height: 10, margin: "6px 0" }}>
      <div
        style={{
          width: `${Math.round(value * 100)}%`,
          background: color,
          height: "100%",
          borderRadius: 4,
          transition: "width 0.4s ease",
        }}
      />
    </div>
  );
}

export default function MLFutures({ iso3 }: { iso3: string }) {
  const [prediction, setPrediction] = useState<CountryPrediction | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .predictCountry(iso3)
      .then(data => {
        if (!cancelled) {
          setPrediction(data);
          setLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setError("Could not load predictions.");
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [iso3]);

  return (
    <div style={{ maxWidth: 800 }}>
      <h2>ML Futures — {iso3}</h2>

      {loading && <p>Loading predictions…</p>}
      {error && <p style={{ color: "red" }}>{error}</p>}

      {prediction && (
        <>
          {!prediction.is_trained && (
            <div
              role="alert"
              style={{
                background: "#fff3e0",
                border: "1px solid #ff9800",
                borderRadius: 6,
                padding: "8px 12px",
                marginBottom: 16,
                fontSize: 13,
                color: "#e65100",
              }}
            >
              Synthetic data — models not yet trained on real Master Panel. Run{" "}
              <code>uv run python src/models/train_all.py</code> to generate real forecasts.
            </div>
          )}

          <div
            aria-label="Compound Risk Score"
            style={{
              background: "#f5f5f5",
              borderRadius: 8,
              padding: 16,
              marginBottom: 24,
              textAlign: "center",
            }}
          >
            <div style={{ fontSize: 48, fontWeight: 700, color: "#37474f" }}>
              {prediction.compound_risk_score.toFixed(1)}
            </div>
            <div style={{ fontSize: 14, color: "#555" }}>
              Compound Risk Score (0–100) · {prediction.year}
            </div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 16, marginBottom: 24 }}>
            {MODELS.map(m => {
              const raw = prediction[m.key];
              const score = typeof raw === "number" ? raw : 0;
              return (
                <div
                  key={m.key}
                  style={{ border: `2px solid ${m.color}`, borderRadius: 8, padding: 16 }}
                >
                  <h3 style={{ margin: "0 0 4px", color: m.color }}>{m.label}</h3>
                  <p style={{ margin: "0 0 4px", fontSize: 13 }}>
                    <strong>Method:</strong> {m.method}
                  </p>
                  <p style={{ margin: "0 0 8px", fontSize: 13 }}>
                    <strong>Predicts:</strong> {m.predicts}
                  </p>
                  <ScoreBar value={score} color={m.color} />
                  <div style={{ fontSize: 13, color: "#555" }}>
                    Score: <strong>{(score * 100).toFixed(1)}%</strong>
                  </div>
                </div>
              );
            })}
          </div>

          <div style={{ background: "#f5f5f5", borderRadius: 8, padding: 16 }}>
            <h3 style={{ margin: "0 0 8px" }}>Compound Risk Score Formula</h3>
            <p style={{ margin: "0 0 8px", fontFamily: "monospace", fontSize: 14 }}>
              CRS = (Scarcity × 0.30) + (Instability × 0.35) + (Migration × 0.35) × 100
            </p>
            <p style={{ margin: 0, fontSize: 13, color: "#555" }}>
              Each component is normalised to [0, 1] before combining. Instability and
              migration carry slightly higher weight because they operate fastest and are
              most relevant to near-term policy response.
            </p>
          </div>
        </>
      )}
    </div>
  );
}
