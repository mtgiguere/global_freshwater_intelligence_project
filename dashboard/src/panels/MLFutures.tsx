/**
 * Panel 5 — ML Futures
 *
 * Forward-looking ML predictions panel. Shows the Phase 4 model outputs for the
 * currently selected country across three dimensions plus the combined Compound Risk
 * Score:
 *
 *   1. Water Scarcity Score (0–1)
 *      Gradient Boosting Regression — predicts log(freshwater per capita) 5 years
 *      ahead. A score near 1.0 means the model expects severe future water scarcity
 *      given current trends in usage, population, and GRACE groundwater anomalies.
 *
 *   2. Instability Probability (0–1)
 *      XGBoost Binary Classifier — predicts P(FSI score jumps >5 points OR a new
 *      armed conflict begins within 3 years). A score of 0.8 means the model assigns
 *      80% probability to significant political deterioration in the near term.
 *
 *   3. Migration Pressure Score (0–1)
 *      Random Forest Regression — predicts log(refugee outflow + 1), normalised to
 *      [0, 1]. A score near 1.0 indicates the model expects large-scale forced
 *      displacement from this country in the coming years.
 *
 *   4. Compound Risk Score (0–100)
 *      Weighted combination: Scarcity × 30% + Instability × 35% + Migration × 35%.
 *      Instability and migration carry slightly more weight because they represent
 *      the fastest-moving, most policy-actionable risks.
 *
 * IMPORTANT — is_trained flag:
 * The FastAPI backend has a synthetic CI fallback that generates plausible-looking
 * scores without real model files. When `is_trained=false`, the API is in this
 * fallback mode and the numbers are illustrative only. A prominent warning banner is
 * shown to communicate this clearly to users. To generate real forecasts, run:
 *   `uv run python src/models/train_all.py`
 * This trains all three models on the Master Panel and saves them to data/models/.
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

/**
 * ScoreBar — horizontal progress-bar visualisation for a normalised score (0–1).
 *
 * Renders a coloured fill proportional to `value` against a grey track. The CSS
 * transition gives a smooth animation when the value changes (e.g. when the user
 * selects a new country), which helps the reader perceive the change rather than
 * just seeing a new static bar.
 *
 * @param props.value - Score in the range [0, 1]. 0 = bar is empty; 1 = bar is full.
 * @param props.color - CSS colour string for the filled portion, e.g. "#1565c0".
 */
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

/**
 * MLFutures component — ML predictions panel for the selected country.
 *
 * @param props.iso3 - ISO 3166-1 alpha-3 country code for the country to predict,
 *   e.g. "SDN" for Sudan. Changing this prop triggers a new API request. The
 *   `cancelled` flag in the useEffect cleanup ensures stale responses from a
 *   previous country do not overwrite the current country's data.
 */
export default function MLFutures({ iso3 }: { iso3: string }) {
  const [prediction, setPrediction] = useState<CountryPrediction | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // `cancelled` is a boolean flag set to true in the cleanup function (the
    // function returned from useEffect). React calls the cleanup when the component
    // unmounts OR when the iso3 prop changes before the previous fetch resolves.
    //
    // Without this pattern, selecting Country A then quickly selecting Country B
    // could result in B's render being overwritten by A's late-arriving response.
    // The flag is cheaper and more reliable than AbortController for this use case.
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
      // Mark any in-flight request as stale. The fetch itself is not aborted
      // (the network request completes), but its result is silently discarded.
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
          {/* is_trained warning banner — shown when the API is in synthetic CI fallback mode.
              The FastAPI backend can run without real model files, but in that mode it
              returns deterministic synthetic scores (seeded by iso3 hash) that are designed
              to look plausible for CI purposes only. They are NOT real predictions.
              The `role="alert"` attribute ensures screen readers announce this to
              accessibility users — it is the most important thing on the page when present. */}
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
