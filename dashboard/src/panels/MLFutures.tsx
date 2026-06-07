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
  why: string;
  how: string;
  interpret: (score: number) => string;
}> = [
  {
    key: "scarcity_score",
    label: "Water Scarcity Forecast",
    method: "Gradient Boosting Regression",
    predicts: "log(freshwater per capita) 5 years ahead",
    color: "#1565c0",
    why: `Water scarcity is a slow-moving crisis that often isn't visible until it's
      too late — aquifers are depleted silently, rivers are over-allocated for
      decades before they run dry. This model was built to give an early-warning
      signal: given current trends in population growth, agricultural water use,
      and groundwater depletion (from GRACE satellites), how much freshwater
      will this country have per person in five years?`,
    how: `Trained on the GFIP Master Panel (274 countries, data through 2024) using
      Gradient Boosting — a technique that builds hundreds of small decision trees,
      each correcting the errors of the last. Features include lagged freshwater
      values (what was the trend?), rolling averages (is it accelerating?),
      population growth (more people = less water per person), GDP per capita
      (wealthier countries invest more in water infrastructure), and GRACE
      groundwater anomalies (is the underground reserve declining?).`,
    interpret: (s) =>
      s >= 0.7
        ? `HIGH RISK (${(s * 100).toFixed(0)}%) — The model forecasts significant water stress. This country is on a trajectory toward serious scarcity, likely driven by population growth, over-extraction, or declining precipitation.`
        : s >= 0.4
        ? `MODERATE RISK (${(s * 100).toFixed(0)}%) — Some scarcity pressure is expected. Water availability is tightening but not yet critical. Policy intervention now could prevent deterioration.`
        : `LOW RISK (${(s * 100).toFixed(0)}%) — The model sees no major scarcity signal in the near term. The country has adequate renewable freshwater relative to its population and usage trajectory.`,
  },
  {
    key: "instability_probability",
    label: "Political Instability Forecast",
    method: "XGBoost Binary Classifier",
    predicts: "P(FSI jump >5 pts OR new conflict within 3 years)",
    color: "#b71c1c",
    why: `Political instability and armed conflict cause immense human suffering and are
      extremely costly to reverse. Early warning systems that can flag deteriorating
      conditions before crisis erupts are one of the most valuable tools in
      international development and peace-building. This model asks: given everything
      we know about this country's water stress, economic health, and existing
      fragility, how likely is it to experience significant political deterioration
      or the onset of new armed conflict in the next three years?`,
    how: `Built using XGBoost — the same algorithm used by leading early-warning systems
      like ACLED's conflict prediction tool. XGBoost handles the high rates of missing
      data in conflict research, captures non-linear interactions (e.g., water scarcity
      only increases conflict risk when economic buffers are already weak), and is
      robust to outliers. The target variable combines two signals: an FSI score jump
      of more than 5 points (rapid state fragility increase) OR the onset of new armed
      conflict per UCDP definition. Features include water stress lags, GDP trajectory,
      existing FSI level, and conflict history.`,
    interpret: (s) =>
      s >= 0.6
        ? `HIGH PROBABILITY (${(s * 100).toFixed(0)}%) — The model assigns a substantial probability to significant political deterioration or conflict onset within 3 years. This is a serious early-warning signal warranting close monitoring and pre-emptive engagement.`
        : s >= 0.3
        ? `MODERATE PROBABILITY (${(s * 100).toFixed(0)}%) — Some elevated risk is present. The country has vulnerability factors that could compound under stress. Preventive diplomacy and economic support are most cost-effective at this stage.`
        : `LOW PROBABILITY (${(s * 100).toFixed(0)}%) — The model sees a relatively stable near-term outlook. Existing institutions and buffers appear adequate to absorb current stresses.`,
  },
  {
    key: "migration_score",
    label: "Displacement Pressure Forecast",
    method: "Random Forest Regression",
    predicts: "log(refugee outflow + 1)",
    color: "#e65100",
    why: `Forced displacement is one of the most severe humanitarian consequences of
      combined water stress, economic failure, and political violence. Over 100 million
      people are currently displaced worldwide — a record high. Understanding which
      countries are most likely to generate large refugee flows in the coming years
      helps international agencies pre-position resources, and helps destination
      countries plan for arrivals. This model links water stress to displacement
      pressure, completing the GFIP causal chain: water → instability → migration.`,
    how: `Built using Random Forest — an ensemble of decision trees ideal for this
      problem because refugee data is highly skewed (most country-years produce zero
      refugees; a few produce millions) and has significant missing values (not all
      countries report to UNHCR consistently). Random Forest handles both properties
      well. Features include freshwater stress indicators, conflict events (UCDP/ACLED),
      FSI fragility scores, economic indicators, and population size. The target is
      log(refugee outflow + 1), which compresses the extreme skew and is back-
      transformed to a normalised 0–1 score for display.`,
    interpret: (s) =>
      s >= 0.6
        ? `HIGH PRESSURE (${(s * 100).toFixed(0)}%) — The model forecasts significant forced displacement from this country. Combined water stress, fragility, and conflict indicators are elevated. International protection and resettlement capacity should be planned.`
        : s >= 0.3
        ? `MODERATE PRESSURE (${(s * 100).toFixed(0)}%) — Some displacement pressure is forecast. Conditions exist that could drive increased refugee outflow if they deteriorate. Monitoring and early humanitarian pre-positioning are warranted.`
        : `LOW PRESSURE (${(s * 100).toFixed(0)}%) — The model does not see major forced displacement signals in the near term. The country's combination of water security, economic stability, and governance capacity is currently adequate.`,
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
      <h2>Risk Forecast — {prediction?.country_name ?? iso3}</h2>
      <p style={{ color: "#555", marginBottom: 16, lineHeight: 1.6 }}>
        These scores are produced by three independent machine learning models trained on the
        GFIP Master Panel — 274 countries, data through 2024. Each model was built to answer a specific
        policy-relevant question about this country's near-term future. The scores are not
        predictions in the sense of "this will definitely happen" — they are <em>risk signals</em>:
        the models have learned what patterns preceded past crises and are flagging whether
        those patterns are present here today.
      </p>
      <p style={{ color: "#777", fontSize: 13, marginBottom: 20, lineHeight: 1.5 }}>
        Together they feed the <strong>Compound Risk Score</strong> — a single 0–100 index
        that summarises this country's overall water-related vulnerability.
        A score above 70 warrants urgent attention; 50–70 calls for active monitoring and
        preventive investment; below 30 indicates the country has healthy buffers.
      </p>

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
                  <h3 style={{ margin: "0 0 8px", color: m.color }}>{m.label}</h3>
                  <ScoreBar value={score} color={m.color} />
                  {/* Plain-language verdict — the so-what, shown immediately after the bar */}
                  <p style={{ margin: "8px 0 10px", fontSize: 14, color: "#1a1a1a",
                    background: "#fafafa", padding: "8px 10px", borderRadius: 4, lineHeight: 1.6 }}>
                    {m.interpret(score)}
                  </p>
                  {/* Method and technical detail — for researchers, tucked away */}
                  <details style={{ fontSize: 13 }}>
                    <summary style={{ cursor: "pointer", color: "#1a3a5c", fontWeight: 600 }}>
                      How does this model work?
                    </summary>
                    <div style={{ marginTop: 8, paddingLeft: 12, borderLeft: `3px solid ${m.color}` }}>
                      <p style={{ margin: "0 0 6px", color: "#666", fontSize: 12 }}>
                        <strong>Method:</strong> {m.method} · <strong>Output:</strong> {m.predicts}
                      </p>
                      <p style={{ margin: "0 0 8px", color: "#444", lineHeight: 1.6 }}>
                        <strong>Why we built it:</strong> {m.why}
                      </p>
                      <p style={{ margin: 0, color: "#444", lineHeight: 1.6 }}>
                        <strong>How it works:</strong> {m.how}
                      </p>
                    </div>
                  </details>
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
