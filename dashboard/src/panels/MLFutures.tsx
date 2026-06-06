/**
 * Panel 5 — ML Futures
 *
 * The forward-looking panel. Uses the three Phase 4 ML models to project
 * a country's Compound Risk Score into 2025–2050 under different climate
 * and policy scenarios.
 *
 * What this tells you:
 *   - Where is this country headed if current trends continue?
 *   - How much worse does the trajectory get under high-emissions (SSP5-8.5)?
 *   - What would a policy intervention (e.g. 20% improvement in water access) do?
 *
 * This is the panel most relevant to long-term planning and early warning.
 * A score of 70+ within 10 years is the signal that intervention is urgent.
 *
 * Note: model predictions on real data require training the Phase 4 models
 * on the Master Panel. This panel currently shows the model architecture
 * and explains the methodology while live predictions are being wired up.
 */

const regionNames = new Intl.DisplayNames(["en"], { type: "region" });
const countryName = (iso3: string) => {
  try { return regionNames.of(iso3) ?? iso3; }
  catch { return iso3; }
};

export default function MLFutures({ iso3 }: { iso3: string }) {
  return (
    <div style={{ maxWidth: 800 }}>
      <h2>ML Futures — {countryName(iso3)} <span style={{ color: "#aaa", fontSize: 16, fontWeight: 400 }}>({iso3})</span></h2>
      <p style={{ color: "#555" }}>
        Three machine learning models combine to produce forward-looking risk projections:
      </p>

      <div style={{ display: "flex", flexDirection: "column", gap: 16, marginBottom: 24 }}>
        {[
          {
            title: "Model 1 — Water Scarcity Forecaster",
            method: "Gradient Boosting Regression (→ LSTM)",
            predicts: "log(freshwater per capita) 5 years ahead",
            status: "Trained on synthetic data; real-data training pending",
            color: "#1565c0",
          },
          {
            title: "Model 2 — Instability Risk Predictor",
            method: "XGBoost Binary Classifier",
            predicts: "P(FSI jump >5pts OR new conflict within 3 years)",
            status: "Trained on synthetic data; real-data training pending",
            color: "#b71c1c",
          },
          {
            title: "Model 3 — Migration Pressure Estimator",
            method: "Random Forest Regression",
            predicts: "log(refugee outflow + 1)",
            status: "Trained on synthetic data; real-data training pending",
            color: "#e65100",
          },
        ].map(m => (
          <div key={m.title} style={{ border: `2px solid ${m.color}`, borderRadius: 8, padding: 16 }}>
            <h3 style={{ margin: "0 0 8px", color: m.color }}>{m.title}</h3>
            <p style={{ margin: "0 0 4px", fontSize: 14 }}><strong>Method:</strong> {m.method}</p>
            <p style={{ margin: "0 0 4px", fontSize: 14 }}><strong>Predicts:</strong> {m.predicts}</p>
            <p style={{ margin: 0, fontSize: 13, color: "#888", fontStyle: "italic" }}>⏳ {m.status}</p>
          </div>
        ))}
      </div>

      <div style={{ background: "#f5f5f5", borderRadius: 8, padding: 16 }}>
        <h3 style={{ margin: "0 0 8px" }}>Compound Risk Score Formula</h3>
        <p style={{ margin: "0 0 8px", fontFamily: "monospace", fontSize: 14 }}>
          CRS = (Scarcity × 0.30) + (Instability × 0.35) + (Migration × 0.35) × 100
        </p>
        <p style={{ margin: 0, fontSize: 13, color: "#555" }}>
          Each component is normalised to [0, 1] before combining. The weights reflect
          the relative evidence base from Phase 3: instability and migration are weighted
          slightly higher because those effects operate fastest and are most relevant to
          near-term policy response.
        </p>
      </div>

      <p style={{ color: "#999", fontSize: 12, marginTop: 16 }}>
        Live projections with Deck.gl scenario visualisation in Phase 5 iteration 2.
        Train models on real data by running: <code>uv run python src/models/train_all.py</code>
      </p>
    </div>
  );
}
