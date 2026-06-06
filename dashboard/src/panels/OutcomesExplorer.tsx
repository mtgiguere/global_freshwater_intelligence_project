/**
 * Panel 2 — Outcomes Explorer
 *
 * Shows the Phase 3 hypothesis testing results: for each of our seven
 * hypotheses (H1–H7), what did the statistical analysis find?
 *
 * What this tells you:
 *   - The effect size (beta): how much does freshwater availability
 *     change each outcome, holding everything else constant?
 *   - Whether the result is statistically significant
 *   - The number of country-years used in the analysis
 *
 * Plain-language interpretation is provided below each result so that
 * a reader without a statistics background can understand what it means.
 *
 * The scatter plots (H1–H7 pairwise) are in Phase 5 iteration 2 using Recharts.
 */

import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { HypothesisResult } from "../api/client";

const interpret = (h: HypothesisResult): string => {
  const dir = h.beta > 0 ? "increases" : "decreases";
  const sig = h.p_value < 0.05 ? "statistically significant" : "directionally consistent but not yet statistically significant";
  return `A unit improvement in ${h.exposure} ${dir} ${h.outcome} by ${Math.abs(h.beta).toFixed(3)} units. This result is ${sig} (p = ${h.p_value.toFixed(3)}, n = ${h.n_obs.toLocaleString()} country-years).`;
};

export default function OutcomesExplorer() {
  const [hypotheses, setHypotheses] = useState<HypothesisResult[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.hypotheses().then(setHypotheses).finally(() => setLoading(false));
  }, []);

  if (loading) return <p>Loading hypothesis results…</p>;

  return (
    <div style={{ maxWidth: 900 }}>
      <h2>Outcomes Explorer — Does Water Drive Human Welfare?</h2>
      <p style={{ color: "#555" }}>
        These results come from two-way fixed effects panel regression — a statistical
        method that controls for everything permanently different between countries
        (geography, history, culture) and asks only: <em>within the same country,
        when water availability changed, did human outcomes change?</em>
        This is the gold standard for establishing causal relationships in
        cross-country panel data.
      </p>

      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        {hypotheses.map(h => (
          <div key={h.id} style={{ border: "1px solid #ddd", borderRadius: 8, padding: 16,
            borderLeft: `4px solid ${h.confirmed ? "#2e7d32" : "#e65100"}` }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
              <span style={{ fontWeight: 700, fontSize: 18, color: "#1a3a5c" }}>{h.id}</span>
              <span style={{ fontWeight: 500 }}>{h.label}</span>
              <span style={{ marginLeft: "auto", background: h.confirmed ? "#e8f5e9" : "#fff3e0",
                color: h.confirmed ? "#2e7d32" : "#e65100", borderRadius: 4, padding: "2px 10px", fontSize: 12 }}>
                {h.confirmed ? "Confirmed" : "Directional"}
              </span>
            </div>
            <div style={{ display: "flex", gap: 24, marginBottom: 8, fontSize: 14, color: "#555" }}>
              <span><strong>β</strong> = {h.beta.toFixed(3)}</span>
              <span><strong>p</strong> = {h.p_value.toFixed(4)}</span>
              <span><strong>n</strong> = {h.n_obs.toLocaleString()}</span>
            </div>
            <p style={{ margin: 0, fontSize: 14, color: "#333" }}>{interpret(h)}</p>
            {h.note && <p style={{ margin: "8px 0 0", fontSize: 12, color: "#888", fontStyle: "italic" }}>{h.note}</p>}
          </div>
        ))}
      </div>
    </div>
  );
}
