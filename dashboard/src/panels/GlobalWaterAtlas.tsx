/**
 * Panel 1 — Global Water Atlas
 *
 * The landing view. Shows every country coloured by its Compound Risk Score
 * (0-100). Red = critical water stress and human vulnerability. Green = stable.
 *
 * What this tells you at a glance:
 *   - Which parts of the world are most water-stressed RIGHT NOW
 *   - How that stress overlaps with fragility, conflict, and poor health
 *   - Where the hidden groundwater crisis (H7) is concentrated
 *
 * Click any country to open its Country Deep Dive.
 *
 * Implementation note: the full Deck.gl WebGL globe is Phase 5 iteration 2.
 * This version uses a colour-coded table that conveys the same information
 * while the geospatial layer is built.
 */

import { useEffect, useState } from "react";
import { api, CountryRisk } from "../api/client";

const riskLabel = (score: number) => {
  if (score >= 70) return { label: "Critical", color: "#c62828" };
  if (score >= 50) return { label: "High",     color: "#e65100" };
  if (score >= 30) return { label: "Elevated", color: "#f9a825" };
  return                  { label: "Low",      color: "#2e7d32" };
};

export default function GlobalWaterAtlas({ onCountrySelect }: { onCountrySelect: (iso3: string) => void }) {
  const [risks, setRisks] = useState<CountryRisk[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.globalRisk()
      .then(data => setRisks(data.sort((a, b) => b.compound_risk_score - a.compound_risk_score)))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <p>Loading global risk data…</p>;

  return (
    <div>
      <h2>Global Water Risk Atlas</h2>
      <p style={{ color: "#555", maxWidth: 720 }}>
        Each country is scored 0–100 on the <strong>Compound Risk Score</strong> —
        a combination of water scarcity (30%), political instability risk (35%),
        and migration pressure (35%). Higher scores indicate greater vulnerability.
        Click a country to explore its full data history.
      </p>

      <div style={{ display: "flex", gap: 16, marginBottom: 16, flexWrap: "wrap" }}>
        {[["Critical (≥70)", "#c62828"], ["High (50–70)", "#e65100"], ["Elevated (30–50)", "#f9a825"], ["Low (<30)", "#2e7d32"]].map(([label, color]) => (
          <span key={label} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13 }}>
            <span style={{ width: 14, height: 14, background: color as string, borderRadius: 2, display: "inline-block" }} />
            {label}
          </span>
        ))}
      </div>

      <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 14 }}>
        <thead>
          <tr style={{ background: "#f5f5f5" }}>
            <th style={{ textAlign: "left", padding: "8px 12px" }}>Country</th>
            <th style={{ textAlign: "left", padding: "8px 12px" }}>Year</th>
            <th style={{ textAlign: "left", padding: "8px 12px" }}>Risk Level</th>
            <th style={{ textAlign: "right", padding: "8px 12px" }}>CRS Score</th>
          </tr>
        </thead>
        <tbody>
          {risks.slice(0, 50).map(r => {
            const { label, color } = riskLabel(r.compound_risk_score);
            return (
              <tr key={r.iso3} style={{ borderBottom: "1px solid #eee", cursor: "pointer" }}
                onClick={() => onCountrySelect(r.iso3)}>
                <td style={{ padding: "8px 12px", fontWeight: 500 }}>{r.iso3}</td>
                <td style={{ padding: "8px 12px", color: "#777" }}>{r.year}</td>
                <td style={{ padding: "8px 12px" }}>
                  <span style={{ background: color, color: "white", borderRadius: 4, padding: "2px 8px", fontSize: 12 }}>{label}</span>
                </td>
                <td style={{ padding: "8px 12px", textAlign: "right", fontWeight: 700, color }}>{r.compound_risk_score}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p style={{ color: "#999", fontSize: 12, marginTop: 8 }}>Showing top 50 countries by risk score. Full Deck.gl map in Phase 5 iteration 2.</p>
    </div>
  );
}
