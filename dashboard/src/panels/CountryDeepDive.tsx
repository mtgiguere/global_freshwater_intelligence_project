/**
 * Panel 3 — Country Deep Dive
 *
 * Triggered by clicking a country on the Global Water Atlas.
 * Shows the full history of one country across all variables.
 *
 * What this tells you:
 *   - How freshwater availability has changed over time in this country
 *   - How that tracks against economic performance, health, and stability
 *   - The country's current Compound Risk Score and trajectory
 *
 * This is the panel most relevant to a policymaker who wants to understand
 * a specific country's water-related vulnerability in depth.
 *
 * Charts (Recharts line charts) are in Phase 5 iteration 2.
 * This version shows the data in a readable table.
 */

import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { CountryDetail } from "../api/client";

const regionNames = new Intl.DisplayNames(["en"], { type: "region" });
const countryName = (iso3: string) => {
  try { return regionNames.of(iso3) ?? iso3; }
  catch { return iso3; }
};

export default function CountryDeepDive({ iso3 }: { iso3: string }) {
  const [detail, setDetail] = useState<CountryDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    api.countryDetail(iso3)
      .then(setDetail)
      .catch(() => setError(`No data found for country code "${iso3}"`))
      .finally(() => setLoading(false));
  }, [iso3]);

  if (loading) return <p>Loading data for {iso3}…</p>;
  if (error)   return <p style={{ color: "#c62828" }}>{error}</p>;
  if (!detail) return null;

  const recent = detail.timeseries.filter(r => r.year >= 1990).reverse();

  return (
    <div style={{ maxWidth: 1000 }}>
      <h2>Country Deep Dive — {countryName(detail.iso3)} <span style={{ color: "#aaa", fontSize: 16, fontWeight: 400 }}>({detail.iso3})</span></h2>
      <p style={{ color: "#555" }}>
        Annual observations from the Master Panel (1990 onwards).
        All variables are sourced from internationally recognised databases:
        FAO AQUASTAT (water), World Bank (economy), WHO (health), FSI (fragility).
      </p>

      <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 13 }}>
        <thead>
          <tr style={{ background: "#1a3a5c", color: "white" }}>
            <th style={{ padding: "8px 10px", textAlign: "left" }}>Year</th>
            <th style={{ padding: "8px 10px", textAlign: "right" }}>Freshwater<br />(m³/person/yr)</th>
            <th style={{ padding: "8px 10px", textAlign: "right" }}>GDP per capita<br />(2015 USD)</th>
            <th style={{ padding: "8px 10px", textAlign: "right" }}>Life Expectancy<br />(years)</th>
            <th style={{ padding: "8px 10px", textAlign: "right" }}>State Fragility<br />(FSI score)</th>
            <th style={{ padding: "8px 10px", textAlign: "right" }}>Conflict<br />(UCDP binary)</th>
          </tr>
        </thead>
        <tbody>
          {recent.map((row, i) => (
            <tr key={row.year} style={{ background: i % 2 === 0 ? "#fafafa" : "white", borderBottom: "1px solid #eee" }}>
              <td style={{ padding: "6px 10px", fontWeight: 600 }}>{row.year}</td>
              <td style={{ padding: "6px 10px", textAlign: "right" }}>
                {row.renewable_freshwater_percap?.toLocaleString(undefined, { maximumFractionDigits: 0 }) ?? "—"}
              </td>
              <td style={{ padding: "6px 10px", textAlign: "right" }}>
                {row.gdp_pc_ppp?.toLocaleString(undefined, { maximumFractionDigits: 0 }) ?? "—"}
              </td>
              <td style={{ padding: "6px 10px", textAlign: "right" }}>
                {row.life_expectancy?.toFixed(1) ?? "—"}
              </td>
              <td style={{ padding: "6px 10px", textAlign: "right" }}>
                {row.fsi_score?.toFixed(1) ?? "—"}
              </td>
              <td style={{ padding: "6px 10px", textAlign: "right" }}>
                {row.ucdp_conflict_binary !== undefined ? (row.ucdp_conflict_binary === 1 ? "Yes" : "No") : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p style={{ color: "#999", fontSize: 12, marginTop: 8 }}>
        "—" indicates data not available for that year. Recharts time-series charts in Phase 5 iteration 2.
      </p>
    </div>
  );
}
