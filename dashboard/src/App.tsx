/**
 * App — Root React component for the GFIP Dashboard.
 *
 * Defines the overall dashboard layout (top navigation bar with CountrySearch, tab
 * bar with four panel buttons, main content area) and manages the single piece of
 * shared state: `selectedIso3` (the ISO3 alpha-3 code of the currently focused
 * country, defaulting to "AFG"). All four panels receive `selectedIso3` as a prop
 * and re-render when it changes.
 *
 * STATE ARCHITECTURE
 * A deliberate design decision: there is only ONE piece of global state — which
 * country the user has selected. Everything else (risk data, time series, predictions)
 * is fetched by the individual panels when they mount or when `selectedIso3` changes.
 * This keeps the root component simple and avoids a global state manager (Redux,
 * Zustand) for what is essentially a single-select interface.
 *
 * SHARED DATA (allCountries)
 * The global risk list (GET /api/v1/global/risk) is loaded once here and passed to
 * CountrySearch for the autocomplete. GlobalWaterAtlas also fetches the same endpoint
 * independently (for the globe colours) — this is a small, intentional duplication
 * that keeps panels self-contained. Both requests are served from the browser cache
 * after the first one completes.
 *
 * PANELS
 * Each panel corresponds to a phase or result type of the GFIP project:
 *   - GlobalWaterAtlas: globe with CRS colours (Phase 5 / Phase 4 ML scores)
 *   - OutcomesExplorer: H1–H7 regression results (Phase 3)
 *   - CountryDeepDive: historical time-series charts (Phase 1 Master Panel)
 *   - MLFutures: per-country ML predictions (Phase 4 models)
 *
 * The goal: a UN analyst, a journalist, or a curious citizen can open this and
 * immediately understand the relationship between water and human welfare without
 * reading a single line of code.
 */

import { useState, useEffect } from "react";
import GlobalWaterAtlas from "./panels/GlobalWaterAtlas";
import OutcomesExplorer from "./panels/OutcomesExplorer";
import CountryDeepDive from "./panels/CountryDeepDive";
import MLFutures from "./panels/MLFutures";
import CountrySearch from "./components/CountrySearch";
import { api } from "./api/client";
import type { CountryRisk } from "./api/client";

type Panel = "atlas" | "outcomes" | "country" | "futures";

const NAV: { id: Panel; label: string }[] = [
  { id: "atlas", label: "Global Water Atlas" },
  { id: "outcomes", label: "Outcomes Explorer" },
  { id: "country", label: "Country Deep Dive" },
  { id: "futures", label: "ML Futures" },
];

/**
 * App component — dashboard shell.
 *
 * Renders the navigation header and routes between panels based on the `active`
 * state. The `country` (selectedIso3) state is shared downward to CountryDeepDive
 * and MLFutures via props; selecting a country in CountrySearch or clicking on the
 * globe both update it.
 */
export default function App() {
  /** Currently active panel tab — determines which panel is rendered in `<main>`. */
  const [active, setActive] = useState<Panel>("atlas");
  /** ISO3 code of the currently selected country; drives CountryDeepDive and MLFutures. */
  const [country, setCountry] = useState<string>("AFG");
  /** Full country risk list, loaded once for the CountrySearch autocomplete. */
  const [allCountries, setAllCountries] = useState<CountryRisk[]>([]);

  // Load the country list once on mount for the CountrySearch autocomplete dropdown.
  // If the API is unavailable (e.g. network error, Render cold start), CountrySearch
  // degrades gracefully — it shows an empty dropdown rather than crashing.
  useEffect(() => {
    api.globalRisk().then(setAllCountries).catch(() => {/* search degrades gracefully to empty */});
  }, []);

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", margin: 0 }}>
      <header style={{ background: "#1a3a5c", color: "white", padding: "12px 24px", display: "flex", alignItems: "center", gap: 24 }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 18 }}>Global Freshwater Intelligence Project</h1>
          <p style={{ margin: 0, fontSize: 12, opacity: 0.7 }}>Water · Stability · Human Welfare · 274 Countries · 1946–2025</p>
        </div>
        <nav style={{ marginLeft: "auto", display: "flex", gap: 8, alignItems: "center" }}>
          {NAV.map(({ id, label }) => (
            <button key={id} onClick={() => setActive(id)}
              style={{ background: active === id ? "#2196f3" : "transparent", color: "white", border: "1px solid rgba(255,255,255,0.3)", borderRadius: 4, padding: "6px 14px", cursor: "pointer", fontSize: 13 }}>
              {label}
            </button>
          ))}
          <CountrySearch
            countries={allCountries}
            onSelect={iso3 => { setCountry(iso3); setActive("country"); }}
          />
        </nav>
      </header>
      <main style={{ padding: 24 }}>
        {active === "atlas"    && <GlobalWaterAtlas onCountrySelect={setCountry} />}
        {active === "outcomes" && <OutcomesExplorer />}
        {active === "country"  && <CountryDeepDive key={country} iso3={country} />}
        {active === "futures"  && <MLFutures iso3={country} />}
      </main>
    </div>
  );
}
