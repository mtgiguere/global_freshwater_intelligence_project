/**
 * GFIP Dashboard — Global Freshwater Intelligence Project
 *
 * Five panels, accessible to every reader — not just developers.
 * The goal: a UN analyst, a journalist, or a curious citizen can open
 * this and immediately understand the relationship between water and
 * human welfare without reading a single line of code.
 */

import { useState } from "react";
import GlobalWaterAtlas from "./panels/GlobalWaterAtlas";
import OutcomesExplorer from "./panels/OutcomesExplorer";
import CountryDeepDive from "./panels/CountryDeepDive";
import MLFutures from "./panels/MLFutures";

type Panel = "atlas" | "outcomes" | "country" | "futures";

const NAV: { id: Panel; label: string }[] = [
  { id: "atlas", label: "Global Water Atlas" },
  { id: "outcomes", label: "Outcomes Explorer" },
  { id: "country", label: "Country Deep Dive" },
  { id: "futures", label: "ML Futures" },
];

export default function App() {
  const [active, setActive] = useState<Panel>("atlas");
  const [country, setCountry] = useState<string>("AFG");

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", margin: 0 }}>
      <header style={{ background: "#1a3a5c", color: "white", padding: "12px 24px", display: "flex", alignItems: "center", gap: 24 }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 18 }}>Global Freshwater Intelligence Project</h1>
          <p style={{ margin: 0, fontSize: 12, opacity: 0.7 }}>Water · Stability · Human Welfare · 274 Countries · 1946–2025</p>
        </div>
        <nav style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
          {NAV.map(({ id, label }) => (
            <button key={id} onClick={() => setActive(id)}
              style={{ background: active === id ? "#2196f3" : "transparent", color: "white", border: "1px solid rgba(255,255,255,0.3)", borderRadius: 4, padding: "6px 14px", cursor: "pointer", fontSize: 13 }}>
              {label}
            </button>
          ))}
        </nav>
      </header>
      <main style={{ padding: 24 }}>
        {active === "atlas"    && <GlobalWaterAtlas onCountrySelect={setCountry} />}
        {active === "outcomes" && <OutcomesExplorer />}
        {active === "country"  && <CountryDeepDive iso3={country} />}
        {active === "futures"  && <MLFutures iso3={country} />}
      </main>
    </div>
  );
}
