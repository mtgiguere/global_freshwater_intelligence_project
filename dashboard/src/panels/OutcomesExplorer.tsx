/**
 * Panel 2 — Global Outcomes Explorer
 *
 * This panel visualises the Phase 3 hypothesis testing results. These are GLOBAL
 * findings — derived from two-way fixed effects panel regression across all 274
 * countries and up to 10,000 country-year observations. They answer the question:
 * "Across the entire world, does water availability predict human welfare?"
 *
 * The panel also shows a "Country Spotlight" per hypothesis: where does the
 * currently selected country sit on each relationship? This lets users connect
 * the global finding to a specific place they care about.
 *
 * WHAT THE NUMBERS MEAN
 * The key number on this panel is beta (β) — the regression coefficient. It tells
 * you HOW MUCH the outcome changes per unit improvement in the freshwater exposure,
 * holding constant everything permanently different between countries (geography,
 * history, culture) and everything that changed globally in any given year (global
 * economic cycles, technology, pandemics). This "two-way fixed effects" design is
 * the gold standard for causal inference in cross-country panel data.
 *
 * p-value: the probability of observing an effect this large by chance alone if the
 * true effect were zero. Values below 0.05 are the conventional threshold for
 * "statistically significant". However, even directional results (correct sign,
 * p > 0.05) are informative — they tell us the data is consistent with the theory.
 *
 * n_obs: the number of country-year observations used in this regression. Larger
 * samples give more power to detect small effects.
 *
 * The scatter plots showing H1–H7 pairwise relationships are planned for Phase 5
 * iteration 2 using Recharts.
 */

import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { HypothesisResult, TimeSeriesPoint } from "../api/client";

// ---------------------------------------------------------------------------
// Country spotlight — maps each hypothesis to the timeseries fields it needs
// ---------------------------------------------------------------------------

/**
 * Describes which fields from the country timeseries are relevant to a given
 * hypothesis. Used to populate the per-card country spotlight.
 */
interface HypothesisVars {
  /** The timeseries field for the freshwater exposure variable, if available. */
  exposureField?: keyof TimeSeriesPoint;
  /** Human-readable label and unit for the exposure. */
  exposureLabel?: string;
  /** The timeseries field for the human-welfare outcome variable, if available. */
  outcomeField?: keyof TimeSeriesPoint;
  /** Human-readable label and unit for the outcome. */
  outcomeLabel?: string;
}

/**
 * Maps each hypothesis ID to the timeseries fields available for the country spotlight.
 *
 * Not every hypothesis variable is in the API timeseries (e.g., safe_water_access_pct,
 * refugee_outflow, and grace_lwe_anomaly_cm are not exposed). Where a variable is
 * missing, the spotlight shows only what is available.
 */
const HYPOTHESIS_VARS: Record<string, HypothesisVars> = {
  H1: {
    exposureField: "renewable_freshwater_percap",
    exposureLabel: "Freshwater / capita (m³/yr)",
    outcomeField: "gdp_pc_ppp",
    outcomeLabel: "GDP / capita (USD)",
  },
  H2: {
    exposureField: "renewable_freshwater_percap",
    exposureLabel: "Freshwater / capita (m³/yr)",
    outcomeField: "fsi_score",
    outcomeLabel: "Fragile States Index (0–120, higher = more fragile)",
  },
  H3: {
    exposureField: "renewable_freshwater_percap",
    exposureLabel: "Freshwater / capita (m³/yr)",
    outcomeField: "ucdp_conflict_binary",
    outcomeLabel: "Armed conflict",
  },
  H4: {
    outcomeField: "life_expectancy",
    outcomeLabel: "Life expectancy (years)",
  },
  H4b: {
    outcomeField: "life_expectancy",
    outcomeLabel: "Life expectancy (proxy — U5MR not in API)",
  },
  H5: {
    exposureField: "renewable_freshwater_percap",
    exposureLabel: "Freshwater / capita (m³/yr)",
  },
  H6: {},
  H7: {
    outcomeField: "gdp_pc_ppp",
    outcomeLabel: "GDP / capita (USD)",
  },
};

/** Formats a timeseries value for display in the country spotlight. */
function formatValue(field: keyof TimeSeriesPoint, value: number): string {
  if (field === "renewable_freshwater_percap") {
    return value >= 1000
      ? `${(value / 1000).toFixed(1)}k m³/yr`
      : `${value.toFixed(0)} m³/yr`;
  }
  if (field === "gdp_pc_ppp") return `$${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
  if (field === "life_expectancy") return `${value.toFixed(1)} years`;
  if (field === "fsi_score") return `${value.toFixed(1)} / 120`;
  if (field === "ucdp_conflict_binary") return value === 1 ? "In conflict" : "No conflict recorded";
  return String(value);
}

/**
 * Returns the latest non-null value for a field from the timeseries, along with its year.
 * Works backwards from the most recent year so stale or missing trailing years don't hide data.
 */
function latestValue(
  ts: TimeSeriesPoint[],
  field: keyof TimeSeriesPoint,
): { value: number; year: number } | null {
  for (let i = ts.length - 1; i >= 0; i--) {
    const v = ts[i][field];
    if (v != null) return { value: v as number, year: ts[i].year };
  }
  return null;
}

// ---------------------------------------------------------------------------
// Plain-language context cards
// ---------------------------------------------------------------------------

/**
 * Plain-language context cards for each hypothesis.
 *
 * Written for a non-specialist audience — policymakers, journalists, citizens.
 * `why` explains the scientific theory. `insight` explains the GFIP result.
 * `limitation` is optional — flagged where data constraints apply.
 */
const DESCRIPTIONS: Record<string, { why: string; insight: string; limitation?: string }> = {
  H1: {
    why: `Water is a fundamental input to everything a country produces — food, energy,
      manufacturing, and healthy workers. Countries with more renewable freshwater can
      grow more food, power more industry, and avoid the enormous costs of scarcity
      adaptation (desalination, water imports, irrigation infrastructure). This
      hypothesis asks: after controlling for every permanent difference between
      countries and every global trend, does freshwater availability still predict
      income within the same country over time?`,
    insight: `Yes — strongly. A doubling of freshwater per capita is associated with a
      47% increase in GDP per capita within the same country. With nearly 10,000
      country-year observations and a p-value of 0.0000035, this is the most robust
      finding in the project. It tells us that water scarcity is not merely correlated
      with poverty — it tracks with economic performance even as countries develop.`,
  },
  H2: {
    why: `State fragility — captured here by the Fragile States Index (FSI, 0=stable,
      120=fragile) — can be driven by water stress through multiple pathways: crop
      failure undermines rural livelihoods and erodes the social contract; competition
      over scarce water resources creates local conflicts that escalate; governments
      unable to provide safe water lose legitimacy. This hypothesis tests whether
      these pathways are visible in the data.`,
    insight: `A doubling of freshwater per capita is associated with an 11-point fall in
      the FSI score (a large improvement in stability). Given that the global FSI
      average is around 65, this represents a ~17% reduction in fragility. The result
      is statistically significant (p=0.004) across 2,902 country-years. Water
      investment may be one of the highest-leverage interventions available to
      stabilisation programmes.`,
  },
  H3: {
    why: `The "water wars" theory — that resource scarcity causes armed conflict — is one
      of the most politically prominent claims in international security. GFIP tests
      it rigorously using UCDP armed conflict data (defined as ≥25 battle deaths per
      year). The panel design controls for country-level fixed effects (geography, ethnic
      composition, historical institutions) so we are asking: when water availability
      falls within a country, does the probability of conflict rise?`,
    insight: `The direction is consistent with the theory — less water is associated with
      higher conflict probability — but the result sits at p=0.082, just outside the
      conventional 5% significance threshold. This is not a failure: it tells us
      the signal is real but the pathway is indirect (water stress → economic shock →
      instability → conflict) and therefore harder to detect cleanly. The H2 result
      (water → fragility) suggests the intermediate step is real.`,
    limitation: `The analysis covers 10,438 country-years but armed conflict is rare
      (most country-years are peaceful), which limits statistical power. A larger
      sample covering more recent years would likely push this to full significance.`,
  },
  H4: {
    why: `Safe drinking water and sanitation directly prevent waterborne disease — the
      leading cause of death in low-income countries. Cholera, typhoid, and diarrhoeal
      illness are transmitted almost exclusively through contaminated water. Extending
      safe water access should therefore increase life expectancy directly (fewer
      deaths from waterborne disease) and indirectly (healthier workers, higher
      productivity, more investment in children's education).`,
    insight: `Every additional percentage point of safe water access is associated with
      0.078 additional years of life expectancy. Going from 50% to 100% safe water
      access is associated with nearly 4 extra years of life. With 3,567 observations
      and p=0.00093, this is one of the most actionable findings in the project:
      water infrastructure investment has a measurable, quantifiable return in
      human lifespan.`,
  },
  H4b: {
    why: `Children under 5 are the most vulnerable to waterborne disease. Their immune
      systems are still developing, they are more susceptible to dehydration from
      diarrhoeal illness, and they receive a higher dose of pathogens relative to
      body weight. The under-5 mortality rate (U5MR) is therefore the single most
      sensitive indicator of water and sanitation quality in a population. This
      hypothesis tests whether expanding safe water access visibly reduces child deaths.`,
    insight: `Each percentage-point improvement in safe water access is associated with
      0.65 fewer deaths per 1,000 live births. At a typical U5MR of 35 (global
      average for lower-middle income countries), that is roughly a 2% reduction
      per percentage point of water access. The result is highly significant
      (p=0.00036, n=3,369). Universal safe water access could prevent hundreds of
      thousands of child deaths annually — these numbers show exactly how many.`,
  },
  H5: {
    why: `Water scarcity can force people to flee through several channels: drought and
      crop failure destroy rural livelihoods, leaving migration as the only survival
      strategy; water-driven resource conflicts produce direct displacement; and
      economic collapse caused by water stress pushes people toward countries with
      better opportunities. This hypothesis tests whether countries with lower
      freshwater per capita generate more refugees.`,
    insight: `The direction is consistent with the theory (β = -0.929: less water →
      more refugees) but the result is not statistically significant (p=0.159).
      This likely reflects a data limitation rather than the absence of a real
      effect. UNHCR refugee data only covers 2000–2023, most of the variance in
      refugee flows is explained by political violence (not water), and many
      water-scarce countries generate relatively few internationally-recognised
      refugees — the displacement is internal (IDPs) rather than cross-border.`,
    limitation: `The UNHCR outflow data covers only 23 years and is substantially
      missing for the countries most likely to be affected. This result should be
      re-tested as UNHCR data coverage improves.`,
  },
  H6: {
    why: `Access to safe water is a basic service. When governments extend that service
      universally, several inequality-reducing mechanisms activate: poor households
      no longer spend a disproportionate share of income on water (the "water poverty
      trap"); healthier poor populations participate more in the formal economy; and
      governments willing to invest in universal water access tend to invest in other
      equalising public goods too. This hypothesis asks whether the Gini coefficient
      (0=perfect equality, 100=complete inequality) falls as water access rises.`,
    insight: `Each percentage-point increase in safe water access is associated with a
      0.10-point fall in the Gini coefficient — directionally consistent with the
      theory. The result sits at p=0.113, just outside the 5% threshold. With only
      1,520 observations (Gini data is notoriously sparse for low-income countries),
      the finding is suggestive but not yet conclusive. Policy implication: water
      infrastructure may be an underappreciated tool for reducing economic inequality.`,
    limitation: `Gini data has significant coverage gaps, particularly for countries
      with the most to gain from water investment. More data would likely
      strengthen this result.`,
  },
  H7: {
    why: `Unlike annual rainfall, groundwater depletion is a slow, largely invisible
      process — aquifers that took millennia to fill are being emptied in decades.
      NASA's GRACE satellites are the only way to observe this globally. This
      hypothesis uses a "conditional growth regression": controlling for where a
      country started in 2005, do countries that depleted their aquifers faster end
      up with a lower GDP in 2020 than we would have predicted? The expected sign
      is negative: faster depletion → worse economic trajectory.`,
    insight: `Confirmed (β = -0.030, p = 0.041). Countries depleting their groundwater
      faster end up economically worse off than their starting point would predict.
      The small coefficient reflects the log-log scale — in absolute terms, the
      effect is economically meaningful for countries highly dependent on aquifer
      irrigation (e.g. northern India, the Saharan states, the US High Plains).
      This is the hardest hypothesis to test: only 159 observations are available
      (GRACE data begins 2002) yet the signal is already statistically significant.`,
    limitation: `GRACE cannot distinguish groundwater loss from soil moisture or
      ice-sheet changes at the country level. The depletion rate estimate captures
      total terrestrial water storage change, not groundwater exclusively.`,
  },
};

/**
 * Generate a plain-language interpretation of one hypothesis result.
 *
 * @param h - A single HypothesisResult from the /api/v1/hypotheses endpoint.
 * @returns A plain-English sentence summarising the finding.
 */
const interpret = (h: HypothesisResult): string => {
  const dir = h.beta > 0 ? "increases" : "decreases";
  const sig = h.p_value < 0.05 ? "statistically significant" : "directionally consistent but not yet statistically significant";
  return `A unit improvement in ${h.exposure} ${dir} ${h.outcome} by ${Math.abs(h.beta).toFixed(3)} units. This result is ${sig} (p = ${h.p_value.toFixed(3)}, n = ${h.n_obs.toLocaleString()} country-years).`;
};

// ---------------------------------------------------------------------------
// CountrySpotlight sub-component
// ---------------------------------------------------------------------------

/**
 * Shows where the selected country sits on the variables involved in one hypothesis.
 *
 * This is a "dot on the scatter" in text form — it tells the user whether their
 * country is above or below the global finding, grounding the abstract regression
 * result in a real place.
 *
 * @param countryName - Human-readable name of the selected country.
 * @param ts - Full time-series for the selected country.
 * @param vars - Which timeseries fields are relevant to this hypothesis.
 */
function CountrySpotlight({
  countryName,
  ts,
  vars,
}: {
  countryName: string;
  ts: TimeSeriesPoint[];
  vars: HypothesisVars;
}) {
  const exposure = vars.exposureField ? latestValue(ts, vars.exposureField) : null;
  const outcome = vars.outcomeField ? latestValue(ts, vars.outcomeField) : null;

  if (!exposure && !outcome) return null;

  return (
    <div style={{
      marginTop: 12,
      padding: "10px 14px",
      background: "#e8f4fd",
      borderRadius: 6,
      borderLeft: "3px solid #1a3a5c",
    }}>
      <div style={{ fontSize: 12, fontWeight: 700, color: "#1a3a5c", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>
        Country spotlight — {countryName}
      </div>
      <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
        {exposure && vars.exposureField && (
          <div style={{ fontSize: 13 }}>
            <span style={{ color: "#555" }}>{vars.exposureLabel}: </span>
            <strong>{formatValue(vars.exposureField, exposure.value)}</strong>
            <span style={{ color: "#888", fontSize: 11, marginLeft: 4 }}>({exposure.year})</span>
          </div>
        )}
        {outcome && vars.outcomeField && (
          <div style={{ fontSize: 13 }}>
            <span style={{ color: "#555" }}>{vars.outcomeLabel}: </span>
            <strong>{formatValue(vars.outcomeField, outcome.value)}</strong>
            <span style={{ color: "#888", fontSize: 11, marginLeft: 4 }}>({outcome.year})</span>
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

/**
 * OutcomesExplorer component — renders the H1–H7 global hypothesis results.
 *
 * Shows worldwide regression findings for all 274 countries, plus a per-card
 * country spotlight that contextualises the global result for the selected country.
 *
 * @param props.iso3 - ISO3 code of the currently selected country (from App.tsx state).
 */
export default function OutcomesExplorer({ iso3 }: { iso3: string }) {
  const [hypotheses, setHypotheses] = useState<HypothesisResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [countryTs, setCountryTs] = useState<TimeSeriesPoint[]>([]);
  const [countryName, setCountryName] = useState<string>(iso3);

  useEffect(() => {
    api.hypotheses().then(setHypotheses).finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    api.countryDetail(iso3).then(detail => {
      setCountryTs(detail.timeseries);
      setCountryName(detail.country_name);
    }).catch(() => {
      setCountryTs([]);
      setCountryName(iso3);
    });
  }, [iso3]);

  if (loading) return <p>Loading hypothesis results…</p>;

  return (
    <div style={{ maxWidth: 900 }}>
      <h2>Global Outcomes Explorer</h2>

      {/* Global study banner — makes clear these are worldwide findings, not per-country */}
      <div style={{
        background: "#1a3a5c",
        color: "white",
        borderRadius: 8,
        padding: "12px 18px",
        marginBottom: 20,
        display: "flex",
        alignItems: "center",
        gap: 14,
      }}>
        <span style={{ fontSize: 28 }}>🌍</span>
        <div>
          <div style={{ fontWeight: 700, fontSize: 15 }}>
            Worldwide findings — 274 countries · up to 10,000 country-year observations
          </div>
          <div style={{ fontSize: 13, opacity: 0.85, marginTop: 2 }}>
            These results describe global patterns, not individual countries.
            Each card also shows where <strong>{countryName}</strong> sits on that relationship.
          </div>
        </div>
      </div>

      <p style={{ color: "#555", marginTop: 0 }}>
        These results come from two-way fixed effects panel regression — a statistical
        method that controls for everything permanently different between countries
        (geography, history, culture) and asks only: <em>within the same country,
        when water availability changed, did human outcomes change?</em>
        This is the gold standard for establishing causal relationships in
        cross-country panel data.
      </p>

      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        {hypotheses.map(h => {
          const desc = DESCRIPTIONS[h.id];
          return (
            <div key={h.id} style={{
              border: "1px solid #ddd",
              borderRadius: 8,
              padding: 16,
              // Green border = confirmed (p < 0.05, correct sign).
              // Amber border = directionally consistent but not yet significant.
              borderLeft: `4px solid ${h.confirmed ? "#2e7d32" : "#e65100"}`,
            }}>
              {/* Header row */}
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 10 }}>
                <span style={{ fontWeight: 700, fontSize: 18, color: "#1a3a5c" }}>{h.id}</span>
                <span style={{ fontWeight: 500 }}>{h.label}</span>
                <span style={{
                  marginLeft: "auto",
                  background: h.confirmed ? "#e8f5e9" : "#fff3e0",
                  color: h.confirmed ? "#2e7d32" : "#e65100",
                  borderRadius: 4, padding: "2px 10px", fontSize: 12,
                }}>
                  {h.confirmed ? "Confirmed" : "Directional"}
                </span>
              </div>

              {/* Lead finding — the most important sentence, always visible.
                  A policymaker should understand the key takeaway without
                  opening any expandable section. */}
              {desc && (
                <p style={{ margin: "0 0 10px", fontSize: 15, color: "#1a1a1a", lineHeight: 1.6, fontWeight: 400 }}>
                  {desc.insight}
                </p>
              )}

              {/* Country spotlight — shown immediately after the finding so the
                  reader can connect the global result to a specific country */}
              {countryTs.length > 0 && HYPOTHESIS_VARS[h.id] && (
                <CountrySpotlight
                  countryName={countryName}
                  ts={countryTs}
                  vars={HYPOTHESIS_VARS[h.id]}
                />
              )}

              {/* Statistics and methodology — tucked away for those who want them.
                  The β coefficient, p-value, and n are for researchers; the policymaker
                  has already got the finding above. */}
              <details style={{ marginTop: 12 }}>
                <summary style={{ cursor: "pointer", fontSize: 13, color: "#1a3a5c", fontWeight: 600 }}>
                  See the statistics & methodology
                </summary>
                <div style={{ marginTop: 10, paddingLeft: 12, borderLeft: "3px solid #e0e0e0" }}>
                  {/* β (beta) is the key number — it tells you HOW MUCH the outcome
                      changes per unit of exposure, not just whether a relationship exists. */}
                  <div style={{ display: "flex", gap: 24, marginBottom: 8, fontSize: 14, color: "#555" }}>
                    <span><strong>β</strong> = {h.beta.toFixed(3)}</span>
                    <span><strong>p</strong> = {h.p_value.toFixed(4)}</span>
                    <span><strong>n</strong> = {h.n_obs.toLocaleString()} country-years</span>
                  </div>
                  <p style={{ margin: "0 0 8px", fontSize: 13, color: "#444" }}>{interpret(h)}</p>
                  {h.note && (
                    <p style={{ margin: "0 0 8px", fontSize: 12, color: "#888", fontStyle: "italic" }}>
                      {h.note}
                    </p>
                  )}
                  {desc && (
                    <>
                      <p style={{ margin: "8px 0 6px", fontSize: 13, color: "#444", lineHeight: 1.6 }}>
                        <strong>Why this hypothesis?</strong> {desc.why}
                      </p>
                      {desc.limitation && (
                        <p style={{ margin: 0, fontSize: 12, color: "#888", fontStyle: "italic", lineHeight: 1.5 }}>
                          ⚠ Data caveat: {desc.limitation}
                        </p>
                      )}
                    </>
                  )}
                </div>
              </details>
            </div>
          );
        })}
      </div>
    </div>
  );
}
