/**
 * Colour mapping utilities for the GFIP Compound Risk Score (CRS).
 *
 * Maps a CRS value (0–100) to an RGBA colour array suitable for use in Deck.gl
 * layer callbacks (getFillColor). The four tiers and their thresholds mirror the
 * bins used throughout the dashboard and the Phase 3 analysis reports, so that a
 * "Critical" country looks the same on the globe, in bar charts, and in printed
 * policy briefs.
 *
 * Colour semantics (matching the LEGEND in GlobalWaterAtlas.tsx):
 *   Critical  ≥ 70  Deep red   — The country is in an acute water-related crisis.
 *                               Water scarcity, fragility, and displacement pressures
 *                               are all elevated simultaneously. Requires urgent
 *                               international attention and/or emergency response.
 *   High      50–70 Orange     — Severe stress across at least one or two dimensions.
 *                               Policy intervention is needed but the situation is
 *                               not yet a full-blown emergency.
 *   Elevated  30–50 Amber      — Significant but manageable stress. Water security
 *                               is under pressure and governance capacity is
 *                               stretched, but institutions are still functioning.
 *                               Preventive investment is most cost-effective here.
 *   Low        < 30 Green      — The country has healthy water and stability buffers.
 *                               No immediate freshwater-related risk signal. Long-term
 *                               monitoring remains warranted given climate trajectory.
 *
 * @module utils/riskColors
 */

import type { CountryRisk } from '../api/client'

/** RGBA colour tuple as expected by Deck.gl getFillColor callbacks. */
export type RGBA = [number, number, number, number]

/**
 * Build an O(1) lookup map from ISO3 code to Compound Risk Score.
 *
 * Deck.gl calls getFillColor on every country feature during every render cycle.
 * Building the Map once (via useMemo in the component) and passing it into the
 * layer avoids an O(n) array scan per country per frame, which matters at 200+
 * countries and 60 fps.
 *
 * @param risks - Array of CountryRisk objects from GET /api/v1/global/risk.
 * @returns A Map keyed on iso3 (e.g. "KEN") with the CRS value (0–100) as the value.
 */
export function buildRiskIndex(risks: CountryRisk[]): Map<string, number> {
  const index = new Map<string, number>()
  for (const r of risks) {
    index.set(r.iso3, r.compound_risk_score)
  }
  return index
}

/**
 * Map a Compound Risk Score to an RGBA fill colour for a Deck.gl GeoJsonLayer.
 *
 * The four thresholds (70 / 50 / 30) are the same bins used in Phase 3 analysis
 * outputs and the dashboard legend. The alpha channel (200 out of 255) leaves the
 * globe's base tile slightly visible through the fill, preserving geographic context.
 * Countries with no CRS data receive a neutral grey with lower opacity so they are
 * clearly distinguished from low-risk green countries.
 *
 * @param score - CRS value 0–100, or undefined if no data exists for the country.
 * @returns RGBA tuple [r, g, b, a] with values 0–255.
 */
export function riskColor(score: number | undefined): RGBA {
  if (score === undefined) return [150, 150, 150, 100]  // No data — neutral grey
  if (score >= 70) return [198, 40, 40, 200]            // Critical — deep red
  if (score >= 50) return [230, 81, 0, 200]             // High — orange
  if (score >= 30) return [249, 168, 37, 200]           // Elevated — amber
  return [46, 125, 50, 200]                             // Low — green
}
