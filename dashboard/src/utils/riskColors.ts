// Maps a Compound Risk Score (0-100) to an RGBA colour array for Deck.gl layers.
// Thresholds match the CRS bins used throughout the dashboard and Phase 3 analysis.

import type { CountryRisk } from '../api/client'

export type RGBA = [number, number, number, number]

// Builds an O(1) lookup from iso3 → compound_risk_score for Deck.gl layer callbacks.
// Called once when risk data arrives; the Map is passed into the layer getFillColor.
export function buildRiskIndex(risks: CountryRisk[]): Map<string, number> {
  const index = new Map<string, number>()
  for (const r of risks) {
    index.set(r.iso3, r.compound_risk_score)
  }
  return index
}

export function riskColor(score: number | undefined): RGBA {
  if (score === undefined) return [150, 150, 150, 100]
  if (score >= 70) return [198, 40, 40, 200]   // Critical — deep red
  if (score >= 50) return [230, 81, 0, 200]    // High — orange
  if (score >= 30) return [249, 168, 37, 200]  // Elevated — amber
  return [46, 125, 50, 200]                    // Low — green
}
