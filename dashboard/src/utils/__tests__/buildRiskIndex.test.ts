import { buildRiskIndex } from '../riskColors'
import type { CountryRisk } from '../../api/client'

// ------------------------------------------------------------
// buildRiskIndex(risks) → Map<iso3, compound_risk_score>
//
// Pre-processes the /api/v1/global/risk response into a Map so
// the Deck.gl GeoJsonLayer can do O(1) colour lookups per country
// feature without scanning the full array on every render frame.
// ------------------------------------------------------------

const make = (iso3: string, score: number): CountryRisk => ({
  iso3,
  country_name: iso3,
  year: 2023,
  compound_risk_score: score,
})

describe('buildRiskIndex', () => {
  it('returns an empty Map for an empty input array', () => {
    const index = buildRiskIndex([])
    expect(index.size).toBe(0)
  })

  it('maps iso3 to compound_risk_score for a single entry', () => {
    const index = buildRiskIndex([make('AFG', 72.4)])
    expect(index.get('AFG')).toBe(72.4)
  })

  it('maps all entries when multiple countries are present', () => {
    const index = buildRiskIndex([make('AFG', 72.4), make('FRA', 18.1), make('IND', 55.3)])
    expect(index.size).toBe(3)
    expect(index.get('FRA')).toBe(18.1)
    expect(index.get('IND')).toBe(55.3)
  })

  it('does not contain keys for countries not in the input', () => {
    const index = buildRiskIndex([make('AFG', 72.4)])
    expect(index.has('USA')).toBe(false)
  })
})
