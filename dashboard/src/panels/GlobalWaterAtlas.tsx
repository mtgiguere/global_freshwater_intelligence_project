/**
 * Panel 1 — Global Water Atlas
 *
 * This is the landing panel of the GFIP dashboard — a 3D interactive globe
 * rendered using Deck.gl's GlobeView (WebGL). Every country is filled with a
 * colour derived from its Compound Risk Score (CRS, 0–100):
 *   - Deep red  (≥70): critical water-related vulnerability
 *   - Orange    (50–70): high stress
 *   - Amber     (30–50): elevated but manageable stress
 *   - Green     (<30): healthy water and stability buffers
 *   - Grey:     no data available for this country
 *
 * Clicking a country updates the `selectedIso3` state in App.tsx (via the
 * onCountrySelect callback), which simultaneously updates the CountrySearch input
 * in the navigation bar and, if the user navigates to the Country Deep Dive panel,
 * loads that country's full historical data.
 *
 * What this panel tells a reader at a glance:
 *   - Which parts of the world face the greatest water-related risk right now.
 *   - How that stress concentrates geographically — Sub-Saharan Africa, South
 *     Asia, and MENA are consistently the most affected regions.
 *   - Where the groundwater depletion crisis (H7) overlaps with fragility and
 *     poverty — visible as clusters of red/orange in arid regions.
 *
 * TDD note: the Deck.gl GlobeView renders via WebGL, which jsdom cannot
 * simulate. The component's data-fetching lifecycle (loading / loaded states)
 * is covered by component tests in __tests__/GlobalWaterAtlas.test.tsx.
 * The colour-mapping logic (riskColor, buildRiskIndex) is fully TDD'd in
 * src/utils/__tests__/. Visual correctness of the globe is verified in the
 * browser — documented in the PR per the project TDD contract.
 *
 * Geographic data: Natural Earth 110m countries via world-atlas (npm).
 * Country colours: driven by the /api/v1/global/risk endpoint.
 */

import { useEffect, useState, useMemo } from 'react'
import DeckGL from '@deck.gl/react'
import { GeoJsonLayer } from '@deck.gl/layers'
// In deck.gl v9, GlobeView became an experimental API and was renamed _GlobeView.
// It is still fully functional — the underscore prefix is deck.gl's convention for
// "not yet stable API, may change in a future minor version". We import it from the
// main deck.gl package which re-exports all experimental views.
import { _GlobeView as GlobeView } from 'deck.gl'
import { feature } from 'topojson-client'
import type { Topology, GeometryCollection } from 'topojson-specification'
import type { GeoJsonProperties } from 'geojson'
import worldAtlas from 'world-atlas/countries-110m.json'
import { api } from '../api/client'
import type { CountryRisk } from '../api/client'
import { riskColor, buildRiskIndex } from '../utils/riskColors'
import { numericToIso3 } from '../utils/numericToIso3'

// Convert the TopoJSON topology to a GeoJSON FeatureCollection once at module
// load — this is a pure transform of a static file, so it never needs to run
// again during the session.
const WORLD_GEOJSON = feature(
  worldAtlas as unknown as Topology,
  (worldAtlas as unknown as Topology<{ countries: GeometryCollection<GeoJsonProperties> }>)
    .objects.countries,
)

const INITIAL_VIEW_STATE = { longitude: 10, latitude: 20, zoom: 0.8 }

const LEGEND: [string, string][] = [
  ['Critical (≥70)', '#c62828'],
  ['High (50–70)',   '#e65100'],
  ['Elevated (30–50)', '#f9a825'],
  ['Low (<30)',      '#2e7d32'],
]

/**
 * GlobalWaterAtlas component — the 3D globe landing panel.
 *
 * @param props.onCountrySelect - Callback fired when the user clicks a country on
 *   the globe. Receives the ISO3 alpha-3 code of the clicked country (e.g. "KEN").
 *   In App.tsx this sets `selectedIso3` and switches to the Country Deep Dive panel.
 */
export default function GlobalWaterAtlas({
  onCountrySelect,
}: {
  onCountrySelect: (iso3: string) => void
}) {
  const [risks, setRisks] = useState<CountryRisk[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.globalRisk()
      .then(setRisks)
      .finally(() => setLoading(false))
  }, [])

  // Build the iso3→score lookup once when risk data arrives, not on every render.
  // Deck.gl calls getFillColor for every country feature on every render frame,
  // so this Map gives O(1) access rather than an O(n) array scan per country.
  const riskIndex = useMemo(() => buildRiskIndex(risks), [risks])

  const layer = new GeoJsonLayer({
    id: 'countries',
    data: WORLD_GEOJSON,
    filled: true,
    stroked: true,
    lineWidthMinPixels: 0.5,
    // GeoJsonLayer fill colour callback — called once per country feature per frame.
    //
    // The world-atlas shapefile identifies countries by ISO 3166-1 *numeric* code
    // (e.g. f.id === 4 for Afghanistan). The GFIP risk index is keyed on
    // ISO 3166-1 *alpha-3* codes (e.g. "AFG"). numericToIso3 bridges the two.
    //
    // If a country has no entry in numericToIso3 (e.g. a disputed territory not in
    // the standard) or no CRS data in the risk index, riskColor receives `undefined`
    // and returns a neutral grey — so no country is ever rendered without a fill.
    getFillColor: (f) => {
      const iso3 = numericToIso3[String(f.id)]
      return riskColor(iso3 ? riskIndex.get(iso3) : undefined)
    },
    getLineColor: [255, 255, 255, 40],
    pickable: true,
    // onClick handler — translates the clicked feature's numeric ID to an alpha-3
    // code and calls onCountrySelect, which updates App.tsx's selectedIso3 state.
    // The guard `if (!object)` handles clicks on the ocean (no feature intersected).
    // The guard `if (iso3)` handles features for territories not in our lookup table.
    onClick: ({ object }) => {
      if (!object) return
      const iso3 = numericToIso3[String(object.id)]
      if (iso3) onCountrySelect(iso3)
    },
    // Tell Deck.gl to recompute fill colours when the risk index changes.
    // Without this, Deck.gl caches the getFillColor results and the globe would
    // not update after the API response arrives.
    updateTriggers: { getFillColor: [riskIndex] },
  })

  if (loading) return <p>Loading global risk data…</p>

  return (
    <div>
      <h2>Global Water Risk Atlas</h2>
      <p style={{ color: '#555', maxWidth: 720 }}>
        Each country is scored 0–100 on the <strong>Compound Risk Score</strong> —
        a combination of water scarcity (30%), political instability risk (35%),
        and migration pressure (35%). Darker red = greater vulnerability.
        Click any country to explore its full historical data.
      </p>

      {/* Colour legend — mirrors the CRS bins from Phase 3 analysis */}
      <div style={{ display: 'flex', gap: 16, marginBottom: 12, flexWrap: 'wrap' }}>
        {LEGEND.map(([label, color]) => (
          <span
            key={label}
            style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13 }}
          >
            <span
              style={{
                width: 14, height: 14, background: color,
                borderRadius: 2, display: 'inline-block',
              }}
            />
            {label}
          </span>
        ))}
      </div>

      {/* Globe container — explicit height required by Deck.gl */}
      <div style={{ position: 'relative', height: 520, borderRadius: 8, overflow: 'hidden' }}>
        <DeckGL
          views={new GlobeView({ id: 'globe' })}
          initialViewState={INITIAL_VIEW_STATE}
          controller={true}
          layers={[layer]}
          style={{ position: 'absolute', inset: 0 }}
        />
      </div>

      <p style={{ color: '#999', fontSize: 12, marginTop: 8 }}>
        {risks.length} countries loaded · scroll to zoom · drag to rotate · click a country to deep-dive
      </p>
    </div>
  )
}
