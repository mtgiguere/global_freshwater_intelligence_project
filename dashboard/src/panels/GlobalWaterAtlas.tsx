/**
 * Panel 1 — Global Water Atlas
 *
 * The landing panel of the GFIP dashboard — an interactive flat world map rendered
 * via Deck.gl's MapView (WebGL). Every country is filled with a colour derived from
 * its Compound Risk Score (CRS, 0–100):
 *   - Deep red  (≥70): critical water-related vulnerability
 *   - Orange    (50–70): high stress
 *   - Amber     (30–50): elevated but manageable stress
 *   - Green     (<30): healthy water and stability buffers
 *   - Grey:     no data available for this country
 *
 * Clicking a country updates `selectedIso3` in App.tsx (via onCountrySelect),
 * simultaneously updating the CountrySearch input and priming the Country Deep
 * Dive and ML Futures panels for that country.
 *
 * NOTE: We use MapView (flat Mercator projection) instead of GlobeView.
 * Deck.gl's _GlobeView is an experimental API in v9.x that produces rendering
 * artefacts: polygon "spiralling" on wrap-around countries, and a visible equator
 * arc line. MapView is fully stable and shows all countries at once — better for
 * scanning a data dashboard quickly.
 *
 * TDD note: Deck.gl renders via WebGL, which jsdom cannot simulate. The component's
 * data-fetching lifecycle is covered in __tests__/GlobalWaterAtlas.test.tsx. The
 * colour-mapping logic is fully TDD'd in src/utils/__tests__/.
 *
 * Geographic data: Natural Earth 110m countries via world-atlas (npm).
 * Country colours: driven by the /api/v1/global/risk endpoint.
 */

import { useEffect, useState, useMemo } from 'react'
import DeckGL from '@deck.gl/react'
import { GeoJsonLayer } from '@deck.gl/layers'
import { MapView } from '@deck.gl/core'
import { feature } from 'topojson-client'
import type { Topology, GeometryCollection } from 'topojson-specification'
import type { GeoJsonProperties } from 'geojson'
import worldAtlas from 'world-atlas/countries-110m.json'
import { api } from '../api/client'
import type { CountryRisk } from '../api/client'
import { riskColor, buildRiskIndex } from '../utils/riskColors'
import { numericToIso3 } from '../utils/numericToIso3'

// Convert the TopoJSON topology to a GeoJSON FeatureCollection once at module
// load — this is a pure transform of a static file, so it never needs to run again.
const WORLD_GEOJSON = feature(
  worldAtlas as unknown as Topology,
  (worldAtlas as unknown as Topology<{ countries: GeometryCollection<GeoJsonProperties> }>)
    .objects.countries,
)

// Zoom level 1.2 shows the full world with a small margin on all sides.
// latitude: 15 centres on the populated landmasses rather than the mid-ocean equator.
const INITIAL_VIEW_STATE = { longitude: 10, latitude: 15, zoom: 1.2 }

const LEGEND: [string, string][] = [
  ['Critical (≥70)', '#c62828'],
  ['High (50–70)',   '#e65100'],
  ['Elevated (30–50)', '#f9a825'],
  ['Low (<30)',      '#2e7d32'],
]

/** Shape of the info panel shown after a country is clicked. */
interface SelectedCountry {
  iso3: string
  name: string
  score: number | undefined
}

/** CRS tier label for the info bar. */
function tierLabel(score: number | undefined): string {
  if (score === undefined) return 'No data'
  if (score >= 70) return 'Critical'
  if (score >= 50) return 'High'
  if (score >= 30) return 'Elevated'
  return 'Low'
}

/**
 * GlobalWaterAtlas component — interactive flat world map panel.
 *
 * @param props.onCountrySelect - Called when a country is clicked. Updates the
 *   shared `selectedIso3` state in App.tsx so other panels stay in sync.
 * @param props.onNavigate - Called when the user clicks a navigation button in
 *   the info bar. Receives the panel id ("country" or "futures") and switches
 *   the active tab in App.tsx.
 */
export default function GlobalWaterAtlas({
  onCountrySelect,
  onNavigate,
}: {
  onCountrySelect: (iso3: string) => void
  onNavigate: (panel: 'country' | 'futures') => void
}) {
  const [risks, setRisks] = useState<CountryRisk[]>([])
  const [loading, setLoading] = useState(true)
  const [selected, setSelected] = useState<SelectedCountry | null>(null)

  useEffect(() => {
    api.globalRisk()
      .then(setRisks)
      .finally(() => setLoading(false))
  }, [])

  // Build two lookup maps once when risk data arrives:
  // riskIndex: iso3 → CRS score  (used by getFillColor on every render frame)
  // nameIndex: iso3 → full country name  (used in the click info bar)
  const riskIndex = useMemo(() => buildRiskIndex(risks), [risks])
  const nameIndex = useMemo(() => {
    const m = new Map<string, string>()
    for (const r of risks) m.set(r.iso3, r.country_name)
    return m
  }, [risks])

  const layer = new GeoJsonLayer({
    id: 'countries',
    data: WORLD_GEOJSON,
    filled: true,
    stroked: true,
    lineWidthMinPixels: 0.5,
    // wrapLongitude splits polygons that cross the ±180° antimeridian before
    // rendering. Without this, Russia, Fiji, and Kiribati produce giant triangles
    // or horizontal streaks across the map width.
    wrapLongitude: true,
    // autoHighlight brightens whichever country the cursor is over, giving
    // immediate visual feedback that the map is interactive.
    autoHighlight: true,
    highlightColor: [255, 255, 255, 80],
    // getFillColor: called once per country feature per render frame.
    // The world-atlas shapefile uses ISO 3166-1 numeric IDs; numericToIso3
    // bridges to the alpha-3 codes used throughout the GFIP data stack.
    getFillColor: (f) => {
      const iso3 = numericToIso3[String(f.id)]
      return riskColor(iso3 ? riskIndex.get(iso3) : undefined)
    },
    getLineColor: [255, 255, 255, 40],
    pickable: true,
    // onClick: set the selected country info bar AND notify App.tsx so that
    // CountryDeepDive and MLFutures are primed for this country.
    onClick: ({ object }) => {
      if (!object) { setSelected(null); return }
      const iso3 = numericToIso3[String(object.id)]
      if (!iso3) return
      const score = riskIndex.get(iso3)
      const name = nameIndex.get(iso3) ?? iso3
      setSelected({ iso3, name, score })
      onCountrySelect(iso3)
    },
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
        <strong> Click any country</strong> to see its score and explore its data.
      </p>

      {/* Colour legend */}
      <div style={{ display: 'flex', gap: 16, marginBottom: 12, flexWrap: 'wrap' }}>
        {LEGEND.map(([label, color]) => (
          <span key={label} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13 }}>
            <span style={{ width: 14, height: 14, background: color, borderRadius: 2, display: 'inline-block' }} />
            {label}
          </span>
        ))}
      </div>

      {/* Map container — explicit height required by Deck.gl */}
      <div style={{ position: 'relative', height: 520, borderRadius: 8, overflow: 'hidden', background: '#d0e8f5' }}>
        <DeckGL
          views={new MapView({ id: 'map', repeat: true })}
          initialViewState={INITIAL_VIEW_STATE}
          controller={true}
          layers={[layer]}
          style={{ position: 'absolute', inset: '0' }}
        />
      </div>

      {/* Country info bar — appears after a country is clicked.
          Shows the country name, CRS score, and navigation buttons so the user
          can jump directly to the Country Deep Dive or ML Futures panel without
          having to find the country again using the search bar. */}
      {selected ? (
        <div style={{
          marginTop: 12,
          padding: '12px 16px',
          background: '#1a3a5c',
          color: 'white',
          borderRadius: 8,
          display: 'flex',
          alignItems: 'center',
          gap: 16,
          flexWrap: 'wrap',
        }}>
          <div>
            <span style={{ fontWeight: 700, fontSize: 16 }}>{selected.name}</span>
            <span style={{ marginLeft: 10, fontSize: 13, opacity: 0.8 }}>
              {selected.iso3}
            </span>
          </div>
          <div style={{
            background: 'rgba(255,255,255,0.15)',
            borderRadius: 6,
            padding: '4px 12px',
            fontSize: 14,
          }}>
            CRS: <strong>{selected.score !== undefined ? selected.score.toFixed(1) : '—'}</strong>
            {' '}· <span style={{ opacity: 0.85 }}>{tierLabel(selected.score)}</span>
          </div>
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
            <button
              onClick={() => onNavigate('country')}
              style={{ background: '#2196f3', color: 'white', border: 'none', borderRadius: 4, padding: '6px 14px', cursor: 'pointer', fontSize: 13 }}
            >
              Country Deep Dive
            </button>
            <button
              onClick={() => onNavigate('futures')}
              style={{ background: '#43a047', color: 'white', border: 'none', borderRadius: 4, padding: '6px 14px', cursor: 'pointer', fontSize: 13 }}
            >
              ML Futures
            </button>
            <button
              onClick={() => setSelected(null)}
              style={{ background: 'transparent', color: 'rgba(255,255,255,0.6)', border: '1px solid rgba(255,255,255,0.3)', borderRadius: 4, padding: '6px 10px', cursor: 'pointer', fontSize: 13 }}
              aria-label="Dismiss country info"
            >
              ✕
            </button>
          </div>
        </div>
      ) : (
        <p style={{ color: '#999', fontSize: 12, marginTop: 8 }}>
          {risks.length} countries loaded · scroll to zoom · drag to pan · click a country to explore
        </p>
      )}
    </div>
  )
}
