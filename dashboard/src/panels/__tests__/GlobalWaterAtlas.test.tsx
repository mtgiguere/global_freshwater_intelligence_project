/**
 * GlobalWaterAtlas component tests.
 *
 * Why TDD is partially applied here (not strict RED/GREEN for rendering):
 * Deck.gl requires WebGL, which jsdom does not support. The DeckGL canvas
 * component is therefore mocked. This lets us TDD the component's
 * data-fetching lifecycle (loading/loaded states) and verify the table
 * is removed. Visual correctness of the globe is verified in the browser.
 */

import { render, screen, waitFor } from '@testing-library/react'
import { vi, beforeEach } from 'vitest'
import GlobalWaterAtlas from '../GlobalWaterAtlas'
import { api } from '../../api/client'

// vi.mock calls are hoisted to the top of the file by Vitest — they run
// before any import, so mocks are in place when the component module loads.

vi.mock('@deck.gl/react', () => ({
  default: () => <canvas data-testid="deckgl-canvas" />,
}))
vi.mock('@deck.gl/layers', () => ({ GeoJsonLayer: vi.fn() }))
vi.mock('@deck.gl/core', () => ({ MapView: vi.fn() }))
vi.mock('world-atlas/countries-110m.json', () => ({
  default: { type: 'Topology', objects: { countries: { type: 'GeometryCollection', geometries: [] } }, arcs: [] },
}))
vi.mock('topojson-client', () => ({
  feature: () => ({ type: 'FeatureCollection', features: [] }),
}))
vi.mock('../../api/client', () => ({
  api: { globalRisk: vi.fn() },
}))

const MOCK_RISKS = [
  { iso3: 'AFG', country_name: 'Afghanistan', year: 2023, compound_risk_score: 72.4 },
  { iso3: 'FRA', country_name: 'France',      year: 2023, compound_risk_score: 18.1 },
]

beforeEach(() => {
  vi.resetAllMocks()
})

describe('GlobalWaterAtlas', () => {
  it('shows a loading indicator while risk data is being fetched', () => {
    // Promise that never resolves — keeps the component in the loading state.
    vi.mocked(api.globalRisk).mockReturnValue(new Promise(() => {}))
    render(<GlobalWaterAtlas onCountrySelect={vi.fn()} />)
    expect(screen.getByText(/loading/i)).toBeInTheDocument()
  })

  it('renders the Deck.gl canvas after data has loaded', async () => {
    vi.mocked(api.globalRisk).mockResolvedValue(MOCK_RISKS)
    render(<GlobalWaterAtlas onCountrySelect={vi.fn()} />)
    await waitFor(() =>
      expect(screen.getByTestId('deckgl-canvas')).toBeInTheDocument()
    )
  })

  it('does not render a data table after the globe is live', async () => {
    // The HTML table was the Phase 5 iteration 1 placeholder — it must be removed.
    vi.mocked(api.globalRisk).mockResolvedValue(MOCK_RISKS)
    render(<GlobalWaterAtlas onCountrySelect={vi.fn()} />)
    await waitFor(() =>
      expect(screen.queryByRole('table')).not.toBeInTheDocument()
    )
  })
})
