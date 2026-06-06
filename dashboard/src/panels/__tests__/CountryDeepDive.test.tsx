/**
 * CountryDeepDive component tests.
 *
 * Why TDD is partially applied (not strict RED/GREEN for chart rendering):
 * Recharts renders to SVG, which jsdom supports. However, the chart internals
 * (axes, lines, tooltips) are complex SVG structures that are fragile to
 * snapshot-test and add no behavioral value. Instead we verify:
 *   - The loading state renders correctly
 *   - The error state renders correctly
 *   - After successful load, chart section headings are present
 *   - The old HTML table is no longer rendered
 * Visual correctness of the charts is verified in the browser.
 */

import { render, screen, waitFor } from '@testing-library/react'
import { vi, beforeEach } from 'vitest'
import CountryDeepDive from '../CountryDeepDive'
import { api } from '../../api/client'

vi.mock('../../api/client', () => ({
  api: { countryDetail: vi.fn() },
}))

// Recharts renders SVG, which jsdom supports — no need to mock it.

const MOCK_DETAIL = {
  iso3: 'AFG',
  country_name: 'Afghanistan',
  timeseries: [
    { year: 2010, renewable_freshwater_percap: 1500, gdp_pc_ppp: 500, life_expectancy: 60.1, fsi_score: 110.2 },
    { year: 2015, renewable_freshwater_percap: 1480, gdp_pc_ppp: 550, life_expectancy: 61.5, fsi_score: 108.7 },
    { year: 2020, renewable_freshwater_percap: 1460, gdp_pc_ppp: 510, life_expectancy: 62.0, fsi_score: 111.1 },
  ],
}

beforeEach(() => {
  vi.resetAllMocks()
})

describe('CountryDeepDive', () => {
  it('shows a loading indicator while country data is being fetched', () => {
    vi.mocked(api.countryDetail).mockReturnValue(new Promise(() => {}))
    render(<CountryDeepDive iso3="AFG" />)
    expect(screen.getByText(/loading/i)).toBeInTheDocument()
  })

  it('shows an error message when the country is not found', async () => {
    vi.mocked(api.countryDetail).mockRejectedValue(new Error('404'))
    render(<CountryDeepDive iso3="XYZ" />)
    await waitFor(() =>
      expect(screen.getByText(/no data found/i)).toBeInTheDocument()
    )
  })

  it('renders chart section headings after data loads', async () => {
    vi.mocked(api.countryDetail).mockResolvedValue(MOCK_DETAIL)
    render(<CountryDeepDive iso3="AFG" />)
    // Wait for loading to finish — the country name appears in the panel heading.
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: /afghanistan/i })).toBeInTheDocument()
    )
    // Each chart has its own <h3> section heading — distinct from table <th> cells.
    expect(screen.getByRole('heading', { name: /freshwater per capita/i })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: /gdp per capita/i })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: /life expectancy/i })).toBeInTheDocument()
  })

  it('does not render an HTML table after charts are live', async () => {
    vi.mocked(api.countryDetail).mockResolvedValue(MOCK_DETAIL)
    render(<CountryDeepDive iso3="AFG" />)
    // Wait for loading to finish first — the absence check only means something
    // once we know the data has arrived and been rendered.
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: /afghanistan/i })).toBeInTheDocument()
    )
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
  })
})
