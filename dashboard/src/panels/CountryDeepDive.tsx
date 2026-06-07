/**
 * Panel 3 — Country Deep Dive
 *
 * Time-series panel showing historical trends for a selected country across four
 * dimensions: renewable freshwater per capita, GDP per capita (PPP), life
 * expectancy, and FSI fragility score. Each metric gets its own line chart so a
 * reader can focus on one relationship at a time without being overwhelmed by a
 * multi-axis overlay.
 *
 * Uses Recharts LineChart with `connectNulls={false}` to display honest data gaps
 * rather than interpolated lines. A gap in the chart where data does not exist is
 * important information for a policymaker — it communicates measurement uncertainty,
 * not just missing pixels.
 *
 * What this panel tells a policymaker or researcher:
 *   - How freshwater availability has changed over time for this specific country
 *     (not global averages — what happened HERE).
 *   - Whether economic performance tracks the water trend (hypothesis H1:
 *     water scarcity → reduced GDP growth).
 *   - Whether life expectancy improved alongside water access growth (H4:
 *     water scarcity → higher under-5 mortality / lower life expectancy).
 *   - Whether state fragility rose during water-scarce periods (H2:
 *     water scarcity → higher FSI score / state fragility).
 *
 * Data sources shown (as labelled in the panel):
 *   - FAO AQUASTAT (freshwater)
 *   - World Bank (economy)
 *   - WHO (health)
 *   - Fund for Peace / FSI (fragility)
 *
 * TDD note: Recharts renders SVG, which jsdom supports. Component tests
 * in __tests__/CountryDeepDive.test.tsx verify loading state, error state,
 * chart heading presence, and absence of the old HTML table.
 * The prepareChartData transformation is fully TDD'd in src/utils/__tests__/.
 */

import { useEffect, useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from 'recharts'
import { api } from '../api/client'
import type { CountryDetail } from '../api/client'
import { prepareChartData } from '../utils/prepareChartData'
import type { ChartRow } from '../utils/prepareChartData'

// Chart definitions — each entry produces one panel with its own axis scale.
// Keeping these as data avoids four nearly-identical JSX blocks.
const CHARTS: {
  key: keyof Omit<ChartRow, 'year'>
  heading: string
  unit: string
  color: string
}[] = [
  {
    key: 'renewable_freshwater_percap',
    heading: 'Freshwater per capita',
    unit: 'm³/person/yr',
    color: '#1565c0',
  },
  {
    key: 'gdp_pc_ppp',
    heading: 'GDP per capita',
    unit: '2015 USD',
    color: '#2e7d32',
  },
  {
    key: 'life_expectancy',
    heading: 'Life expectancy',
    unit: 'years',
    color: '#6a1b9a',
  },
  {
    key: 'fsi_score',
    heading: 'State fragility (FSI)',
    unit: '0-120, higher = more fragile',
    color: '#c62828',
  },
]

/**
 * CountryDeepDive component — historical trend charts for one country.
 *
 * @param props.iso3 - ISO 3166-1 alpha-3 country code for the country to display,
 *   e.g. "KEN" for Kenya. Changing this prop triggers a fresh API fetch and re-renders
 *   all four charts. In App.tsx the component is keyed on `country` so that React
 *   unmounts and remounts it (resetting loading state) on each new selection.
 */
export default function CountryDeepDive({ iso3 }: { iso3: string }) {
  const [detail, setDetail] = useState<CountryDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    // `active` flag prevents a stale async response from updating state after the
    // component has unmounted or the iso3 prop has changed. Without this guard,
    // rapidly switching countries could cause a response from a previous country to
    // overwrite the data for the current one — a classic React async state race.
    let active = true
    api.countryDetail(iso3)
      .then(data  => { if (active) { setDetail(data);  setLoading(false) } })
      .catch(()   => { if (active) { setError(`No data found for country code "${iso3}"`); setLoading(false) } })
    return () => { active = false }
  }, [iso3])

  if (loading) return <p>Loading data for {iso3}…</p>
  if (error)   return <p style={{ color: '#c62828' }}>{error}</p>
  if (!detail) return null

  const chartData = prepareChartData(detail.timeseries)

  return (
    <div style={{ maxWidth: 900 }}>
      <h2>
        Country Deep Dive — {detail.country_name}{' '}
        <span style={{ color: '#aaa', fontSize: 16, fontWeight: 400 }}>({detail.iso3})</span>
      </h2>
      <p style={{ color: '#555', marginBottom: 24 }}>
        Annual observations from the Master Panel (1990 onwards).
        Sources: FAO AQUASTAT (water), World Bank (economy), WHO (health), FSI (fragility).
        Missing years appear as gaps in the line.
      </p>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 32 }}>
        {CHARTS.map(({ key, heading, unit, color }) => (
          <div key={key}>
            <h3 style={{ margin: '0 0 4px', color: '#1a3a5c', fontSize: 15 }}>{heading}</h3>
            <p style={{ margin: '0 0 8px', fontSize: 12, color: '#888' }}>{unit}</p>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={chartData} margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                <XAxis dataKey="year" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} width={60} />
                <Tooltip
                  formatter={(value) =>
                    value == null ? 'No data' : Number(value).toLocaleString()
                  }
                />
                <Line
                  type="monotone"
                  dataKey={key}
                  stroke={color}
                  strokeWidth={2}
                  dot={false}
                  // connectNulls={false} is intentional — do NOT change this to true.
                  //
                  // When this is false, Recharts draws a visible break in the line
                  // wherever the data value is null. This is scientifically honest:
                  // for policymakers and researchers reading these charts, a gap is
                  // important information. It means "we don't have a measurement for
                  // this country in this year" — which could reflect a real data
                  // collection failure, a conflict that disrupted reporting, or a
                  // country that didn't exist yet (e.g. South Sudan before 2011).
                  //
                  // Connecting across gaps (connectNulls=true) would imply a smooth
                  // trend across years where we simply have no data — misleading to
                  // any reader who doesn't know to look for the missing point markers.
                  connectNulls={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ))}
      </div>

      <p style={{ color: '#999', fontSize: 12, marginTop: 16 }}>
        {chartData.length} years of data (1990–present). "—" indicates data not available for that year.
      </p>
    </div>
  )
}
