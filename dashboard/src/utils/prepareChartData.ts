import type { TimeSeriesPoint } from '../api/client'

// Each row passed to Recharts — null signals a genuine gap in the data.
// Recharts renders null as a break in the line, making data gaps honest
// rather than silently interpolated.
export interface ChartRow {
  year: number
  renewable_freshwater_percap: number | null
  gdp_pc_ppp: number | null
  life_expectancy: number | null
  fsi_score: number | null
}

export function prepareChartData(timeseries: TimeSeriesPoint[]): ChartRow[] {
  return timeseries
    .filter(p => p.year >= 1990)
    .sort((a, b) => a.year - b.year)
    .map(p => ({
      year: p.year,
      renewable_freshwater_percap: p.renewable_freshwater_percap ?? null,
      gdp_pc_ppp: p.gdp_pc_ppp ?? null,
      life_expectancy: p.life_expectancy ?? null,
      fsi_score: p.fsi_score ?? null,
    }))
}
