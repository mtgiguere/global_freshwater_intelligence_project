import { prepareChartData } from '../prepareChartData'
import type { TimeSeriesPoint } from '../../api/client'

// ------------------------------------------------------------
// prepareChartData(timeseries) → ChartRow[]
//
// Transforms the raw API TimeSeriesPoint array into the shape
// Recharts expects:
//   - sorted ascending by year (charts read left-to-right)
//   - undefined values replaced with null (Recharts renders gaps
//     for null but skips undefined entirely, leaving the line
//     connected across missing data — both are valid choices;
//     null gives honest gaps)
//   - filtered to 1990 onwards (pre-1990 data is very sparse)
// ------------------------------------------------------------

const pt = (year: number, overrides: Partial<TimeSeriesPoint> = {}): TimeSeriesPoint => ({
  year,
  ...overrides,
})

describe('prepareChartData', () => {
  it('returns an empty array for empty input', () => {
    expect(prepareChartData([])).toEqual([])
  })

  it('sorts rows ascending by year so the chart reads left-to-right', () => {
    const data = [pt(2010), pt(2005), pt(2015)]
    const result = prepareChartData(data)
    expect(result.map(r => r.year)).toEqual([2005, 2010, 2015])
  })

  it('converts undefined metric values to null for honest gap rendering', () => {
    const data = [pt(2000, { renewable_freshwater_percap: undefined })]
    const result = prepareChartData(data)
    expect(result[0].renewable_freshwater_percap).toBeNull()
  })

  it('converts undefined for all four metrics', () => {
    const data = [pt(2000)]  // no overrides — all metrics are undefined
    const row = prepareChartData(data)[0]
    expect(row.gdp_pc_ppp).toBeNull()
    expect(row.life_expectancy).toBeNull()
    expect(row.fsi_score).toBeNull()
  })

  it('filters out rows before 1990', () => {
    const data = [pt(1985), pt(1989), pt(1990), pt(2000)]
    const years = prepareChartData(data).map(r => r.year)
    expect(years).not.toContain(1985)
    expect(years).not.toContain(1989)
    expect(years).toContain(1990)
  })

  it('preserves defined numeric values unchanged', () => {
    const data = [pt(2010, {
      renewable_freshwater_percap: 1500.5,
      gdp_pc_ppp: 850.0,
      life_expectancy: 63.2,
      fsi_score: 88.4,
    })]
    const row = prepareChartData(data)[0]
    expect(row.renewable_freshwater_percap).toBe(1500.5)
    expect(row.gdp_pc_ppp).toBe(850.0)
    expect(row.life_expectancy).toBe(63.2)
    expect(row.fsi_score).toBe(88.4)
  })
})
