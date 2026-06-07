/**
 * Data preparation utilities for the CountryDeepDive Recharts line charts.
 *
 * Transforms raw TimeSeriesPoint arrays from the API into the flat ChartRow format
 * that Recharts expects, handling missing values in a way that produces honest
 * chart visualisations rather than misleading interpolated lines.
 *
 * @module utils/prepareChartData
 */

import type { TimeSeriesPoint } from '../api/client'

/**
 * A single row in the Recharts chart data array.
 *
 * All metric fields use `number | null` rather than `number | undefined`.
 * Recharts distinguishes between the two: `undefined` is treated as "field not
 * present" and may be interpolated; `null` is treated as "no data for this year"
 * and renders as a visible break in the line. Using `null` is intentional — a gap
 * in the chart is more honest than connecting the line across years where no
 * observation exists.
 */
export interface ChartRow {
  /** Calendar year, e.g. 2015. */
  year: number
  /**
   * FAO AQUASTAT: total renewable freshwater per person (m³/person/year).
   * Null indicates no AQUASTAT record for this country-year.
   */
  renewable_freshwater_percap: number | null
  /**
   * World Bank: GDP per capita in constant 2015 USD (PPP-adjusted).
   * Null indicates no World Bank record for this country-year.
   */
  gdp_pc_ppp: number | null
  /**
   * WHO: life expectancy at birth in years.
   * Null indicates no WHO record for this country-year.
   */
  life_expectancy: number | null
  /**
   * Fund for Peace: Fragile States Index score (0–120, higher = more fragile).
   * Null indicates no FSI record for this country-year.
   */
  fsi_score: number | null
}

/**
 * Transform a raw API timeseries into chart-ready rows for Recharts.
 *
 * Applies three transformations:
 *   1. Filters to 1990-onwards — data before 1990 is too sparse and methodologically
 *      inconsistent to display alongside modern records.
 *   2. Sorts ascending by year — Recharts requires chronological order to draw lines
 *      correctly.
 *   3. Converts `undefined` to `null` for all metric fields — see the ChartRow
 *      interface comment for why null is the correct signal for missing data.
 *
 * @param timeseries - Raw time-series array from the CountryDetail API response.
 * @returns Array of ChartRow objects ready to pass as `data` to a Recharts LineChart.
 */
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
