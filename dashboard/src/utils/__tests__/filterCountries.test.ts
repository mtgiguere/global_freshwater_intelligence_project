import { filterCountries } from '../filterCountries'
import type { CountryRisk } from '../../api/client'

// ------------------------------------------------------------
// filterCountries(query, countries) → CountryRisk[]
//
// Filters the global risk list to countries whose name or iso3
// code contains the query string (case-insensitive).
// Used to power the search input in the navigation bar.
//
// Design choices:
//   - Case-insensitive: a user typing "afg" finds "AFG"
//   - Matches on both country_name AND iso3: "fra" finds France,
//     "franc" also finds France — two entry points for the same record
//   - Empty query returns the full list (no filter applied)
//   - Results preserve the input order (caller decides sorting)
// ------------------------------------------------------------

const make = (iso3: string, country_name: string): CountryRisk => ({
  iso3,
  country_name,
  year: 2023,
  compound_risk_score: 50,
})

const COUNTRIES = [
  make('AFG', 'Afghanistan'),
  make('FRA', 'France'),
  make('IND', 'India'),
  make('NGA', 'Nigeria'),
]

describe('filterCountries', () => {
  it('returns the full list when the query is empty', () => {
    expect(filterCountries('', COUNTRIES)).toHaveLength(4)
  })

  it('matches on country name (case-insensitive)', () => {
    const result = filterCountries('france', COUNTRIES)
    expect(result).toHaveLength(1)
    expect(result[0].iso3).toBe('FRA')
  })

  it('matches on iso3 code (case-insensitive)', () => {
    const result = filterCountries('afg', COUNTRIES)
    expect(result).toHaveLength(1)
    expect(result[0].iso3).toBe('AFG')
  })

  it('matches a partial name prefix', () => {
    const result = filterCountries('nig', COUNTRIES)
    expect(result[0].iso3).toBe('NGA')
  })

  it('is case-insensitive for mixed-case input', () => {
    expect(filterCountries('INDIA', COUNTRIES)).toHaveLength(1)
    expect(filterCountries('india', COUNTRIES)).toHaveLength(1)
    expect(filterCountries('InDiA', COUNTRIES)).toHaveLength(1)
  })

  it('returns an empty array when nothing matches', () => {
    expect(filterCountries('zzz', COUNTRIES)).toHaveLength(0)
  })

  it('preserves input order of matching results', () => {
    // Both France and Afghanistan match 'a' (contains 'a')
    const result = filterCountries('a', COUNTRIES)
    const isos = result.map(c => c.iso3)
    // AFG appears before FRA in input — must stay that way
    expect(isos.indexOf('AFG')).toBeLessThan(isos.indexOf('FRA'))
  })
})
