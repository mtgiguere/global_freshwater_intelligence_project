import type { CountryRisk } from '../api/client'

// Filters the global risk list to entries matching the query against country
// name or ISO3 code. Case-insensitive. Empty query = no filter.
export function filterCountries(query: string, countries: CountryRisk[]): CountryRisk[] {
  if (!query) return countries
  const q = query.toLowerCase()
  return countries.filter(
    c =>
      c.country_name.toLowerCase().includes(q) ||
      c.iso3.toLowerCase().includes(q),
  )
}
