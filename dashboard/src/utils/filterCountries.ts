/**
 * Utility for filtering the global country risk list in the CountrySearch component.
 *
 * Kept as a pure function (no React dependencies) so it can be tested in isolation
 * via Vitest without mounting a component. The CountrySearch component calls this
 * on every keystroke; keeping it lightweight ensures a responsive autocomplete.
 *
 * @module utils/filterCountries
 */

import type { CountryRisk } from '../api/client'

/**
 * Filter the global risk list to entries whose country name or ISO3 code matches
 * the search query.
 *
 * Matching rules:
 *   - Case-insensitive substring match on the full country name (e.g. "ban" matches
 *     "Bangladesh", "Albania", "Lebanon").
 *   - Case-insensitive match on the ISO3 code (e.g. "ken" matches "KEN").
 *   - An empty or whitespace-only query returns the full list unfiltered, so
 *     the CountrySearch dropdown can show all countries when first focused.
 *
 * @param query - The user's current search string. May be empty.
 * @param countries - The full list of CountryRisk entries loaded from the API.
 * @returns A filtered subset of `countries`. If `query` is empty, the original
 *   array reference is returned (no copy), avoiding unnecessary re-renders.
 */
export function filterCountries(query: string, countries: CountryRisk[]): CountryRisk[] {
  if (!query) return countries
  const q = query.toLowerCase()
  return countries.filter(
    c =>
      c.country_name.toLowerCase().includes(q) ||
      c.iso3.toLowerCase().includes(q),
  )
}
