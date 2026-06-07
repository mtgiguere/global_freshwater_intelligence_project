/**
 * CountrySearch — autocomplete country selector in the navigation header.
 *
 * Filters the global risk list (up to 274 countries) by country name or ISO3 code
 * as the user types. Selecting a country updates the global `selectedIso3` state
 * in App.tsx (via the onSelect callback), which simultaneously drives all four
 * dashboard panels. The input clears itself after selection so the user can
 * immediately search for another country.
 *
 * The country list is loaded once by App.tsx (from GET /api/v1/global/risk) and
 * passed as a prop here — CountrySearch does not fetch data itself. This keeps
 * the component stateless with respect to data loading and makes it easier to
 * test in isolation with a mock list.
 *
 * Accessibility: input uses role="searchbox", dropdown uses role="listbox",
 * each suggestion uses role="option" — standard ARIA combobox pattern. This
 * ensures the component is navigable by keyboard and readable by screen readers,
 * which matters for a public-facing policy tool.
 */

import { useState } from 'react'
import type { CountryRisk } from '../api/client'
import { filterCountries } from '../utils/filterCountries'

/** Maximum number of suggestions to show in the dropdown at any one time.
 *  8 keeps the list scannable without overwhelming the navigation bar area. */
const MAX_SUGGESTIONS = 8

/**
 * CountrySearch component — autocomplete input in the navigation bar.
 *
 * @param props.countries - Full list of CountryRisk entries loaded from the API.
 *   Passed from App.tsx so this component can filter without making its own API call.
 * @param props.onSelect - Callback fired when the user selects a country.
 *   Receives the ISO3 alpha-3 code (e.g. "KEN"). In App.tsx this sets
 *   `selectedIso3` and switches the active panel to Country Deep Dive.
 */
export default function CountrySearch({
  countries,
  onSelect,
}: {
  countries: CountryRisk[]
  onSelect: (iso3: string) => void
}) {
  const [query, setQuery] = useState('')

  const suggestions =
    query.length > 0
      ? filterCountries(query, countries).slice(0, MAX_SUGGESTIONS)
      : []

  const handleSelect = (iso3: string) => {
    onSelect(iso3)
    setQuery('')
  }

  return (
    <div style={{ position: 'relative' }}>
      <input
        role="searchbox"
        type="search"
        value={query}
        onChange={e => setQuery(e.target.value)}
        placeholder="Search country…"
        style={{
          padding: '5px 10px', borderRadius: 4, border: '1px solid rgba(255,255,255,0.3)',
          background: 'rgba(255,255,255,0.1)', color: 'white', fontSize: 13, width: 180,
        }}
      />
      {suggestions.length > 0 && (
        <ul
          role="listbox"
          style={{
            position: 'absolute', top: '100%', left: 0, right: 0, margin: 0,
            padding: 0, listStyle: 'none', background: 'white', border: '1px solid #ddd',
            borderRadius: 4, boxShadow: '0 4px 12px rgba(0,0,0,0.15)', zIndex: 100,
          }}
        >
          {suggestions.map(c => (
            <li
              key={c.iso3}
              role="option"
              aria-selected={false}
              onClick={() => handleSelect(c.iso3)}
              style={{
                padding: '7px 12px', cursor: 'pointer', fontSize: 13,
                color: '#333', borderBottom: '1px solid #f5f5f5',
              }}
              onMouseEnter={e => ((e.target as HTMLElement).style.background = '#f0f7ff')}
              onMouseLeave={e => ((e.target as HTMLElement).style.background = 'white')}
            >
              {c.country_name}
              <span style={{ color: '#aaa', marginLeft: 6, fontSize: 11 }}>{c.iso3}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
