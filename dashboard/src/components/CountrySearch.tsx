/**
 * CountrySearch — autocomplete input for the navigation bar.
 *
 * Accepts the full global risk list as props (already loaded by the parent),
 * filters it as the user types using filterCountries(), and calls onSelect(iso3)
 * when the user picks an entry. Clears itself after selection.
 *
 * Accessibility: input uses role="searchbox", dropdown uses role="listbox",
 * each suggestion uses role="option" — standard combobox pattern.
 */

import { useState } from 'react'
import type { CountryRisk } from '../api/client'
import { filterCountries } from '../utils/filterCountries'

const MAX_SUGGESTIONS = 8

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
