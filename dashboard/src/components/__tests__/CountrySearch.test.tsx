/**
 * CountrySearch component tests.
 *
 * The search input sits in the header nav and fires onSelect(iso3) when the
 * user picks a country from the dropdown suggestions.
 */

import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi, beforeEach } from 'vitest'
import CountrySearch from '../CountrySearch'
import type { CountryRisk } from '../../api/client'

const make = (iso3: string, country_name: string): CountryRisk => ({
  iso3, country_name, year: 2023, compound_risk_score: 50,
})

const COUNTRIES = [
  make('AFG', 'Afghanistan'),
  make('FRA', 'France'),
  make('IND', 'India'),
]

const onSelect = vi.fn()

beforeEach(() => vi.resetAllMocks())

describe('CountrySearch', () => {
  it('renders a search input', () => {
    render(<CountrySearch countries={COUNTRIES} onSelect={onSelect} />)
    expect(screen.getByRole('searchbox')).toBeInTheDocument()
  })

  it('shows no suggestions when the input is empty', () => {
    render(<CountrySearch countries={COUNTRIES} onSelect={onSelect} />)
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument()
  })

  it('shows matching suggestions as the user types', async () => {
    render(<CountrySearch countries={COUNTRIES} onSelect={onSelect} />)
    await userEvent.type(screen.getByRole('searchbox'), 'fra')
    await waitFor(() =>
      expect(screen.getByRole('option', { name: /france/i })).toBeInTheDocument()
    )
  })

  it('does not show non-matching countries in suggestions', async () => {
    render(<CountrySearch countries={COUNTRIES} onSelect={onSelect} />)
    await userEvent.type(screen.getByRole('searchbox'), 'fra')
    await waitFor(() => screen.getByRole('option', { name: /france/i }))
    expect(screen.queryByRole('option', { name: /afghanistan/i })).not.toBeInTheDocument()
  })

  it('calls onSelect with the iso3 code when a suggestion is clicked', async () => {
    render(<CountrySearch countries={COUNTRIES} onSelect={onSelect} />)
    await userEvent.type(screen.getByRole('searchbox'), 'ind')
    await waitFor(() => screen.getByRole('option', { name: /india/i }))
    await userEvent.click(screen.getByRole('option', { name: /india/i }))
    expect(onSelect).toHaveBeenCalledWith('IND')
  })

  it('clears the input and hides suggestions after a selection', async () => {
    render(<CountrySearch countries={COUNTRIES} onSelect={onSelect} />)
    await userEvent.type(screen.getByRole('searchbox'), 'ind')
    await waitFor(() => screen.getByRole('option', { name: /india/i }))
    await userEvent.click(screen.getByRole('option', { name: /india/i }))
    expect(screen.getByRole('searchbox')).toHaveValue('')
    expect(screen.queryByRole('listbox')).not.toBeInTheDocument()
  })
})
