/**
 * MLFutures component tests.
 *
 * Why TDD is partially applied (not strict RED/GREEN for score bar rendering):
 * The score bars are CSS-driven visual elements — jsdom supports them but
 * testing exact pixel widths adds no behavioral value and creates fragile tests.
 * Instead we verify:
 *   - The loading state renders while data is being fetched
 *   - The synthetic-data warning banner appears when is_trained=false
 *   - The Compound Risk Score value is displayed after a successful load
 *   - An error message is shown when the endpoint fails
 * Visual correctness of the score bars is verified in the browser.
 */

import { render, screen, waitFor } from '@testing-library/react'
import { vi, beforeEach } from 'vitest'
import MLFutures from '../MLFutures'
import { api } from '../../api/client'

vi.mock('../../api/client', () => ({
  api: { predictCountry: vi.fn() },
}))

const MOCK_PREDICTION = {
  iso3: 'AFG',
  country_name: 'Afghanistan',
  year: 2025,
  scarcity_score: 0.78,
  instability_probability: 0.91,
  migration_score: 0.85,
  compound_risk_score: 84.2,
  is_trained: false,
}

beforeEach(() => {
  vi.resetAllMocks()
})

describe('MLFutures', () => {
  it('shows a loading indicator while predictions are being fetched', () => {
    vi.mocked(api.predictCountry).mockReturnValue(new Promise(() => {}))
    render(<MLFutures iso3="AFG" />)
    expect(screen.getByText(/loading/i)).toBeInTheDocument()
  })

  it('shows the synthetic-data warning banner when is_trained is false', async () => {
    vi.mocked(api.predictCountry).mockResolvedValue(MOCK_PREDICTION)
    render(<MLFutures iso3="AFG" />)
    await waitFor(() =>
      expect(screen.getByRole('alert')).toBeInTheDocument()
    )
    expect(screen.getByRole('alert')).toHaveTextContent(/synthetic data/i)
  })

  it('displays the Compound Risk Score after a successful load', async () => {
    vi.mocked(api.predictCountry).mockResolvedValue(MOCK_PREDICTION)
    render(<MLFutures iso3="AFG" />)
    await waitFor(() =>
      expect(screen.getByLabelText(/compound risk score/i)).toBeInTheDocument()
    )
    expect(screen.getByLabelText(/compound risk score/i)).toHaveTextContent('84.2')
  })

  it('shows an error message when the prediction endpoint fails', async () => {
    vi.mocked(api.predictCountry).mockRejectedValue(new Error('404'))
    render(<MLFutures iso3="ZZZ" />)
    await waitFor(() =>
      expect(screen.getByText(/could not load predictions/i)).toBeInTheDocument()
    )
  })
})
