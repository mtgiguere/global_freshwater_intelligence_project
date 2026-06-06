/**
 * GFIP API client.
 *
 * All data fetching goes through this module. Change the base URL here
 * to point at local dev, staging, or production without touching components.
 */

const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export interface CountryRisk {
  iso3: string;
  year: number;
  compound_risk_score: number;
  scarcity_component?: number;
  instability_component?: number;
  migration_component?: number;
}

export interface TimeSeriesPoint {
  year: number;
  renewable_freshwater_percap?: number;
  gdp_pc_ppp?: number;
  life_expectancy?: number;
  fsi_score?: number;
  ucdp_conflict_binary?: number;
  compound_risk_score?: number;
}

export interface CountryDetail {
  iso3: string;
  timeseries: TimeSeriesPoint[];
}

export interface HypothesisResult {
  id: string;
  label: string;
  exposure: string;
  outcome: string;
  beta: number;
  p_value: number;
  n_obs: number;
  confirmed: boolean;
  note?: string;
}

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`);
  if (!response.ok) {
    throw new Error(`API error ${response.status}: ${path}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  globalRisk: () => get<CountryRisk[]>("/api/v1/global/risk"),
  countryDetail: (iso3: string) =>
    get<CountryDetail>(`/api/v1/country/${iso3}`),
  hypotheses: () => get<HypothesisResult[]>("/api/v1/hypotheses"),
};
