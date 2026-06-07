/**
 * GFIP API client — single point of contact between the React dashboard and the
 * FastAPI backend.
 *
 * ALL API calls go through this module so that:
 *   1. The base URL can be changed in one place (via the VITE_API_URL environment
 *      variable) without touching any dashboard component.
 *   2. TypeScript types are enforced on every response — components receive typed
 *      data rather than raw `unknown` JSON blobs.
 *   3. HTTP error handling lives here, keeping components free of fetch boilerplate.
 *
 * In local development the API runs at http://localhost:8000 (uvicorn default).
 * In production it is deployed to Render; the VITE_API_URL env var is set in the
 * Vercel project settings to point at the Render service URL.
 *
 * @module api/client
 */

/**
 * DEPLOYMENT MODES
 *
 * VITE_API_URL set (local dev or live API deployment):
 *   All requests go to the FastAPI server at that URL.
 *   e.g. VITE_API_URL=http://localhost:8000 for local development.
 *
 * VITE_API_URL not set (GitHub Pages / static deployment):
 *   Requests are rewritten to pre-generated JSON files under /data/.
 *   Run `python scripts/generate_static.py` to produce these files,
 *   then commit them before deploying. No backend required.
 *
 * Switching between modes requires only the VITE_API_URL env var —
 * no component code changes.
 */
const BASE_URL = import.meta.env.VITE_API_URL ?? null;

/** True when building for static/GitHub Pages deployment (no live API). */
const STATIC = BASE_URL === null;

/**
 * The Compound Risk Score for a single country, as returned by GET /api/v1/global/risk.
 *
 * compound_risk_score is the headline number (0–100) shown on the globe:
 * it combines scarcity (30%), instability (35%), and migration (35%) components.
 * The component fields are optional because the synthetic CI fallback may omit them.
 */
export interface CountryRisk {
  /** ISO 3166-1 alpha-3 country code, e.g. "AFG". Primary join key throughout GFIP. */
  iso3: string;
  /** Human-readable country name resolved from the iso3 code, e.g. "Afghanistan". */
  country_name: string;
  /** Reference year for this risk snapshot — typically the most recent data year. */
  year: number;
  /** Compound Risk Score 0–100. Higher = greater water-related vulnerability. */
  compound_risk_score: number;
  /** Water scarcity component score (0–1), normalised before combining. */
  scarcity_component?: number;
  /** Political instability component score (0–1), normalised before combining. */
  instability_component?: number;
  /** Forced migration component score (0–1), normalised before combining. */
  migration_component?: number;
}

/**
 * A single year of Master Panel data for one country, as nested inside CountryDetail.
 *
 * All fields except `year` are optional because the Master Panel uses an outer join
 * across 10 data sources — not every source covers every country-year. Missing values
 * are represented as `undefined` here and `null` in the Recharts chart data (see
 * prepareChartData.ts) so that line charts show honest gaps rather than interpolated
 * lines.
 */
export interface TimeSeriesPoint {
  /** Calendar year, e.g. 2019. */
  year: number;
  /** FAO AQUASTAT: total renewable freshwater resources per person (m³/person/year). */
  renewable_freshwater_percap?: number;
  /** World Bank: GDP per capita in constant 2015 USD (PPP-adjusted). */
  gdp_pc_ppp?: number;
  /** WHO: life expectancy at birth in years. */
  life_expectancy?: number;
  /** Fund for Peace: Fragile States Index score (0–120, higher = more fragile). */
  fsi_score?: number;
  /** UCDP: binary indicator, 1 if armed conflict was recorded in this country-year. */
  ucdp_conflict_binary?: number;
  /** Compound Risk Score snapshot for this year (0–100). */
  compound_risk_score?: number;
}

/**
 * Full historical record for one country, as returned by GET /api/v1/country/{iso3}.
 *
 * The `timeseries` array covers all years for which at least one data source has an
 * observation — typically 1946 to the most recent year in the Master Panel (2025 or
 * the last AQUASTAT/World Bank update). Years with no data at all for a country are
 * omitted entirely.
 */
export interface CountryDetail {
  /** ISO 3166-1 alpha-3 country code. */
  iso3: string;
  /** Human-readable country name. */
  country_name: string;
  /** Year-by-year observations from the Master Panel, sorted ascending by year. */
  timeseries: TimeSeriesPoint[];
}

/**
 * The Phase 3 regression result for one hypothesis, as returned by GET /api/v1/hypotheses.
 *
 * GFIP tested seven hypotheses (H1–H7) using two-way fixed effects panel regression.
 * The key number is `beta` — the marginal effect of the exposure on the outcome,
 * holding country and year fixed. A confirmed result has p_value < 0.05.
 */
export interface HypothesisResult {
  /** Hypothesis identifier, e.g. "H1". */
  id: string;
  /** Short plain-language label, e.g. "Water scarcity reduces economic growth". */
  label: string;
  /** Name of the independent variable (the freshwater-side measure). */
  exposure: string;
  /** Name of the dependent variable (the human-welfare outcome). */
  outcome: string;
  /**
   * Regression coefficient (beta): how much the outcome changes per unit
   * increase in the exposure, holding all else constant. This is the key
   * number — it tells you HOW MUCH freshwater stress matters, not just
   * whether it matters at all.
   */
  beta: number;
  /** Two-tailed p-value. Values < 0.05 are considered statistically significant. */
  p_value: number;
  /** Number of country-year observations used in this regression. */
  n_obs: number;
  /** True if p_value < 0.05 AND the sign of beta matches the predicted direction. */
  confirmed: boolean;
  /** Optional plain-language note about caveats or methodological choices. */
  note?: string;
}

/**
 * ML model predictions for one country, as returned by GET /api/v1/predict/{iso3}.
 *
 * The three component scores come from the Phase 4 ML models (GBR, XGBoost, RF).
 * The compound_risk_score is the weighted combination (scarcity 30%, instability 35%,
 * migration 35%) scaled to 0–100.
 *
 * IMPORTANT: when `is_trained` is false, the API is returning synthetic CI fallback
 * values generated without real model files. The dashboard shows a warning banner in
 * this case so users understand they are looking at illustrative data, not real
 * forecasts. Run `uv run python src/models/train_all.py` to generate real predictions.
 */
export interface CountryPrediction {
  /** ISO 3166-1 alpha-3 country code. */
  iso3: string;
  /** Human-readable country name. */
  country_name: string;
  /** Reference year for this prediction. */
  year: number;
  /**
   * Water scarcity score (0–1). Output of the Gradient Boosting Regression model
   * that predicts log(freshwater per capita) 5 years ahead, normalised to [0, 1].
   */
  scarcity_score: number;
  /**
   * Political instability probability (0–1). Output of the XGBoost binary classifier
   * that predicts P(FSI jump >5 points OR new conflict onset within 3 years).
   */
  instability_probability: number;
  /**
   * Migration pressure score (0–1). Output of the Random Forest regression model
   * that predicts log(refugee outflow + 1), normalised to [0, 1].
   */
  migration_score: number;
  /** Compound Risk Score (0–100), combining all three components. */
  compound_risk_score: number;
  /** False when the API is returning synthetic CI fallback data rather than real model output. */
  is_trained: boolean;
}

/**
 * Generic typed HTTP GET helper. All methods on the `api` object delegate here.
 *
 * In static mode (VITE_API_URL not set), the caller passes the pre-generated
 * JSON file path directly. In live mode, it passes the API endpoint path which
 * is prefixed with BASE_URL.
 *
 * @param path - URL path to fetch. Static mode: data/... Live mode: /api/v1/...
 * @returns A Promise resolving to the JSON response body, typed as T.
 * @throws Error with a descriptive message if the HTTP response status is not 2xx.
 */
async function get<T>(path: string): Promise<T> {
  // In static mode, paths are relative to the app's base URL (import.meta.env.BASE_URL).
  // Vite sets BASE_URL to the VITE_BASE_PATH at build time — e.g.
  // '/global_freshwater_intelligence_project/' on GitHub Pages, '/' locally.
  // Using absolute '/data/...' would resolve from the domain root and 404 on Pages.
  const url = STATIC ? `${import.meta.env.BASE_URL}${path}` : `${BASE_URL}${path}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`API error ${response.status}: ${url}`);
  }
  return response.json() as Promise<T>;
}

/**
 * The GFIP API surface exposed to all dashboard components.
 *
 * Import this object in components rather than constructing fetch calls directly.
 * Every method returns a typed Promise; callers should handle rejections (e.g. via
 * `.catch()` in useEffect hooks) to degrade gracefully when the API is unavailable.
 */
export const api = {
  /**
   * Fetches the Compound Risk Score for all countries.
   * Live: GET /api/v1/global/risk
   * Static: /data/global-risk.json
   */
  globalRisk: () =>
    get<CountryRisk[]>(STATIC ? "data/global-risk.json" : "/api/v1/global/risk"),

  /**
   * Fetches the full Master Panel time-series for one country.
   * Live: GET /api/v1/country/{iso3}
   * Static: /data/country/{ISO3}.json
   *
   * @param iso3 - ISO 3166-1 alpha-3 country code, e.g. "KEN".
   */
  countryDetail: (iso3: string) =>
    get<CountryDetail>(
      STATIC
        ? `data/country/${iso3.toUpperCase()}.json`
        : `/api/v1/country/${iso3}`
    ),

  /**
   * Fetches the Phase 3 hypothesis testing results (H1–H7).
   * Live: GET /api/v1/hypotheses
   * Static: /data/hypotheses.json
   */
  hypotheses: () =>
    get<HypothesisResult[]>(STATIC ? "data/hypotheses.json" : "/api/v1/hypotheses"),

  /**
   * Fetches Phase 4 ML model predictions for one country.
   * Live: GET /api/v1/predict/{iso3}
   * Static: /data/predict/{ISO3}.json
   *
   * @param iso3 - ISO 3166-1 alpha-3 country code, e.g. "SDN".
   */
  predictCountry: (iso3: string) =>
    get<CountryPrediction>(
      STATIC
        ? `data/predict/${iso3.toUpperCase()}.json`
        : `/api/v1/predict/${iso3}`
    ),
};
