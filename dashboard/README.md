# GFIP Dashboard

React 18 + TypeScript frontend for the Global Freshwater Intelligence Project.

**Live: https://mtgiguere.github.io/global_freshwater_intelligence_project/**

---

## Running locally

```bash
npm install
npm run dev     # -> http://localhost:5173
```

The dashboard needs the API running at `http://localhost:8000` for live data.
See the root [README.md](../README.md) for API setup instructions.

If the API is not running, the dashboard still works — it falls back to the
pre-generated static JSON files in `public/data/` automatically.

---

## Deployment modes

| Mode | How | Data source |
|------|-----|-------------|
| Local dev | `npm run dev` + API running | Live FastAPI at `VITE_API_URL` |
| GitHub Pages | Push to `main` — auto-deploy | Pre-generated JSON in `public/data/` |
| Live API | Set `VITE_API_URL` at build time | Live FastAPI server |

The switch between modes is controlled by the `VITE_API_URL` environment variable:
- **Set** — calls the live API
- **Not set** — reads from `public/data/*.json`

---

## Updating the static data

The pre-generated JSON files in `public/data/` are committed to the repo and
served directly by GitHub Pages. Regenerate them whenever the Master Panel or
trained models change:

```bash
# From the project root (not from dashboard/)
$env:PYTHONPATH = "."; python scripts/generate_static.py   # PowerShell
PYTHONPATH=. python scripts/generate_static.py             # bash/zsh
```

Then commit and push — GitHub Actions redeploys automatically.

---

## Scripts

| Script | What it does |
|--------|-------------|
| `npm run dev` | Start Vite dev server with hot reload |
| `npm run build` | Production build to `dist/` |
| `npm test` | Run Vitest + React Testing Library tests |
| `npm run lint` | ESLint |

---

## Structure

```
dashboard/
  public/
    data/                     # Pre-generated static JSON (committed to repo)
      global-risk.json        # CRS for all 179 countries
      hypotheses.json         # H1-H7 regression results
      country/{ISO3}.json     # Historical time-series per country
      predict/{ISO3}.json     # ML predictions per country
  src/
    api/
      client.ts               # All API calls — static vs live mode handled here
    components/
      CountrySearch.tsx        # Autocomplete country selector
    panels/
      GlobalWaterAtlas.tsx     # Deck.gl world map + country click interaction
      OutcomesExplorer.tsx     # H1-H7 global findings + per-country spotlight
      CountryDeepDive.tsx      # Historical time-series charts (Recharts)
      MLFutures.tsx            # ML risk forecasts per country
    utils/
      riskColors.ts            # CRS score to fill colour mapping
      numericToIso3.ts         # Numeric country ID to ISO3 lookup
    App.tsx                    # Root: navigation, shared selectedIso3 state
```

---

## Tests

Tests live alongside components in `src/**/__tests__/`.
Deck.gl (WebGL) is mocked — data-fetching lifecycle and UI state are tested with RTL.

```bash
npm test              # run all
npm test -- --watch   # watch mode
```
