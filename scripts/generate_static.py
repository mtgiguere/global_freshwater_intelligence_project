"""Generate static JSON files for GitHub Pages deployment.

This script calls every FastAPI endpoint internally (no HTTP server needed)
and writes the responses as static JSON files to dashboard/public/data/.

Run this locally whenever the Master Panel or trained models are updated:

    python scripts/generate_static.py

Then commit the generated files — the GitHub Actions deploy workflow just
builds the React app and serves them from GitHub Pages. No backend required.

When VITE_API_URL is NOT set in the dashboard build, the React app fetches
from /data/... (these files). When VITE_API_URL IS set, it calls the live
FastAPI server. Flipping between the two modes requires only that env var.

Prerequisites:
    - Run from the project root (not from scripts/)
    - data/processed/master_panel.parquet must exist
    - data/models/*.joblib should exist for real predictions (synthetic
      fallback is used automatically if not present)

PYTHONPATH must include the project root so src.api can be imported:
    $env:PYTHONPATH = "."; python scripts/generate_static.py   # PowerShell
    PYTHONPATH=. python scripts/generate_static.py             # bash/zsh
"""

import json
import sys
import time
from pathlib import Path

# Ensure src.api is importable regardless of how the script is invoked.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from src.api.main import app  # noqa: E402

OUTPUT = ROOT / "dashboard" / "public" / "data"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_json(path: Path, data: object) -> None:
    """Write data as pretty-printed JSON, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def ok(path: Path) -> None:
    print(f"  ok  {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    client = TestClient(app, raise_server_exceptions=True)
    start = time.time()

    print(f"\nGFIP static generator — writing to {OUTPUT.relative_to(ROOT)}/\n")

    # 1. Global risk — one file, drives the globe colours
    r = client.get("/api/v1/global/risk")
    r.raise_for_status()
    risks = r.json()
    write_json(OUTPUT / "global-risk.json", risks)
    ok(OUTPUT / "global-risk.json")

    is_trained = any(c.get("compound_risk_score", 0) != 50.0 for c in risks)
    print(f"     {len(risks)} countries  |  is_trained={is_trained}\n")

    # 2. Hypotheses — one file, drives the Outcomes Explorer
    r = client.get("/api/v1/hypotheses")
    r.raise_for_status()
    write_json(OUTPUT / "hypotheses.json", r.json())
    ok(OUTPUT / "hypotheses.json")
    print()

    # 3. Per-country files — country detail + ML predictions
    iso3_list = sorted(c["iso3"] for c in risks)
    print(f"Generating {len(iso3_list)} country files (detail + predict)…\n")

    detail_ok = predict_ok = 0
    for iso3 in iso3_list:
        # Country historical time-series
        r = client.get(f"/api/v1/country/{iso3}")
        if r.status_code == 200:
            write_json(OUTPUT / "country" / f"{iso3}.json", r.json())
            detail_ok += 1
        else:
            print(f"  !  country/{iso3} → {r.status_code}")

        # ML predictions
        r = client.get(f"/api/v1/predict/{iso3}")
        if r.status_code == 200:
            write_json(OUTPUT / "predict" / f"{iso3}.json", r.json())
            predict_ok += 1
        else:
            print(f"  !  predict/{iso3} → {r.status_code}")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  {detail_ok}/{len(iso3_list)} country detail files")
    print(f"  {predict_ok}/{len(iso3_list)} prediction files")
    print("\nNext steps:")
    print("  git add dashboard/public/data/")
    print("  git commit -m 'data: regenerate static JSON'")
    print("  git push")


if __name__ == "__main__":
    main()
