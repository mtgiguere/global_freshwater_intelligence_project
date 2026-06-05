"""Download ACLED political violence event data via the ACLED API.

Requires ACLED_API_KEY and ACLED_EMAIL in .env
Register at https://acleddata.com/register/ (free academic access).

NOTE: API key activation may take 1-2 business days after registration.
"""

import os
from pathlib import Path

import pandas as pd
import pycountry
import requests
from dotenv import load_dotenv

load_dotenv()

_VALID_ISO3 = {c.alpha_3 for c in pycountry.countries}
_ACLED_API = "https://api.acleddata.com/acled/read/"
_YEAR_START = 1997
_YEAR_END = 2024


def _get_credentials() -> tuple[str, str]:
    key = os.environ.get("ACLED_API_KEY", "")
    email = os.environ.get("ACLED_EMAIL", "")
    if not key or not email:
        raise OSError(
            "ACLED_API_KEY and ACLED_EMAIL not set. "
            "Register at https://acleddata.com/register/ then add to .env"
        )
    return key, email


def download_acled(dest_dir: Path) -> Path:
    """Download all ACLED events and aggregate to country-year panel."""
    dest = dest_dir / "acled_events.csv"
    if dest.exists():
        print("  ACLED — already exists, skipping")
        return dest

    key, email = _get_credentials()
    dest_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    page = 1

    print(f"  ACLED — fetching {_YEAR_START}-{_YEAR_END}...")
    while True:
        r = requests.get(
            _ACLED_API,
            params={
                "key": key,
                "email": email,
                "year": f"{_YEAR_START}|{_YEAR_END}",
                "fields": "iso3|country|event_date|fatalities",
                "limit": 10000,
                "page": page,
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()

        if data.get("status") != 200:
            raise RuntimeError(f"ACLED API error: {data.get('messages', data)}")

        records = data.get("data", [])
        if not records:
            break

        all_records.extend(records)
        print(f"  page {page}: {len(records)} events (total so far: {len(all_records)})")
        page += 1

    df = pd.DataFrame(all_records)
    df.to_csv(dest, index=False)
    print(f"  ACLED — saved {len(df):,} events -> {dest}")
    return dest


if __name__ == "__main__":
    raw_dir = Path(__file__).parents[3] / "data" / "raw" / "acled"
    print("Downloading ACLED event data...")
    result = download_acled(raw_dir)
    print(f"\nDone. File at {result}")
