"""Download NASA GRACE/GRACE-FO mascon data via Earthdata.

Uses the JPL RL06.3 Mascon solution — a single netCDF4 file containing
monthly groundwater storage anomalies (cm LWE) on a 0.5° global grid.

Requires NASA_EARTHDATA_TOKEN in .env (generate at https://urs.earthdata.nasa.gov)
"""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

# JPL RL06.3 mascon — combined GRACE + GRACE-FO (2002-present)
_GRACE_GRANULE_API = (
    "https://cmr.earthdata.nasa.gov/search/granules.json"
    "?short_name=TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4"
    "&sort_key=-start_date&page_size=1"
)


def _get_token() -> str:
    token = os.environ.get("NASA_EARTHDATA_TOKEN", "")
    if not token:
        raise OSError(
            "NASA_EARTHDATA_TOKEN not set. "
            "Generate a token at https://urs.earthdata.nasa.gov/profile then add to .env"
        )
    return token


def download_grace(dest_dir: Path) -> Path:
    """Download the latest JPL GRACE mascon netCDF4 file."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    token = _get_token()
    headers = {"Authorization": f"Bearer {token}"}

    # Find the latest granule URL via CMR
    print("  GRACE — searching CMR for latest granule...")
    r = requests.get(_GRACE_GRANULE_API, headers=headers, timeout=30)
    r.raise_for_status()
    granules = r.json()["feed"]["entry"]
    if not granules:
        raise RuntimeError("No GRACE granules found via CMR search")

    # Get the download URL for the netCDF4 file
    links = granules[0].get("links", [])
    nc_links = [lk["href"] for lk in links if lk.get("href", "").endswith(".nc")]
    if not nc_links:
        raise RuntimeError(f"No .nc download link in granule: {granules[0].get('id')}")

    url = nc_links[0]
    filename = url.split("/")[-1]
    dest = dest_dir / filename

    if dest.exists():
        print(f"  GRACE — {filename} already exists, skipping")
        return dest

    print(f"  GRACE — downloading {filename} (large file, ~160MB)...")
    with requests.get(url, headers=headers, stream=True, timeout=300) as dl:
        dl.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in dl.iter_content(chunk_size=65536):
                f.write(chunk)

    print(f"  GRACE — saved ({dest.stat().st_size:,} bytes)")
    return dest


if __name__ == "__main__":
    raw_dir = Path(__file__).parents[3] / "data" / "raw" / "grace"
    print("Downloading GRACE mascon data...")
    result = download_grace(raw_dir)
    print(f"\nDone. File at {result}")
