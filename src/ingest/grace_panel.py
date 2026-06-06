"""Load the pre-processed GRACE country-year groundwater anomaly panel.

The source CSV is produced by src/ingest/download/grace_process.py, which
aggregates the NASA GRACE mascon netCDF4 to country-level annual means.
"""

import pandas as pd

_REQUIRED_COLUMNS = ["iso3", "year", "grace_lwe_anomaly_cm"]


def load_grace_panel(path) -> pd.DataFrame:
    """Load GRACE country-year groundwater anomaly panel."""
    df = pd.read_csv(path)

    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["year"] = df["year"].astype("int64")
    return df
