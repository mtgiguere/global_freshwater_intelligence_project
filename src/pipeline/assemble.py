"""Assemble the GFIP Master Panel from all Phase 1 ingest outputs.

Loads each cleaned source, joins on [iso3, year], validates against
Phase 1 exit criteria, and writes the Master Panel to data/processed/.

Run from repo root:
    uv run python src/pipeline/assemble.py
"""

from pathlib import Path

import pandas as pd

from src.ingest.aquastat import load_aquastat
from src.ingest.fsi import load_fsi
from src.ingest.grace_panel import load_grace_panel
from src.ingest.ucdp import load_ucdp
from src.ingest.undesa import load_undesa
from src.ingest.unhcr import load_unhcr
from src.ingest.unodc import load_unodc
from src.ingest.who import load_who
from src.ingest.worldbank import load_worldbank
from src.pipeline.master_panel import build_master_panel
from src.pipeline.validate import validate_master_panel

RAW = Path(__file__).parents[2] / "data" / "raw"
PROCESSED = Path(__file__).parents[2] / "data" / "processed"

_WB_INDICATORS = [
    "NY.GDP.PCAP.KD",
    "SI.POV.GINI",
    "NV.AGR.TOTL.ZS",
    "SH.H2O.SMDW.ZS",
]


def _load_worldbank_combined() -> pd.DataFrame:
    """Load and merge all World Bank indicator files."""
    dfs = []
    for code in _WB_INDICATORS:
        path = RAW / "worldbank" / f"{code}.csv"
        if path.exists():
            dfs.append(load_worldbank(path))
    if not dfs:
        raise FileNotFoundError("No World Bank CSV files found in data/raw/worldbank/")
    import functools

    return functools.reduce(
        lambda left, right: left.merge(right, on=["iso3", "year"], how="outer"), dfs
    )


def assemble(save_parquet: bool = True) -> pd.DataFrame:
    """Load all sources, join, validate, and return the Master Panel."""
    print("Loading sources...")
    sources = []

    def _try(name: str, loader_fn):
        try:
            df = loader_fn()
            print(f"  {name}: {len(df):,} rows, {df.iso3.nunique()} countries")
            sources.append(df)
        except FileNotFoundError as e:
            print(f"  {name}: SKIPPED — {e}")

    _try("AQUASTAT", lambda: load_aquastat(RAW / "aquastat" / "aquastat_all.csv"))
    _try("World Bank", _load_worldbank_combined)
    _try("UN DESA", lambda: load_undesa(RAW / "undesa" / "undesa_population.csv"))
    _try("FSI", lambda: load_fsi(RAW / "fragilestatesindex" / "fsi_combined.csv"))
    _try("UNODC", lambda: load_unodc(RAW / "unodc" / "unodc_homicide.csv"))
    _try("WHO", lambda: load_who(RAW / "GHDx" / "who_health.csv"))
    _try("UNHCR", lambda: load_unhcr(RAW / "unhcr" / "unhcr_displacement.csv"))
    _try("UCDP", lambda: load_ucdp(RAW / "ucdp" / "ucdp_conflicts.csv"))
    _try("GRACE", lambda: load_grace_panel(RAW / "grace" / "grace_country_year.csv"))

    print(f"\nAssembling Master Panel from {len(sources)} sources...")
    panel = build_master_panel(sources)
    print(f"  Panel shape: {panel.shape[0]:,} rows x {panel.shape[1]} columns")
    print(f"  Countries: {panel.iso3.nunique()} | Years: {panel.year.min()}-{panel.year.max()}")

    print("\nValidating Phase 1 exit criteria...")
    validate_master_panel(panel)
    print("  OK — all exit criteria passed")

    if save_parquet:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        out = PROCESSED / "master_panel.parquet"
        panel.to_parquet(out, index=False)
        print(f"\nSaved: {out} ({out.stat().st_size:,} bytes)")

    return panel


if __name__ == "__main__":
    panel = assemble()
    print("\nColumn summary:")
    for col in sorted(panel.columns):
        n_null = panel[col].isna().sum()
        coverage = 1 - n_null / len(panel)
        print(f"  {col:<40} {coverage:5.1%} coverage")
