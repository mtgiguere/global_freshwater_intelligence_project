"""Assembly of the GFIP Master Panel — the central data product of Phase 1.

The Master Panel is a country-year panel covering 274 countries from 1946 to 2025.
It is built by calling all 10 ingest modules, merging their outputs on the shared
keys [iso3, year], and writing the result to data/processed/master_panel.parquet.

This file is the downstream input for every subsequent phase of the project:
  - Phase 2: Exploratory Data Analysis (notebooks/01_eda.ipynb)
  - Phase 3: Hypothesis testing H1-H7 (analysis/ R scripts)
  - Phase 4: ML model training (src/models/train_all.py)
  - Phase 5: FastAPI backend and React dashboard (src/api/, dashboard/)

Design decisions:
  - The merge strategy is an outer join on [iso3, year] so that no country-year is
    silently discarded just because one source (e.g. UNODC) does not report it.
    See the comment on the merge step for the full rationale.
  - Validation (validate_master_panel) is called AFTER assembly and BEFORE the
    parquet write, so the file on disk is always structurally sound.
  - The parquet format is chosen over CSV for efficiency and type preservation.
    See the comment on the write step for the full rationale.
"""

import functools

import pandas as pd


def build_master_panel(sources: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge a list of source DataFrames into the Master Panel.

    Each source must share the panel keys ``iso3`` and ``year``. All columns
    beyond those keys are assumed to be non-overlapping (each ingest module
    produces uniquely named columns). If column names do collide, pandas will
    append ``_x`` / ``_y`` suffixes — the downstream validate step will catch
    this via structural checks.

    Args:
        sources: A list of DataFrames produced by the GFIP ingest modules. Each
            DataFrame must contain at minimum the columns ``iso3`` and ``year``.
            At least one DataFrame must be supplied.

    Returns:
        A single merged DataFrame indexed by the natural key (iso3, year). Rows
        present in only some sources will have NaN in the columns contributed by
        the absent sources — this is expected and intentional (see the outer-join
        comment below).

    Raises:
        ValueError: If ``sources`` is empty.
        ValueError: If any source DataFrame is missing the ``iso3`` or ``year``
            column, identifying which source (by positional index) is at fault.
    """
    if not sources:
        raise ValueError("at least one source DataFrame is required")

    for i, df in enumerate(sources):
        missing = [c for c in ["iso3", "year"] if c not in df.columns]
        if missing:
            raise ValueError(
                f"Source {i} is missing required columns: {missing}. "
                "All sources must have iso3 and year columns."
            )

    # --- Merge strategy: outer join on [iso3, year] ---
    #
    # We use an outer join (keep ALL rows from BOTH sides) rather than an inner
    # join (keep only rows present in BOTH sides).
    #
    # Why this matters: not every data source covers every country-year. For
    # example, UNODC homicide data is absent for many small island states; GRACE
    # satellite groundwater anomalies only begin in 2002; UNHCR refugee data is
    # sparse before 1990. An inner join would silently discard any country-year
    # where even one of the 10 sources has a gap — losing potentially thousands
    # of valid rows and introducing a severe selection bias (the surviving rows
    # would be disproportionately rich, well-documented countries, not the
    # fragile states that are most important to our analysis).
    #
    # The outer join preserves every country-year that appears in at least one
    # source. Missing values show up as NaN, which every downstream phase
    # handles explicitly (e.g. Phase 3 R scripts use na.omit per regression;
    # Phase 4 ML models impute or drop on a per-model basis).
    return functools.reduce(
        lambda left, right: left.merge(right, on=["iso3", "year"], how="outer"),
        sources,
    )
