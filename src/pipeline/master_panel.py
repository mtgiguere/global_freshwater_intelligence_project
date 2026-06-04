import functools

import pandas as pd


def build_master_panel(sources: list[pd.DataFrame]) -> pd.DataFrame:
    if not sources:
        raise ValueError("at least one source DataFrame is required")

    for i, df in enumerate(sources):
        missing = [c for c in ["iso3", "year"] if c not in df.columns]
        if missing:
            raise ValueError(
                f"Source {i} is missing required columns: {missing}. "
                "All sources must have iso3 and year columns."
            )

    return functools.reduce(
        lambda left, right: left.merge(right, on=["iso3", "year"], how="outer"),
        sources,
    )
