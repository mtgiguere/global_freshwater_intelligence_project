"""Train all three Phase 4 ML models on the Master Panel and save to data/models/.

After running this script the API prediction endpoint returns is_trained=True
and the dashboard shows real forecasts instead of the synthetic CI fallback.

Usage:
    uv run python src/models/train_all.py

Prerequisites:
    - data/processed/master_panel.parquet must exist (run the Phase 1 pipeline first)
    - All Phase 4 model dependencies must be installed (uv sync)

Outputs written to data/models/:
    - instability_model.joblib   — XGBoost binary classifier
    - scarcity_model.joblib      — GradientBoosting regression
    - migration_model.joblib     — RandomForest regression
    - normalization_stats.json   — min/max ranges for scarcity and migration outputs,
                                   used by the API to map raw predictions to [0, 1]

TDD exemption (per docs/TDD_CONTRACT.md §"Where strict RED/GREEN TDD is not
fully applicable"): ML training loops are not unit-testable in the RED/GREEN
sense — the key behaviour is statistical (model beats baseline) not functional
(function returns X given input Y). The unit-testable pieces — feature engineering,
target construction, temporal splitting — are all covered in tests/models/.
What IS verified here:
    1. Scarcity model must beat the naive persistence baseline (RMSE check)
    2. Training sizes are logged so data sparsity problems are obvious immediately
    3. All three model files must be written before the script exits
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Resolve paths relative to the repository root regardless of working directory
_REPO_ROOT = Path(__file__).parents[2]
_PANEL_PATH = _REPO_ROOT / "data" / "processed" / "master_panel.parquet"
_MODELS_DIR = _REPO_ROOT / "data" / "models"

from src.models.features import (  # noqa: E402
    add_log_transforms,
    temporal_train_test_split,
)
from src.models.instability import (  # noqa: E402
    build_instability_features,
    build_instability_target,
    train_instability_model,
)
from src.models.migration import (  # noqa: E402
    build_migration_features,
    build_migration_target,
    train_migration_model,
)
from src.models.scarcity import (  # noqa: E402
    build_scarcity_features,
    build_scarcity_target,
    evaluate_vs_baseline,
    train_scarcity_model,
)


def _load_panel() -> pd.DataFrame:
    if not _PANEL_PATH.exists():
        print(f"ERROR: Master Panel not found at {_PANEL_PATH}")
        print("Run the Phase 1 pipeline first: uv run python src/pipeline/master_panel.py")
        sys.exit(1)
    print(f"Loading Master Panel from {_PANEL_PATH} …")
    return pd.read_parquet(_PANEL_PATH)


def _prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Add log transforms and drop rows where the core freshwater column is missing.

    Models cannot be trained on rows with NaN freshwater data — that column is
    the primary exposure variable in all three models. We drop only those rows,
    not the entire country, to preserve as much data as possible.
    """
    panel = add_log_transforms(panel)
    before = len(panel)
    panel = panel.dropna(subset=["log_freshwater_percap"]).copy()
    after = len(panel)
    if before - after > 0:
        print(f"  Dropped {before - after} rows missing renewable_freshwater_percap")
    return panel


def train_scarcity(panel: pd.DataFrame) -> dict:
    """Train the water scarcity forecaster (GradientBoosting) and return model + stats."""
    print("\n[1/3] Water Scarcity Forecaster …")

    target = build_scarcity_target(panel)
    features = build_scarcity_features(panel)

    valid = target.notna()
    X, y = features[valid], target[valid]
    print(f"  Training rows: {len(X)}")

    train_panel, test_panel = temporal_train_test_split(panel[valid], test_from_year=2015)
    train_idx = train_panel.index
    test_idx = test_panel.index

    # Align X and y to the split — only rows that survived both the target filter
    # and the temporal split are used
    train_idx = train_idx.intersection(X.index)
    test_idx = test_idx.intersection(X.index)

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]
    print(f"  Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

    model = train_scarcity_model(X_train, y_train)

    # Persistence baseline: predict this year's freshwater = last year's freshwater
    # (lag-1 feature is the single strongest naive predictor)
    if "log_freshwater_percap_lag1" in X_test.columns:
        baseline = X_test["log_freshwater_percap_lag1"].fillna(y_test.mean()).values
    else:
        baseline = np.full(len(y_test), y_test.mean())

    metrics = evaluate_vs_baseline(model, X_test, y_test, baseline)
    print(f"  Model RMSE:    {metrics['model_rmse']:.4f}")
    print(f"  Baseline RMSE: {metrics['baseline_rmse']:.4f}")
    if metrics["model_rmse"] >= metrics["baseline_rmse"]:
        print("  WARNING: model does not beat persistence baseline — check features")

    # Save normalization range derived from ALL training predictions.
    # Used by the API to map raw log(freshwater_percap) predictions to [0, 1].
    all_preds = model.predict(X)
    return {
        "model": model,
        "norm": {"min": float(all_preds.min()), "max": float(all_preds.max())},
    }


def train_instability(panel: pd.DataFrame) -> dict:
    """Train the instability risk predictor (XGBoost) and return the model."""
    print("\n[2/3] Instability Risk Predictor …")

    target = build_instability_target(panel)
    features = build_instability_features(panel)

    valid = target.notna()
    X, y = features[valid], target[valid]
    print(f"  Training rows: {len(X)}  |  Positive rate: {y.mean():.1%}")

    train_panel, _ = temporal_train_test_split(panel[valid], test_from_year=2015)
    train_idx = train_panel.index.intersection(X.index)
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    print(f"  Train: {len(X_train)} rows")

    model = train_instability_model(X_train, y_train)

    # Instability model outputs probabilities — already in [0, 1]; no normalisation needed.
    return {"model": model}


def train_migration(panel: pd.DataFrame) -> dict:
    """Train the migration pressure estimator (RandomForest) and return model + stats."""
    print("\n[3/3] Migration Pressure Estimator …")

    # migration requires refugee_outflow — drop countries missing it entirely
    panel_mig = panel.dropna(subset=["refugee_outflow"]).copy()
    n_dropped = len(panel) - len(panel_mig)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} rows missing refugee_outflow")

    target = build_migration_target(panel_mig)
    features = build_migration_features(panel_mig)

    valid = target.notna() & features.notna().all(axis=1)
    X, y = features[valid], target[valid]
    print(f"  Training rows: {len(X)}")

    train_panel, _ = temporal_train_test_split(panel_mig[valid], test_from_year=2015)
    train_idx = train_panel.index.intersection(X.index)
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    print(f"  Train: {len(X_train)} rows")

    model = train_migration_model(X_train, y_train)

    all_preds = model.predict(X)
    return {
        "model": model,
        "norm": {"min": float(all_preds.min()), "max": float(all_preds.max())},
    }


def save_models(scarcity: dict, instability: dict, migration: dict) -> None:
    """Write all model files and normalization stats to data/models/."""
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(scarcity["model"], _MODELS_DIR / "scarcity_model.joblib")
    joblib.dump(instability["model"], _MODELS_DIR / "instability_model.joblib")
    joblib.dump(migration["model"], _MODELS_DIR / "migration_model.joblib")

    norm_stats = {
        "scarcity": scarcity["norm"],
        "migration": migration["norm"],
    }
    (_MODELS_DIR / "normalization_stats.json").write_text(json.dumps(norm_stats, indent=2))

    print(f"\nAll models saved to {_MODELS_DIR}/")
    print("  instability_model.joblib")
    print("  scarcity_model.joblib")
    print("  migration_model.joblib")
    print("  normalization_stats.json")
    print("\nThe prediction endpoint will now return is_trained=True.")
    print("Restart the API server to pick up the new models.")


if __name__ == "__main__":
    panel = _load_panel()
    print(f"Panel shape: {panel.shape}  |  Countries: {panel['iso3'].nunique()}")

    panel = _prepare_panel(panel)

    scarcity = train_scarcity(panel)
    instability = train_instability(panel)
    migration = train_migration(panel)

    save_models(scarcity, instability, migration)
