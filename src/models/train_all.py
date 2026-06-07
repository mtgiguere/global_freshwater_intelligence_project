"""Train all three Phase 4 ML models on the Master Panel and save to data/models/.

Full workflow:
    1. Load ``data/processed/master_panel.parquet`` — the unified 17,000-row panel
       of 274 countries from 1946 to present, produced by the Phase 1 pipeline.
    2. Add log transforms (log_freshwater_percap, log_gdp_pc_ppp, log_population)
       to prepare the features that all three models share.
    3. Drop rows where the primary exposure variable (log_freshwater_percap) is
       missing — these rows cannot inform any of the three models.
    4. Train three models on a temporal split (pre-2015 train / 2015+ test):
         a. Water Scarcity Forecaster (GradientBoostingRegressor)
            — predicts log freshwater per capita 5 years ahead
         b. Instability Risk Predictor (XGBoost binary classifier)
            — predicts P(FSI jump OR conflict onset within 3 years)
         c. Migration Pressure Estimator (RandomForest regression)
            — predicts log(refugee outflow + 1)
    5. Evaluate the scarcity model against the persistence baseline to confirm it
       has learned something useful (RMSE check — see ``train_scarcity``).
    6. Save model files and normalization stats to ``data/models/``.

After running this script:
    - The API prediction endpoint (``GET /api/v1/predict/{iso3}``) returns
      ``is_trained=True`` and serves real model forecasts.
    - The dashboard MLFutures panel shows real predictions instead of the
      synthetic confidence-interval fallback.
    - Restart the API server after training to pick up the new model files.

Usage:
    uv run python src/models/train_all.py

Prerequisites:
    - data/processed/master_panel.parquet must exist (run the Phase 1 pipeline first:
      ``uv run python src/pipeline/master_panel.py``)
    - All Phase 4 model dependencies must be installed (uv sync)

Outputs written to data/models/:
    - instability_model.joblib    — XGBoost binary classifier (pickled via joblib)
    - scarcity_model.joblib       — GradientBoosting regressor (pickled via joblib)
    - migration_model.joblib      — RandomForest regressor (pickled via joblib)
    - normalization_stats.json    — min/max ranges for scarcity and migration outputs,
                                    used by the API to map raw model predictions to [0, 1]
                                    for the Compound Risk Score calculation.

Why normalization_stats.json is critical:
    The Compound Risk Score requires each model's output to be normalised to [0, 1]
    using min-max scaling. The min and max used in that normalisation MUST be the
    same values observed during training — if the API re-normalises based on the
    current prediction batch, scores will be inconsistent across requests (a country
    might score 0.9 in one API call and 0.3 in another depending on which other
    countries happen to be in the same batch). Saving the training-time min/max to
    a file and loading it in the API ensures consistent, comparable scores.

TDD exemption (per docs/TDD_CONTRACT.md §"Where strict RED/GREEN TDD is not
fully applicable"):
    ML training loops are not unit-testable in the RED/GREEN sense — the key
    behaviour is statistical (model beats baseline) not functional (function
    returns X given input Y). There is no sensible way to write a failing unit
    test that, when passed, proves the gradient boosting algorithm has learned
    the correct relationship between freshwater depletion and future water stress.
    The unit-testable pieces — feature engineering, target construction, temporal
    splitting, output schema validation — are all covered in ``tests/models/``.
    What IS verified in this script at runtime:
        1. Scarcity model beats the naive persistence baseline (RMSE < baseline RMSE).
           A failure here should halt investigation before production deployment.
        2. Training sizes are printed so data-sparsity problems are immediately obvious.
        3. All three model files are written before the script exits (joblib handles this).
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
    """Load the Master Panel parquet file, exiting early with a helpful error if absent.

    Returns:
        The Master Panel as a DataFrame (~17,000 rows x 35+ columns).

    Raises:
        SystemExit: If the parquet file does not exist. The error message explains
            exactly which command to run to generate the file, so the user is not
            left guessing.
    """
    if not _PANEL_PATH.exists():
        print(f"ERROR: Master Panel not found at {_PANEL_PATH}")
        print("Run the Phase 1 pipeline first: uv run python src/pipeline/master_panel.py")
        sys.exit(1)
    print(f"Loading Master Panel from {_PANEL_PATH} …")
    return pd.read_parquet(_PANEL_PATH)


def _prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Add log transforms and drop rows where the core freshwater column is missing.

    Preparation steps:
        1. Add log-transformed columns (log_freshwater_percap, log_gdp_pc_ppp,
           log_population) via ``add_log_transforms``.
        2. Drop rows where ``log_freshwater_percap`` is NaN.

    Why drop on freshwater only?
        ``log_freshwater_percap`` is the primary exposure variable in all three
        models — scarcity predicts it directly, instability and migration use it
        as a key feature. A row with no freshwater data provides essentially no
        signal on the dimension we care most about and would introduce noise.
        Other columns (GDP, conflict, FSI) may still be missing after this step;
        those are handled by the individual model feature builders via zero-fill.

    We drop individual ROWS with missing freshwater, not entire COUNTRIES. This
    preserves data for countries that have incomplete time coverage (e.g. South
    Sudan, which only exists as a country from 2011) while removing only the
    specific years for which the key variable is unavailable.

    Args:
        panel: Raw Master Panel DataFrame as loaded from parquet.

    Returns:
        A copy of ``panel`` with log-transformed columns added and rows missing
        ``log_freshwater_percap`` removed. The row count reduction is printed
        so any unexpected data loss is immediately visible.
    """
    panel = add_log_transforms(panel)
    before = len(panel)
    panel = panel.dropna(subset=["log_freshwater_percap"]).copy()
    after = len(panel)
    if before - after > 0:
        print(f"  Dropped {before - after} rows missing renewable_freshwater_percap")
    return panel


def train_scarcity(panel: pd.DataFrame) -> dict:
    """Train the water scarcity forecaster (GradientBoosting) and return model + stats.

    Workflow:
        1. Build the 5-year-ahead log freshwater target for each country-year.
        2. Build the feature matrix (lagged and rolling water/economic variables).
        3. Split at 2015: train on pre-2015 data, evaluate on 2015 onward.
        4. Train the GradientBoostingRegressor on the training split.
        5. Compare model RMSE to persistence baseline RMSE on the test split.
           Print a warning if the model does not beat the baseline.
        6. Compute and return the normalisation range from the full training dataset
           (not just the test split) — this is what the API will use to map raw
           predictions to [0, 1] consistently across all future requests.

    Persistence baseline explanation:
        The persistence baseline predicts: "in 5 years, freshwater per capita will
        be the same as it is today." This is the simplest conceivable forecasting
        strategy. A trained model that cannot beat it has not learned the direction
        of change — it should be considered broken. The baseline uses the lag-1
        feature (last year's value) as a proxy for "today's value" because that is
        what is available in the feature matrix at test time.

    RMSE interpretation (lower is better):
        RMSE is measured in log-scale units. An RMSE of 0.3 means the model's
        predictions are, on average, about 0.3 natural-log units off from the true
        future value — roughly a 35% error on the original m³/person/year scale.

    Normalization range:
        The min and max of all_preds (predictions over the full valid dataset, not
        just the test split) are stored and later saved to normalization_stats.json.
        The API loads these values to apply the SAME min-max scaling used at training
        time when computing the Compound Risk Score. If the API normalised on the fly
        using the current batch's min/max, scores would shift between API calls and
        be incomparable across countries and time periods.

    Args:
        panel: Prepared Master Panel (log transforms added, NaN freshwater dropped).

    Returns:
        A dict with two keys:
            ``model`` — the fitted GradientBoostingRegressor
            ``norm``  — dict with ``min`` and ``max`` of training-time predictions,
                        for consistent normalisation in the API
    """
    print("\n[1/3] Water Scarcity Forecaster …")

    target = build_scarcity_target(panel)
    features = build_scarcity_features(panel)

    # Align to common index — feature engineering (lags/rolling) drops early rows,
    # so features and target may have different indices after build.
    common = features.index.intersection(target.index)
    features, target = features.loc[common], target.loc[common]
    valid = target.notna()
    # X and y follow universal ML naming conventions (PEP8 exemption)
    X, y = features[valid], target[valid]
    print(f"  Training rows: {len(X)}")

    train_panel, test_panel = temporal_train_test_split(panel.loc[X.index], test_from_year=2015)
    train_idx = train_panel.index
    test_idx = test_panel.index

    # Align X and y to the split — only rows that survived both the target filter
    # and the temporal split are used. The intersection handles edge cases where
    # a row appears in panel[valid] but not in features[valid] due to NaN differences.
    train_idx = train_idx.intersection(X.index)
    test_idx = test_idx.intersection(X.index)

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]
    print(f"  Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

    model = train_scarcity_model(X_train, y_train)

    # Persistence baseline: predict "5 years from now = same as last year."
    # We use lag-1 as the baseline prediction because it is the value the model
    # itself receives as a feature — if the model simply copied lag-1, it would
    # match this baseline exactly. The actual persistence prediction would be
    # lag-5, but that column is not in the feature set; lag-1 is the closest
    # available proxy and produces a conservative (easy-to-beat) baseline.
    if "log_freshwater_percap_lag1" in X_test.columns:
        baseline = X_test["log_freshwater_percap_lag1"].fillna(y_test.mean()).values
    else:
        baseline = np.full(len(y_test), y_test.mean())

    metrics = evaluate_vs_baseline(model, X_test, y_test, baseline)
    # R² is not printed here because evaluate_vs_baseline returns RMSE only.
    # RMSE interpretation: lower = better predictions on the log freshwater scale.
    print(f"  Model RMSE:    {metrics['model_rmse']:.4f}")
    print(f"  Baseline RMSE: {metrics['baseline_rmse']:.4f}")
    if metrics["model_rmse"] >= metrics["baseline_rmse"]:
        print("  WARNING: model does not beat persistence baseline — check features")

    # Compute normalization range from ALL valid predictions (not just test split).
    # The API uses these min/max values to convert raw predictions to [0, 1].
    # Using the full valid set (not just training) gives a more stable range that
    # encompasses the full distribution the model is likely to encounter.
    all_preds = model.predict(X)
    return {
        "model": model,
        "norm": {"min": float(all_preds.min()), "max": float(all_preds.max())},
    }


def train_instability(panel: pd.DataFrame) -> dict:
    """Train the instability risk predictor (XGBoost classifier) and return the model.

    Workflow:
        1. Build the forward-looking binary target (1 = instability onset within
           3 years, 0 = no onset).
        2. Build the feature matrix (lagged water, economic, and conflict variables).
        3. Train on the pre-2015 split only (no test-set evaluation here, because
           AUC evaluation for binary classifiers requires sufficient positive
           examples in the test set — the ~5-year test window may not have enough
           instability onsets in some runs to produce a reliable AUC estimate).
        4. Return the model. No normalization stats needed: XGBoost's
           ``predict_proba`` already returns values in [0, 1].

    The positive rate printed here (e.g. "Positive rate: 18.3%") shows what
    fraction of training country-years experienced instability onset within the
    3-year horizon. If this falls below ~5%, the class imbalance correction
    (``scale_pos_weight``) becomes critical. If it is above ~40%, the target
    definition may be too broad (too many "positives").

    Temporal split: pre-2015 training is used here (rather than a shorter window)
    to maximize the number of positive examples in training. Instability onsets
    are rare — the more historical data the model sees, the better it can learn
    the pre-conditions that distinguish peaceful from at-risk countries.

    Args:
        panel: Prepared Master Panel (log transforms added, NaN freshwater dropped).

    Returns:
        A dict with one key:
            ``model`` — the fitted XGBClassifier
        (No ``norm`` key — instability probabilities are already in [0, 1].)
    """
    print("\n[2/3] Instability Risk Predictor …")

    target = build_instability_target(panel)
    features = build_instability_features(panel)

    # Align to common index — same reason as scarcity: lag features drop early rows.
    common = features.index.intersection(target.index)
    features, target = features.loc[common], target.loc[common]
    valid = target.notna()
    # X and y follow universal ML naming conventions (PEP8 exemption)
    X, y = features[valid], target[valid]
    # Positive rate = fraction of country-years where instability onset was observed.
    # Typical value: 10-25%. Very low rates mean class imbalance is severe.
    print(f"  Training rows: {len(X)}  |  Positive rate: {y.mean():.1%}")

    train_panel, _ = temporal_train_test_split(panel.loc[X.index], test_from_year=2015)
    train_idx = train_panel.index.intersection(X.index)
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    print(f"  Train: {len(X_train)} rows")

    model = train_instability_model(X_train, y_train)

    # Instability model outputs probabilities via predict_proba — already in [0, 1];
    # no min/max normalisation step is needed for the Compound Risk Score.
    return {"model": model}


def train_migration(panel: pd.DataFrame) -> dict:
    """Train the migration pressure estimator (RandomForest) and return model + stats.

    Workflow:
        1. Drop rows where ``refugee_outflow`` is NaN. Unlike the other two models,
           which can zero-fill missing features, the migration model CANNOT treat
           unreported refugee data as zero — missing reports are not the same as
           peaceful situations. (See the ``migration.py`` module docstring for the
           full rationale.)
        2. Build the log(refugee_outflow + 1) target and the feature matrix.
        3. Filter to rows where both target and all features are available.
        4. Train on the pre-2015 split using ``train_migration_model``.
        5. Compute and return the normalisation range from all valid predictions.

    Why the additional ``features.notna().all(axis=1)`` filter?
        The migration feature matrix includes ``log_refugee_outflow_lag1`` — the
        lagged own outflow, which requires two consecutive years of UNHCR data.
        For countries that begin reporting mid-series, the first valid year will
        have a NaN lag-1 feature even after zero-fill. The
        ``features.notna().all(axis=1)`` check catches any residual NaNs that
        zero-fill missed (this should rarely trigger but is a safety net).

    Normalization range:
        As with scarcity, the training-time min/max of all valid predictions is
        saved to normalization_stats.json. The API loads these to ensure
        consistent [0, 1] scaling across all prediction requests.

    Args:
        panel: Prepared Master Panel (log transforms added, NaN freshwater dropped).

    Returns:
        A dict with two keys:
            ``model`` — the fitted RandomForestRegressor
            ``norm``  — dict with ``min`` and ``max`` of training-time predictions,
                        for consistent normalisation in the API
    """
    print("\n[3/3] Migration Pressure Estimator …")

    # Migration requires refugee_outflow — drop rows where it is missing entirely.
    # This is a deliberate design choice: missing UNHCR data is NOT the same as
    # zero refugee outflow (see migration.py module docstring for full explanation).
    panel_mig = panel.dropna(subset=["refugee_outflow"]).copy()
    n_dropped = len(panel) - len(panel_mig)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} rows missing refugee_outflow")

    target = build_migration_target(panel_mig)
    features = build_migration_features(panel_mig)

    # Align to common index before filtering — lag features drop early rows.
    common = features.index.intersection(target.index)
    features, target = features.loc[common], target.loc[common]
    valid = target.notna() & features.notna().all(axis=1)
    # X and y follow universal ML naming conventions (PEP8 exemption)
    X, y = features[valid], target[valid]
    print(f"  Training rows: {len(X)}")

    train_panel, _ = temporal_train_test_split(panel_mig[valid], test_from_year=2015)
    train_idx = train_panel.index.intersection(X.index)
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    print(f"  Train: {len(X_train)} rows")

    model = train_migration_model(X_train, y_train)

    # Compute normalisation range from all valid predictions (same reasoning as scarcity).
    all_preds = model.predict(X)
    return {
        "model": model,
        "norm": {"min": float(all_preds.min()), "max": float(all_preds.max())},
    }


def save_models(scarcity: dict, instability: dict, migration: dict) -> None:
    """Write all model files and normalization stats to data/models/.

    Files written:
        instability_model.joblib
            — Pickled XGBClassifier. Loaded by the API's ``/predict/{iso3}``
              endpoint to compute instability probabilities.

        scarcity_model.joblib
            — Pickled GradientBoostingRegressor. Loaded by the API to predict
              log freshwater per capita 5 years ahead.

        migration_model.joblib
            — Pickled RandomForestRegressor. Loaded by the API to predict
              log(refugee_outflow + 1).

        normalization_stats.json
            — JSON file with the structure:
                {
                  "scarcity":  {"min": <float>, "max": <float>},
                  "migration": {"min": <float>, "max": <float>}
                }
              These min/max values are the range of predictions from the
              TRAINING data. The API uses them to map raw model predictions
              to [0, 1] via min-max normalisation before combining them into
              the Compound Risk Score.

              Why is this file critical?
              If the API normalised on-the-fly using the current request's
              min/max, then:
                  a) Scores would change between API calls as different country
                     subsets are queried — a country might score 0.8 in one
                     call and 0.4 in another.
                  b) Single-country predictions would always normalise to 0 or 1
                     (the only point is trivially both the min and the max).
              Using training-time normalization stats solves both problems by
              anchoring the scale to the full distribution observed during training.

              Instability is excluded because its output (probability from
              predict_proba) is already in [0, 1] without any normalisation.

    joblib is used instead of pickle directly because joblib handles numpy arrays
    (embedded in scikit-learn and XGBoost models) more efficiently than pickle,
    producing smaller files and faster load times.

    Args:
        scarcity: dict returned by ``train_scarcity``, containing keys
            ``model`` and ``norm``.
        instability: dict returned by ``train_instability``, containing key
            ``model``.
        migration: dict returned by ``train_migration``, containing keys
            ``model`` and ``norm``.
    """
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
