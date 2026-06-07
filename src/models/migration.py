"""Migration Pressure Estimator — Model 3 of GFIP Phase 4.

What this model predicts:
    The log of refugee outflow (people fleeing a country in a given year).
    "Refugee outflow" here uses the UNHCR definition: people who have crossed an
    international border and been registered as refugees or asylum seekers.
    It does not include internally displaced people (IDPs), who are covered
    separately in the Master Panel under ``idp_count``.

    Higher predicted outflow = more forced displacement pressure = higher
    migration risk component in the Compound Risk Score.

    Target variable: log(refugee_outflow + 1)
        — The +1 ensures log(0) = 0 for countries with zero recorded outflows,
          rather than log(0) = undefined. The log transform is applied because
          refugee outflows follow an extremely heavy-tailed distribution: most
          country-years have near-zero outflows, while a handful of countries
          (Syria in 2013-2019, Afghanistan, South Sudan, Venezuela) generate
          outflows of hundreds of thousands or millions. Without the transform,
          the model would be completely dominated by those extreme cases and
          would perform poorly on the 90%+ of countries with moderate outflows.

Why water stress causes migration — the causal pathway:
    Water scarcity does not typically cause migration directly. The chain is:
        1. Drought / aquifer depletion → agricultural failure
        2. Agricultural failure → rural income collapse + food insecurity
        3. Food insecurity → social tension and conflict escalation
        4. Conflict escalation → forced displacement
    This indirect pathway is why the model includes both water stress AND conflict
    features: the two interact multiplicatively, not additively. Countries that are
    both water-stressed AND experiencing conflict are at disproportionately higher
    migration risk than countries with only one of those conditions.

Why Random Forest rather than Gradient Boosting?
    Random Forest is used here (versus Gradient Boosting for scarcity) for two
    specific reasons:
        1. Missingness robustness: UNHCR refugee data has substantial reporting
           gaps, especially for smaller countries and earlier years. Random
           Forest's ensemble averaging is more robust to the resulting noise
           than gradient boosting's sequential error-correction approach.
        2. Skewed distribution handling: refugee outflows are extremely right-skewed.
           Random Forest's voting/averaging mechanism naturally handles cases where
           most trees predict "low outflow" and a few trees correctly identify the
           high-outflow regime, producing calibrated ensemble predictions.

Zero vs. unreported — why we DROP rather than impute missing refugee_outflow:
    In the UNHCR data, a missing refugee_outflow value does NOT mean zero outflow.
    It means the data was not reported by the country for that year. Imputing zero
    would create false negatives — treating unreported situations as peaceful —
    and would systematically downwardly bias the model's predictions for
    data-sparse countries. Dropping these rows is the conservative, honest choice;
    it means the model is trained and evaluated only on situations where we have
    ground-truth evidence. The API handles this by falling back to synthetic CI
    predictions for countries with insufficient UNHCR coverage.

Phase 4 iteration 2 note:
    The project plan also specifies a gravity model component for bilateral flow
    estimation (predicting which destination countries receive flows from which
    origin countries). That extension is planned for a future iteration.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.models.features import add_lag_features, add_rolling_features

# Features used by the migration pressure estimator — all lagged or rolling.
# Feature selection rationale (each feature maps to one step in the water→migration
# causal chain described in the module docstring above):
#
#   log_freshwater_percap_lag1
#       — Lagged water availability. The UPSTREAM driver: countries with
#         very low freshwater per capita face chronic agricultural stress that
#         creates underlying displacement pressure even before acute conflict.
#
#   log_freshwater_percap_roll3_mean
#       — Three-year rolling mean of water availability.
#         A multi-year drought trend (declining rolling mean) is more likely
#         to trigger permanent displacement than a single bad year, from which
#         communities can recover. This feature captures whether the water
#         situation is deteriorating structurally, not just this year.
#
#   log_gdp_pc_ppp_lag1
#       — Lagged income per capita.
#         Wealthier countries have buffers — food imports, social safety nets,
#         emergency water infrastructure — that absorb agricultural shocks
#         without displacement. Low income per capita removes these buffers and
#         makes the same water shock far more likely to trigger migration.
#
#   log_population_lag1
#       — Lagged population size.
#         Even at the same per-capita outflow rate, a larger country produces
#         a larger absolute number of refugees. Population size also correlates
#         with land competition and resource conflict intensity.
#
#   ucdp_conflict_binary_lag1 / lag2
#       — Whether an armed conflict was active in the preceding one and two years.
#         Active conflict is the single strongest proximate driver of forced
#         displacement. Including both lags captures ongoing conflicts (both = 1)
#         versus post-conflict volatility (lag1 = 0, lag2 = 1), where return
#         migration may still be unsafe and secondary displacement continues.
#
#   fsi_score_lag1
#       — Lagged Fragile States Index score (0 = stable, 120 = most fragile).
#         Captures STATE FRAGILITY broadly — including governance failure,
#         human rights violations, and demographic pressure — beyond just armed
#         conflict. Countries with high FSI scores that have NOT yet experienced
#         formal UCDP-coded conflict may still generate significant refugee
#         flows from persecution, detention, and structural oppression.
#
#   safe_water_access_pct_lag1
#       — Percentage of population with safe water access.
#         A proxy for service delivery capacity. When this falls below ~60%,
#         it signals a governance breakdown that typically precedes displacement
#         even before open conflict begins.
#
#   log_refugee_outflow_lag1
#       — Lagged own refugee outflow (log scale).
#         Displacement is strongly serially correlated: once a country starts
#         generating refugees, it tends to continue for years (conflict duration,
#         continued persecution, deteriorating conditions). This is the most
#         powerful predictor in the model at short horizons. At longer horizons,
#         structural features like freshwater and FSI become relatively more
#         important as proximate drivers of continued outflow.
_FEATURE_COLS = [
    "log_freshwater_percap_lag1",
    "log_freshwater_percap_roll3_mean",
    "log_gdp_pc_ppp_lag1",
    "log_population_lag1",
    "ucdp_conflict_binary_lag1",
    "ucdp_conflict_binary_lag2",
    "fsi_score_lag1",
    "safe_water_access_pct_lag1",
    "log_refugee_outflow_lag1",  # own past outflow is a strong predictor
]


def build_migration_target(panel: pd.DataFrame) -> pd.Series:
    """Build log(refugee_outflow + 1) as the migration pressure target variable.

    The log-plus-one transform serves two purposes:
        1. Compresses the heavy tail: refugee outflows are extraordinarily skewed.
           Syria in 2013 generated ~2.5 million outflows; the median country-year
           has near-zero outflow. On the raw scale, the model's loss function would
           be dominated by Syria and a handful of other crisis states, degrading
           performance for the 95% of country-years with moderate outflows. On the
           log scale, Syria's 2.5 million becomes ~14.7, and a country with 1,000
           outflows becomes 6.9 — comparable magnitudes that the model can learn
           from equally.
        2. Handles zeros gracefully: log(0) is mathematically undefined, and many
           country-years have zero recorded refugee outflows. Adding 1 before taking
           the log maps zero to 0.0 (a valid, interpretable value) and preserves
           the ordering (more outflow = higher log-transformed target).

    The ``clip(lower=0)`` before the transform handles any negative values that
    might appear in the data due to statistical adjustments in the UNHCR source
    (e.g. net-flow corrections) — physically impossible values are clipped to zero.

    Args:
        panel: Master Panel DataFrame. Must contain the column ``refugee_outflow``
            (raw UNHCR outflow count, in persons per year). Rows with NaN values
            should be excluded before calling this function (see ``train_migration``
            in ``train_all.py`` for how this is handled).

    Returns:
        A pandas Series of the same length as ``panel`` containing
        log(refugee_outflow + 1) for each row. Values are in [0, ∞).
    """
    return np.log(panel["refugee_outflow"].clip(lower=0) + 1)


def build_migration_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix for the migration pressure estimator.

    Constructs lagged and rolling features from the Master Panel and returns
    only the columns listed in ``_FEATURE_COLS`` (those present in the data).

    Key constraint: the raw ``refugee_outflow`` column is the TARGET, not a
    feature. Including it as a feature would be perfect leakage — the model
    would simply learn "log(outflow this year) ≈ log(outflow this year + 1)".
    Only the lagged version (``log_refugee_outflow_lag1``) is permitted as a
    feature, representing information available one year BEFORE the prediction.

    The ``log_refugee_outflow`` intermediate column is constructed internally
    from the raw ``refugee_outflow`` before lagging. This is necessary because
    the lagging step operates on the log-transformed series; applying the lag
    to the raw series and then logging would give different (incorrect) results.

    Missing values in features are filled with 0. See the module docstring for
    the rationale: zero vs. unreported is a meaningful distinction for the
    TARGET variable, but for FEATURES, zero-fill is acceptable because most
    feature variables genuinely are near-zero for peaceful, non-stressed countries.

    Args:
        panel: Master Panel DataFrame that has already had log transforms applied
            (``add_log_transforms`` called) and has been filtered to rows with
            non-null ``refugee_outflow`` values. Must contain ``iso3`` and ``year``.

    Returns:
        A DataFrame with one row per (iso3, year) and columns corresponding to
        the subset of ``_FEATURE_COLS`` present after feature construction.
        All NaN values are filled with 0.
    """
    p = panel.copy()
    p["log_refugee_outflow"] = np.log(p["refugee_outflow"].clip(lower=0) + 1)

    p = add_lag_features(
        p,
        cols=[
            "log_freshwater_percap",
            "log_gdp_pc_ppp",
            "log_population",
            "ucdp_conflict_binary",
            "fsi_score",
            "safe_water_access_pct",
            "log_refugee_outflow",
        ],
        lags=[1, 2],
    )
    p = add_rolling_features(
        p,
        cols=["log_freshwater_percap"],
        windows=[3],
    )
    available = [c for c in _FEATURE_COLS if c in p.columns]
    return p[available].fillna(0)


def train_migration_model(
    X: pd.DataFrame,
    y: pd.Series,
) -> RandomForestRegressor:
    """Train a Random Forest regression model on the migration pressure target.

    Random Forest builds an ensemble of decision trees independently (in parallel),
    each trained on a random bootstrap sample of the data with a random subset of
    features considered at each split. The final prediction is the average of all
    trees. This is different from Gradient Boosting (used for scarcity), where
    trees are built sequentially to correct each other's errors.

    Why Random Forest for migration, not Gradient Boosting?
        1. Missing data robustness: UNHCR data has significant reporting gaps.
           Random Forest's bootstrap averaging smooths over the noise introduced
           by missing values in a way that sequential error-correction (gradient
           boosting) does not — each bootstrapped tree sees a different subset of
           the data, preventing any single gap from dominating.
        2. Threshold effects: refugee crises are threshold phenomena — a country
           is generating near-zero outflows and then suddenly millions. Random
           Forest's recursive binary partitioning is well-adapted to finding these
           "tipping point" boundaries in feature space (e.g. "conflict = 1 AND
           freshwater below threshold X → very high outflow").
        3. Interpretability under skew: the averaging of many independently
           trained trees produces well-calibrated estimates across the full range
           of the distribution, including the zero-inflation at the lower end
           and the extreme outliers at the upper end.

    Hyperparameter notes:
        n_estimators=200  — 200 independent trees. More trees reduce variance
                            but with diminishing returns beyond ~200 for this
                            dataset size.
        max_depth=6       — Slightly deeper than instability (depth 4) to capture
                            the multi-step causal chain (water → crop failure →
                            conflict → migration) which requires more splits.
        n_jobs=-1         — Use all available CPU cores. Random Forest is
                            "embarrassingly parallel" (trees are independent) so
                            this gives a significant speedup on multi-core machines.
        random_state=42   — Fixed seed for reproducibility.

    Args:
        X: Feature matrix (n_samples x n_features). Must not contain NaN.
            Variable names X and y follow universal ML conventions (PEP8 exemption).
        y: Continuous target vector (n_samples,) — log(refugee_outflow + 1).
            Must be the same length as ``X``.

    Returns:
        A fitted ``RandomForestRegressor`` instance. Call ``model.predict(X)``
        to obtain log-scale refugee outflow predictions for new data.
    """
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def predict_migration(
    model: RandomForestRegressor,
    X: pd.DataFrame,
) -> np.ndarray:
    """Return predicted log(refugee_outflow + 1) for each row in X.

    Predictions are in log-plus-one scale. To recover an approximate refugee
    count, apply exp(prediction) - 1. For example:
        prediction = 7.0  →  exp(7.0) - 1  ≈ 1,096 refugees
        prediction = 13.0 →  exp(13.0) - 1 ≈ 442,413 refugees

    These are intended as ordinal risk indicators, not precise census counts.
    The absolute numbers should be interpreted with caution; the relative
    ranking of countries is more reliable.

    Args:
        model: A fitted ``RandomForestRegressor`` as returned by
            ``train_migration_model``.
        X: Feature matrix with the same columns used during training. Variable
            names X and y follow universal ML conventions (PEP8 exemption).

    Returns:
        A 1-D NumPy array of log-scale refugee outflow predictions, one per
        row of ``X``. Higher values indicate greater predicted migration
        pressure from the country.
    """
    return model.predict(X)
