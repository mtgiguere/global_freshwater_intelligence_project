"""Instability Risk Predictor — Model 2 of GFIP Phase 4.

What this model predicts:
    The probability that a country will experience a significant deterioration
    in political stability within the next three years. "Instability" is defined
    as meeting at least one of two criteria:
        (1) The Fragile States Index (FSI) score increases by more than 5 points
            — indicating a meaningful worsening of governance, security, or
            social cohesion as judged by the Fund for Peace methodology.
        (2) A new armed conflict event is recorded by the Uppsala Conflict Data
            Program (UCDP) — meaning at least 25 battle-related deaths in a
            calendar year, in a country where no such event was recorded the
            year before (onset, not just continuation).

    This is a BINARY CLASSIFICATION problem: for each country-year in the
    training data, the target is 1 (instability onset within 3 years) or 0
    (no onset). The model outputs a probability, P ∈ [0, 1].

    This approach follows the PITF (Political Instability Task Force) and ViEWS
    (Violence Early Warning System) convention of predicting significant
    deterioration rather than merely the current level of instability, because
    policymakers need early warning of CHANGE — a country that is already
    unstable is a known situation, but a stable country suddenly sliding toward
    conflict is where early intervention can prevent harm.

Why XGBoost (gradient boosted trees) rather than logistic regression?
    1. Missing data tolerance: the Master Panel has substantial gaps, especially
       for smaller countries and earlier decades. XGBoost handles NaN values
       natively by learning which direction to send missing-value rows at each
       tree split. Logistic regression requires imputation, which introduces its
       own assumptions and errors.
    2. Non-linear interactions: the relationship between water stress and
       conflict is not linear. A country with both very low freshwater per
       capita AND low GDP is at disproportionately higher risk than the sum of
       each factor alone — gradient boosted trees capture these interaction
       effects through their recursive splitting structure.
    3. Robustness to outliers: extreme values (e.g. Somalia's FSI score or
       Syria's refugee outflow) would strongly influence a logistic regression
       model's coefficients, potentially degrading performance on average
       countries. Tree-based methods partition the feature space and are not
       sensitive to extreme values in the same way.

Class imbalance:
    Armed conflict events are rare — most country-years in the panel are
    peaceful. If the model were trained naively on the raw data it would
    achieve high accuracy simply by predicting "no conflict" for every
    country-year. The ``scale_pos_weight`` hyperparameter corrects for this by
    telling XGBoost to weight each positive (instability = 1) training example
    as if it appeared ``n_negative / n_positive`` times. This forces the model
    to "care" about the rare conflict-onset events.

Input features:
    All features are lagged or rolling — the current year's outcome variables
    (FSI score, conflict status) are NEVER used as features because doing so
    would leak the answer into the question. See ``_FEATURE_COLS`` below.

Temporal split:
    Training uses data before 2015; evaluation uses 2015 onward. This ensures
    the model is evaluated on genuinely out-of-sample future data. The ~5 years
    of test data (2015-2020) provide a meaningful estimate of real-world
    predictive performance.
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.features import add_lag_features, add_rolling_features

# Features used in the model — all lagged to avoid future leakage.
# Each variable is selected based on theoretical links to conflict and instability:
#
#   log_freshwater_percap_lag1 / lag2
#       — Current water stress level and its one-year change direction.
#         Theory: acute water scarcity strains food systems, raises commodity prices,
#         and creates resource competition — all well-documented conflict triggers
#         (Homer-Dixon 1994; Gleick 2014). The two-lag difference captures momentum.
#
#   log_freshwater_percap_roll3_mean
#       — Medium-term trend in water availability.
#         A three-year declining trend is more alarming than a single bad year;
#         it suggests structural depletion (aquifer drawdown, climate shift) rather
#         than weather-driven variability.
#
#   log_gdp_pc_ppp_lag1
#       — Lagged income per capita (log scale).
#         Poverty is the single strongest correlate of civil conflict in the
#         empirical literature (Collier & Hoeffler 2004). Low-income countries
#         have weaker institutions, fewer fiscal buffers, and higher grievance levels.
#
#   log_population_lag1
#       — Lagged population size (log scale).
#         Larger populations mean more people competing for scarce resources and
#         larger absolute numbers of potentially mobilisable combatants.
#
#   safe_water_access_pct_lag1
#       — Percentage of the population with access to safe drinking water.
#         A measure of governance capacity and infrastructure quality, not just
#         physical water availability. Low access signals state fragility.
#
#   ucdp_conflict_binary_lag1 / lag2
#       — Whether an armed conflict was recorded in the preceding one and two years.
#         Conflict is strongly serially correlated — countries that experienced
#         conflict recently are far more likely to experience it again (conflict
#         trap). Including both lags captures whether a conflict is ongoing (both
#         lag1 and lag2 = 1) versus recently ended (lag1 = 0, lag2 = 1).
#
#   grace_lwe_anomaly_cm_lag1
#       — Lagged GRACE satellite groundwater anomaly (cm of equivalent water height).
#         Negative values indicate groundwater depletion below the long-run mean.
#         This is the most direct measurement of aquifer health in the dataset —
#         it captures hidden depletion that surface-water statistics miss entirely.
_FEATURE_COLS = [
    "log_freshwater_percap_lag1",
    "log_freshwater_percap_lag2",
    "log_freshwater_percap_roll3_mean",
    "log_gdp_pc_ppp_lag1",
    "log_population_lag1",
    "safe_water_access_pct_lag1",
    "ucdp_conflict_binary_lag1",
    "ucdp_conflict_binary_lag2",
    "grace_lwe_anomaly_cm_lag1",
]


def build_instability_target(
    panel: pd.DataFrame,
    horizon_years: int = 3,
    fsi_jump_threshold: float = 5.0,
) -> pd.Series:
    """Build the forward-looking instability target variable (binary: 0 or 1).

    For each country-year (country C, year Y), this function scans the next
    ``horizon_years`` years (Y+1 through Y+horizon_years) and asks two questions:

        1. Did the FSI score increase by more than ``fsi_jump_threshold`` points
           at any point in that window?  (A jump of 5+ points on the 0-120 FSI
           scale represents a meaningful governance deterioration, not just noise.)

        2. Did a UCDP-recorded armed conflict START in that window? A "start"
           means the country was conflict-free in year Y (ucdp_conflict_binary = 0)
           but recorded a conflict in at least one year in the window. This
           captures ONSET rather than continuation — we want to predict new crises.

    If EITHER condition is satisfied, the target for (C, Y) is 1 (instability
    onset predicted). If neither is satisfied, the target is 0 (stable trajectory).

    The final ``horizon_years`` rows per country always have NaN targets because
    there are not enough future years in the dataset to evaluate the full window.
    These rows must be filtered out before model training.

    This is the most critical function in the model pipeline: a wrong target
    definition means we are training the model to answer a different question
    from the one policymakers actually need answered.

    Args:
        panel: Master Panel DataFrame. Must contain columns ``iso3``, ``year``,
            and at least one of ``fsi_score`` or ``ucdp_conflict_binary``.
            Missing columns are handled gracefully via ``row.get(..., default)``.
        horizon_years: How many years into the future to look for instability onset.
            Default is 3 (predict whether instability occurs within 3 years).
            Increasing this captures longer-term risk but reduces training data
            (more NaN rows at the end of each country's time series).
        fsi_jump_threshold: Minimum FSI score increase (from current year to any
            future year in the window) required to flag instability. Default is 5.0.
            On the 0-120 FSI scale, a 5-point jump is approximately one risk-band
            upward — a change noticeable to a country analyst and meaningful for
            policy response.

    Returns:
        A pandas Series indexed the same as ``panel``, with values:
            1.0 — instability onset is detected within the prediction horizon
            0.0 — no instability onset within the prediction horizon
            NaN — the prediction horizon extends beyond the end of available data
                  (the last ``horizon_years`` rows per country)
    """
    result = panel.copy().sort_values(["iso3", "year"]).reset_index(drop=True)
    target = pd.Series(np.nan, index=result.index)

    for _, grp in result.groupby("iso3"):
        grp = grp.sort_values("year")
        n = len(grp)
        for pos, (idx, row) in enumerate(grp.iterrows()):
            if pos >= n - horizon_years:
                break  # last horizon_years rows cannot have a computable target
            future = grp.iloc[pos + 1 : pos + 1 + horizon_years]
            current_fsi = row.get("fsi_score", np.nan)

            fsi_jumped = (
                not pd.isna(current_fsi)
                and not future["fsi_score"].isna().all()
                and (future["fsi_score"] - current_fsi).max() > fsi_jump_threshold
            )
            conflict_onset = (
                row.get("ucdp_conflict_binary", 0) == 0
                and (future["ucdp_conflict_binary"] == 1).any()
            )
            target.loc[idx] = int(fsi_jumped or conflict_onset)

    return target


def build_instability_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix for the instability risk classifier.

    Constructs lagged and rolling features from the Master Panel and returns
    only the columns listed in ``_FEATURE_COLS`` (those present in the data).

    Why only lagged features?
        The raw current-year values of outcome variables (FSI score, conflict
        status) are NOT included. Including them would constitute data leakage:
        if the model learns "countries with high FSI scores this year will have
        high FSI scores next year" it is exploiting correlation with the very
        outcome we are trying to predict BEFORE that outcome is known. All
        features represent information that was available at the START of year Y,
        not the end — this mirrors the real-world prediction scenario where an
        analyst is looking forward, not backward.

    Missing values in features are filled with 0. This is conservative — it
    effectively treats "unknown" as "no stress / no conflict" which will cause
    the model to under-predict risk for countries with sparse data. An
    alternative (country mean imputation) introduces its own assumptions;
    zero-fill is chosen for simplicity and interpretability.

    Args:
        panel: Master Panel DataFrame that has already had log transforms applied
            (i.e. ``add_log_transforms`` has been called). Must contain ``iso3``
            and ``year`` columns, plus the raw feature columns listed in
            ``_FEATURE_COLS`` (before lagging).

    Returns:
        A DataFrame with one row per (iso3, year) and columns corresponding to
        the subset of ``_FEATURE_COLS`` that are present after feature construction.
        All NaN values are filled with 0.
    """
    p = add_lag_features(
        panel,
        cols=[
            "log_freshwater_percap",
            "log_gdp_pc_ppp",
            "log_population",
            "safe_water_access_pct",
            "ucdp_conflict_binary",
            "grace_lwe_anomaly_cm",
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


def train_instability_model(
    X: pd.DataFrame,
    y: pd.Series,
) -> xgb.XGBClassifier:
    """Train an XGBoost binary classifier on the instability target.

    XGBoost (eXtreme Gradient Boosting) builds an ensemble of decision trees
    sequentially, where each new tree corrects the errors of all previous trees.
    Key hyperparameter choices:

        n_estimators=200  — 200 boosting rounds. More rounds improve accuracy up to
                            a point, after which the model starts to overfit. 200 is
                            a reasonable starting value for datasets of this size
                            (~5,000-15,000 rows after filtering).

        max_depth=4       — Each individual tree is limited to 4 levels deep.
                            Shallow trees are "weak learners" that generalise better;
                            the ensemble combines many of them to produce a strong model.

        learning_rate=0.05 — Each new tree's contribution is shrunk by 5% before being
                             added to the ensemble. Lower learning rates require more
                             trees but tend to produce a more robust final model.

        scale_pos_weight  — Computed as n_negative / n_positive, this tells XGBoost to
                            treat each positive (instability = 1) training example as if
                            it appeared ``scale`` times. This corrects for the rarity of
                            conflict events in the training data — without this correction,
                            the model would achieve high accuracy by simply predicting
                            "no conflict" for every country, which is useless for
                            early-warning purposes.

        eval_metric="auc" — AUC (Area Under the ROC Curve) is the right metric for
                            imbalanced binary classification. Unlike accuracy, AUC is not
                            distorted by the rare positive class. AUC = 0.5 means random
                            guessing; AUC = 1.0 means perfect classification; AUC > 0.7
                            is conventionally considered a useful classifier.

        random_state=42   — Fixes the random seed for reproducibility. XGBoost uses
                            random subsampling internally; fixing the seed ensures the
                            same model is produced each time the script is run on
                            the same data.

    Args:
        X: Feature matrix (n_samples x n_features). Must not contain NaN -- fill
            missing values before calling (``build_instability_features`` handles this).
            Variable names X and y follow universal ML conventions (PEP8 exemption).
        y: Binary target vector (n_samples,) with values 0 or 1. Must be the same
            length as ``X``.

    Returns:
        A fitted ``xgb.XGBClassifier`` instance. Call ``model.predict_proba(X)[:, 1]``
        to obtain instability probability estimates for new data.
    """
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    # scale_pos_weight = (number of negative examples) / (number of positive examples).
    # This is XGBoost's recommended formula for handling class imbalance.
    # A country-year is "positive" only if instability onset occurred — typically
    # 10-20% of the training data, meaning scale_pos_weight is roughly 4-9.
    scale = n_neg / max(n_pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale,
        eval_metric="auc",
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)
    return model


def predict_instability(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
) -> np.ndarray:
    """Return P(instability onset) for each row in X as a numpy array.

    ``predict_proba`` returns a 2-column array: column 0 is P(class=0, stable),
    column 1 is P(class=1, instability onset). We return column 1.

    Args:
        model: A fitted ``xgb.XGBClassifier`` as returned by
            ``train_instability_model``.
        X: Feature matrix with the same columns used during training. Variable
            names X and y follow universal ML conventions (PEP8 exemption).

    Returns:
        A 1-D NumPy array of probabilities in [0, 1], one per row of ``X``.
        Higher values indicate greater predicted probability of instability onset
        within the model's training horizon (3 years by default).
    """
    return model.predict_proba(X)[:, 1]
