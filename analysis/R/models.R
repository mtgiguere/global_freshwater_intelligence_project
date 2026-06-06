library(fixest)
library(dplyr)

#' Compute per-country groundwater depletion rate (cm LWE per year) from GRACE data.
#'
#' Fits a simple linear trend (lm) to grace_lwe_anomaly_cm vs year per country.
#' Negative slope = aquifer depletion. Positive slope = recharge.
#'
#' @param panel Data frame containing the master panel.
#' @return Data frame with iso3 and grace_depletion_rate_cm_yr columns.
compute_grace_trend <- function(panel) {
  panel |>
    filter(!is.na(grace_lwe_anomaly_cm)) |>
    group_by(iso3) |>
    filter(n() >= 5) |>  # require at least 5 years of data
    summarise(
      grace_depletion_rate_cm_yr = coef(lm(grace_lwe_anomaly_cm ~ year))[["year"]],
      n_grace_years = n(),
      .groups = "drop"
    )
}

#' Fit a two-way fixed effects panel model.
#'
#' Controls for all time-invariant country characteristics (country FE)
#' and all country-invariant year shocks (year FE). This isolates the
#' within-country, within-year variation in the exposure variable.
#'
#' @param panel Data frame containing the master panel.
#' @param outcome Character. Name of the outcome column (e.g. "log_gdp_pc_ppp").
#' @param exposure Character. Name of the exposure column (e.g. "log_freshwater_percap").
#' @param controls Character vector of additional control variable names.
#' @return A fixest model object.
fit_twoway_fe <- function(panel, outcome, exposure, controls = character(0)) {
  rhs <- paste(c(exposure, controls), collapse = " + ")
  formula_str <- sprintf("%s ~ %s | iso3 + year", outcome, rhs)
  fml <- as.formula(formula_str)
  feols(fml, data = panel, cluster = ~iso3)
}
