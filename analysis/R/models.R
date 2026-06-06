library(fixest)

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
