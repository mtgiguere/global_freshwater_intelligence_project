library(fixest)
library(modelsummary)

source(here::here("R", "load_panel.R"))
source(here::here("R", "models.R"))

build_results_table <- function(panel, output_path = NULL) {
  panel$log_refugee_outflow <- log(panel$refugee_outflow + 1)

  models <- list(
    "H1: log(GDP)" = fit_twoway_fe(
      panel, "log_gdp_pc_ppp", "log_freshwater_percap"
    ),
    "H2: Fragility" = fit_twoway_fe(
      panel, "fsi_score", "log_freshwater_percap"
    ),
    "H3: Conflict" = fit_twoway_fe(
      panel, "ucdp_conflict_binary", "log_freshwater_percap"
    ),
    "H4: Life exp." = fit_twoway_fe(
      panel, "life_expectancy", "safe_water_access_pct", "log_gdp_pc_ppp"
    ),
    "H4b: U5MR" = fit_twoway_fe(
      panel, "u5mr", "safe_water_access_pct", "log_gdp_pc_ppp"
    ),
    "H5: log(Refugees)" = fit_twoway_fe(
      panel, "log_refugee_outflow", "log_freshwater_percap", "log_gdp_pc_ppp"
    ),
    "H6: Gini" = fit_twoway_fe(
      panel, "gini", "safe_water_access_pct", "log_gdp_pc_ppp"
    )
  )

  coef_map <- c(
    "log_freshwater_percap" = "log(Freshwater per capita)",
    "safe_water_access_pct" = "Safe water access (%)",
    "log_gdp_pc_ppp"        = "log(GDP per capita)"
  )

  gof_map <- c("nobs", "adj.r.squared", "FE: iso3", "FE: year")

  tbl <- modelsummary(
    models,
    coef_map    = coef_map,
    gof_map     = gof_map,
    stars       = c("*" = 0.1, "**" = 0.05, "***" = 0.01),
    title       = "GFIP Phase 3: Two-Way Fixed Effects Results (Country + Year FE, Clustered SE)",
    notes       = "All models include country and year fixed effects. Standard errors clustered by country.",
    output      = if (is.null(output_path)) "markdown" else output_path
  )

  tbl
}
