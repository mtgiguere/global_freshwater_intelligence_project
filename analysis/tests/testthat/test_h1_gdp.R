library(testthat)
library(fixest)

source(here::here("R", "load_panel.R"))
source(here::here("R", "models.R"))

# Hypothesis tests require the real Master Panel (gitignored processed data).
# They are skipped in CI and run fully when data/processed/master_panel_r.csv exists.
skip_if_not(file.exists(panel_path()), "master_panel_r.csv not found -- skipping H1 tests")
panel <- load_panel()

test_that("fit_twoway_fe returns a fixest object", {
  m <- fit_twoway_fe(panel, outcome = "log_gdp_pc_ppp", exposure = "log_freshwater_percap")
  expect_s3_class(m, "fixest")
})

test_that("fit_twoway_fe includes country and year fixed effects", {
  m <- fit_twoway_fe(panel, outcome = "log_gdp_pc_ppp", exposure = "log_freshwater_percap")
  expect_true("iso3" %in% names(fixef(m)))
  expect_true("year" %in% names(fixef(m)))
})

test_that("fit_twoway_fe exposure coefficient is in the model", {
  m <- fit_twoway_fe(panel, outcome = "log_gdp_pc_ppp", exposure = "log_freshwater_percap")
  coefs <- coef(m)
  expect_true("log_freshwater_percap" %in% names(coefs))
})

test_that("fit_twoway_fe produces a positive H1 coefficient after FE controls", {
  # Within-country: more freshwater in a given year should associate with higher GDP.
  # The cross-sectional bivariate r was negative due to oil-country confounding.
  # Two-way FE removes that confound — the within-country coefficient should be positive.
  m <- fit_twoway_fe(panel, outcome = "log_gdp_pc_ppp", exposure = "log_freshwater_percap")
  beta <- coef(m)[["log_freshwater_percap"]]
  expect_gt(beta, 0)
})

test_that("fit_twoway_fe uses at least 1000 observations for H1", {
  m <- fit_twoway_fe(panel, outcome = "log_gdp_pc_ppp", exposure = "log_freshwater_percap")
  expect_gte(nobs(m), 1000L)
})
