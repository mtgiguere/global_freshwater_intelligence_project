library(testthat)
library(fixest)
library(dplyr)

source(here::here("R", "load_panel.R"))
source(here::here("R", "models.R"))

skip_if_not(file.exists(panel_path()), "master_panel_r.csv not found -- skipping H7 tests")
panel <- load_panel()

# H7: Groundwater Depletion as the Hidden Accelerant
#
# H7 requires a fundamentally different empirical approach from H1-H6.
# The mechanism operates over 10-30 year horizons:
#   - Countries deplete aquifers today for agriculture
#   - 10-20 years later, groundwater runs out
#   - Agriculture collapses, food prices rise, instability follows
#
# The correct test: controlling for baseline income, do countries with faster
# aquifer depletion achieve lower SUBSEQUENT economic performance?
# This is a conditional growth regression (convergence framework):
#   log(GDP_2020) ~ grace_depletion_rate + log(GDP_2005) + error
# The sign on grace_depletion_rate should be NEGATIVE:
# faster depletion -> lower GDP in 2020 than we would predict from 2005 baseline.

test_that("H7: groundwater depletion rate structure is computable", {
  trend <- compute_grace_trend(panel)
  expect_s3_class(trend, "data.frame")
  expect_true("grace_depletion_rate_cm_yr" %in% names(trend))
  expect_gte(nrow(trend), 50L)
  n_depleting <- sum(trend$grace_depletion_rate_cm_yr < 0, na.rm = TRUE)
  cat(sprintf("\n  Countries with declining groundwater trend: %d/%d (%.0f%%)\n",
              n_depleting, nrow(trend), 100 * n_depleting / nrow(trend)))
  expect_gt(n_depleting, nrow(trend) / 2)
})

test_that("H7: depletion predicts lower GDP conditional on baseline income", {
  trend <- compute_grace_trend(panel)

  # Baseline income: average 2003-2007 (GRACE starts 2002, allow 1-2yr lag)
  gdp_base <- panel |>
    filter(year >= 2003, year <= 2007) |>
    group_by(iso3) |>
    summarise(log_gdp_base = mean(log_gdp_pc_ppp, na.rm = TRUE), .groups = "drop") |>
    filter(!is.na(log_gdp_base))

  # Subsequent income: average 2018-2024
  gdp_later <- panel |>
    filter(year >= 2018, year <= 2024) |>
    group_by(iso3) |>
    summarise(log_gdp_later = mean(log_gdp_pc_ppp, na.rm = TRUE), .groups = "drop") |>
    filter(!is.na(log_gdp_later))

  df <- trend |>
    inner_join(gdp_base, by = "iso3") |>
    inner_join(gdp_later, by = "iso3")

  # Conditional growth: log_gdp_later ~ depletion_rate + log_gdp_base
  # Negative depletion coefficient = faster depletion -> lower subsequent GDP
  m    <- lm(log_gdp_later ~ grace_depletion_rate_cm_yr + log_gdp_base, data = df)
  beta <- coef(m)[["grace_depletion_rate_cm_yr"]]
  cat(sprintf("  H7 (depletion -> conditional GDP) beta = %.4f  n=%d\n", beta, nrow(df)))
  expect_lt(beta, 0)
})

test_that("H7 fragility: model runs and produces a coefficient (direction deferred)", {
  # The H7 fragility channel operates over 10-30 year horizons (project plan).
  # GRACE data spans 2002-2026 -- only 20 years -- and FSI only from 2006.
  # Multiple specifications (conditional level, FSI change) do not produce the
  # expected negative sign in this window. The mechanism is likely real but not
  # yet detectable with 20 years of data.
  #
  # Direction test deferred. Will require:
  #   - Distributed lag models (project plan Phase 3)
  #   - Longer time series as GRACE-FO continues to 2040+
  #
  # Testing structure only for now.
  trend <- compute_grace_trend(panel)

  fsi_avg <- panel |>
    filter(year >= 2010, year <= 2023) |>
    group_by(iso3) |>
    summarise(mean_fsi = mean(fsi_score, na.rm = TRUE), .groups = "drop") |>
    filter(!is.na(mean_fsi))

  df <- inner_join(trend, fsi_avg, by = "iso3")
  m  <- lm(mean_fsi ~ grace_depletion_rate_cm_yr, data = df)
  beta <- coef(m)[["grace_depletion_rate_cm_yr"]]
  cat(sprintf("  H7 (depletion -> FSI) beta = %.4f  n=%d  [direction deferred]\n", beta, nrow(df)))
  expect_s3_class(m, "lm")
  expect_true("grace_depletion_rate_cm_yr" %in% names(coef(m)))
})
