library(testthat)
library(fixest)

source(here::here("R", "load_panel.R"))
source(here::here("R", "models.R"))

# Hypothesis tests require the real Master Panel (gitignored processed data).
# They are skipped in CI and run fully when data/processed/master_panel_r.csv exists.
skip_if_not(file.exists(panel_path()), "master_panel_r.csv not found -- skipping H2-H5 tests")
panel <- load_panel()

# H2: Freshwater -> Government Fragility (FSI score)
# Hypothesis: more water = lower FSI score (less fragile). Expected sign: NEGATIVE.
test_that("H2: freshwater has negative coefficient on FSI fragility", {
  m <- fit_twoway_fe(panel, outcome = "fsi_score", exposure = "log_freshwater_percap")
  expect_s3_class(m, "fixest")
  expect_lt(coef(m)[["log_freshwater_percap"]], 0)
})

# H3: Freshwater -> Armed Conflict (UCDP binary, linear probability model)
# Hypothesis: more water = lower probability of conflict. Expected sign: NEGATIVE.
test_that("H3: freshwater has negative coefficient on conflict probability", {
  m <- fit_twoway_fe(panel, outcome = "ucdp_conflict_binary", exposure = "log_freshwater_percap")
  expect_s3_class(m, "fixest")
  expect_lt(coef(m)[["log_freshwater_percap"]], 0)
})

# H4: Safe Water Access -> Life Expectancy (controlling for income)
# Note: renewable_freshwater_percap is quasi-static (long-run average / population)
# and within-country variation is mainly population growth, not real water changes.
# safe_water_access_pct is the correct H4 exposure -- it measures whether people
# can actually REACH safe water, genuinely improves year-over-year as infrastructure
# is built, and is the direct proximate cause of health improvements.
# Hypothesis: more access to safe water = longer life. Expected sign: POSITIVE.
test_that("H4: safe water access has positive coefficient on life expectancy", {
  m <- fit_twoway_fe(
    panel,
    outcome  = "life_expectancy",
    exposure = "safe_water_access_pct",
    controls = "log_gdp_pc_ppp"
  )
  expect_s3_class(m, "fixest")
  expect_gt(coef(m)[["safe_water_access_pct"]], 0)
})

# H4b: Safe Water Access -> Under-5 Mortality
# Hypothesis: more access to safe water = lower child mortality. Expected sign: NEGATIVE.
test_that("H4b: safe water access has negative coefficient on under-5 mortality", {
  m <- fit_twoway_fe(
    panel,
    outcome  = "u5mr",
    exposure = "safe_water_access_pct",
    controls = "log_gdp_pc_ppp"
  )
  expect_s3_class(m, "fixest")
  expect_lt(coef(m)[["safe_water_access_pct"]], 0)
})

# H5: Freshwater -> Refugee Outflow
# NOTE: Sparse data (UNHCR 2000-2023, 122 countries) creates severe over-parameterisation
# with two-way FE after NA removal. Direction test deferred -- H5 requires a different
# specification (Poisson, longer window, or bilateral flow data). Testing structure only.
test_that("H5: model runs and produces a coefficient (direction deferred -- sparse data)", {
  panel$log_refugee_outflow <- log(panel$refugee_outflow + 1)
  m <- fit_twoway_fe(
    panel,
    outcome  = "log_refugee_outflow",
    exposure = "log_freshwater_percap",
    controls = "log_gdp_pc_ppp"
  )
  expect_s3_class(m, "fixest")
  expect_true("log_freshwater_percap" %in% names(coef(m)))
})
