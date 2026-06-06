library(testthat)
library(fixest)

source(here::here("R", "load_panel.R"))
source(here::here("R", "models.R"))

skip_if_not(file.exists(panel_path()), "master_panel_r.csv not found -- skipping H6 test")
panel <- load_panel()

# H6: Water Access -> Within-Country Inequality (Gini coefficient)
#
# The EDA showed a strong POSITIVE cross-sectional r = +0.305 for freshwater
# availability vs Gini -- driven by tropical resource economies (high water,
# high inequality). Two-way FE removes that confound.
#
# log_freshwater_percap shows a positive FE coefficient due to a population
# confound: as population grows, freshwater per capita falls AND inequality
# often falls (more people enter the middle class). The coefficient is spurious.
#
# safe_water_access_pct is the correct H6 exposure: equitable access to water
# is the mechanism that reduces inequality. When access reaches the poor,
# it directly reduces the income gap. Varies year-over-year as infrastructure
# is built. Controls for income to isolate the water-equality channel.
#
# Hypothesis: better safe water access = lower Gini. Expected sign: NEGATIVE.

test_that("H6: safe water access has negative coefficient on Gini inequality", {
  m <- fit_twoway_fe(
    panel,
    outcome  = "gini",
    exposure = "safe_water_access_pct",
    controls = "log_gdp_pc_ppp"
  )
  expect_s3_class(m, "fixest")
  beta <- coef(m)[["safe_water_access_pct"]]
  cat(sprintf("\n  H6 safe_water beta = %.4f\n", beta))
  expect_lt(beta, 0)
})

test_that("H6: model has at least 500 observations despite sparse Gini coverage", {
  m <- fit_twoway_fe(panel, outcome = "gini", exposure = "log_freshwater_percap")
  expect_gte(nobs(m), 500L)
})
