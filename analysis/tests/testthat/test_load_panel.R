library(testthat)

source(here::here("R", "load_panel.R"))

# Structural tests use a small synthetic fixture -- no real data required.
# Runs in CI and locally.
fixture <- here::here("tests", "fixtures", "panel_fixture.csv")

test_that("load_panel returns a data frame", {
  panel <- load_panel(fixture)
  expect_s3_class(panel, "data.frame")
})

test_that("load_panel has iso3 and year columns", {
  panel <- load_panel(fixture)
  expect_true("iso3" %in% names(panel))
  expect_true("year" %in% names(panel))
})

test_that("load_panel year column is integer", {
  panel <- load_panel(fixture)
  expect_true(is.integer(panel$year) || is.numeric(panel$year))
})

test_that("load_panel has the primary exposure variable", {
  panel <- load_panel(fixture)
  expect_true("renewable_freshwater_percap" %in% names(panel))
  expect_true("log_freshwater_percap" %in% names(panel))
})

test_that("load_panel has no duplicate country-year rows", {
  panel <- load_panel(fixture)
  n_dupes <- sum(duplicated(panel[, c("iso3", "year")]))
  expect_equal(n_dupes, 0L)
})

test_that("load_panel coverage for primary exposure is at least 60pct in 1990+ rows", {
  panel <- load_panel(fixture)
  recent <- panel[panel$year >= 1990, ]
  coverage <- mean(!is.na(recent$renewable_freshwater_percap))
  expect_gte(coverage, 0.60)
})
