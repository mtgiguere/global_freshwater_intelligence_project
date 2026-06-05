# GFIP Phase 3 — Hypothesis Testing (R)

Panel regression and causal inference for hypotheses H1–H7.

## Setup

Install R (≥ 4.3), then from the `analysis/` directory:

```r
install.packages("renv")
renv::restore()   # installs all packages from renv.lock
```

If setting up for the first time (no `renv.lock` yet):

```r
renv::init()
renv::snapshot()
```

## Running tests

```r
devtools::test()
```

## Linting

```r
lintr::lint_package()
```

## Key packages

| Package | Purpose |
|---------|---------|
| `fixest` | Fixed effects panel regression with Driscoll-Kraay SEs |
| `AER` | Instrumental variable estimation |
| `modelsummary` | Publication-quality regression tables |
| `sandwich` | Robust standard errors |
| `lmtest` | Coefficient tests and Hausman test |
| `tidyverse` | Data manipulation and visualisation |
| `arrow` | Read Master Panel parquet files from Python pipeline |
| `broom` | Tidy model output |

## TDD in R

The same strict TDD discipline applies here as in Python. Use `testthat`:

```r
# 1. Write the test in tests/testthat/test_<topic>.R
test_that("two-way FE controls for country and year", {
  # arrange — synthetic panel
  # act
  # assert
})

# 2. Run it — confirm RED
devtools::test()

# 3. Write minimum implementation in R/
# 4. Run it — confirm GREEN
```

All analytical functions in `R/` must have corresponding tests in `tests/testthat/`.
Scripts that produce outputs (tables, figures) are not tested directly — but any
helper function they call must be.
