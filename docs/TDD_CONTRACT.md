# The TDD Contract
## Why We Do Test-Driven Development — Evidence From This Codebase

This document exists because of a deliberate experiment.

During the development of the Geo-Fluid Dynamics Engine, Claude was given significant
autonomy and explicitly allowed to use TAD (Test-After Development) instead of TDD.
The results were analyzed in a systematic retrospective. What follows is the evidence —
not theory, not best practice citations, but specific bugs, specific tests, and specific
lines of code from this project that prove why TDD is not optional.

**If you are a new Claude Code session starting work on this project: read this first.**
**If you are inclined to skip TDD "just this once": re-read this.**

---

## What TAD Looks Like (What We Did Wrong)

TAD means: design in your head → write implementation → write tests that confirm
what you just wrote → push.

The tests look fine. They pass. Coverage looks good. But something is wrong:
**the tests are asking "does this code do what I just wrote?" not "does this code do
what it SHOULD do?"** That is a completely different question.

Here is the evidence from this project.

---

## Bug #1: Column Name Drift (adoption_prob → adoption_probability)

**What happened:**

`SurvivalDiffusionModel.get_adoption_frontier()` was implemented and returned a column
named `adoption_prob`. Separately, `frontier_map.build_layer()` was implemented expecting
a column named `adoption_probability`. Both were tested individually and both passed.

When CI wired them together via the API test, it crashed:
```
ValueError: Missing required columns: ['on_frontier', 'adoption_probability']
```

**Why TAD caused it:**

I wrote `get_adoption_frontier()` and tested it knowing I named the column `adoption_prob`.
I wrote `frontier_map.build_layer()` and tested it knowing I expected `adoption_probability`.
Both tests confirmed what I just coded. Neither test specified the contract between them.

**What TDD would have done:**

Write this test first, before implementing either function:
```python
def test_frontier_output_is_consumable_by_frontier_map():
    frontier = model.get_adoption_frontier()
    # This test specifies the contract between two functions
    result = frontier_map.build_layer(frontier)
    assert result["type"] == "FeatureCollection"
```

That test would fail immediately on the first implementation attempt because the column
name would be wrong. The contract would be specified before either function was written.

---

## Bug #2: Empty DataFrame KeyError (node_classifier)

**What happened:**

`classify_nodes()` was called with an empty adoption_events DataFrame.
`pd.DataFrame([])` with an empty list produces a DataFrame with **zero columns**, not
zero rows with the expected columns. So `adoption_events["topic_id"]` raised `KeyError`.

CI output:
```
KeyError: 'topic_id'
modules/gravity_engine/processing/node_classifier.py:87
```

**Why TAD caused it:**

I wrote `_compute_county_stats()` and tested it with non-empty adoption_events.
The happy path worked. I never asked "what does an empty adoption_events look like?"
because I was thinking about the implementation I just wrote, not the contract.

**What TDD would have done:**

Before writing a single line of `_compute_county_stats()`, write this test:
```python
def test_classify_nodes_with_no_adopters_for_topic():
    """Topic 2 has no adopters — function must not crash."""
    ae = _make_adoption_events({}, topic_id=2)  # empty dict → empty DataFrame
    result = classify_nodes(panel, out_c, in_c, fips_index, ae, topic_id=2)
    assert result["node_type"].isin(NODE_TYPES).all()
```

Watch it fail. Then write the guard. The bug never ships.

---

## Bug #3: FIPS Sort Order (The Comment That Lied)

**What happened:**

In `test_spatial_weights.py`, the test comment read:
```python
fips_order = sorted(panel["fips"].unique())
sup_vec = np.array(support)   # ordered: 29169, 29183, 29189 alphabetically
```

The comment said alphabetically. The code used `support` which was in **insertion order**
`[29169, 29189, 29183]`. The test had been wrong since it was written. CI found it:

```
assert np.float64(0.15000000000000002) < 1e-06
```

**Why TAD caused it:**

I wrote the test knowing the implementation sorts alphabetically. I wrote the assertion
knowing what I intended. I wrote the comment to remind myself. Then I wrote the wrong code
in the assertion because I was thinking about `support` in the order I created it, not in
the order the implementation would use it.

**What TDD would have done:**

Write the test first with an explicit mapping that makes the expected output unambiguous:
```python
fips_to_support = {"29169": 0.3, "29189": 0.6, "29183": 0.9}
fips_order = sorted(panel["fips"].unique())
sup_vec = np.array([fips_to_support[f] for f in fips_order])
```

The explicit mapping cannot lie. The comment can.

---

## Bug #4: The np.True_ Identity Check

**What happened:**

Three tests in `test_rolloff.py` failed with:
```
assert np.True_ is True   →   AssertionError
assert np.False_ is False →   AssertionError
```

**Why TAD caused it:**

I was thinking about boolean logic when I wrote the tests. I knew the implementation
would produce True/False values. I used `is True` because that's what you write when
you're thinking "this should be True." I wasn't thinking about what pandas actually
returns — numpy scalars, not Python singletons.

**What TDD would have done:**

If you write the test first and actually RUN it to confirm it fails (the RED step),
you immediately discover that the implementation doesn't exist yet, and you think carefully
about what the assertion syntax should be. The RED step forces you to think about the
assertion before the implementation.

The correct pattern — never use `is True` or `is False` with pandas/numpy values:
```python
assert result["is_false_bastion"]        # truthiness check — works with np.bool_
assert not result["is_false_bastion"]
assert result["is_false_bastion"] == True  # value equality — not identity
```

---

## Bug #5: Eight Hollow Tests That Cannot Fail

**What happened:**

The retrospective analysis found 8 tests with conditional guards that let the
core assertion bypass entirely. The test runs, it passes, coverage is counted.
But the thing being tested is never actually verified.

Example:
```python
def test_wombling_value_equals_support_difference():
    """With support [0.8, 0.3], wombling_value should be 0.5."""
    ...
    if len(result) > 0:           # ← hollow guard
        assert abs(result.iloc[0]["wombling_value"] - 0.5) < 1e-6
```

If `result` is empty — because the adjacency detection has a bug — this test passes.
The bug ships. The test reported 100% on this line.

**Why TAD caused it:**

I knew the implementation sometimes returns empty results for edge cases. I added the
guard to avoid flaky tests. But I was protecting the implementation from the test,
which is exactly backwards. Tests are supposed to catch bugs. Guards prevent that.

**What TDD would have done:**

Write the test first. It specifies that for two adjacent polygons with supports 0.8
and 0.3, the output must be non-empty and must have wombling_value ≈ 0.5. Full stop.
```python
def test_wombling_value_equals_support_difference():
    gdf = _make_adjacent_counties()
    W = np.array([[0, 1], [1, 0]], dtype=float)
    support = pd.Series({"A": 0.8, "B": 0.3})
    result = lattice_wombling(gdf, support, W, ["A", "B"])
    assert len(result) == 1, "Two adjacent counties must produce one boundary"
    assert abs(result.iloc[0]["wombling_value"] - 0.5) < 1e-6
```

No guard. If the function doesn't detect the boundary, the test fails loudly.

---

## What 100% Coverage Actually Means

Coverage tells you which lines were **executed**. It does not tell you which behaviors
were **verified**.

Example: `compute_ifs(0.5, 0.5)` achieves 100% line coverage of `compute_ifs`.
It does not test:
- `compute_ifs(float('nan'), 0.5)` → returns NaN silently
- `compute_ifs(-0.1, 0.5)` → `np.clip` handles it, but was that specified?
- `compute_ifs(0.0, 0.5)` → geometric mean property, verified?
- `compute_ifs(1.0, 1.0)` → what does full-danger look like?

86% coverage with TAD ≠ 86% of behavior verified.
70% coverage with TDD > 86% coverage with TAD.

The number that matters is: **what fraction of the behavioral contract is specified
in tests before the code is written?**

---

## The Actual TDD Discipline — Not Theory, Instructions

### The Sequence (Non-Negotiable)

```
1. Write one test. It must describe behavior you want, not code you will write.
2. Run the test. CONFIRM IT FAILS (RED). If it passes, the test is wrong.
3. Write the minimum code to make it pass. Not more.
4. Run the test. CONFIRM IT PASSES (GREEN).
5. Refactor if needed. Run tests again.
6. Commit. The commit message describes the behavior added, not the code written.
7. Repeat.
```

The RED step is not optional. If you skip it, you are doing TAD.

### Pre-Commit — Run the Linters Locally First

CI catching a linter error is not a minor inconvenience. It means you pushed
broken code, wasted pipeline minutes, and added a noise commit ("fix lint") to
the history. The fix is simple: run the linters before committing, not after.

```bash
# Before every commit — Python
uv run ruff check .
uv run pytest --no-header -q

# Before every commit — R (when analysis/ files changed)
Rscript --vanilla -e "lintr::lint_package()"
```

If any of these fail, do not commit. Fix the issue first.

### Comments Are for Every Reader — Not Just Developers

This project is deliberately public. Its audience is not only software engineers.
It includes policymakers, researchers, journalists, students, and anyone who cares
about freshwater and human welfare. The code should be legible to a motivated
non-developer who is willing to read carefully.

**This means: long, explanatory comments are a feature, not a code smell.**

When implementing scientific methodology — a hypothesis test, a model specification,
a data transformation decision — write a comment that explains:
- What the code is doing in plain language
- WHY this approach was chosen
- What the expected result means
- What the limitation is
- What alternative was considered and rejected

```r
# H7 requires a fundamentally different empirical approach from H1-H6.
# The mechanism operates over 10-30 year horizons:
#   - Countries deplete aquifers today for agriculture
#   - 10-20 years later, groundwater runs out
#   - Agriculture collapses, food prices rise, instability follows
#
# The correct test: controlling for baseline income, do countries with faster
# aquifer depletion achieve lower SUBSEQUENT economic performance?
```

This is not over-commenting. This is open science. A UN analyst, a journalist,
or a student should be able to read this and understand what we are testing and why.

**Linters that fight this goal should be configured away — not obeyed.**

`commented_code_linter` (R) is disabled in `.lintr` because it would flag
methodological commentary as violations. The linter serves the project, not the
other way around. When a lint rule conflicts with the project's purpose, disable
the rule and document why.

### Red Flag Signals — Stop and Fix Before Proceeding

If you find yourself writing any of the following, you are not doing TDD:

```python
# RED FLAG 1: Skip guard without confirmed implementation
pytest.skip("not implemented")  # wrong — this is the TDD stub; DON'T add the code yet

# RED FLAG 2: Conditional guard hiding assertion
if len(result) > 0:
    assert something  # wrong — design fixture so result is always non-empty

# RED FLAG 3: Seed-specific assertion
rng = np.random.default_rng(42)  # wrong — use property-based testing for algorithms
assert some_trend_holds()         #        or construct data so the property is guaranteed

# RED FLAG 4: Identity check with pandas/numpy
assert result["col"] is True     # wrong — always
assert result["col"] is False    # wrong — always

# RED FLAG 5: Test name that describes implementation
def test_calls_psycopg2_connect():  # wrong — tests describe behavior not mechanism
def test_uses_groupby():

# RED FLAG 6: Test written after the function it tests
# (you wrote parse(), then wrote test_parse() — that is TAD)
```

### What a Good Test Looks Like

A good test is written **before the implementation** and specifies:
1. Given this specific, deterministic input
2. When this function is called
3. Then this exact output is produced (or this exact error is raised)

```python
def test_adoption_year_is_first_crossing():
    """Specifies: the adoption year is the FIRST year support crosses the threshold.
    Not the last. Not any year. The first."""
    support = pd.DataFrame({
        "fips": ["29169"] * 4,
        "topic_id": [1] * 4,
        "year": [2012, 2016, 2020, 2024],
        "support_pct": [0.40, 0.48, 0.52, 0.58],
    })
    events = compute_adoption_events(support, topic_id=1, threshold_pct=0.50)
    row = events[events["fips"] == "29169"].iloc[0]
    assert row["first_adoption_year"] == 2020  # not 2024; not 2016
```

This test was written before the implementation. It fails before the implementation.
It specifies exactly one thing. It has no guards. It has no seeds. It has one expected value.

### Using Hypothesis for Algorithmic Code

For mathematical functions (IFS, CCI, spatial weights, survival probability),
seed-based testing is always wrong. Use Hypothesis:

```python
from hypothesis import given, strategies as st

@given(
    cci=st.floats(0.0, 1.0, allow_nan=False),
    ili=st.floats(0.0, 1.0, allow_nan=False),
)
def test_ifs_always_in_0_1(cci, ili):
    """Property: IFS is always in [0, 1] for valid inputs."""
    assert 0.0 <= compute_ifs(cci, ili) <= 1.0
```

Hypothesis will find the edge cases you didn't think of.

### Property Tests Emerge Naturally From Strict TDD

You do not always need Hypothesis to write a property test. When you ask
"what must be true about this output regardless of the specific input?" — you are
already thinking in properties. Strict TDD surfaces this naturally.

During GFIP Phase 1 GRACE ingest development, the spatial aggregation test was written as:

```python
def test_load_grace_area_weighted_mean_of_constant_equals_that_constant():
    """Property: area-weighted mean of a spatially uniform field must equal the field value.
    This holds regardless of country shape or latitude.
    """
    for value in [-3.5, 0.0, 2.8]:
        ds = _make_dataset(value=value)
        result = load_grace(ds, shapes)
        assert abs(result.iloc[0]["grace_lwe_anomaly_cm"] - value) < 1e-6
```

This test was not designed. It emerged from asking "what must always be true about
area-weighted mean?" — the answer is: a uniform field must return the field value.
That property is independent of grid resolution, country shape, or latitude.

If the cos(lat) weights were wrong, this test would catch it for any input value.
A seed-based test (`assert result == 1.847...`) would only catch it for that one case.

**The pattern:** When testing mathematical or spatial functions, ask:
- What invariant must hold for all valid inputs?
- What relationship must be preserved regardless of the specific values?
- What property would be violated if my algorithm is wrong?

That question — not the specific expected output — is the test. It is more powerful
than any hand-calculated expected value, and it emerges naturally from strict TDD
because strict TDD forces you to think about behavior before implementation.

### When a Test Is Immediately GREEN — That Is Also Information

During GFIP Phase 1 development, two tests passed without driving any code change:
- Year column is integer dtype (pivot preserves int64 automatically)
- Tiny country with no grid cells gets NaN (weighted mean of empty mask = NaN naturally)

When a test you write is immediately GREEN, it means one of two things:
1. The behavior was already guaranteed by your implementation choice — the test
   is still valuable as a regression guard, confirming the guarantee is real.
2. The test is redundant — it confirms something another test already covers.

In strict TDD, an immediately GREEN test is not a failure. It is the process telling
you something about your implementation that you could not have known without running
it. Write the test. See GREEN. Note the reason. Move on.

---

## The Conversation You Will Have

At some point you will think:

> "This is a simple function. I know exactly what it does. Writing the test first
> is just busywork. I'll write the test after — it'll be faster."

That is exactly what happened in this project. Every time. The function that felt
simple had a bug. The test written after documented the bug. CI found it later.

The bugs above were not from complex functions. They were from:
- A column being named `adoption_prob` instead of `adoption_probability`
- An empty DataFrame having zero columns instead of zero rows
- A comment that said "alphabetically" while the code did insertion order
- A boolean being `np.True_` instead of `True`

None of these required deep thinking to get right. All of them required writing the
test first so the contract was specified before the code was written.

---

## The Standing Instruction for This Project

**For every new function added to this codebase:**

1. Write the test in the test file. Run pytest. Confirm RED.
2. Write the implementation. Run pytest. Confirm GREEN.
3. Then and only then, open a PR.

**For every bug fix:**
1. Write a test that reproduces the bug. Confirm RED.
2. Fix the bug. Confirm GREEN.
3. The test stays in the suite permanently.

**For every edge case you think of while implementing:**
1. Stop implementing. Write the edge case test first.
2. Run it. Confirm RED (or discover the implementation already handles it).
3. Continue.

This is not about discipline or methodology. This is about the specific, documented
bugs in this codebase that would not exist if we had done this.

---

## Summary of Evidence

| Bug | How Found | TDD Prevention |
|-----|-----------|----------------|
| `adoption_prob` vs `adoption_probability` | CI failure after PR merge | Interface contract test written before either function |
| Empty DataFrame KeyError in node_classifier | CI failure | Test with empty adoption_events before implementation |
| FIPS sort order mismatch | CI failure | Explicit mapping in test rather than `np.array(support)` |
| `np.True_ is True` assertion failure | CI failure | RED step forces you to think about assertion syntax |
| 8 hollow tests with conditional guards | Manual retrospective | No guards — design fixture to guarantee non-empty result |
| 20+ seed-dependent assertions | Manual retrospective | Hypothesis property tests |
| Column naming inconsistency (node types) | Manual retrospective | Enum defined in test file before any implementation |

None of these were caught by the tests. They were caught by CI or by a human reviewing.
**Tests that don't catch bugs are documentation, not verification.**

---

*This document was written after the fact as an honest retrospective.*
*Its purpose is to prevent future sessions from repeating the same patterns.*
*The evidence in it is real. The bugs were real. The fixes were real.*
*Do the work in the right order.*

---

## Just-In-Time Programming — The Other Half of the Discipline

TDD tells you *how* to build. Just-In-Time (JIT) programming tells you *what* to build
and *when*. Together they are the same discipline from two angles.

**The JIT rule:**

> Write only the code that a currently failing test demands.
> Do not write code for needs that do not yet exist.

This sounds obvious. It is not practiced. Here is what violating it looks like:

```python
# VIOLATION: writing a helper "because it might be useful later"
def filter_variables(df, variables):   # no test demands this as a public function
    ...                                # it exists because I imagined a future caller

def validate_schema(df):               # same — planned upfront, not test-driven
    ...
```

Both functions were written during the GFIP AQUASTAT batch approach because the
developer anticipated they would be needed. No test demanded them as public functions.
The strict TDD pass eliminated both — not because they were wrong, but because they
were *premature*.

**The tell:**

If you are about to write a function and you cannot point to a currently failing test
that demands it — stop. The function does not belong yet.

When the need arrives, a test will fail. That failing test is your permission to write
the function. Not before.

**Why this matters:**

Every speculative public function is API surface that has to be maintained, tested,
documented, and kept consistent with the rest of the codebase. Speculative functions
that never get called are pure cost. Speculative functions that *do* get called but
were designed for an imagined use case often get called *wrong*.

JIT is not laziness. It is the discipline of trusting that the tests will tell you
what to build, in the order you need to build it.

**The connection to TDD:**

TDD enforces JIT automatically. You cannot write speculative code and have it be tested,
because the test is supposed to come first. If you find yourself writing code before the
test, you are violating both TDD and JIT simultaneously.

The sequence "write test → write minimum code → repeat" *is* just-in-time programming.
The "minimum code" step is JIT: not one line more than the failing test requires.

**A RED FLAG specific to JIT violations:**

```python
# RED FLAG 7: Function with no failing test demanding it
def _compute_auxiliary_stats(df):   # who called this? what test failed without it?
    ...

# RED FLAG 8: Public function that is only called by one private function
def parse_raw_csv(path):            # if only load_aquastat() calls this, it should
    ...                             # be private or inlined — no external test demands it
```

---

## What We Learned: Batch TDD vs Strict TDD

This section documents a deliberate experiment run during GFIP Phase 1 development.
The same module (AQUASTAT ingest) was written twice: once with batch TDD ("tests-first
design") and once with strict TDD (one test, RED, minimum code, GREEN, repeat).

### The experiment

**Batch approach:** All 17 tests written at once. RED confirmed once as a batch (ImportError).
Full implementation written at once. GREEN confirmed.

**Strict TDD:** One test at a time. RED confirmed individually. Minimum code to pass that
one test. GREEN confirmed. Next test.

### What changed

| | Batch | Strict TDD |
|---|---|---|
| Public functions | 5 | 1 |
| Lines of implementation | 52 | 27 |
| Tests | 17 | 8 |
| Branch coverage | 99% | 100% |

### Why the designs diverged

**1. Starting from consumer behavior collapses the API.**

Batch approach started from "what functions do I need?" and produced 5 public functions:
`parse_raw_csv`, `filter_variables`, `pivot_to_wide`, `map_country_codes`, `validate_schema`.

Strict TDD started from "what does the consumer want?" and produced 1 public function:
`load_aquastat`. The internal steps became private implementation details.

The consumer cannot call the pipeline steps in the wrong order. They cannot forget to call
`validate_schema`. The function guarantees its output is valid. The batch API could not.

**2. Error handling location is driven by the test, not by habit.**

Batch approach: `map_country_codes` returned NaN silently. `validate_schema` caught it
downstream. Two public functions the consumer had to remember to chain.

Strict TDD: The test said "load_aquastat raises if any country cannot be mapped." So the
check lives inside `load_aquastat`. Fail-fast, co-located with the failure, impossible to skip.

**3. Some planned functions never needed to exist.**

`validate_schema` was a public function in the batch approach because it was planned upfront.
No test ever demanded it as a public function. Strict TDD never created it.

**4. One test was immediately GREEN — and that is information.**

The year-is-integer test passed without any code change. In the batch approach this could not
be known, so a defensive cast was written anyway. Strict TDD revealed the cast was unnecessary.
When a test you write is immediately GREEN, the behavior was already guaranteed. This is not
a failure of TDD — it is TDD giving you information about your implementation.

**5. Fewer tests, but higher quality.**

17 tests → 8 tests. The batch approach tested each internal function separately. When those
functions are private, those tests verify implementation mechanics, not behavior. The 8 strict
TDD tests each verify one consumer-visible behavior. All 8 drove a code change or confirmed
a behavioral guarantee.

### The lesson

> "Tests-first design" produces the design you planned.
> Strict TDD produces the design the behavior demands.
>
> They are not the same design. The strict TDD design is simpler, better encapsulated,
> and has fewer failure modes — not because the developer was smarter, but because
> each test forced the question: "what is the minimum interface that satisfies this
> one behavior?" The answer is always simpler than what you planned.

---

## Methodological Notes — Data Limitations Discovered During EDA

These are not engineering bugs. They are scientific observations about the data that
shape how the Phase 3 analysis must be designed and interpreted.

### Annual Averages Mask Seasonal Water Stress

**Discovery (Phase 2 EDA):**
The primary exposure variable — `renewable_freshwater_percap` from AQUASTAT — is an
annual average. This makes countries like India, Pakistan, and the Sahel *appear*
adequately watered when they are effectively bone-dry for 6–9 months of the year.

**The mechanism:**
Monsoon climates receive the majority of annual precipitation in a 60–90 day window.
When the monsoon arrives, the ground is baked hard from months of heat. Water runs off
as floods rather than infiltrating to recharge aquifers. The annual "total" looks fine.
The lived reality is severe seasonal scarcity for most of the year.

**Why this matters for each hypothesis:**
- H1–H4: Cross-sectional correlations between freshwater and human outcomes are
  artificially weakened because the annual average misclassifies seasonally-arid
  countries as water-secure.
- H7 (groundwater): Monsoon countries are among the heaviest aquifer extractors
  precisely *because* surface water is seasonally unreliable. GRACE data reveals
  the depletion that annual averages hide.

**What this means for Phase 3:**
1. Annual freshwater per capita is a necessary but insufficient exposure variable.
2. Fixed effects panel regression partially addresses this by controlling for
   time-invariant country characteristics (including climate type).
3. The SPEI (Standardised Precipitation-Evapotranspiration Index) from CMIP6/WorldClim
   data captures drought duration and intensity at monthly resolution and should be
   included as a supplementary exposure variable in Phase 3 models.
4. A `seasonal_aridity_flag` (see below) should be computed and used as a moderating
   variable — effects of freshwater stress may be stronger in seasonally arid countries.

### The Seasonal Aridity Flag — Plan

**Definition:** A country-year is seasonally arid if it experiences more than 6 months
per year with average precipitation below 50mm (roughly 1.6mm/day — the standard
meteorological dry month threshold).

**Data source:** WorldClim 2.1 — 30-year average monthly precipitation at 2.5 arc-minute
resolution, available as free GeoTIFF downloads at worldclim.org.

**Implementation plan:**
1. Download 12 monthly precipitation GeoTIFFs from WorldClim 2.1.
2. Aggregate each raster to country level (area-weighted mean, same approach as GRACE).
3. For each country: count months where monthly_precip_mm < 50.
4. Output variables:
   - `dry_months_count` — integer 0–12, number of months below threshold
   - `seasonal_aridity_flag` — boolean, True if dry_months_count > 6
5. Add both to the Master Panel as static country-level features (WorldClim is a
   long-term climatological average, not time-varying — join on iso3 only, not year).

**Expected findings:**
Countries flagged: most of MENA, Sahel, Horn of Africa, Pakistan, northwestern India,
Central Asia, northern Mexico, southwestern USA, parts of Brazil (Nordeste), Australia
(interior). These are exactly the countries where the annual freshwater average
misrepresents lived water scarcity.

**Phase 3 moderation analysis:**
Run regressions separately for `seasonal_aridity_flag == True` and `== False`.
The hypothesis is that H1–H5 effect sizes will be significantly larger in the
seasonally-arid group — because in these countries, a reduction in annual freshwater
represents a reduction in an already marginal and unreliable supply, not just a
reduction from abundance.
