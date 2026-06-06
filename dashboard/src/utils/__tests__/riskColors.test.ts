import { riskColor } from '../riskColors'

// ------------------------------------------------------------
// riskColor(score) → [R, G, B, A]
//
// Maps a Compound Risk Score (0-100) to an RGBA array for use
// as Deck.gl layer fill colours.
//
// Thresholds (from GlobalWaterAtlas legend, matching Phase 3 CRS bins):
//   ≥ 70  → Critical  (deep red)
//   ≥ 50  → High      (orange)
//   ≥ 30  → Elevated  (amber)
//   < 30  → Low       (green)
//   undefined → No data (grey, lower alpha so base map shows through)
// ------------------------------------------------------------

describe('riskColor', () => {
  it('returns deep red for a Critical score (≥70)', () => {
    const [r, g, b, a] = riskColor(75)
    expect(r).toBeGreaterThan(150)   // red channel dominant
    expect(g).toBeLessThan(80)
    expect(b).toBeLessThan(80)
    expect(a).toBeGreaterThan(0)
  })

  it('returns orange for a High score (50–69)', () => {
    const [r, g, b, a] = riskColor(55)
    expect(r).toBeGreaterThan(150)   // red + green channels → orange
    expect(g).toBeGreaterThan(0)
    expect(b).toBeLessThan(50)
    expect(a).toBeGreaterThan(0)
  })

  it('returns amber for an Elevated score (30–49)', () => {
    const [r, g, b, a] = riskColor(40)
    expect(r).toBeGreaterThan(150)
    expect(g).toBeGreaterThan(100)   // amber has a high green channel
    expect(b).toBeLessThan(80)
    expect(a).toBeGreaterThan(0)
  })

  it('returns green for a Low score (<30)', () => {
    const [r, g, b, a] = riskColor(20)
    expect(r).toBeLessThan(100)
    expect(g).toBeGreaterThan(80)    // green channel dominant
    expect(b).toBeLessThan(100)
    expect(a).toBeGreaterThan(0)
  })

  it('returns grey with lower alpha when score is undefined (no data)', () => {
    const [r, g, b, a] = riskColor(undefined)
    // All channels equal → grey
    expect(r).toBe(g)
    expect(g).toBe(b)
    // Alpha lower than coloured countries so the base map shows through
    expect(a).toBeLessThan(150)
  })

  // Boundary conditions — these are exactly the threshold values.
  // An off-by-one error here silently miscategorises countries.
  it('score of exactly 70 is Critical, not High', () => {
    // Critical is deep red; High is orange. Both have high R, but Critical has
    // much lower G channel (40 vs 81) — so comparing full tuples is clearest.
    expect(riskColor(70)).toEqual([198, 40, 40, 200])
    expect(riskColor(69)).toEqual([230, 81, 0, 200])
  })

  it('score of exactly 50 is High, not Elevated', () => {
    const high     = riskColor(50)
    const elevated = riskColor(49)
    expect(high).not.toEqual(elevated)
  })

  it('score of exactly 30 is Elevated, not Low', () => {
    const elevated = riskColor(30)
    const low      = riskColor(29)
    expect(elevated).not.toEqual(low)
  })
})
