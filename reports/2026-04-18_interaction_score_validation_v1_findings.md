# Interaction Score Validation v1 Findings - 2026-04-18

## Setup

- Synthetic calibration source:
  - `results/structured_hybrid_probe_v1/results.json`
- Semi-real zero-shot target:
  - `results/semireal_transfer_probe_v1/results.json`
- Structured advantage target:
  - `A(w) = MSE_coord(w) - min_s MSE_structured_s(w)`
- Score candidates:
  - `S_cpl`
  - `S_irr`
  - `S_diag`

The goal of this probe was to test the new practically useful claim in the repository:

> can we extract a transferable interaction score that predicts when structured bias helps?

## Main Result

Not yet.

The first generation of scores explains the synthetic ladder, but does **not** transfer zero-shot to the semi-real benchmark.

That makes this a clean negative result for the current score designs.

## Decisive Pattern

### `S_cpl`

- Synthetic Spearman: `-0.829`
- Synthetic sign accuracy: `1.00`
- Semi-real Spearman: `+0.000`
- Semi-real zero-shot sign accuracy: `0.00`

Interpretation:

`S_cpl` is an excellent synthetic regime separator, but it completely fails as a zero-shot transfer score.

This means raw commutator-over-drift is not enough to explain the richer world shift.

### `S_diag`

- Synthetic Spearman: `-0.829`
- Synthetic sign accuracy: `1.00`
- Semi-real Spearman: `+0.000`
- Semi-real zero-shot sign accuracy: `0.00`

Interpretation:

The most triple-native score, diagonal defect concentration, behaves almost identically to `S_cpl` in the current setup:

- it cleanly explains the synthetic ladder;
- it says essentially nothing useful about the semi-real winner.

This is important because it rules out the easy optimistic reading:

> just measuring “how diagonal the defect is” will already give a transferable regime detector.

### `S_irr`

- Synthetic Spearman: `-0.486`
- Synthetic sign accuracy: `0.67`
- Semi-real Spearman: `+0.300`
- Semi-real zero-shot sign accuracy: `0.20`

Interpretation:

`S_irr` is weaker on synthetic, but slightly less brittle on semi-real.

That makes it the most promising of the three in principle, but its actual predictive quality is still too poor to claim usefulness.

## What This Means

The practical-object hypothesis survives only in a weaker form.

The repository no longer supports:

> we already have a transferable interaction score.

It now supports only:

> we have score candidates that reveal the synthetic regime structure, but not yet the cross-world transferable one.

So the missing object is likely not a raw world-level commutator proxy alone.

It is probably one of:

- a normalized score that factors out world complexity,
- a score measured in a learned shared representation rather than raw pixels,
- or a joint score that combines interaction strength with “structured recoverability”.

## Research Consequence

This result is useful because it prevents another false consolidation.

The next score iteration should **not** just add more scalar heuristics in pixel space.

The strongest next score test would be:

1. fit a shared encoder across synthetic and semi-real families,
2. measure interaction geometry in that shared representation,
3. test whether the resulting score predicts structured advantage better than the current raw-pixel proxies.

That is a much tighter and more defensible next step than continuing to hand-design world-level scores.
