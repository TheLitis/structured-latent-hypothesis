# Transfer Law Strict Follow-Up

## Why This Follow-Up Matters

`budget_v1` showed the first partial practical win:

- some budgeted transfer tasks beat trivial baselines;
- but those wins were only validated under leave-one-seed-out.

That left a real concern:

> maybe the criterion is only exploiting seed-level regularities and does not survive harder shifts.

This cycle tested exactly that.

## What Was Made Stricter

Three axes were tightened simultaneously:

1. **harder cross-validation**
   - leave-one-seed-out
   - leave-one-world-out
   - leave-one-family-out
2. **asymmetric cost sweep**
   - `fp/fn = 2/1, 3/1, 5/1`
3. **explicit abstain / escalate policy**
   - binary decision
   - abstain band around the threshold with explicit abstain costs

## Main Result

The budgeted-transfer claim does **not** collapse under stricter validation.

It survives in a narrower but still meaningful form.

### Stable surviving regimes

- `within_budget_6`
  - survives under `world` and `family` holdout;
  - strongest recurring score: `diag_interaction`
- `within_budget_8`
  - survives under `seed`, `world`, and `family` holdout;
  - strongest recurring score: `diag_interaction`
- `within_budget_10`
  - survives under `seed` and `family` holdout, but fails under `world` holdout at the main `3/1` cost pair
- `safe_regret_1e-05`
  - survives under `seed`, `world`, and `family` holdout;
  - strongest recurring score family: `diag_residual` / `diag_joint_sum`

### What changed about abstention

Abstention is **not** a universal win.

But it is no longer empty:

- on `within_budget_8`, abstention improves the best binary policy under harder splits:
  - `world` holdout: `0.339286` vs binary `0.357143`
  - `family` holdout: `0.267857` vs binary `0.380952`

So the abstain band is not just decorative. It can reduce cost in the mid-budget regime when the split becomes harsher.

## New Best Interpretation

The project still does **not** support:

- a universal structured router;
- a universal transfer law for all regimes;
- a replacement architecture claim.

But it now supports a stronger and more defensible statement than before:

> triple-derived interaction geometry can provide a **cost-sensitive criterion for budgeted adaptation and safe structured transfer**, and that criterion survives some nontrivial world-level generalization.

That is a materially stronger claim than:

- “the score correlates with coupling”;
- or “the score works on seed resamples”.

## Practical Shape Of The Result

The pattern is now:

- **strict / zero-regret tasks** still fail;
- **mid-budget / bounded-regret tasks** are where the criterion becomes useful;
- **diag-based scores** dominate the practical wins;
- **abstain** helps mainly in the uncertain middle regime, not everywhere.

That is exactly the kind of regime map a real deployment criterion should have.

## Most Honest Next Step

The next step should not be another broad sweep.

It should be a targeted escalation policy benchmark:

1. use `diag_interaction`, `diag_joint_sum`, and `diag_residual` only;
2. define a concrete policy:
   - accept structured path,
   - abstain/escalate budget,
   - or fall back;
3. evaluate total cost under a simple decision pipeline.

If that holds, the project’s strongest surviving path becomes:

> a transferable **decision law** for when structured transfer is safe, risky, or needs escalation.
