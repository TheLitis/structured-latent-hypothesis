# Commutator Ladder Scale Nested V1 Findings - 2026-04-18

## Setup

- Benchmark family: `matched_scale_alpha`
- Split: `cartesian_blocks`
- Hyperparameter protocol: nested inner validation on a structured inner split
- Selection metric: inner validation reconstruction MSE
- Variants:
  - `baseline`
  - `smooth`
  - fixed `cfp_l0.010`
  - fixed `cfp_l0.050`
  - `step_selected`
  - `cfp_selected`

## Main Result

CFP still survives the stronger protocol in this family, but not as a universal winner.

## What Holds Up

- At `matched_scale_0.10`, the nested-selected CFP variant is best: `0.000852` vs `0.001010` for baseline.
- At `matched_scale_0.20`, CFP still helps, although the best result is the fixed `cfp_l0.010`: `0.002046` vs `0.002107` for baseline.
- At `matched_scale_0.35`, CFP still gives the best result through the fixed `cfp_l0.010`: `0.002740` vs `0.002991` for baseline.

## What Changed

- The selected CFP variant is not stable across seeds. Chosen lambdas spread over `0.005`, `0.010`, `0.020`, `0.050`, and `0.100`.
- At `matched_scale_0.50`, `step_selected` is best: `0.002776` vs `0.002891` for baseline, while `cfp_selected` degrades to `0.003025`.
- `step_selected` is clearly worse than CFP in the middle of the ladder, but it is not a trivial loser anymore.

## Interpretation

This strengthens a narrower claim and weakens a broader one.

The surviving honest statement is:

> in the audited scale family, CFP retains a real advantage in the mild-to-moderate regime, but part of the late-ladder gain can be matched or exceeded by a step-only prior, and the nested selector is not yet stable.

## Research Decision

- Keep `scale` as the best evidence family.
- Do not compare against external popular baselines yet.
- First stabilize model selection, because the current nested CFP selector is too seed-sensitive to be the final comparison protocol.
