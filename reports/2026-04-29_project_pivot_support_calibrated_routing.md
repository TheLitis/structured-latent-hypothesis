# Project Pivot: Support-Calibrated Adaptive Routing

## Decision

The original three-point / commutator geometry branch is closed as a global discovery claim.

It produced useful hypotheses, but the strict baseline gauntlet showed that the unique support-commutator decision criterion is not competitive with simpler support-set baselines.

## Closed Claims

- Strict affine lattice / equal spacing is too brittle.
- CFP is not a universal latent prior.
- Structured latent variants do not reliably beat coordinate/fallback baselines.
- Optimizer-geometry did not clear its predefined improvement gate.
- Source-only synthetic-to-semi-real transfer failed.
- Support-commutator routing is not uniquely useful after comparison against validation-loss, conformal-rank, and router-margin baselines.

## Surviving Mechanism

The surviving mechanism is not the original geometry. It is:

> support-set model selection under context shift.

In the current context-transfer setup, a small target-domain support set can calibrate routing between:

- structured branch,
- fallback branch,
- escalation.

The strongest current baselines are:

- `validation_loss`,
- `conformal_validation_rank`,
- `router_margin`.

## New Active Claim

> A small target-domain support set can calibrate a deployable adaptive routing policy that remains useful across held-out contexts and deployment cost models.

This claim is narrower, but practical. It points toward agent systems, model routers, tool-use fallback policies, and world-model adaptation rather than a new latent-space law.

## Next Gate

The next gate is cost robustness.

A routing policy is useful only if it remains better than trivial actions when the deployment cost model changes:

- structured failure becomes more expensive,
- fallback delay becomes more expensive,
- escalation becomes cheap,
- escalation becomes expensive.

If support-calibrated routing only works for one hand-picked cost model, the pivot is still too narrow.
