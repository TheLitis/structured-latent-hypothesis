# structured-latent-hypothesis

This repository started from a simple three-point geometric observation and tested whether its mixed-difference / commutator structure could become a useful ML principle.

The original global claim is now **closed as unsupported**:

- strict affine lattice / equal spacing did not survive;
- CFP was not a universal latent prior;
- optimizer-geometry and structured-latent variants did not clear their gates;
- support-commutator routing was beaten by simpler support-validation and router-margin baselines.

The active project has pivoted to:

> **support-calibrated adaptive routing under context shift**

The practical question is no longer "does the three-point law define a new latent space?" It is:

> Given a small support set in a new context, can we choose safely between a structured branch, a fallback branch, and escalation?

## Current Status

The strongest current result is the baseline gauntlet:

- source-only synthetic-to-semi-real transfer still fails;
- target calibration is useful;
- `validation_loss`, `conformal_validation_rank`, and `router_margin` beat the triple-derived support-commutator criterion;
- the useful mechanism is support-set model selection, not the original commutator score.

See:

- `reports/2026-04-29_support_contrast_baseline_gauntlet_v1_findings.md`
- `results/support_contrast_baseline_gauntlet_v1/summary.md`

## Active Research Claim

The new claim is deliberately narrower:

> A small target-domain support set can calibrate a deployable routing policy that chooses between structured transfer, fallback adaptation, and escalation under context shift.

This is useful only if it beats trivial policies and remains stable across:

- held-out worlds,
- seeds,
- context families,
- deployment cost models.

## Closed Claim

The original three-point observation remains documented as project origin, but it is not the active claim.

It led to the mixed finite difference:

`Delta_i Delta_j z ~= 0`

which is the discrete statement that factor steps approximately commute. This was a useful hypothesis generator, but not a winning decision criterion in the current context-transfer benchmarks.

## Important Artifacts

- `docs/research_note.md`: historical CFP note.
- `reports/2026-04-29_project_pivot_support_calibrated_routing.md`: pivot report.
- `scripts/run_support_contrast_baseline_gauntlet.py`: strict baseline comparison that closed the unique commutator claim.
- `scripts/run_support_routing_cost_robustness.py`: active support-routing cost robustness probe.
- `src/structured_latent_hypothesis/support_contrast.py`: support-set feature extraction and decision-policy helpers.

## Quick Start

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python .\scripts\run_support_contrast_baseline_gauntlet.py
.venv\Scripts\python .\scripts\run_support_routing_cost_robustness.py
```

## Research Rule

No claim survives because it is mathematically elegant. A claim survives only if it beats strong baselines under the same calibration, cost, and holdout protocol.
