# Multi-Vector Reframe - 2026-04-18

## Core Reformulation

The project should no longer be framed as:

> a search for a universal latent-space replacement derived from the triple-point geometry.

The strongest current formulation is:

> triple-point mathematics gives a way to define, measure, and use **low interaction curvature / near-commuting transport**.

That object can then be studied across several distinct research vectors without collapsing them into one over-broad claim.

## Central Practical Object

The most promising practically useful object in the current repository is an **interaction score**.

Not a new universal latent module, and not yet a new optimizer.

The proposed object is:

> a scalar measure of path-dependence / coupling strength that predicts when structured bias is appropriate and when a richer unrestricted model is needed.

Why this is the strongest near-term object:

- [2026-04-18_transport_pair_probe_findings.md](/D:/Experiment/reports/2026-04-18_transport_pair_probe_findings.md) already showed that gain tracks separability vs path dependence.
- [2026-04-18_interaction_ladder_probe_v2_findings.md](/D:/Experiment/reports/2026-04-18_interaction_ladder_probe_v2_findings.md) established a real regime boundary.
- [2026-04-18_semireal_transfer_probe_v1_findings.md](/D:/Experiment/reports/2026-04-18_semireal_transfer_probe_v1_findings.md) showed that the structured hierarchy survives qualitatively on richer scenes, even though practical superiority does not.
- [results/crossworld_diagnostic_v1/report.md](/D:/Experiment/results/crossworld_diagnostic_v1/report.md) shows that the best structured gap flips sign across worlds and that structured variant ranking itself shifts across families.

This makes `interaction score -> regime selection` the cleanest current route to practical utility.

## Research Vector 1: Separable Transport

Central object:

- `z_{ij} = a_i + b_j + r_{ij}` as a low-coupling backbone.

Repository status:

- Confirmed that the direct additive branch is real and not a weak-baseline artifact.
- Confirmed that it is useful on matched separable worlds and low/moderate coupling ladders.
- Refuted as a universal winner on strong-coupling and semi-real settings.

Primary evidence:

- [2026-04-18_direct_separable_probe_v2_findings.md](/D:/Experiment/reports/2026-04-18_direct_separable_probe_v2_findings.md)
- [2026-04-18_interaction_ladder_probe_v2_findings.md](/D:/Experiment/reports/2026-04-18_interaction_ladder_probe_v2_findings.md)
- [2026-04-18_semireal_transfer_probe_v1_findings.md](/D:/Experiment/reports/2026-04-18_semireal_transfer_probe_v1_findings.md)

Next experiment:

- cross-world backbone transfer:
  train a separable backbone on one family, freeze `a_i, b_j`, and compare how much residual adaptation is needed on a sibling family versus `coord_latent`.

## Research Vector 2: Interaction Geometry

Central object:

- not the cell residual itself, but the **defect field**:
  - mixed difference `Delta_i Delta_j z`
  - diagonal defect `h(i + j)`

Repository status:

- Confirmed that direct defect parameterization is stronger than the earlier affine and free-cell views.
- Confirmed that `hankel` helps in the boundary regime and `curvature_field` helps on the early tail.
- Refuted that any one fixed defect class solves the whole ladder or yields a semi-real win.

Primary evidence:

- [2026-04-18_curvature_defect_probe_v1_findings.md](/D:/Experiment/reports/2026-04-18_curvature_defect_probe_v1_findings.md)
- [2026-04-18_structured_hybrid_probe_v1_findings.md](/D:/Experiment/reports/2026-04-18_structured_hybrid_probe_v1_findings.md)
- [2026-04-18_semireal_transfer_probe_v1_findings.md](/D:/Experiment/reports/2026-04-18_semireal_transfer_probe_v1_findings.md)

Next experiment:

- interaction-map recovery:
  measure how well learned defect fields correlate with known ground-truth coupling maps, not just reconstruction error.

## Research Vector 3: Adaptive Regime Selection

Central object:

- an explicit regime selector that chooses between separable, diagonal-defect, curvature-field, and unrestricted branches.

Repository status:

- Confirmed that a regime boundary exists.
- Refuted that either a fixed structured sum or a small generic coordinate adapter solves the problem.

Primary evidence:

- [2026-04-18_interaction_ladder_probe_v2_findings.md](/D:/Experiment/reports/2026-04-18_interaction_ladder_probe_v2_findings.md)
- [2026-04-18_structured_hybrid_probe_v1_findings.md](/D:/Experiment/reports/2026-04-18_structured_hybrid_probe_v1_findings.md)
- [2026-04-18_structured_coord_adapter_probe_v1_findings.md](/D:/Experiment/reports/2026-04-18_structured_coord_adapter_probe_v1_findings.md)

Next experiment:

- sparse mixture-of-structures with an explicit gate and entropy/sparsity control, tested both on the synthetic ladder and semi-real family.

## Research Vector 4: Optimization Geometry

Central object:

- geometry of updates rather than geometry of latent values:
  - low mixed-curvature update bases
  - commutator-corrected split updates

Repository status:

- No direct empirical confirmation yet.
- But the current repo has already refuted the strongest latent-replacement reading, which makes optimization geometry a clean place to test the same mathematics at a more fundamental layer.

Primary basis:

- current negative results on universal representation replacement
- existing theoretical direction captured in prior CFP notes and optimizer sketches

Next experiment:

- synthetic optimizer benchmark on quadratic and warped-separable objectives with controlled coupling, comparing `Adam`, linear subspace, low-mixed-curvature basis, and curvature-corrected split updates.

## Practical Interpretation

The project should currently be presented in two layers:

1. **Confirmed research signal**
   Triple-derived geometry captures a real low-interaction object, visible in defect fields, regime boundaries, and structured bias on synthetic ladders.

2. **Unconfirmed practical claim**
   Current structured models do not yet outperform strong unrestricted baselines on the semi-real benchmark.

So the honest near-term goal is not:

> prove that triple-point structure replaces modern latent spaces.

It is:

> extract a transferable interaction/coupling object that can drive regime-aware model selection, diagnostics, and possibly later optimization or operator learning.

## Current Recommendation

If the project must move in “all vectors at once”, the correct execution order is:

1. keep the representational line alive, but stop over-claiming it;
2. prioritize the `interaction score` / `regime detector` object as the nearest useful artifact;
3. open a dedicated optimizer-geometry branch rather than forcing every idea back into latent-space replacement;
4. use cross-world diagnostics as the filter for whether any future claim really transfers between worlds.
