# Next Discovery Program Summary

## Scope

This cycle tested the two remaining serious discovery paths after the latent-space replacement claim had already weakened:

1. `context transfer`: whether triple-derived near-commutativity predicts transferability between actions/skills and contexts;
2. `optimizer geometry`: whether low mixed-curvature structure helps optimization on low-coupling objectives.

The locked interpretation rule for this cycle was strict: a result counts only if the gain is specific to the low-coupling regime. Global improvements or non-specific improvements do not count as evidence for the idea.

## Context-Transfer Branch

### What held up

- The operator benchmark now exists as a real parallel branch, not a toy sketch:
  - synthetic transition families `context_commuting_alpha` and `context_coupled_alpha`;
  - held-out `context x action` combinations;
  - zero-shot one-step and rollout metrics;
  - frozen-action adaptation benchmark with support/query split.
- The structured residual branch does learn a nontrivial interaction signal.
  - In `operator_diag_residual`, holdout interaction magnitude rises almost monotonically on the coupled ladder.
  - Zero-shot holdout interaction vs coupled `alpha` gives Spearman `+0.96`.
  - Adapted residual norm vs coupled `alpha` gives Spearman `+0.71`.

This means the triple-derived structure still carries diagnostic information about interaction strength.

### What did not hold up

The main discovery claim for this branch was:

> near-commuting structure should make structured operator models specifically better in low/moderate coupling, and that advantage should shrink or reverse at high coupling.

That pattern did **not** appear cleanly.

- In zero-shot one-step transfer, the structured models do **not** show a meaningful low-coupling edge over `full_transition`.
  - On the full ladder, low-coupling average deltas versus `full_transition` are slightly negative:
    - `commuting_operator`: `-5.0e-7`
    - `operator_plus_residual`: `-8.2e-7`
    - `operator_diag_residual`: `-5.1e-7`
- In the coupled family, the best structured zero-shot wins appear at higher coupling, not lower coupling.
  - `context_coupled_0.75`: `operator_plus_residual` `0.000454` vs `full_transition` `0.000490`
  - `context_coupled_1.00`: structured branch loses again
- In adaptation, structured models often need fewer steps than `full_transition`, but that advantage does **not** collapse as coupling grows.
  - For `operator_plus_residual`, the average steps-to-target advantage in the coupled family is:
    - low `alpha <= 0.35`: `+2.75` steps
    - high `alpha >= 0.75`: `+4.83` steps

So the branch does not validate the intended regime law. The transfer-cost advantage is not low-coupling-specific; if anything, parts of it remain strong or even stronger in the high-coupling tail.

### Context verdict

The strong claim is **not confirmed**:

> triple-derived near-commutativity predicts transferability between actions and contexts in a low-coupling-specific way.

What survives is narrower:

> triple-derived structured residuals provide a usable interaction diagnostic inside operator-style transfer worlds.

That is meaningful, but it is not yet the discovery claim this branch was designed to prove.

## Optimizer-Geometry Branch

### What held up

- The benchmark is now real and reproducible:
  - quadratic coupling ladder in `d=128`, rank `8`;
  - `adam_full`, `random_subspace_diag`, `oja_subspace_diag`, `oja_subspace_full`, `low_mixed_curvature_basis`;
  - metrics for final loss, AUC, off-diagonal curvature, and order sensitivity.
- The proposed method does reduce mixed curvature exactly as intended.
  - `low_mixed_curvature_basis` drives `C_off` to `0.000000` across the full ladder.
  - It also improves over `oja_subspace_diag` on final loss.

### What did not hold up

The continuation gate for this branch was explicit:

> beat `oja_subspace_diag` by at least `10%` median final loss on `alpha <= 0.35`, and that win must coincide with lower mixed curvature.

The curvature condition passed, but the performance condition did not.

- At `alpha=0.20`:
  - `oja_subspace_diag` median final loss: `0.048860`
  - `low_mixed_curvature_basis` median final loss: `0.047183`
  - relative gain: `+3.43%`

That is directionally correct but well below the `10%` gate.

### Optimizer verdict

The fallback discovery claim is **not confirmed**.

What survives is:

> low mixed-curvature bases are structurally cleaner and mildly better than Oja+diag in this synthetic quadratic family, but not by enough to justify continuing this as a breakthrough path.

## Repo-Wide Interpretation

After this cycle, the project state is:

- **closed**
  - strict affine lattice / equal spacing as the transferable law;
  - universal latent-space CFP win;
  - raw pixel-level interaction scores as a transferable detector;
  - optimizer geometry as a strong replacement-style discovery path, at least in the current falsification benchmark.
- **still alive**
  - separable transport / path independence as a synthetic law;
  - bounded-regime usefulness of structured backbones in earlier static probes;
  - structured interaction magnitude as a diagnostic object;
  - context/operator benchmarks as a place where the triple-derived math can still be measured cleanly.

## Most Honest Next Move

The project should now pivot away from:

- "find the model class that replaces standard latent spaces";
- "find the optimizer variant that replaces standard training."

It should pivot toward:

> **diagnostic law / transfer criterion**

The strongest surviving object is not a universal architecture, but a measurement:

- interaction magnitude;
- commutator defect;
- adaptation residual norm;
- steps-to-target under frozen-action transfer.

The next hypothesis should therefore be narrower and sharper:

> triple-derived interaction geometry predicts when transfer will be cheap, when adaptation budget will be small, and when a structured bias is safe to use.

That is still scientifically valuable. It is just a different kind of discovery than the one originally hoped for.
