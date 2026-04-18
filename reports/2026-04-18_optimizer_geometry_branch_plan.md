# Optimizer Geometry Branch Plan - 2026-04-18

## Aim

Open a separate branch of the project where the triple-point mathematics is tested on **update geometry** rather than on latent-space replacement.

The central question is:

> can low mixed-curvature update bases or commutator-aware updates improve optimization in the low-coupling regime?

## Minimal Benchmark

### Objective family

Use a quadratic coupling ladder in `d = 128` with a fixed active subspace of rank `k = 8`.

For each coupling value `alpha`:

`L_alpha(theta) = 0.5 * theta^T Q_alpha theta`

with

`Q_alpha = U (diag(lambda) + alpha * C) U^T + lambda_perp * (I - U U^T)`

where:

- `U` is a fixed random orthonormal `128 x 8` basis,
- `diag(lambda)` is a diagonal spectrum such as `1 .. 16`,
- `C` is a symmetric zero-diagonal `8 x 8` coupling matrix normalized to unit Frobenius norm,
- `alpha` controls mixed curvature inside the active subspace.

Suggested ladder:

- `0.00, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00`

### Baselines

- `Full Adam`
- `Random-k diagonal subspace`
- `PCA/Oja-k diagonal subspace`
- `PCA/Oja-k full-k preconditioner`
- `Low-mixed-curvature basis` (proposed branch)

## Protocol

- Same `theta_0` per seed for all methods
- Fixed step budget, e.g. `T = 300`
- Basis refresh every `10` steps
- Equal basis-update budget for proposed and PCA/Oja methods
- `10-20` seeds per coupling value

## Metrics

- final loss after `T` steps
- loss AUC over the trajectory
- off-diagonal curvature score:
  - `C_off(U) = sum_{i != j} (u_i^T H u_j)^2 / ||U^T H U||_F^2`
- order sensitivity:
  - difference between forward and reverse coordinate update order

## Falsifiable Criterion

This branch should only continue if, at fixed rank `k` and equal update budget:

- the proposed low-mixed-curvature basis beats `PCA/Oja diagonal` by at least `10%` median final loss on `alpha <= 0.35`,
- and that gain is accompanied by a smaller `C_off(U)`.

If `C_off` improves but loss/AUC do not improve in the low-coupling regime, the optimizer-geometry branch should be closed.

## If The Branch Succeeds

Only after passing the quadratic ladder should the branch move to:

- warped-separable objectives,
- then possibly commutator-corrected split updates on multi-term losses.
