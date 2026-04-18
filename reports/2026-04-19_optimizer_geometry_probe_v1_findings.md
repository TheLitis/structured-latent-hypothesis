# Optimizer Geometry Probe v1

## Result

- Low-coupling check `alpha=0.20`: `oja_subspace_diag` median final loss `0.048860`, `low_mixed_curvature_basis` median final loss `0.047183`, relative gain `+3.43%`.
- Stronger-coupling check `alpha=0.75`: `oja_subspace_diag` mean final loss `0.049777`, `low_mixed_curvature_basis` mean final loss `0.047317`.

## Interpretation

The branch continues only if low-mixed-curvature basis wins in the low-coupling regime and that win coincides with lower off-diagonal curvature.
