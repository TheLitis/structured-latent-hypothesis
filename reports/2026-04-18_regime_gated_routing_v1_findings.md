# Regime-Gated Routing v1

## Result

- Best deployable router: `router_pairA_curv_hankel_shared_joint`.
- Pooled routed mse: `0.000596`.
- Pooled gain over always-coord: `+0.000047`.
- Pooled regret to pair oracle: `0.000000`.
- Semi-real structured route rate: `0.00`.

## Interpretation

Shared-latent gating now has practical value: it preserves coord-like behavior on semi-real while recovering structured gains on the synthetic boundary regime.

The result should still be treated as a coarse gate, not a continuous regime estimator, because within-family semi-real ranking remains unresolved.
