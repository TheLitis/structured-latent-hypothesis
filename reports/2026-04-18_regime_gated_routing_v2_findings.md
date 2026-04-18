# Regime-Gated Routing v2

## Result

- best deployable router: `synthetic_only_train_pairA_curv_hankel` with pooled routed mse `0.000597` and semi-real gain over coord `+0.000000`.
- `shared_train` pair A pooled routed mse: `0.000606`.
- `shared_train` pair A semi-real gain over always-coord: `-0.000018`.
- `synthetic_only_train` pair A pooled routed mse: `0.000597`.
- `synthetic_only_train` pair A semi-real gain over always-coord: `+0.000000`.

## Interpretation

Routing v2 still yields a deployable coarse gate, but the transfer-separation check failed: synthetic-only training beats shared training on the semi-real side, so the shared-latent portability claim is weakened.

This should still be treated as a coarse gate only. The result does not upgrade the score into a continuous regime estimator.
