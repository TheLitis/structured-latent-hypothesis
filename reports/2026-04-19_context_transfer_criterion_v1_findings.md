# Context Transfer Criterion v1

## Result

- Best regret-tracking score: `diag_joint_sum` with all-row Spearman `+0.623`.
- Best transfer-difficulty score: `diag_joint_sum` with all-row Spearman `+0.265`.
- Best leave-one-seed-out router: `plus_joint_sum` with routed rollout@5 `0.000285` versus `always_full` `0.000285`; no improvement over always_full.

## Interpretation

This probe asks a narrower question than the earlier architecture claims: whether triple-derived interaction scores can serve as a usable transfer criterion. A candidate counts as practical only if it both tracks regret/difficulty and beats `always_full` in leave-one-seed-out routing.
