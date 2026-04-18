# Context Transfer Budget v1

## Result

- `within_budget_4`: best candidate `diag_joint_sum` with LOO cost `0.357143` vs trivial baseline `0.285714`; does not beat trivial baseline. Balanced accuracy `0.485`.
- `within_budget_6`: best candidate `diag_joint_prod` with LOO cost `0.476190` vs trivial baseline `0.500000`; beats trivial baseline. Balanced accuracy `0.556`.
- `within_budget_8`: best candidate `diag_interaction` with LOO cost `0.333333` vs trivial baseline `0.571429`; beats trivial baseline. Balanced accuracy `0.625`.
- `within_budget_10`: best candidate `diag_joint_sum` with LOO cost `0.238095` vs trivial baseline `0.285714`; beats trivial baseline. Balanced accuracy `0.745`.
- `safe_regret_0`: best candidate `plus_residual` with LOO cost `0.214286` vs trivial baseline `0.071429`; does not beat trivial baseline. Balanced accuracy `0.556`.
- `safe_regret_1e-05`: best candidate `diag_residual` with LOO cost `0.166667` vs trivial baseline `0.500000`; beats trivial baseline. Balanced accuracy `0.902`.
- `safe_regret_5e-05`: best candidate `diag_interaction` with LOO cost `0.142857` vs trivial baseline `0.142857`; does not beat trivial baseline. Balanced accuracy `0.583`.

## Interpretation

This probe asks a practical question: can triple-derived scores support budget allocation or safe-use decisions under asymmetric error costs? A task only counts as deployable if its learned threshold beats the best trivial baseline in leave-one-seed-out evaluation.
