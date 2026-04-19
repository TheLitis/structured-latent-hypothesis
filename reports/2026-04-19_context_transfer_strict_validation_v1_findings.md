# Context Transfer Strict Validation v1

## Result

- `within_budget_6` / `seed`: best binary `diag_joint_prod` cost `0.476190` (beats trivial baseline); best abstain `diag_interaction` cost `0.500000` (does not beat trivial baseline), coverage `1.000`.
- `within_budget_6` / `world`: best binary `diag_interaction` cost `0.404762` (beats trivial baseline); best abstain `diag_interaction` cost `0.404762` (beats trivial baseline), coverage `1.000`.
- `within_budget_6` / `family`: best binary `diag_interaction` cost `0.404762` (beats trivial baseline); best abstain `diag_joint_prod` cost `0.458333` (beats trivial baseline), coverage `0.833`.
- `within_budget_8` / `seed`: best binary `diag_interaction` cost `0.333333` (beats trivial baseline); best abstain `diag_interaction` cost `0.333333` (beats trivial baseline), coverage `1.000`.
- `within_budget_8` / `world`: best binary `diag_interaction` cost `0.357143` (beats trivial baseline); best abstain `diag_interaction` cost `0.339286` (beats trivial baseline), coverage `0.881`.
- `within_budget_8` / `family`: best binary `diag_interaction` cost `0.380952` (beats trivial baseline); best abstain `diag_interaction` cost `0.267857` (beats trivial baseline), coverage `0.643`.
- `within_budget_10` / `seed`: best binary `diag_joint_sum` cost `0.238095` (beats trivial baseline); best abstain `diag_joint_sum` cost `0.238095` (beats trivial baseline), coverage `1.000`.
- `within_budget_10` / `world`: best binary `diag_interaction` cost `0.309524` (does not beat trivial baseline); best abstain `diag_residual` cost `0.309524` (does not beat trivial baseline), coverage `1.000`.
- `within_budget_10` / `family`: best binary `diag_joint_sum` cost `0.261905` (beats trivial baseline); best abstain `diag_joint_sum` cost `0.261905` (beats trivial baseline), coverage `1.000`.
- `safe_regret_1e-05` / `seed`: best binary `diag_residual` cost `0.166667` (beats trivial baseline); best abstain `diag_residual` cost `0.166667` (beats trivial baseline), coverage `1.000`.
- `safe_regret_1e-05` / `world`: best binary `diag_joint_sum` cost `0.238095` (beats trivial baseline); best abstain `diag_residual` cost `0.345238` (beats trivial baseline), coverage `0.952`.
- `safe_regret_1e-05` / `family`: best binary `diag_residual` cost `0.285714` (beats trivial baseline); best abstain `diag_residual` cost `0.285714` (beats trivial baseline), coverage `1.000`.

## Interpretation

This probe is the strict version of the budget claim. It only counts if the criterion survives harder cross-validation splits and keeps beating the best trivial baseline under asymmetric costs, with or without abstention.
