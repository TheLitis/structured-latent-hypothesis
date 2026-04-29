# Support Contrast Baseline Gauntlet v1

## Result

- Support-commutator first success budget: `5`.
- Validation-loss first success budget: `2`.
- Conformal-rank validation first success budget: `2`.
- Router-margin first success budget: `2`.
- Leader at 5 calibration worlds: `validation_loss` with delta `-0.310937` and win rate `0.828`.
- Leader at 6 calibration worlds: `validation_loss` with delta `-0.321289` and win rate `0.812`.
- Best source-only external family: `validation_loss` with delta `+0.325000`.
- Unique support-commutator claim survives: `False`.

## Interpretation

This is the strict comparison cycle. A claim survives only if the support-commutator family beats validation-loss, uncertainty/conformal, and router-margin baselines at small target calibration budgets.

The unique support-commutator claim does not survive this gauntlet. The support-commutator policy becomes useful with target calibration, but simpler support validation and router-margin baselines reach the practical gate earlier and with lower cost.

The practical lesson is narrower: the useful mechanism is support-set model selection/routing under context shift. The triple-derived commutator scores can remain diagnostic features, but they are not currently the best decision criterion.

The entropy baseline here is a regression proxy from support-loss scale and adaptation-curve variance, not a classifier softmax entropy.
