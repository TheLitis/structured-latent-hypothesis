# Context Adaptation Probe v1

## Result

- `context_commuting_0.20` adaptation gain: `commuting_operator` `+0.000001`, `operator_plus_residual` `+0.000001`, `full_transition` `+0.000000`.
- `context_coupled_0.35` steps-to-target: `commuting_operator` `6.67`, `operator_plus_residual` `5.00`, `full_transition` `5.33`.
- `context_coupled_0.75` residual norm final: `commuting_operator` `0.000000`, `operator_plus_residual` `0.001448`, `full_transition` `0.000000`.

## Interpretation

This probe is the transfer-cost check for the context branch: structured models should need less adaptation in low coupling and lose that advantage as coupling grows.
