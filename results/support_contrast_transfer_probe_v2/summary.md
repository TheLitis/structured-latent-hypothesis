# Support Contrast Transfer Probe v2

## Synthetic-Trained External Policy

Policy cost: `1.150000`
Best trivial cost: `0.450000`
Delta: `+0.700000`
Wins best trivial: `False`

## Semi-Real In-Domain CV

| Group | Cost | Best Trivial | Delta | Wins |
| --- | ---: | ---: | ---: | --- |
| seed | 0.750000 | 0.450000 | +0.300000 | False |
| world | 0.125000 | 0.450000 | -0.325000 | True |
| family | 0.125000 | 0.450000 | -0.325000 | True |

## Score Correlations

| Domain | Score | Regret Spearman | Fallback Steps Spearman | Structured Steps Spearman |
| --- | --- | ---: | ---: | ---: |
| synthetic | score_contrast | +0.314 | -0.317 | -0.083 |
| synthetic | score_gain_delta_1 | -0.519 | +0.213 | -0.735 |
| synthetic | score_gain_delta_8 | -0.599 | +0.071 | -0.426 |
| synthetic | score_gain_ratio_1 | +0.280 | +0.138 | -0.142 |
| synthetic | score_gain_ratio_8 | -0.269 | -0.690 | -0.432 |
| synthetic | score_instability | -0.361 | +0.157 | -0.013 |
| synthetic | score_residual_normalized | -0.776 | -0.167 | -0.466 |
| semireal | score_contrast | +0.575 | +0.111 | +0.291 |
| semireal | score_gain_delta_1 | +0.009 | +0.711 | -0.522 |
| semireal | score_gain_delta_8 | +0.408 | +0.196 | +0.014 |
| semireal | score_gain_ratio_1 | -0.119 | +0.814 | -0.567 |
| semireal | score_gain_ratio_8 | +0.346 | +0.279 | -0.028 |
| semireal | score_instability | +nan | +nan | +nan |
| semireal | score_residual_normalized | -0.498 | -0.059 | -0.277 |
