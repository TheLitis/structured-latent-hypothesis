# Support Contrast Transfer Probe v1

## Synthetic-Trained External Policy

Policy cost: `3.000000`
Best trivial cost: `0.450000`
Delta: `+2.550000`
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
| synthetic | score_contrast | +0.380 | -0.502 | -0.163 |
| synthetic | score_gain_delta_1 | -0.054 | +0.769 | -0.461 |
| synthetic | score_gain_delta_8 | -0.419 | +0.726 | -0.164 |
| synthetic | score_gain_ratio_1 | +0.465 | +0.545 | -0.019 |
| synthetic | score_gain_ratio_8 | +0.110 | -0.682 | -0.311 |
| synthetic | score_instability | -0.525 | +0.124 | -0.084 |
| synthetic | score_residual_normalized | -0.596 | +0.231 | -0.001 |
| semireal | score_contrast | +0.575 | +0.111 | +0.291 |
| semireal | score_gain_delta_1 | +0.009 | +0.711 | -0.522 |
| semireal | score_gain_delta_8 | +0.408 | +0.196 | +0.014 |
| semireal | score_gain_ratio_1 | -0.119 | +0.814 | -0.567 |
| semireal | score_gain_ratio_8 | +0.346 | +0.279 | -0.028 |
| semireal | score_instability | +nan | +nan | +nan |
| semireal | score_residual_normalized | -0.498 | -0.059 | -0.277 |
