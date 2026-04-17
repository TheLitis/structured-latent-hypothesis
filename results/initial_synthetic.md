# Initial Synthetic Benchmark

This report compares baseline, generic smoothness, CFP-only, and CFP-plus-step regularization.

## World: commutative

| Variant | Test Recon MSE | Comm Error | Latent Std |
| --- | ---: | ---: | ---: |
| baseline | 0.000906 +/- 0.000219 | 0.002042 +/- 0.000303 | 0.782713 +/- 0.028546 |
| smooth | 0.001930 +/- 0.000134 | 0.000201 +/- 0.000026 | 0.376303 +/- 0.022445 |
| comm | 0.001110 +/- 0.000270 | 0.000674 +/- 0.000127 | 0.563421 +/- 0.042779 |
| comm_step | 0.002106 +/- 0.000353 | 0.000107 +/- 0.000016 | 0.386325 +/- 0.016788 |

## World: noncommutative

| Variant | Test Recon MSE | Comm Error | Latent Std |
| --- | ---: | ---: | ---: |
| baseline | 0.002136 +/- 0.000050 | 0.017538 +/- 0.004890 | 1.730866 +/- 0.023559 |
| smooth | 0.002803 +/- 0.000043 | 0.004003 +/- 0.001248 | 0.853520 +/- 0.121903 |
| comm | 0.002340 +/- 0.000022 | 0.017400 +/- 0.007711 | 1.770613 +/- 0.258397 |
| comm_step | 0.003183 +/- 0.000204 | 0.003146 +/- 0.001694 | 0.781702 +/- 0.089542 |

## Reading Guide

- A good CFP outcome is lower held-out reconstruction error in the commutative world.
- The non-commutative control should not show the same pattern of gain.
- If `smooth` matches `comm`, the effect is probably not specific to CFP.
