# Support Contrast Transfer Probe v2

## Result

- External synthetic-to-semi-real policy cost `1.150000` vs best trivial `0.450000`; delta `+0.700000`.
- Semi-real in-domain CV wins: `2/3`.
- Synthetic label rates: safe `0.786`, budget `0.786`.
- Semi-real label rates: safe `0.400`, budget `0.850`.

## Interpretation

This probe fixes the v1 source-label issue by extending the synthetic source ladder to `alpha=1.50/2.00`. That creates unsafe structured cases: synthetic `task_safe` drops from `1.000` to `0.786`.

The result improves but does not pass the external portability gate. Synthetic-to-semi-real delta improves from `+2.550000` in v1 to `+0.700000` in v2, while semi-real in-domain CV wins on `world` and `family` holdout. So support contrast is useful inside the semi-real domain, but it still does not solve synthetic-to-semi-real calibration transfer.
