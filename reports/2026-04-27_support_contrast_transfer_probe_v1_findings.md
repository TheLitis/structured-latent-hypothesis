# Support Contrast Transfer Probe v1

## Result

- External synthetic-to-semi-real policy cost `3.000000` vs best trivial `0.450000`; delta `+2.550000`.
- Semi-real in-domain CV wins: `2/3`.
- Synthetic label rates: safe `1.000`, budget `0.900`.
- Semi-real label rates: safe `0.400`, budget `0.850`.

## Interpretation

This probe checks whether support-set contrast fixes the representation-scale failure of raw residual scores. The first version still has a source-label problem: synthetic `task_safe` is `1.000`, so the policy has no meaningful unsafe structured examples to learn from.

That makes the external failure informative but not decisive. The stress-source v2 probe extends the synthetic ladder to add unsafe cases.
