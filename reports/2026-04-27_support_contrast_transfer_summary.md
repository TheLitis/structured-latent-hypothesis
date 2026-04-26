# Support Contrast Transfer Summary - 2026-04-27

## What Was Tested

The support-contrast branch tested whether a small support-set probing protocol can replace raw residual norms as the transfer decision score.

The target claim was:

> compare structured and fallback adaptation on the same support set, so the score is less sensitive to representation scale and visual domain shift.

## Result

The strong external claim is not confirmed.

- v1 synthetic source had a degenerate safe label distribution: `task_safe = 1.000`.
- v2 added stress synthetic worlds up to `alpha=2.00`, dropping synthetic `task_safe` to `0.786`.
- External synthetic-to-semi-real policy improved but still lost:
  - v1 delta to best trivial: `+2.550000`
  - v2 delta to best trivial: `+0.700000`
- Semi-real in-domain CV in v2 won on `world` and `family`, but not `seed`.

## Interpretation

Support contrast is a better direction than raw residual norms, but it is still not externally portable in the current form.

The surviving signal is:

> support contrast can help after target-domain calibration, but synthetic source calibration does not transfer to semi-real context worlds.

The next test should not be another threshold sweep. It should introduce a calibration layer or domain-invariant normalization of the support-contrast score.
