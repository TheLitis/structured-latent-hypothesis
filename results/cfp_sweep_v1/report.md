# CFP Lambda Sweep

Split strategy: `cartesian_blocks`

## Observations

- `commutative`: best CFP variant is `cfp_l0.010` with test recon `0.001031` versus baseline `0.000884`.
- `noncommutative`: best CFP variant is `cfp_l0.010` with test recon `0.002447` versus baseline `0.002599`.

## Plots

![Test reconstruction vs lambda](test_recon_vs_lambda.png)

![Holdout commutativity error vs lambda](comm_error_vs_lambda.png)

![Pareto plot](pareto_holdout.png)

![Train vs holdout commutativity](train_vs_holdout_comm.png)

![commutative heatmaps](commutative_error_heatmaps.png)
![noncommutative heatmaps](noncommutative_error_heatmaps.png)
