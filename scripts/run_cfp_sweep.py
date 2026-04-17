from __future__ import annotations

import argparse
from pathlib import Path

from structured_latent_hypothesis.plotting import (
    best_cfp_variant,
    load_results,
    plot_error_heatmaps,
    plot_metric_vs_lambda,
    plot_pareto,
    plot_train_vs_holdout_comm,
)
from structured_latent_hypothesis.synthetic import run_benchmark_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CFP lambda sweep with compositional holdouts and generate plots.")
    parser.add_argument("--output-dir", default="results/cfp_sweep")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=20)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 11, 29])
    parser.add_argument("--worlds", nargs="+", default=["commutative", "noncommutative"])
    parser.add_argument("--lambda-comm-values", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.10, 0.18])
    parser.add_argument("--split-strategy", default="cartesian_blocks")
    return parser.parse_args()


def build_variant_recipes(lambda_values: list[float]) -> dict[str, dict[str, float]]:
    recipes: dict[str, dict[str, float]] = {
        "baseline": {},
        "smooth": {"lambda_smooth": 0.05},
    }
    for value in lambda_values:
        recipes[f"cfp_l{value:0.3f}"] = {"lambda_comm": value}
    return recipes


def write_report(results_path: Path, output_dir: Path) -> None:
    results = load_results(results_path)
    recon_plot = output_dir / "test_recon_vs_lambda.png"
    comm_plot = output_dir / "comm_error_vs_lambda.png"
    plot_metric_vs_lambda(results, "test_recon_mse", "Held-out reconstruction vs CFP strength", recon_plot)
    plot_metric_vs_lambda(results, "comm_error_holdout_cells", "Holdout-cell commutativity error vs CFP strength", comm_plot)
    pareto_plot = output_dir / "pareto_holdout.png"
    geometry_plot = output_dir / "train_vs_holdout_comm.png"
    plot_pareto(results, pareto_plot)
    plot_train_vs_holdout_comm(results, geometry_plot)

    observations: list[str] = []
    heatmaps: list[str] = []
    for world in results["worlds"]:
        best_variant = best_cfp_variant(results, world)
        best_test = results["summary"][world][best_variant]["test_recon_mse"]["mean"]
        baseline = results["summary"][world]["baseline"]["test_recon_mse"]["mean"]
        heatmap_path = output_dir / f"{world}_error_heatmaps.png"
        plot_error_heatmaps(results, world, heatmap_path)
        heatmaps.append(f"![{world} heatmaps]({heatmap_path.name})")
        observations.append(
            f"- `{world}`: best CFP variant is `{best_variant}` with test recon `{best_test:.6f}` versus baseline `{baseline:.6f}`."
        )

    report_lines = [
        "# CFP Lambda Sweep",
        "",
        f"Split strategy: `{results['split_strategy']}`",
        "",
        "## Observations",
        "",
        *observations,
        "",
        "## Plots",
        "",
        "![Test reconstruction vs lambda](test_recon_vs_lambda.png)",
        "",
        "![Holdout commutativity error vs lambda](comm_error_vs_lambda.png)",
        "",
        "![Pareto plot](pareto_holdout.png)",
        "",
        "![Train vs holdout commutativity](train_vs_holdout_comm.png)",
        "",
        *heatmaps,
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_recipes = build_variant_recipes(args.lambda_comm_values)
    variants = list(variant_recipes.keys())

    results_path = output_dir / "results.json"
    markdown_path = output_dir / "summary.md"
    run_benchmark_suite(
        seeds=args.seeds,
        worlds=args.worlds,
        variants=variants,
        variant_recipes=variant_recipes,
        output_json=str(results_path),
        output_markdown=str(markdown_path),
        split_strategy=args.split_strategy,
        grid_size=args.grid_size,
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
    )
    write_report(results_path, output_dir)


if __name__ == "__main__":
    main()
