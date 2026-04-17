from __future__ import annotations

import argparse
from pathlib import Path

from structured_latent_hypothesis.plotting import load_results, plot_gain_vs_commutator, plot_metric_vs_lambda, plot_pareto
from structured_latent_hypothesis.synthetic import run_benchmark_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a matched commutator ladder benchmark.")
    parser.add_argument("--output-dir", default="results/commutator_ladder_v1")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=20)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 11, 29])
    parser.add_argument(
        "--worlds",
        nargs="+",
        default=["matched_comm_0.00", "matched_comm_0.10", "matched_comm_0.20", "matched_comm_0.35", "matched_comm_0.50"],
    )
    parser.add_argument("--split-strategy", default="cartesian_blocks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_recipes = {
        "baseline": {},
        "smooth": {"lambda_smooth": 0.05},
        "cfp_l0.010": {"lambda_comm": 0.01},
        "cfp_l0.050": {"lambda_comm": 0.05},
    }
    variants = list(variant_recipes.keys())
    results_path = output_dir / "results.json"
    summary_path = output_dir / "summary.md"
    run_benchmark_suite(
        seeds=args.seeds,
        worlds=args.worlds,
        variants=variants,
        variant_recipes=variant_recipes,
        output_json=str(results_path),
        output_markdown=str(summary_path),
        split_strategy=args.split_strategy,
        grid_size=args.grid_size,
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
    )

    results = load_results(results_path)
    plot_gain_vs_commutator(results, output_dir / "gain_vs_commutator.png")
    plot_pareto(results, output_dir / "pareto.png")
    plot_metric_vs_lambda(results, "test_recon_mse", "Held-out reconstruction across the matched ladder", output_dir / "test_recon.png")

    observations = []
    for world in results["worlds"]:
        magnitude = results["world_metadata"][world]["ground_truth_commutator"]
        baseline = results["summary"][world]["baseline"]["test_recon_mse"]["mean"]
        cfp_small = results["summary"][world]["cfp_l0.010"]["test_recon_mse"]["mean"]
        cfp_mid = results["summary"][world]["cfp_l0.050"]["test_recon_mse"]["mean"]
        observations.append(
            f"- `{world}`: commutator `{magnitude:.6f}`, baseline `{baseline:.6f}`, cfp_l0.010 `{cfp_small:.6f}`, cfp_l0.050 `{cfp_mid:.6f}`."
        )

    report_lines = [
        "# Matched Commutator Ladder V1",
        "",
        f"Split strategy: `{results['split_strategy']}`",
        "",
        "## Observations",
        "",
        *observations,
        "",
        "## Plots",
        "",
        "![Gain vs commutator](gain_vs_commutator.png)",
        "",
        "![Pareto](pareto.png)",
        "",
        "![Held-out reconstruction](test_recon.png)",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
