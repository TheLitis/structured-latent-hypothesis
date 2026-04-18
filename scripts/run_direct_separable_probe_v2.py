from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from structured_latent_hypothesis.direct_separable import run_direct_benchmark_suite
from structured_latent_hypothesis.plotting import load_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a direct separable benchmark with a coordinate baseline.")
    parser.add_argument("--output-dir", default="results/direct_separable_probe_v2")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=20)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 11, 29])
    parser.add_argument(
        "--worlds",
        nargs="+",
        default=[
            "stepcurve_1.00",
            "stepcurve_path_1.00",
            "stepcurve_2.00",
            "stepcurve_path_2.00",
            "stepcurve_4.00",
            "stepcurve_path_4.00",
        ],
    )
    parser.add_argument("--split-strategy", default="cartesian_blocks")
    parser.add_argument("--selection-mode", choices=["manual", "nested"], default="nested")
    parser.add_argument("--lambda-grid", type=float, nargs="+", default=[0.001, 0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--inner-train-fraction", type=float, default=0.72)
    return parser.parse_args()


def format_lambda(value: float) -> str:
    return f"{value:0.3f}"


def build_variant_recipes(args: argparse.Namespace) -> dict[str, dict]:
    recipes: dict[str, dict] = {
        "cell_latent": {"model_type": "cell"},
        "coord_latent": {"model_type": "coord"},
        "additive_only": {"model_type": "additive"},
        "additive_resid_l0.010": {"model_type": "additive_residual", "lambda_residual": 0.01},
    }
    if args.selection_mode == "nested":
        candidates = {
            f"additive_resid_candidate_l{format_lambda(value)}": {
                "model_type": "additive_residual",
                "lambda_residual": float(value),
            }
            for value in args.lambda_grid
        }
        recipes["additive_resid_selected"] = {
            "model_type": "additive_residual",
            "selection": {
                "metric": "test_recon_mse",
                "inner_train_fraction": float(args.inner_train_fraction),
                "candidates": candidates,
            },
        }
    return recipes


def selection_summary(results: dict, world: str, variant: str) -> str | None:
    chosen = []
    for run in results["runs"]:
        config = run["config"]
        if config["world"] == world and config["variant"] == variant and "selection" in run:
            chosen.append(run["selection"]["chosen_candidate"])
    if not chosen:
        return None
    counts = Counter(chosen)
    return ", ".join(f"{name} x{counts[name]}" for name in sorted(counts))


def world_family(world: str) -> str:
    return "path" if "path" in world else "separable"


def plot_metric(results: dict, output_path: str | Path, metric: str, title: str, ylabel: str) -> None:
    worlds = results["worlds"]
    x_positions = range(len(worlds))
    figure, axis = plt.subplots(1, 1, figsize=(10, 4.8))
    for variant in results["variants"]:
        values = [results["summary"][world][variant][metric]["mean"] for world in worlds]
        axis.plot(list(x_positions), values, marker="o", linewidth=2.0, label=variant)
    axis.set_xticks(list(x_positions))
    axis.set_xticklabels(worlds, rotation=30, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variant_recipes = build_variant_recipes(args)
    variants = list(variant_recipes.keys())
    results_path = output_dir / "results.json"
    summary_path = output_dir / "summary.md"

    run_direct_benchmark_suite(
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
    plot_metric(
        results,
        output_dir / "test_recon_by_world.png",
        metric="test_recon_mse",
        title="Direct separable models with coordinate baseline",
        ylabel="test recon mse",
    )
    plot_metric(
        results,
        output_dir / "generalization_gap_by_world.png",
        metric="generalization_gap",
        title="Generalization gap across matched worlds",
        ylabel="test-train recon gap",
    )
    plot_metric(
        results,
        output_dir / "residual_norm_by_world.png",
        metric="residual_norm_train",
        title="Residual interaction norm across worlds",
        ylabel="train residual norm",
    )

    observations = []
    for world in results["worlds"]:
        meta = results["world_metadata"][world]
        parts = [
            f"family `{world_family(world)}`",
            f"step_drift `{meta['ground_truth_step_drift']:.6f}`",
            f"commutator `{meta['ground_truth_commutator']:.6f}`",
            f"cell_latent `{results['summary'][world]['cell_latent']['test_recon_mse']['mean']:.6f}`",
            f"coord_latent `{results['summary'][world]['coord_latent']['test_recon_mse']['mean']:.6f}`",
            f"additive_only `{results['summary'][world]['additive_only']['test_recon_mse']['mean']:.6f}`",
            f"additive_resid_l0.010 `{results['summary'][world]['additive_resid_l0.010']['test_recon_mse']['mean']:.6f}`",
        ]
        if "additive_resid_selected" in results["summary"][world]:
            value = results["summary"][world]["additive_resid_selected"]["test_recon_mse"]["mean"]
            parts.append(
                f"additive_resid_selected `{value:.6f}` ({selection_summary(results, world, 'additive_resid_selected')})"
            )
        observations.append(f"- `{world}`: " + ", ".join(parts) + ".")

    report_lines = [
        "# Direct Separable Probe v2",
        "",
        f"Split strategy: `{results['split_strategy']}`",
        f"Selection mode: `{args.selection_mode}`",
        "",
        "## Observations",
        "",
        *observations,
        "",
        "## Plots",
        "",
        "![Test reconstruction by world](test_recon_by_world.png)",
        "",
        "![Generalization gap by world](generalization_gap_by_world.png)",
        "",
        "![Residual norm by world](residual_norm_by_world.png)",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
