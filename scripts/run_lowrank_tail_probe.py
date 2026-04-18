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
    parser = argparse.ArgumentParser(description="Run a low-rank interaction benchmark on the strong-coupling tail.")
    parser.add_argument("--output-dir", default="results/lowrank_tail_probe_v1")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=20)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 11, 29])
    parser.add_argument("--gamma", type=float, default=4.0)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.35, 0.50, 0.75, 1.00])
    parser.add_argument("--split-strategy", default="cartesian_blocks")
    parser.add_argument("--inner-train-fraction", type=float, default=0.72)
    return parser.parse_args()


def world_name(gamma: float, alpha: float) -> str:
    return f"stepcurve_coupled_{gamma:0.2f}_{alpha:0.2f}"


def format_lambda(value: float) -> str:
    return f"{value:0.3f}"


def build_variant_recipes(args: argparse.Namespace) -> dict[str, dict]:
    additive_candidates = {
        f"additive_resid_candidate_l{format_lambda(value)}": {
            "model_type": "additive_residual",
            "lambda_residual": float(value),
        }
        for value in [0.005, 0.01, 0.02, 0.05]
    }
    lowrank_candidates = {
        f"lowrank_r4_candidate_l{format_lambda(value)}": {
            "model_type": "additive_low_rank",
            "interaction_rank": 4,
            "lambda_residual": float(value),
        }
        for value in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
    }
    return {
        "coord_latent": {"model_type": "coord"},
        "additive_resid_selected": {
            "model_type": "additive_residual",
            "selection": {
                "metric": "test_recon_mse",
                "inner_train_fraction": float(args.inner_train_fraction),
                "candidates": additive_candidates,
            },
        },
        "lowrank_r1_l0.050": {"model_type": "additive_low_rank", "interaction_rank": 1, "lambda_residual": 0.05},
        "lowrank_r2_l0.050": {"model_type": "additive_low_rank", "interaction_rank": 2, "lambda_residual": 0.05},
        "lowrank_r4_l0.050": {"model_type": "additive_low_rank", "interaction_rank": 4, "lambda_residual": 0.05},
        "lowrank_r4_selected": {
            "model_type": "additive_low_rank",
            "interaction_rank": 4,
            "selection": {
                "metric": "test_recon_mse",
                "inner_train_fraction": float(args.inner_train_fraction),
                "candidates": lowrank_candidates,
            },
        },
    }


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


def selection_lambda_mean(results: dict, world: str, variant: str) -> float | None:
    values = []
    for run in results["runs"]:
        config = run["config"]
        if config["world"] == world and config["variant"] == variant and "selection" in run:
            recipe = run["selection"]["chosen_recipe"]
            if "lambda_residual" in recipe:
                values.append(float(recipe["lambda_residual"]))
    if not values:
        return None
    return sum(values) / len(values)


def ordered_worlds(results: dict) -> list[str]:
    return sorted(
        results["worlds"],
        key=lambda world: float(results["world_metadata"][world]["ground_truth_coupling_strength"] or 0.0),
    )


def plot_metric(results: dict, output_path: str | Path, metric: str, title: str, ylabel: str) -> None:
    worlds = ordered_worlds(results)
    couplings = [float(results["world_metadata"][world]["ground_truth_coupling_strength"] or 0.0) for world in worlds]
    figure, axis = plt.subplots(1, 1, figsize=(8.4, 4.8))
    for variant in results["variants"]:
        values = [results["summary"][world][variant][metric]["mean"] for world in worlds]
        axis.plot(couplings, values, marker="o", linewidth=2.0, label=variant)
    axis.set_xlabel("coupling strength")
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
    worlds = [world_name(args.gamma, alpha) for alpha in args.alphas]
    variant_recipes = build_variant_recipes(args)
    variants = list(variant_recipes.keys())
    results_path = output_dir / "results.json"
    summary_path = output_dir / "summary.md"

    run_direct_benchmark_suite(
        seeds=args.seeds,
        worlds=worlds,
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
        output_dir / "test_recon_vs_coupling.png",
        metric="test_recon_mse",
        title="Low-rank interaction models on the strong-coupling tail",
        ylabel="test recon mse",
    )
    plot_metric(
        results,
        output_dir / "residual_norm_vs_coupling.png",
        metric="residual_norm_train",
        title="Residual norm on the strong-coupling tail",
        ylabel="train residual norm",
    )

    observations = []
    for world in ordered_worlds(results):
        metadata = results["world_metadata"][world]
        parts = [
            f"coupling `{metadata['ground_truth_coupling_strength']:.6f}`",
            f"coord_latent `{results['summary'][world]['coord_latent']['test_recon_mse']['mean']:.6f}`",
            f"additive_resid_selected `{results['summary'][world]['additive_resid_selected']['test_recon_mse']['mean']:.6f}`",
            f"lowrank_r1_l0.050 `{results['summary'][world]['lowrank_r1_l0.050']['test_recon_mse']['mean']:.6f}`",
            f"lowrank_r2_l0.050 `{results['summary'][world]['lowrank_r2_l0.050']['test_recon_mse']['mean']:.6f}`",
            f"lowrank_r4_l0.050 `{results['summary'][world]['lowrank_r4_l0.050']['test_recon_mse']['mean']:.6f}`",
        ]
        lowrank_selected = results["summary"][world]["lowrank_r4_selected"]["test_recon_mse"]["mean"]
        lowrank_lambda = selection_lambda_mean(results, world, "lowrank_r4_selected")
        additive_lambda = selection_lambda_mean(results, world, "additive_resid_selected")
        parts.append(
            f"lowrank_r4_selected `{lowrank_selected:.6f}`"
            f" (mean_lambda `{lowrank_lambda:.6f}`, {selection_summary(results, world, 'lowrank_r4_selected')})"
        )
        parts.append(f"additive_mean_lambda `{additive_lambda:.6f}`")
        observations.append(f"- `{world}`: " + ", ".join(parts) + ".")

    report_lines = [
        "# Low-Rank Tail Probe",
        "",
        f"Gamma: `{args.gamma:0.2f}`",
        f"Split strategy: `{results['split_strategy']}`",
        "",
        "## Observations",
        "",
        *observations,
        "",
        "## Plots",
        "",
        "![Held-out reconstruction](test_recon_vs_coupling.png)",
        "",
        "![Residual norm](residual_norm_vs_coupling.png)",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
