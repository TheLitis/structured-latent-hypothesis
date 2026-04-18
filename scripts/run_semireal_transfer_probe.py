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
    parser = argparse.ArgumentParser(description="Run semi-real compositional transfer probe.")
    parser.add_argument("--output-dir", default="results/semireal_transfer_probe_v1")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=24)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=340)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 11, 29])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.00, 0.20, 0.35, 0.50, 0.75])
    parser.add_argument("--split-strategy", default="cartesian_blocks")
    parser.add_argument("--inner-train-fraction", type=float, default=0.72)
    return parser.parse_args()


def world_name(alpha: float) -> str:
    return f"semireal_coupled_{alpha:0.2f}"


def selection_grid(model_type: str, interaction_rank: int | None = None) -> dict[str, dict]:
    base = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
    recipes = {}
    for value in base:
        key = f"l{value:0.3f}"
        recipe = {"model_type": model_type, "lambda_residual": float(value)}
        if interaction_rank is not None:
            recipe["interaction_rank"] = int(interaction_rank)
        recipes[key] = recipe
    return recipes


def selected_recipe(model_type: str, interaction_rank: int | None, inner_train_fraction: float) -> dict:
    recipe: dict[str, object] = {"model_type": model_type}
    if interaction_rank is not None:
        recipe["interaction_rank"] = interaction_rank
    recipe["selection"] = {
        "metric": "test_recon_mse",
        "inner_train_fraction": float(inner_train_fraction),
        "candidates": selection_grid(model_type, interaction_rank=interaction_rank),
    }
    return recipe


def build_variant_recipes(args: argparse.Namespace) -> dict[str, dict]:
    return {
        "coord_latent": {"model_type": "coord"},
        "additive_resid_selected": selected_recipe("additive_residual", None, args.inner_train_fraction),
        "curv_hankel_r4_selected": selected_recipe("additive_curvature_hankel", 4, args.inner_train_fraction),
        "curvature_field_r4_selected": selected_recipe("additive_curvature_field", 4, args.inner_train_fraction),
        "operator_diag_r2_selected": selected_recipe("additive_operator_diag", 2, args.inner_train_fraction),
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


def ordered_worlds(results: dict) -> list[str]:
    return sorted(results["worlds"], key=lambda world: float(results["world_metadata"][world]["ground_truth_coupling_strength"] or 0.0))


def plot_metric(results: dict, output_path: str | Path, metric: str, title: str, ylabel: str) -> None:
    worlds = ordered_worlds(results)
    couplings = [float(results["world_metadata"][world]["ground_truth_coupling_strength"] or 0.0) for world in worlds]
    figure, axis = plt.subplots(1, 1, figsize=(8.8, 4.8))
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
    worlds = [world_name(alpha) for alpha in args.alphas]
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
    plot_metric(results, output_dir / "test_recon_vs_coupling.png", "test_recon_mse", "Semi-real compositional transfer", "test recon mse")
    plot_metric(results, output_dir / "generalization_gap_vs_coupling.png", "generalization_gap", "Generalization gap on semi-real scenes", "test-train gap")
    plot_metric(results, output_dir / "residual_norm_vs_coupling.png", "residual_norm_train", "Train interaction norm on semi-real scenes", "train interaction norm")

    observations = []
    for world in ordered_worlds(results):
        meta = results["world_metadata"][world]
        parts = [f"coupling `{meta['ground_truth_coupling_strength']:.6f}`"]
        for variant in results["variants"]:
            value = results["summary"][world][variant]["test_recon_mse"]["mean"]
            text = f"{variant} `{value:.6f}`"
            summary = selection_summary(results, world, variant)
            if summary:
                text += f" ({summary})"
            parts.append(text)
        observations.append(f"- `{world}`: " + ", ".join(parts) + ".")

    report_lines = [
        "# Semi-Real Transfer Probe",
        "",
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
        "![Generalization gap](generalization_gap_vs_coupling.png)",
        "",
        "![Interaction norm](residual_norm_vs_coupling.png)",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
