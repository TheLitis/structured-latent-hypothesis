from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from structured_latent_hypothesis.plotting import load_results, plot_pareto
from structured_latent_hypothesis.synthetic import run_benchmark_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the step-stationarity probe derived from the original triplet geometry.")
    parser.add_argument("--output-dir", default="results/step_stationarity_probe_v1")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=20)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 11, 29])
    parser.add_argument(
        "--worlds",
        nargs="+",
        default=["stepcurve_1.00", "stepcurve_1.50", "stepcurve_2.00", "stepcurve_3.00", "stepcurve_4.00"],
    )
    parser.add_argument("--split-strategy", default="cartesian_blocks")
    parser.add_argument("--selection-mode", choices=["manual", "nested"], default="nested")
    parser.add_argument("--lambda-grid", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05, 0.10])
    parser.add_argument("--inner-train-fraction", type=float, default=0.72)
    return parser.parse_args()


def format_lambda(value: float) -> str:
    return f"{value:0.3f}"


def build_variant_recipes(args: argparse.Namespace) -> dict[str, dict]:
    recipes: dict[str, dict] = {
        "baseline": {},
        "smooth": {"lambda_smooth": 0.05},
        "cfp_l0.010": {"lambda_comm": 0.01},
        "affine_l0.010_s0.005": {"lambda_comm": 0.01, "lambda_step": 0.005},
    }
    if args.selection_mode == "nested":
        cfp_candidates = {
            f"cfp_candidate_l{format_lambda(value)}": {"lambda_comm": float(value)}
            for value in args.lambda_grid
        }
        step_candidates = {
            f"step_candidate_l{format_lambda(value)}": {"lambda_step": float(value)}
            for value in args.lambda_grid
        }
        affine_step_values = [float(value) for value in args.lambda_grid[:2]]
        affine_candidates = {
            f"affine_candidate_c{format_lambda(comm_value)}_s{format_lambda(step_value)}": {
                "lambda_comm": float(comm_value),
                "lambda_step": float(step_value),
            }
            for comm_value in args.lambda_grid
            for step_value in affine_step_values
            if float(step_value) <= float(comm_value)
        }
        recipes["step_selected"] = {
            "selection": {
                "metric": "test_recon_mse",
                "inner_train_fraction": float(args.inner_train_fraction),
                "candidates": step_candidates,
            }
        }
        recipes["cfp_selected"] = {
            "selection": {
                "metric": "test_recon_mse",
                "inner_train_fraction": float(args.inner_train_fraction),
                "candidates": cfp_candidates,
            }
        }
        recipes["affine_selected"] = {
            "selection": {
                "metric": "test_recon_mse",
                "inner_train_fraction": float(args.inner_train_fraction),
                "candidates": affine_candidates,
            }
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


def plot_gain_vs_step_drift(results: dict, output_path: str | Path) -> None:
    worlds = results["worlds"]
    x_values = [results["world_metadata"][world]["ground_truth_step_drift"] for world in worlds]
    figure, axis = plt.subplots(1, 1, figsize=(7, 4.5))

    for variant in results["variants"]:
        if variant == "baseline":
            continue
        gains = []
        for world in worlds:
            baseline = results["summary"][world]["baseline"]["test_recon_mse"]["mean"]
            variant_value = results["summary"][world][variant]["test_recon_mse"]["mean"]
            gains.append(baseline - variant_value)
        axis.plot(x_values, gains, marker="o", linewidth=2.0, label=variant)

    axis.axhline(0.0, color="#444444", linestyle="--", linewidth=1.2)
    axis.set_xlabel("ground-truth step-drift magnitude")
    axis.set_ylabel("baseline - variant test recon mse")
    axis.set_title("Gain vs step-drift magnitude")
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
    plot_gain_vs_step_drift(results, output_dir / "gain_vs_step_drift.png")
    plot_pareto(results, output_dir / "pareto.png")

    observations = []
    for world in results["worlds"]:
        metadata = results["world_metadata"][world]
        parts = [
            f"commutator `{metadata['ground_truth_commutator']:.6f}`",
            f"step_drift `{metadata['ground_truth_step_drift']:.6f}`",
            f"baseline `{results['summary'][world]['baseline']['test_recon_mse']['mean']:.6f}`",
            f"cfp_l0.010 `{results['summary'][world]['cfp_l0.010']['test_recon_mse']['mean']:.6f}`",
            f"affine_l0.010_s0.005 `{results['summary'][world]['affine_l0.010_s0.005']['test_recon_mse']['mean']:.6f}`",
        ]
        if "step_selected" in results["summary"][world]:
            parts.append(
                f"step_selected `{results['summary'][world]['step_selected']['test_recon_mse']['mean']:.6f}`"
                f" ({selection_summary(results, world, 'step_selected')})"
            )
        if "cfp_selected" in results["summary"][world]:
            parts.append(
                f"cfp_selected `{results['summary'][world]['cfp_selected']['test_recon_mse']['mean']:.6f}`"
                f" ({selection_summary(results, world, 'cfp_selected')})"
            )
        if "affine_selected" in results["summary"][world]:
            parts.append(
                f"affine_selected `{results['summary'][world]['affine_selected']['test_recon_mse']['mean']:.6f}`"
                f" ({selection_summary(results, world, 'affine_selected')})"
            )
        observations.append(f"- `{world}`: " + ", ".join(parts) + ".")

    report_lines = [
        "# Step Stationarity Probe",
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
        "![Gain vs step drift](gain_vs_step_drift.png)",
        "",
        "![Pareto](pareto.png)",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
