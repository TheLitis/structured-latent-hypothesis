from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from structured_latent_hypothesis.plotting import load_results, plot_pareto
from structured_latent_hypothesis.synthetic import run_benchmark_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the separable-vs-path-dependent transport probe.")
    parser.add_argument("--output-dir", default="results/transport_pair_probe_v1")
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


def base_world_name(world: str) -> str:
    return world.replace("stepcurve_path_", "").replace("stepcurve_", "")


def world_family(world: str) -> str:
    return "path" if "stepcurve_path_" in world else "separable"


def plot_pairwise_gain(results: dict, output_path: str | Path, variant: str) -> None:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for world in results["worlds"]:
        drift = results["world_metadata"][world]["ground_truth_step_drift"]
        baseline = results["summary"][world]["baseline"]["test_recon_mse"]["mean"]
        value = results["summary"][world][variant]["test_recon_mse"]["mean"]
        grouped[world_family(world)].append((drift, baseline - value))

    figure, axis = plt.subplots(1, 1, figsize=(7, 4.5))
    colors = {"separable": "#1f77b4", "path": "#d62728"}
    for family, points in grouped.items():
        points = sorted(points)
        axis.plot([point[0] for point in points], [point[1] for point in points], marker="o", linewidth=2.0, color=colors[family], label=family)

    axis.axhline(0.0, color="#444444", linestyle="--", linewidth=1.2)
    axis.set_xlabel("ground-truth step-drift magnitude")
    axis.set_ylabel(f"baseline - {variant} test recon mse")
    axis.set_title(f"{variant} gain on separable vs path-dependent worlds")
    axis.grid(alpha=0.25)
    axis.legend()
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
    plot_pairwise_gain(results, output_dir / "cfp_selected_pair_gain.png", "cfp_selected")
    plot_pairwise_gain(results, output_dir / "affine_selected_pair_gain.png", "affine_selected")
    plot_pareto(results, output_dir / "pareto.png")

    observations = []
    for world in results["worlds"]:
        metadata = results["world_metadata"][world]
        parts = [
            f"family `{world_family(world)}`",
            f"step_drift `{metadata['ground_truth_step_drift']:.6f}`",
            f"commutator `{metadata['ground_truth_commutator']:.6f}`",
            f"baseline `{results['summary'][world]['baseline']['test_recon_mse']['mean']:.6f}`",
            f"cfp_l0.010 `{results['summary'][world]['cfp_l0.010']['test_recon_mse']['mean']:.6f}`",
            f"affine_l0.010_s0.005 `{results['summary'][world]['affine_l0.010_s0.005']['test_recon_mse']['mean']:.6f}`",
        ]
        for variant in ["step_selected", "cfp_selected", "affine_selected"]:
            if variant in results["summary"][world]:
                parts.append(
                    f"{variant} `{results['summary'][world][variant]['test_recon_mse']['mean']:.6f}`"
                    f" ({selection_summary(results, world, variant)})"
                )
        observations.append(f"- `{world}`: " + ", ".join(parts) + ".")

    report_lines = [
        "# Transport Pair Probe",
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
        "![CFP selected pair gain](cfp_selected_pair_gain.png)",
        "",
        "![Affine selected pair gain](affine_selected_pair_gain.png)",
        "",
        "![Pareto](pareto.png)",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
