from __future__ import annotations

import argparse
from collections import Counter
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
    parser.add_argument("--family", choices=["ramp", "scale", "rotation"], default="ramp")
    parser.add_argument("--worlds", nargs="+", default=None)
    parser.add_argument("--split-strategy", default="cartesian_blocks")
    parser.add_argument("--selection-mode", choices=["manual", "nested"], default="manual")
    parser.add_argument("--lambda-grid", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05, 0.10])
    parser.add_argument("--inner-train-fraction", type=float, default=0.72)
    return parser.parse_args()


def default_worlds(family: str) -> list[str]:
    if family == "ramp":
        prefix = "matched_comm"
        values = (0.00, 0.10, 0.20, 0.35, 0.50)
    elif family == "scale":
        prefix = "matched_scale"
        values = (0.00, 0.10, 0.20, 0.35, 0.50)
    else:
        prefix = "matched_rotate"
        values = (0.00, 5.00, 10.00, 20.00, 30.00)
    return [f"{prefix}_{value:0.2f}" for value in values]


def format_lambda(value: float) -> str:
    return f"{value:0.3f}"


def build_variant_recipes(args: argparse.Namespace) -> dict[str, dict]:
    recipes: dict[str, dict] = {
        "baseline": {},
        "smooth": {"lambda_smooth": 0.05},
        "cfp_l0.010": {"lambda_comm": 0.01},
        "cfp_l0.050": {"lambda_comm": 0.05},
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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    worlds = args.worlds or default_worlds(args.family)

    variant_recipes = build_variant_recipes(args)
    variants = list(variant_recipes.keys())
    results_path = output_dir / "results.json"
    summary_path = output_dir / "summary.md"
    run_benchmark_suite(
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
    plot_gain_vs_commutator(results, output_dir / "gain_vs_commutator.png")
    plot_pareto(results, output_dir / "pareto.png")
    plot_metric_vs_lambda(results, "test_recon_mse", "Held-out reconstruction across the matched ladder", output_dir / "test_recon.png")

    observations = []
    for world in results["worlds"]:
        magnitude = results["world_metadata"][world]["ground_truth_commutator"]
        baseline = results["summary"][world]["baseline"]["test_recon_mse"]["mean"]
        parts = [f"commutator `{magnitude:.6f}`", f"baseline `{baseline:.6f}`"]
        if "smooth" in results["summary"][world]:
            smooth = results["summary"][world]["smooth"]["test_recon_mse"]["mean"]
            parts.append(f"smooth `{smooth:.6f}`")
        if "cfp_l0.010" in results["summary"][world]:
            cfp_small = results["summary"][world]["cfp_l0.010"]["test_recon_mse"]["mean"]
            parts.append(f"cfp_l0.010 `{cfp_small:.6f}`")
        if "cfp_l0.050" in results["summary"][world]:
            cfp_mid = results["summary"][world]["cfp_l0.050"]["test_recon_mse"]["mean"]
            parts.append(f"cfp_l0.050 `{cfp_mid:.6f}`")
        if "step_selected" in results["summary"][world]:
            step_selected = results["summary"][world]["step_selected"]["test_recon_mse"]["mean"]
            step_choices = selection_summary(results, world, "step_selected")
            parts.append(f"step_selected `{step_selected:.6f}` ({step_choices})")
        if "cfp_selected" in results["summary"][world]:
            cfp_selected = results["summary"][world]["cfp_selected"]["test_recon_mse"]["mean"]
            cfp_choices = selection_summary(results, world, "cfp_selected")
            parts.append(f"cfp_selected `{cfp_selected:.6f}` ({cfp_choices})")
        observations.append(f"- `{world}`: " + ", ".join(parts) + ".")

    report_lines = [
        f"# Matched Commutator Ladder ({args.family})",
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
