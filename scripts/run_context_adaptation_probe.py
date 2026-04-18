from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from structured_latent_hypothesis.context_transfer import run_context_transfer_suite
from structured_latent_hypothesis.plotting import load_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the context-transfer adaptation benchmark.")
    parser.add_argument("--output-dir", default="results/context_adaptation_probe_v1")
    parser.add_argument("--report-path", default="reports/2026-04-19_context_adaptation_probe_v1_findings.md")
    parser.add_argument("--context-count", type=int, default=5)
    parser.add_argument("--state-count", type=int, default=12)
    parser.add_argument("--action-count", type=int, default=4)
    parser.add_argument("--rollout-length", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=20)
    parser.add_argument("--latent-dim", type=int, default=24)
    parser.add_argument("--context-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--adapt-steps", type=int, default=40)
    parser.add_argument("--adapt-lr", type=float, default=6e-2)
    parser.add_argument("--support-fraction", type=float, default=0.35)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 11, 29])
    parser.add_argument(
        "--worlds",
        nargs="+",
        default=[
            "context_commuting_0.00",
            "context_commuting_0.10",
            "context_commuting_0.20",
            "context_commuting_0.35",
            "context_commuting_0.50",
            "context_commuting_0.75",
            "context_commuting_1.00",
            "context_coupled_0.00",
            "context_coupled_0.10",
            "context_coupled_0.20",
            "context_coupled_0.35",
            "context_coupled_0.50",
            "context_coupled_0.75",
            "context_coupled_1.00",
        ],
    )
    return parser.parse_args()


def build_variant_recipes(args: argparse.Namespace) -> dict[str, dict]:
    return {
        "full_transition": {
            "model_type": "full_transition",
            "lambda_residual": 0.0,
            "adapt_steps": args.adapt_steps,
            "adapt_lr": args.adapt_lr,
            "support_fraction": args.support_fraction,
        },
        "commuting_operator": {
            "model_type": "commuting_operator",
            "lambda_residual": 0.0,
            "adapt_steps": args.adapt_steps,
            "adapt_lr": args.adapt_lr,
            "support_fraction": args.support_fraction,
        },
        "operator_plus_residual": {
            "model_type": "operator_plus_residual",
            "interaction_rank": 4,
            "lambda_residual": 2e-3,
            "adapt_steps": args.adapt_steps,
            "adapt_lr": args.adapt_lr,
            "support_fraction": args.support_fraction,
        },
        "operator_diag_residual": {
            "model_type": "operator_diag_residual",
            "interaction_rank": 4,
            "lambda_residual": 2e-3,
            "adapt_steps": args.adapt_steps,
            "adapt_lr": args.adapt_lr,
            "support_fraction": args.support_fraction,
        },
    }


def family(world: str) -> str:
    return "commuting" if "commuting" in world else "coupled"


def coupling(world: str) -> float:
    return float(world.rsplit("_", maxsplit=1)[-1])


def closest_world(worlds: list[str], family_name: str, target: float) -> str:
    family_worlds = [world for world in worlds if family(world) == family_name]
    return min(family_worlds, key=lambda world: abs(coupling(world) - target))


def plot_metric(results: dict, output_path: Path, metric: str, title: str, ylabel: str) -> None:
    worlds = results["worlds"]
    figure, axis = plt.subplots(1, 1, figsize=(11.0, 4.8))
    for variant in results["variants"]:
        values = [results["summary"][world][variant][metric]["mean"] for world in worlds]
        axis.plot(range(len(worlds)), values, marker="o", linewidth=1.9, label=variant)
    axis.set_xticks(range(len(worlds)))
    axis.set_xticklabels(worlds, rotation=28, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_report(output_dir: Path, report_path: Path, results: dict) -> None:
    summary = results["summary"]
    low_world = closest_world(results["worlds"], "commuting", 0.20)
    boundary_world = closest_world(results["worlds"], "coupled", 0.35)
    tail_world = closest_world(results["worlds"], "coupled", 0.75)
    findings_lines = [
        "# Context Adaptation Probe v1",
        "",
        "## Result",
        "",
        f"- `{low_world}` adaptation gain: `commuting_operator` `{summary[low_world]['commuting_operator']['adaptation_adaptation_gain']['mean']:+.6f}`, `operator_plus_residual` `{summary[low_world]['operator_plus_residual']['adaptation_adaptation_gain']['mean']:+.6f}`, `full_transition` `{summary[low_world]['full_transition']['adaptation_adaptation_gain']['mean']:+.6f}`.",
        f"- `{boundary_world}` steps-to-target: `commuting_operator` `{summary[boundary_world]['commuting_operator']['adaptation_steps_to_target']['mean']:.2f}`, `operator_plus_residual` `{summary[boundary_world]['operator_plus_residual']['adaptation_steps_to_target']['mean']:.2f}`, `full_transition` `{summary[boundary_world]['full_transition']['adaptation_steps_to_target']['mean']:.2f}`.",
        f"- `{tail_world}` residual norm final: `commuting_operator` `{summary[tail_world]['commuting_operator']['adaptation_residual_norm_final']['mean']:.6f}`, `operator_plus_residual` `{summary[tail_world]['operator_plus_residual']['adaptation_residual_norm_final']['mean']:.6f}`, `full_transition` `{summary[tail_world]['full_transition']['adaptation_residual_norm_final']['mean']:.6f}`.",
        "",
        "## Interpretation",
        "",
        "This probe is the transfer-cost check for the context branch: structured models should need less adaptation in low coupling and lose that advantage as coupling grows.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(findings_lines) + "\n", encoding="utf-8")

    report_lines = [
        "# Context Adaptation Probe v1",
        "",
        "## Plots",
        "",
        "![Adaptation gain](adaptation_gain.png)",
        "",
        "![Steps to target](adaptation_steps.png)",
        "",
        "![Residual norm final](adaptation_residual_norm.png)",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    summary_path = output_dir / "summary.md"

    run_context_transfer_suite(
        seeds=args.seeds,
        worlds=args.worlds,
        variants=list(build_variant_recipes(args).keys()),
        variant_recipes=build_variant_recipes(args),
        output_json=str(results_path),
        output_markdown=str(summary_path),
        context_count=args.context_count,
        state_count=args.state_count,
        action_count=args.action_count,
        rollout_length=args.rollout_length,
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        context_dim=args.context_dim,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        evaluate_adaptation=True,
    )

    results = load_results(results_path)
    plot_metric(results, output_dir / "adaptation_gain.png", "adaptation_adaptation_gain", "Context adaptation gain", "gain over zero-shot")
    plot_metric(results, output_dir / "adaptation_steps.png", "adaptation_steps_to_target", "Context adaptation steps-to-target", "steps")
    plot_metric(results, output_dir / "adaptation_residual_norm.png", "adaptation_residual_norm_final", "Adapted residual norm", "residual norm")
    write_report(output_dir, Path(args.report_path), results)


if __name__ == "__main__":
    main()
