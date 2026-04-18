from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from structured_latent_hypothesis.context_transfer import run_context_transfer_suite
from structured_latent_hypothesis.plotting import load_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the context-operator zero-shot benchmark.")
    parser.add_argument("--output-dir", default="results/context_operator_probe_v1")
    parser.add_argument("--report-path", default="reports/2026-04-19_context_operator_probe_v1_findings.md")
    parser.add_argument("--context-count", type=int, default=5)
    parser.add_argument("--state-count", type=int, default=12)
    parser.add_argument("--action-count", type=int, default=4)
    parser.add_argument("--rollout-length", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=20)
    parser.add_argument("--latent-dim", type=int, default=24)
    parser.add_argument("--context-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=220)
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


def build_variant_recipes() -> dict[str, dict]:
    return {
        "full_transition": {"model_type": "full_transition", "lambda_residual": 0.0},
        "commuting_operator": {"model_type": "commuting_operator", "lambda_residual": 0.0},
        "operator_plus_residual": {
            "model_type": "operator_plus_residual",
            "interaction_rank": 4,
            "lambda_residual": 2e-3,
        },
        "operator_diag_residual": {
            "model_type": "operator_diag_residual",
            "interaction_rank": 4,
            "lambda_residual": 2e-3,
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
    worlds = results["worlds"]
    summary = results["summary"]
    observations = []
    for world in worlds:
        parts = [
            f"family `{family(world)}`",
            f"coupling `{coupling(world):.2f}`",
            f"full `{summary[world]['full_transition']['zero_shot_one_step_mse']['mean']:.6f}`",
            f"commuting `{summary[world]['commuting_operator']['zero_shot_one_step_mse']['mean']:.6f}`",
            f"plus_resid `{summary[world]['operator_plus_residual']['zero_shot_one_step_mse']['mean']:.6f}`",
            f"diag_resid `{summary[world]['operator_diag_residual']['zero_shot_one_step_mse']['mean']:.6f}`",
        ]
        observations.append(f"- `{world}`: " + ", ".join(parts) + ".")

    report_lines = [
        "# Context Operator Probe v1",
        "",
        f"Split strategy: `{results['split_strategy']}`",
        "",
        "## Observations",
        "",
        *observations,
        "",
        "## Plots",
        "",
        "![Zero-shot one-step](zero_shot_one_step.png)",
        "",
        "![Zero-shot rollout@5](zero_shot_rollout5.png)",
        "",
        "![Holdout interaction norm](holdout_interaction.png)",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    low_world = closest_world(worlds, "commuting", 0.20)
    high_world = closest_world(worlds, "coupled", 0.75)
    findings_lines = [
        "# Context Operator Probe v1",
        "",
        "## Result",
        "",
        f"- Low-coupling commuting check `{low_world}`: `commuting_operator` `{summary[low_world]['commuting_operator']['zero_shot_one_step_mse']['mean']:.6f}`, `operator_plus_residual` `{summary[low_world]['operator_plus_residual']['zero_shot_one_step_mse']['mean']:.6f}`, `full_transition` `{summary[low_world]['full_transition']['zero_shot_one_step_mse']['mean']:.6f}`.",
        f"- Strong-coupling check `{high_world}`: `commuting_operator` `{summary[high_world]['commuting_operator']['zero_shot_one_step_mse']['mean']:.6f}`, `operator_plus_residual` `{summary[high_world]['operator_plus_residual']['zero_shot_one_step_mse']['mean']:.6f}`, `full_transition` `{summary[high_world]['full_transition']['zero_shot_one_step_mse']['mean']:.6f}`.",
        "",
        "## Interpretation",
        "",
        "This probe should be read as the zero-shot part of the context-transfer claim. Adaptation cost is reported separately in the adaptation probe.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(findings_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    summary_path = output_dir / "summary.md"

    run_context_transfer_suite(
        seeds=args.seeds,
        worlds=args.worlds,
        variants=list(build_variant_recipes().keys()),
        variant_recipes=build_variant_recipes(),
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
        evaluate_adaptation=False,
    )

    results = load_results(results_path)
    plot_metric(results, output_dir / "zero_shot_one_step.png", "zero_shot_one_step_mse", "Context transfer zero-shot one-step", "zero-shot one-step mse")
    plot_metric(results, output_dir / "zero_shot_rollout5.png", "zero_shot_rollout5_mse", "Context transfer zero-shot rollout@5", "rollout@5 mse")
    plot_metric(results, output_dir / "holdout_interaction.png", "interaction_norm_holdout", "Holdout interaction norm", "interaction norm")
    write_report(output_dir, Path(args.report_path), results)


if __name__ == "__main__":
    main()
