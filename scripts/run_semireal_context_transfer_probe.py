from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from structured_latent_hypothesis.context_transfer import run_context_transfer_suite
from structured_latent_hypothesis.plotting import load_results
from structured_latent_hypothesis.transfer_criterion import (
    annotate_transfer_tasks,
    build_context_transfer_rows,
    evaluate_transfer_decision_policy,
    load_json,
    select_transfer_decision_policy,
)


SAFE_SCORE_KEYS = ["score_residual", "score_joint_sum"]
BUDGET_SCORE_KEYS = ["score_interaction", "score_joint_sum", "score_joint_prod"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run semi-real context-transfer and external policy validation.")
    parser.add_argument("--output-dir", default="results/semireal_context_transfer_v1")
    parser.add_argument("--report-path", default="reports/2026-04-26_semireal_context_transfer_v1_findings.md")
    parser.add_argument("--followup-path", default="reports/2026-04-26_semireal_context_transfer_followup.md")
    parser.add_argument("--train-operator-results", default="results/context_operator_probe_v1/results.json")
    parser.add_argument("--train-adaptation-results", default="results/context_adaptation_probe_v1/results.json")
    parser.add_argument("--context-count", type=int, default=5)
    parser.add_argument("--state-count", type=int, default=10)
    parser.add_argument("--action-count", type=int, default=4)
    parser.add_argument("--rollout-length", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=18)
    parser.add_argument("--latent-dim", type=int, default=20)
    parser.add_argument("--context-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--adapt-steps", type=int, default=30)
    parser.add_argument("--adapt-lr", type=float, default=6e-2)
    parser.add_argument("--support-fraction", type=float, default=0.35)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 11])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.00, 0.20, 0.35, 0.75])
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--regret-tolerance", type=float, default=1e-5)
    parser.add_argument("--structured-violation-cost", type=float, default=5.0)
    parser.add_argument("--fallback-overbudget-cost", type=float, default=3.0)
    parser.add_argument("--escalate-needed-cost", type=float, default=0.5)
    parser.add_argument("--escalate-unneeded-cost", type=float, default=1.0)
    return parser.parse_args()


def world_name(family: str, alpha: float) -> str:
    return f"semireal_context_{family}_{alpha:0.2f}"


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


def alpha(world: str) -> float:
    return float(world.rsplit("_", maxsplit=1)[-1])


def plot_metric(results: dict, output_path: Path, metric: str, title: str, ylabel: str) -> None:
    worlds = sorted(results["worlds"], key=lambda item: (family(item), alpha(item)))
    figure, axis = plt.subplots(1, 1, figsize=(10.5, 4.8))
    xs = list(range(len(worlds)))
    for variant in results["variants"]:
        values = [results["summary"][world][variant][metric]["mean"] for world in worlds]
        axis.plot(xs, values, marker="o", linewidth=1.8, label=variant)
    axis.set_xticks(xs)
    axis.set_xticklabels(worlds, rotation=28, ha="right", fontsize=8)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def external_policy_eval(args: argparse.Namespace, semireal_results: dict) -> dict:
    train_operator = load_json(args.train_operator_results)
    train_adaptation = load_json(args.train_adaptation_results)
    train_rows = build_context_transfer_rows(train_operator, train_adaptation)
    semireal_rows = build_context_transfer_rows(semireal_results, semireal_results)

    annotated_train = annotate_transfer_tasks(
        train_rows,
        step_budgets=[args.budget],
        regret_tolerances=[args.regret_tolerance],
    )
    annotated_semireal = annotate_transfer_tasks(
        semireal_rows,
        step_budgets=[args.budget],
        regret_tolerances=[args.regret_tolerance],
    )
    diag_train = [row for row in annotated_train if row["variant"] == "operator_diag_residual"]
    diag_semireal = [row for row in annotated_semireal if row["variant"] == "operator_diag_residual"]
    tolerance_key = format(args.regret_tolerance, ".0e") if args.regret_tolerance > 0 else "0"
    safe_label_key = f"task_safe_regret_{tolerance_key}"
    budget_label_key = f"task_within_budget_{args.budget}"

    selected = select_transfer_decision_policy(
        diag_train,
        safe_score_keys=SAFE_SCORE_KEYS,
        budget_score_keys=BUDGET_SCORE_KEYS,
        safe_label_key=safe_label_key,
        budget_label_key=budget_label_key,
        structured_violation_cost=args.structured_violation_cost,
        fallback_overbudget_cost=args.fallback_overbudget_cost,
        escalate_needed_cost=args.escalate_needed_cost,
        escalate_unneeded_cost=args.escalate_unneeded_cost,
    )
    metrics = evaluate_transfer_decision_policy(
        diag_semireal,
        safe_score_key=selected["safe_score_key"],
        budget_score_key=selected["budget_score_key"],
        safe_threshold=selected["safe_classifier"]["threshold"],
        safe_direction=selected["safe_classifier"]["direction"],
        safe_band=selected["safe_classifier"]["band"],
        budget_threshold=selected["budget_classifier"]["threshold"],
        budget_direction=selected["budget_classifier"]["direction"],
        budget_band=selected["budget_classifier"]["band"],
        safe_label_key=safe_label_key,
        budget_label_key=budget_label_key,
        structured_violation_cost=args.structured_violation_cost,
        fallback_overbudget_cost=args.fallback_overbudget_cost,
        escalate_needed_cost=args.escalate_needed_cost,
        escalate_unneeded_cost=args.escalate_unneeded_cost,
    )
    best_trivial = min(
        metrics["always_structured_cost"],
        metrics["always_fallback_cost"],
        metrics["always_escalate_cost"],
    )
    return {
        "safe_label_key": safe_label_key,
        "budget_label_key": budget_label_key,
        "selected": selected,
        "metrics": {key: value for key, value in metrics.items() if key != "per_row"},
        "best_trivial_cost": best_trivial,
        "delta_to_best_trivial": metrics["average_cost"] - best_trivial,
        "wins_best_trivial": metrics["average_cost"] < best_trivial - 1e-12,
        "per_row": metrics["per_row"],
    }


def write_outputs(output_dir: Path, report_path: Path, followup_path: Path, results: dict, policy: dict) -> None:
    worlds = sorted(results["worlds"], key=lambda item: (family(item), alpha(item)))
    summary_lines = [
        "# Semi-Real Context Transfer v1",
        "",
        "## External Policy",
        "",
        f"Policy cost: `{policy['metrics']['average_cost']:.6f}`",
        f"Best trivial cost: `{policy['best_trivial_cost']:.6f}`",
        f"Delta: `{policy['delta_to_best_trivial']:+.6f}`",
        f"Wins best trivial: `{policy['wins_best_trivial']}`",
        "",
        "## Transfer Metrics",
        "",
        "| World | Full Rollout@5 | Diag Rollout@5 | Full Steps | Diag Steps | Diag Interaction |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for world in worlds:
        full = results["summary"][world]["full_transition"]
        diag = results["summary"][world]["operator_diag_residual"]
        summary_lines.append(
            "| "
            + world
            + f" | {full['zero_shot_rollout5_mse']['mean']:.6f}"
            + f" | {diag['zero_shot_rollout5_mse']['mean']:.6f}"
            + f" | {full['adaptation_steps_to_target']['mean']:.2f}"
            + f" | {diag['adaptation_steps_to_target']['mean']:.2f}"
            + f" | {diag['interaction_norm_holdout']['mean']:.6f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    (output_dir / "report.md").write_text(
        "\n".join(
            [
                "# Semi-Real Context Transfer v1",
                "",
                "## Plots",
                "",
                "![Rollout@5](zero_shot_rollout5.png)",
                "",
                "![Adaptation steps](adaptation_steps.png)",
                "",
                "![Holdout interaction](holdout_interaction.png)",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    findings = [
        "# Semi-Real Context Transfer v1",
        "",
        "## Result",
        "",
        f"- External policy cost `{policy['metrics']['average_cost']:.6f}` vs best trivial `{policy['best_trivial_cost']:.6f}`; delta `{policy['delta_to_best_trivial']:+.6f}`.",
        f"- Action rates: `structured={policy['metrics']['action_rates']['structured']:.3f}`, `fallback={policy['metrics']['action_rates']['fallback']:.3f}`, `escalate={policy['metrics']['action_rates']['escalate']:.3f}`.",
        "",
        "## Interpretation",
        "",
        "This is the first semi-real transition check for the transfer decision law. It uses a policy selected on the earlier synthetic context benchmark and evaluates it on visually richer context-transfer worlds.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(findings) + "\n", encoding="utf-8")

    followup = [
        "# Semi-Real Context Transfer Follow-Up",
        "",
        "## Main Result",
        "",
        f"The synthetic-calibrated decision law {'beats' if policy['wins_best_trivial'] else 'does not beat'} the best trivial baseline on the semi-real context-transfer probe.",
        "",
        "## Interpretation",
        "",
        "A win here would be the first sign that the transfer criterion is not only an internal synthetic artifact. A loss means the next step must address representation/domain shift before claiming practical transfer utility.",
    ]
    followup_path.parent.mkdir(parents=True, exist_ok=True)
    followup_path.write_text("\n".join(followup) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    worlds = [world_name(family_name, value) for family_name in ("commuting", "coupled") for value in args.alphas]
    variant_recipes = build_variant_recipes(args)
    variants = list(variant_recipes.keys())
    results_path = output_dir / "results.json"

    run_context_transfer_suite(
        seeds=args.seeds,
        worlds=worlds,
        variants=variants,
        variant_recipes=variant_recipes,
        output_json=str(results_path),
        output_markdown=None,
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
    policy = external_policy_eval(args, results)
    results["external_policy"] = policy
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_metric(results, output_dir / "zero_shot_rollout5.png", "zero_shot_rollout5_mse", "Semi-real context rollout@5", "rollout@5 mse")
    plot_metric(results, output_dir / "adaptation_steps.png", "adaptation_steps_to_target", "Semi-real context adaptation steps", "steps")
    plot_metric(results, output_dir / "holdout_interaction.png", "interaction_norm_holdout", "Semi-real context interaction norm", "interaction norm")
    write_outputs(output_dir, Path(args.report_path), Path(args.followup_path), results, policy)


if __name__ == "__main__":
    main()
