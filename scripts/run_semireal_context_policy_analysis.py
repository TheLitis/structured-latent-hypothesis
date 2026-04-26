from __future__ import annotations

import argparse
import json
from pathlib import Path

from structured_latent_hypothesis.transfer_criterion import (
    annotate_transfer_tasks,
    build_context_transfer_rows,
    cross_validate_transfer_decision_policy,
    load_json,
    spearman_correlation,
)


SAFE_SCORE_KEYS = ["score_residual", "score_joint_sum"]
BUDGET_SCORE_KEYS = ["score_interaction", "score_joint_sum", "score_joint_prod"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze semi-real context policy calibration without retraining.")
    parser.add_argument("--semireal-results", default="results/semireal_context_transfer_v1/results.json")
    parser.add_argument("--output-dir", default="results/semireal_context_policy_analysis_v1")
    parser.add_argument("--report-path", default="reports/2026-04-26_semireal_context_policy_analysis_v1_findings.md")
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--regret-tolerance", type=float, default=1e-5)
    parser.add_argument("--structured-violation-cost", type=float, default=5.0)
    parser.add_argument("--fallback-overbudget-cost", type=float, default=3.0)
    parser.add_argument("--escalate-needed-cost", type=float, default=0.5)
    parser.add_argument("--escalate-unneeded-cost", type=float, default=1.0)
    return parser.parse_args()


def compact_metrics(metrics: dict) -> dict:
    compact = {key: value for key, value in metrics.items() if key != "per_group"}
    compact["per_group"] = []
    for row in metrics["per_group"]:
        compact["per_group"].append({key: value for key, value in row.items() if key != "per_row"})
    return compact


def best_trivial(metrics: dict) -> float:
    return min(
        metrics["always_structured_cost_mean"],
        metrics["always_fallback_cost_mean"],
        metrics["always_escalate_cost_mean"],
    )


def score_correlations(rows: list[dict]) -> dict:
    targets = {
        "structured_rollout5_regret": [
            row["structured_zero_shot_rollout5_mse"] - row["full_zero_shot_rollout5_mse"] for row in rows
        ],
        "full_adaptation_steps": [float(row["full_adaptation_steps"]) for row in rows],
        "structured_adaptation_steps": [float(row["structured_adaptation_steps"]) for row in rows],
    }
    scores = {
        "score_interaction": [float(row["score_interaction"]) for row in rows],
        "score_residual": [float(row["score_residual"]) for row in rows],
        "score_joint_sum": [float(row["score_joint_sum"]) for row in rows],
        "score_joint_prod": [float(row["score_joint_prod"]) for row in rows],
    }
    return {
        score_name: {
            target_name: spearman_correlation(score_values, target_values)
            for target_name, target_values in targets.items()
        }
        for score_name, score_values in scores.items()
    }


def main() -> None:
    args = parse_args()
    semireal_results = load_json(args.semireal_results)
    rows = build_context_transfer_rows(semireal_results, semireal_results)
    annotated = annotate_transfer_tasks(rows, step_budgets=[args.budget], regret_tolerances=[args.regret_tolerance])
    diag_rows = [row for row in annotated if row["variant"] == "operator_diag_residual"]
    tolerance_key = format(args.regret_tolerance, ".0e") if args.regret_tolerance > 0 else "0"
    safe_label_key = f"task_safe_regret_{tolerance_key}"
    budget_label_key = f"task_within_budget_{args.budget}"

    cv = {}
    for group in ("seed", "world", "family"):
        metrics = cross_validate_transfer_decision_policy(
            diag_rows,
            group_key=group,
            safe_score_keys=SAFE_SCORE_KEYS,
            budget_score_keys=BUDGET_SCORE_KEYS,
            safe_label_key=safe_label_key,
            budget_label_key=budget_label_key,
            structured_violation_cost=args.structured_violation_cost,
            fallback_overbudget_cost=args.fallback_overbudget_cost,
            escalate_needed_cost=args.escalate_needed_cost,
            escalate_unneeded_cost=args.escalate_unneeded_cost,
        )
        cv[group] = {
            "metrics": compact_metrics(metrics),
            "best_trivial_cost": best_trivial(metrics),
            "delta_to_best_trivial": metrics["average_cost_mean"] - best_trivial(metrics),
            "wins": metrics["average_cost_mean"] < best_trivial(metrics) - 1e-12,
        }

    correlations = score_correlations(diag_rows)
    results = {
        "budget": args.budget,
        "regret_tolerance": args.regret_tolerance,
        "safe_label_key": safe_label_key,
        "budget_label_key": budget_label_key,
        "cross_validation": cv,
        "score_correlations": correlations,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    summary_lines = [
        "# Semi-Real Context Policy Analysis v1",
        "",
        "| Group | Policy Cost | Best Trivial | Delta | Wins | Structured Rate | Fallback Rate | Escalate Rate |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for group, row in cv.items():
        metrics = row["metrics"]
        summary_lines.append(
            "| "
            + group
            + f" | {metrics['average_cost_mean']:.6f}"
            + f" | {row['best_trivial_cost']:.6f}"
            + f" | {row['delta_to_best_trivial']:+.6f}"
            + f" | {row['wins']}"
            + f" | {metrics['structured_rate_mean']:.3f}"
            + f" | {metrics['fallback_rate_mean']:.3f}"
            + f" | {metrics['escalate_rate_mean']:.3f} |"
        )
    summary_lines.extend(
        [
            "",
            "## Score Correlations",
            "",
            "| Score | Regret Spearman | Full Steps Spearman | Structured Steps Spearman |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for score, values in correlations.items():
        summary_lines.append(
            "| "
            + score
            + f" | {values['structured_rollout5_regret']:+.3f}"
            + f" | {values['full_adaptation_steps']:+.3f}"
            + f" | {values['structured_adaptation_steps']:+.3f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text("# Semi-Real Context Policy Analysis v1\n\nSee `summary.md`.\n", encoding="utf-8")

    best_group = min(cv.items(), key=lambda item: item[1]["delta_to_best_trivial"])
    findings = [
        "# Semi-Real Context Policy Analysis v1",
        "",
        "## Result",
        "",
        f"- Best in-domain split: `{best_group[0]}` with delta `{best_group[1]['delta_to_best_trivial']:+.6f}`.",
        f"- Groups won: `{sum(int(row['wins']) for row in cv.values())}/3`.",
        f"- Best regret correlation: `{max(correlations, key=lambda key: correlations[key]['structured_rollout5_regret'])}` = `{max(value['structured_rollout5_regret'] for value in correlations.values()):+.3f}`.",
        "",
        "## Interpretation",
        "",
        "This analysis checks whether the semi-real failure is only an external-threshold problem. If in-domain cross-validation also loses to trivial baselines, the next step must address the score representation itself, not just calibration.",
    ]
    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_path).write_text("\n".join(findings) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
