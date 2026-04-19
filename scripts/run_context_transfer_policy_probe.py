from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from structured_latent_hypothesis.transfer_criterion import (
    annotate_transfer_tasks,
    build_context_transfer_rows,
    cross_validate_transfer_decision_policy,
    load_json,
)


SAFE_SCORE_KEYS = ["score_residual", "score_joint_sum"]
BUDGET_SCORE_KEYS = ["score_interaction", "score_joint_sum", "score_joint_prod"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the explicit transfer decision pipeline benchmark.")
    parser.add_argument("--operator-results", default="results/context_operator_probe_v1/results.json")
    parser.add_argument("--adaptation-results", default="results/context_adaptation_probe_v1/results.json")
    parser.add_argument("--output-dir", default="results/context_transfer_policy_v1")
    parser.add_argument("--report-path", default="reports/2026-04-19_context_transfer_policy_v1_findings.md")
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--regret-tolerance", type=float, default=1e-5)
    parser.add_argument("--structured-violation-cost", type=float, default=5.0)
    parser.add_argument("--fallback-overbudget-cost", type=float, default=2.0)
    parser.add_argument("--escalate-needed-cost", type=float, default=1.0)
    parser.add_argument("--escalate-unneeded-cost", type=float, default=1.5)
    return parser.parse_args()


def plot_costs(results: dict, output_path: Path) -> None:
    groups = ["seed", "world", "family"]
    pipeline = [results["cross_validation"][group]["average_cost_mean"] for group in groups]
    oracle = [results["cross_validation"][group]["oracle_average_cost_mean"] for group in groups]
    fallback = [results["cross_validation"][group]["always_fallback_cost_mean"] for group in groups]
    escalate = [results["cross_validation"][group]["always_escalate_cost_mean"] for group in groups]
    structured = [results["cross_validation"][group]["always_structured_cost_mean"] for group in groups]

    figure, axis = plt.subplots(1, 1, figsize=(8.8, 4.8))
    xs = list(range(len(groups)))
    axis.plot(xs, pipeline, marker="o", linewidth=1.9, label="policy")
    axis.plot(xs, oracle, marker="*", linewidth=1.4, label="oracle")
    axis.plot(xs, fallback, marker="s", linewidth=1.4, label="always_fallback")
    axis.plot(xs, escalate, marker="^", linewidth=1.4, label="always_escalate")
    axis.plot(xs, structured, marker="d", linewidth=1.4, label="always_structured")
    axis.set_xticks(xs)
    axis.set_xticklabels(groups)
    axis.set_ylabel("average cost")
    axis.set_title("Explicit transfer decision policy")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_action_rates(results: dict, output_path: Path) -> None:
    groups = ["seed", "world", "family"]
    structured = [results["cross_validation"][group]["structured_rate_mean"] for group in groups]
    fallback = [results["cross_validation"][group]["fallback_rate_mean"] for group in groups]
    escalate = [results["cross_validation"][group]["escalate_rate_mean"] for group in groups]

    figure, axis = plt.subplots(1, 1, figsize=(8.8, 4.8))
    xs = list(range(len(groups)))
    axis.plot(xs, structured, marker="o", linewidth=1.9, label="structured")
    axis.plot(xs, fallback, marker="s", linewidth=1.9, label="fallback")
    axis.plot(xs, escalate, marker="^", linewidth=1.9, label="escalate")
    axis.set_xticks(xs)
    axis.set_xticklabels(groups)
    axis.set_ylabel("rate")
    axis.set_title("Policy action rates")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_report(output_dir: Path, report_path: Path, results: dict) -> None:
    report_lines = [
        "# Context Transfer Policy v1",
        "",
        "## Plots",
        "",
        "![Policy costs](policy_costs.png)",
        "",
        "![Action rates](policy_action_rates.png)",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    findings = ["# Context Transfer Policy v1", "", "## Result", ""]
    for group in ("seed", "world", "family"):
        metrics = results["cross_validation"][group]
        best_baseline = min(
            metrics["always_structured_cost_mean"],
            metrics["always_fallback_cost_mean"],
            metrics["always_escalate_cost_mean"],
        )
        verdict = "beats best trivial baseline" if metrics["average_cost_mean"] < best_baseline - 1e-12 else "does not beat best trivial baseline"
        findings.append(
            f"- `{group}`: policy cost `{metrics['average_cost_mean']:.6f}` vs best trivial `{best_baseline:.6f}`; {verdict}. Action mix `structured={metrics['structured_rate_mean']:.3f}`, `fallback={metrics['fallback_rate_mean']:.3f}`, `escalate={metrics['escalate_rate_mean']:.3f}`."
        )
    findings.extend(
        [
            "",
            "## Interpretation",
            "",
            "This probe evaluates the strongest current practical claim: whether triple-derived scores can drive an explicit accept / fallback / escalate policy with lower total cost than trivial policies.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(findings) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    operator_results = load_json(args.operator_results)
    adaptation_results = load_json(args.adaptation_results)
    rows = build_context_transfer_rows(operator_results, adaptation_results)
    annotated = annotate_transfer_tasks(rows, step_budgets=[args.budget], regret_tolerances=[args.regret_tolerance])
    diag_rows = [row for row in annotated if row["variant"] == "operator_diag_residual"]

    tolerance_key = format(args.regret_tolerance, ".0e") if args.regret_tolerance > 0 else "0"
    safe_label_key = f"task_safe_regret_{tolerance_key}"
    budget_label_key = f"task_within_budget_{args.budget}"

    cross_validation = {}
    for group_key in ("seed", "world", "family"):
        cross_validation[group_key] = cross_validate_transfer_decision_policy(
            diag_rows,
            group_key=group_key,
            safe_score_keys=SAFE_SCORE_KEYS,
            budget_score_keys=BUDGET_SCORE_KEYS,
            safe_label_key=safe_label_key,
            budget_label_key=budget_label_key,
            structured_violation_cost=args.structured_violation_cost,
            fallback_overbudget_cost=args.fallback_overbudget_cost,
            escalate_needed_cost=args.escalate_needed_cost,
            escalate_unneeded_cost=args.escalate_unneeded_cost,
        )

    results = {
        "budget": args.budget,
        "regret_tolerance": args.regret_tolerance,
        "structured_violation_cost": args.structured_violation_cost,
        "fallback_overbudget_cost": args.fallback_overbudget_cost,
        "escalate_needed_cost": args.escalate_needed_cost,
        "escalate_unneeded_cost": args.escalate_unneeded_cost,
        "safe_label_key": safe_label_key,
        "budget_label_key": budget_label_key,
        "safe_score_keys": SAFE_SCORE_KEYS,
        "budget_score_keys": BUDGET_SCORE_KEYS,
        "cross_validation": cross_validation,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary_lines = [
        "# Context Transfer Policy v1",
        "",
        "| Group | Policy Cost | Oracle | Always Structured | Always Fallback | Always Escalate | Structured Rate | Fallback Rate | Escalate Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in ("seed", "world", "family"):
        metrics = cross_validation[group]
        summary_lines.append(
            "| "
            + group
            + f" | {metrics['average_cost_mean']:.6f}"
            + f" | {metrics['oracle_average_cost_mean']:.6f}"
            + f" | {metrics['always_structured_cost_mean']:.6f}"
            + f" | {metrics['always_fallback_cost_mean']:.6f}"
            + f" | {metrics['always_escalate_cost_mean']:.6f}"
            + f" | {metrics['structured_rate_mean']:.3f}"
            + f" | {metrics['fallback_rate_mean']:.3f}"
            + f" | {metrics['escalate_rate_mean']:.3f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    plot_costs(results, output_dir / "policy_costs.png")
    plot_action_rates(results, output_dir / "policy_action_rates.png")
    write_report(output_dir, Path(args.report_path), results)


if __name__ == "__main__":
    main()
