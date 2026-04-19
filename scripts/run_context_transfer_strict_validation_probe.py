from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from structured_latent_hypothesis.transfer_criterion import (
    CANDIDATE_SPECS,
    analyze_context_transfer_budget,
    build_context_transfer_rows,
    cross_validate_abstain_by_group,
    cross_validate_classifier_by_group,
    load_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict group-based validation for transfer-law budget criteria.")
    parser.add_argument("--operator-results", default="results/context_operator_probe_v1/results.json")
    parser.add_argument("--adaptation-results", default="results/context_adaptation_probe_v1/results.json")
    parser.add_argument("--output-dir", default="results/context_transfer_strict_validation_v1")
    parser.add_argument("--report-path", default="reports/2026-04-19_context_transfer_strict_validation_v1_findings.md")
    parser.add_argument("--step-budgets", type=int, nargs="+", default=[6, 8, 10])
    parser.add_argument("--regret-tolerances", type=float, nargs="+", default=[1e-5])
    parser.add_argument("--false-positive-costs", type=float, nargs="+", default=[2.0, 3.0, 5.0])
    parser.add_argument("--false-negative-cost", type=float, default=1.0)
    parser.add_argument("--abstain-positive-cost", type=float, default=0.75)
    parser.add_argument("--abstain-negative-cost", type=float, default=0.25)
    return parser.parse_args()


def build_task_specs(step_budgets: list[int], regret_tolerances: list[float]) -> dict[str, str]:
    task_specs: dict[str, str] = {}
    for budget in step_budgets:
        task_specs[f"within_budget_{budget}"] = f"task_within_budget_{budget}"
    for tolerance in regret_tolerances:
        suffix = format(tolerance, ".0e") if tolerance > 0 else "0"
        task_specs[f"safe_regret_{suffix}"] = f"task_safe_regret_{suffix}"
    return task_specs


def analyze_strict_validation(
    rows: list[dict],
    *,
    step_budgets: list[int],
    regret_tolerances: list[float],
    false_positive_costs: list[float],
    false_negative_cost: float,
    abstain_positive_cost: float,
    abstain_negative_cost: float,
) -> dict:
    base = analyze_context_transfer_budget(
        rows,
        step_budgets=step_budgets,
        regret_tolerances=regret_tolerances,
        false_positive_cost=false_positive_costs[0],
        false_negative_cost=false_negative_cost,
    )
    annotated_rows = base["rows"]
    task_specs = build_task_specs(step_budgets, regret_tolerances)
    analyses: dict[str, dict] = {}
    for candidate_name, (variant, score_suffix) in CANDIDATE_SPECS.items():
        variant_rows = [row for row in annotated_rows if row["variant"] == variant]
        score_key = f"score_{score_suffix}"
        task_results: dict[str, dict] = {}
        for task_name, label_key in task_specs.items():
            cost_results: dict[str, dict] = {}
            for fp_cost in false_positive_costs:
                cost_key = f"fp{fp_cost:.1f}_fn{false_negative_cost:.1f}"
                cv_results: dict[str, dict] = {}
                for group_key in ("seed", "world", "family"):
                    binary = cross_validate_classifier_by_group(
                        variant_rows,
                        score_key,
                        label_key,
                        group_key=group_key,
                        false_positive_cost=fp_cost,
                        false_negative_cost=false_negative_cost,
                    )
                    abstain = cross_validate_abstain_by_group(
                        variant_rows,
                        score_key,
                        label_key,
                        group_key=group_key,
                        false_positive_cost=fp_cost,
                        false_negative_cost=false_negative_cost,
                        abstain_positive_cost=abstain_positive_cost,
                        abstain_negative_cost=abstain_negative_cost,
                    )
                    cv_results[group_key] = {"binary": binary, "abstain": abstain}
                cost_results[cost_key] = cv_results
            task_results[task_name] = cost_results
        analyses[candidate_name] = {"variant": variant, "score_key": score_key, "tasks": task_results}
    return {
        "rows": annotated_rows,
        "task_specs": task_specs,
        "false_positive_costs": false_positive_costs,
        "false_negative_cost": false_negative_cost,
        "abstain_positive_cost": abstain_positive_cost,
        "abstain_negative_cost": abstain_negative_cost,
        "analyses": analyses,
    }


def pick_best(results: dict, task_name: str, cost_key: str, group_key: str, mode: str) -> tuple[str, dict]:
    return min(
        results["analyses"].items(),
        key=lambda item: item[1]["tasks"][task_name][cost_key][group_key][mode]["average_cost_mean"],
    )


def render_summary(results: dict) -> str:
    lines = ["# Context Transfer Strict Validation", ""]
    for task_name in results["task_specs"]:
        lines.extend(
            [
                f"## Task: {task_name}",
                "",
                "| Cost Pair | Group | Best Binary | Binary Cost | Best Abstain | Abstain Cost | Trivial Baseline |",
                "| --- | --- | --- | ---: | --- | ---: | ---: |",
            ]
        )
        for fp_cost in results["false_positive_costs"]:
            cost_key = f"fp{fp_cost:.1f}_fn{results['false_negative_cost']:.1f}"
            for group_key in ("seed", "world", "family"):
                best_binary_name, best_binary = pick_best(results, task_name, cost_key, group_key, "binary")
                best_abstain_name, best_abstain = pick_best(results, task_name, cost_key, group_key, "abstain")
                binary_cost = best_binary["tasks"][task_name][cost_key][group_key]["binary"]["average_cost_mean"]
                abstain_cost = best_abstain["tasks"][task_name][cost_key][group_key]["abstain"]["average_cost_mean"]
                binary_trivial = min(
                    best_binary["tasks"][task_name][cost_key][group_key]["binary"]["always_positive_cost_mean"],
                    best_binary["tasks"][task_name][cost_key][group_key]["binary"]["always_negative_cost_mean"],
                )
                abstain_trivial = min(
                    best_abstain["tasks"][task_name][cost_key][group_key]["abstain"]["always_positive_cost_mean"],
                    best_abstain["tasks"][task_name][cost_key][group_key]["abstain"]["always_negative_cost_mean"],
                    best_abstain["tasks"][task_name][cost_key][group_key]["abstain"]["always_abstain_cost_mean"],
                )
                lines.append(
                    "| "
                    + cost_key
                    + f" | {group_key}"
                    + f" | {best_binary_name}"
                    + f" | {binary_cost:.6f}"
                    + f" | {best_abstain_name}"
                    + f" | {abstain_cost:.6f}"
                    + f" | {min(binary_trivial, abstain_trivial):.6f} |"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def plot_best_costs(results: dict, output_path: Path, task_name: str, cost_key: str) -> None:
    groups = ["seed", "world", "family"]
    binary_costs = [pick_best(results, task_name, cost_key, group, "binary")[1]["tasks"][task_name][cost_key][group]["binary"]["average_cost_mean"] for group in groups]
    abstain_costs = [pick_best(results, task_name, cost_key, group, "abstain")[1]["tasks"][task_name][cost_key][group]["abstain"]["average_cost_mean"] for group in groups]
    trivial_costs = []
    for group in groups:
        _, best_abstain = pick_best(results, task_name, cost_key, group, "abstain")
        abstain = best_abstain["tasks"][task_name][cost_key][group]["abstain"]
        _, best_binary = pick_best(results, task_name, cost_key, group, "binary")
        binary = best_binary["tasks"][task_name][cost_key][group]["binary"]
        trivial_costs.append(
            min(
                binary["always_positive_cost_mean"],
                binary["always_negative_cost_mean"],
                abstain["always_positive_cost_mean"],
                abstain["always_negative_cost_mean"],
                abstain["always_abstain_cost_mean"],
            )
        )

    figure, axis = plt.subplots(1, 1, figsize=(8.8, 4.8))
    xs = list(range(len(groups)))
    axis.plot(xs, binary_costs, marker="o", linewidth=1.8, label="best binary")
    axis.plot(xs, abstain_costs, marker="s", linewidth=1.8, label="best abstain")
    axis.plot(xs, trivial_costs, marker="^", linewidth=1.8, label="best trivial")
    axis.set_xticks(xs)
    axis.set_xticklabels(groups)
    axis.set_ylabel("average cost")
    axis.set_title(f"{task_name} @ {cost_key}")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_report(output_dir: Path, report_path: Path, results: dict) -> None:
    primary_cost_key = f"fp3.0_fn{results['false_negative_cost']:.1f}"
    tasks = list(results["task_specs"])
    report_lines = ["# Context Transfer Strict Validation v1", "", "## Plots", ""]
    for task_name in tasks:
        plot_name = f"{task_name}_{primary_cost_key}.png"
        report_lines.extend([f"![{task_name}]({plot_name})", ""])
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    findings = ["# Context Transfer Strict Validation v1", "", "## Result", ""]
    for task_name in tasks:
        for group_key in ("seed", "world", "family"):
            best_binary_name, best_binary = pick_best(results, task_name, primary_cost_key, group_key, "binary")
            best_abstain_name, best_abstain = pick_best(results, task_name, primary_cost_key, group_key, "abstain")
            binary = best_binary["tasks"][task_name][primary_cost_key][group_key]["binary"]
            abstain = best_abstain["tasks"][task_name][primary_cost_key][group_key]["abstain"]
            trivial = min(
                binary["always_positive_cost_mean"],
                binary["always_negative_cost_mean"],
                abstain["always_positive_cost_mean"],
                abstain["always_negative_cost_mean"],
                abstain["always_abstain_cost_mean"],
            )
            binary_note = "beats trivial baseline" if binary["average_cost_mean"] < trivial - 1e-12 else "does not beat trivial baseline"
            abstain_note = "beats trivial baseline" if abstain["average_cost_mean"] < trivial - 1e-12 else "does not beat trivial baseline"
            findings.append(
                f"- `{task_name}` / `{group_key}`: best binary `{best_binary_name}` cost `{binary['average_cost_mean']:.6f}` ({binary_note}); best abstain `{best_abstain_name}` cost `{abstain['average_cost_mean']:.6f}` ({abstain_note}), coverage `{abstain['coverage_mean']:.3f}`."
            )
    findings.extend(
        [
            "",
            "## Interpretation",
            "",
            "This probe is the strict version of the budget claim. It only counts if the criterion survives harder cross-validation splits and keeps beating the best trivial baseline under asymmetric costs, with or without abstention.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(findings) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    operator_results = load_json(args.operator_results)
    adaptation_results = load_json(args.adaptation_results)
    rows = build_context_transfer_rows(operator_results, adaptation_results)
    results = analyze_strict_validation(
        rows,
        step_budgets=args.step_budgets,
        regret_tolerances=args.regret_tolerances,
        false_positive_costs=args.false_positive_costs,
        false_negative_cost=args.false_negative_cost,
        abstain_positive_cost=args.abstain_positive_cost,
        abstain_negative_cost=args.abstain_negative_cost,
    )
    results["operator_results"] = str(args.operator_results)
    results["adaptation_results"] = str(args.adaptation_results)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(render_summary(results), encoding="utf-8")
    primary_cost_key = f"fp3.0_fn{args.false_negative_cost:.1f}"
    for task_name in results["task_specs"]:
        plot_best_costs(results, output_dir / f"{task_name}_{primary_cost_key}.png", task_name, primary_cost_key)
    write_report(output_dir, Path(args.report_path), results)


if __name__ == "__main__":
    main()
