from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from structured_latent_hypothesis.transfer_criterion import (
    analyze_context_transfer_budget,
    build_context_transfer_rows,
    load_json,
    render_budget_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cost-sensitive budget and safe-use criteria on context-transfer results.")
    parser.add_argument("--operator-results", default="results/context_operator_probe_v1/results.json")
    parser.add_argument("--adaptation-results", default="results/context_adaptation_probe_v1/results.json")
    parser.add_argument("--output-dir", default="results/context_transfer_budget_v1")
    parser.add_argument("--report-path", default="reports/2026-04-19_context_transfer_budget_v1_findings.md")
    parser.add_argument("--step-budgets", type=int, nargs="+", default=[4, 6, 8, 10])
    parser.add_argument("--regret-tolerances", type=float, nargs="+", default=[0.0, 1e-5, 5e-5])
    parser.add_argument("--false-positive-cost", type=float, default=3.0)
    parser.add_argument("--false-negative-cost", type=float, default=1.0)
    return parser.parse_args()


def plot_task_costs(results: dict, output_path: Path, task_name: str) -> None:
    names = list(results["analyses"].keys())
    loo = [results["analyses"][name]["tasks"][task_name]["average_cost_mean"] for name in names]
    pos = [results["analyses"][name]["tasks"][task_name]["always_positive_cost_mean"] for name in names]
    neg = [results["analyses"][name]["tasks"][task_name]["always_negative_cost_mean"] for name in names]
    xs = range(len(names))

    figure, axis = plt.subplots(1, 1, figsize=(10.5, 4.8))
    axis.plot(xs, loo, marker="o", linewidth=1.8, label="loo classifier")
    axis.plot(xs, pos, marker="s", linewidth=1.4, label="always positive")
    axis.plot(xs, neg, marker="^", linewidth=1.4, label="always negative")
    axis.set_xticks(list(xs))
    axis.set_xticklabels(names, rotation=24, ha="right")
    axis.set_ylabel("average cost")
    axis.set_title(task_name)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_report(output_dir: Path, report_path: Path, results: dict) -> None:
    task_summaries = []
    for task_name in results["task_specs"]:
        best = min(
            results["analyses"].items(),
            key=lambda item: item[1]["tasks"][task_name]["average_cost_mean"],
        )
        task = best[1]["tasks"][task_name]
        baseline = min(task["always_positive_cost_mean"], task["always_negative_cost_mean"])
        delta = task["average_cost_mean"] - baseline
        task_summaries.append((task_name, best[0], task["average_cost_mean"], baseline, delta, task["balanced_accuracy_mean"]))

    report_lines = [
        "# Context Transfer Budget v1",
        "",
        "## Plots",
        "",
    ]
    for task_name, *_ in task_summaries:
        plot_name = f"{task_name}.png"
        report_lines.extend([f"![{task_name}]({plot_name})", ""])
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    findings_lines = [
        "# Context Transfer Budget v1",
        "",
        "## Result",
        "",
    ]
    for task_name, candidate, cost_mean, baseline, delta, balanced_accuracy in task_summaries:
        verdict = "beats trivial baseline" if delta < -1e-12 else "does not beat trivial baseline"
        findings_lines.append(
            f"- `{task_name}`: best candidate `{candidate}` with LOO cost `{cost_mean:.6f}` vs trivial baseline `{baseline:.6f}`; {verdict}. Balanced accuracy `{balanced_accuracy:.3f}`."
        )
    findings_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This probe asks a practical question: can triple-derived scores support budget allocation or safe-use decisions under asymmetric error costs? A task only counts as deployable if its learned threshold beats the best trivial baseline in leave-one-seed-out evaluation.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(findings_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    operator_results = load_json(args.operator_results)
    adaptation_results = load_json(args.adaptation_results)
    rows = build_context_transfer_rows(operator_results, adaptation_results)
    results = analyze_context_transfer_budget(
        rows,
        step_budgets=args.step_budgets,
        regret_tolerances=args.regret_tolerances,
        false_positive_cost=args.false_positive_cost,
        false_negative_cost=args.false_negative_cost,
    )
    results["operator_results"] = str(args.operator_results)
    results["adaptation_results"] = str(args.adaptation_results)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(render_budget_markdown(results), encoding="utf-8")
    for task_name in results["task_specs"]:
        plot_task_costs(results, output_dir / f"{task_name}.png", task_name)
    write_report(output_dir, Path(args.report_path), results)


if __name__ == "__main__":
    main()
