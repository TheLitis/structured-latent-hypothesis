from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from structured_latent_hypothesis.transfer_criterion import (
    annotate_transfer_tasks,
    build_context_transfer_rows,
    cross_validate_transfer_decision_policy_cost_shift,
    load_json,
)


SAFE_SCORE_KEYS = ["score_residual", "score_joint_sum"]
BUDGET_SCORE_KEYS = ["score_interaction", "score_joint_sum", "score_joint_prod"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate transfer policy robustness under unseen cost ratios.")
    parser.add_argument("--operator-results", default="results/context_operator_probe_v1/results.json")
    parser.add_argument("--adaptation-results", default="results/context_adaptation_probe_v1/results.json")
    parser.add_argument("--source-sweep", default="results/context_transfer_policy_sweep_v1/results.json")
    parser.add_argument("--output-dir", default="results/context_transfer_cost_shift_v1")
    parser.add_argument("--report-path", default="reports/2026-04-26_context_transfer_cost_shift_v1_findings.md")
    parser.add_argument("--followup-path", default="reports/2026-04-26_transfer_law_cost_shift_followup.md")
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--regret-tolerance", type=float, default=1e-5)
    return parser.parse_args()


def cost_tuple(config: dict) -> tuple[float, float, float, float]:
    return (
        float(config["structured_violation_cost"]),
        float(config["fallback_overbudget_cost"]),
        float(config["escalate_needed_cost"]),
        float(config["escalate_unneeded_cost"]),
    )


def source_config_payload(config: dict) -> dict:
    svc, foc, enc, euc = cost_tuple(config)
    return {
        "structured_violation_cost": svc,
        "fallback_overbudget_cost": foc,
        "escalate_needed_cost": enc,
        "escalate_unneeded_cost": euc,
        "source_delta_sum": float(config.get("delta_sum", 0.0)),
    }


def build_target_configs() -> list[dict]:
    configs: list[dict] = []
    for svc, foc, enc, euc in product([4.0, 6.0], [2.25, 2.75, 3.25], [0.75, 1.25], [1.25, 1.75]):
        configs.append(
            {
                "structured_violation_cost": svc,
                "fallback_overbudget_cost": foc,
                "escalate_needed_cost": enc,
                "escalate_unneeded_cost": euc,
            }
        )
    return configs


def compact_metrics(metrics: dict) -> dict:
    compact = {key: value for key, value in metrics.items() if key != "per_group"}
    compact["per_group"] = []
    for row in metrics["per_group"]:
        compact["per_group"].append({key: value for key, value in row.items() if key != "per_row"})
    return compact


def evaluate_source_target(
    rows: list[dict],
    *,
    budget_label_key: str,
    safe_label_key: str,
    source: dict,
    target: dict,
) -> dict:
    groups = {}
    for group in ("seed", "world", "family"):
        metrics = cross_validate_transfer_decision_policy_cost_shift(
            rows,
            group_key=group,
            safe_score_keys=SAFE_SCORE_KEYS,
            budget_score_keys=BUDGET_SCORE_KEYS,
            safe_label_key=safe_label_key,
            budget_label_key=budget_label_key,
            train_structured_violation_cost=source["structured_violation_cost"],
            train_fallback_overbudget_cost=source["fallback_overbudget_cost"],
            train_escalate_needed_cost=source["escalate_needed_cost"],
            train_escalate_unneeded_cost=source["escalate_unneeded_cost"],
            eval_structured_violation_cost=target["structured_violation_cost"],
            eval_fallback_overbudget_cost=target["fallback_overbudget_cost"],
            eval_escalate_needed_cost=target["escalate_needed_cost"],
            eval_escalate_unneeded_cost=target["escalate_unneeded_cost"],
        )
        best_trivial = min(
            metrics["always_structured_cost_mean"],
            metrics["always_fallback_cost_mean"],
            metrics["always_escalate_cost_mean"],
        )
        groups[group] = {
            "metrics": compact_metrics(metrics),
            "best_trivial_cost": best_trivial,
            "delta_to_best_trivial": metrics["average_cost_mean"] - best_trivial,
            "wins": metrics["average_cost_mean"] < best_trivial - 1e-12,
        }
    return {
        "source": source,
        "target": target,
        "groups": groups,
        "all_group_win": all(value["wins"] for value in groups.values()),
        "delta_sum": sum(value["delta_to_best_trivial"] for value in groups.values()),
        "worst_group_delta": max(value["delta_to_best_trivial"] for value in groups.values()),
        "win_count": sum(int(value["wins"]) for value in groups.values()),
    }


def summarize_sources(evaluations: list[dict]) -> list[dict]:
    by_source: dict[tuple[float, float, float, float], list[dict]] = {}
    for evaluation in evaluations:
        by_source.setdefault(cost_tuple(evaluation["source"]), []).append(evaluation)

    summaries: list[dict] = []
    for source_key, rows in by_source.items():
        source = rows[0]["source"]
        summaries.append(
            {
                "source": source,
                "target_count": len(rows),
                "all_group_win_count": sum(int(row["all_group_win"]) for row in rows),
                "mean_delta_sum": sum(row["delta_sum"] for row in rows) / len(rows),
                "worst_group_delta": max(row["worst_group_delta"] for row in rows),
                "mean_worst_group_delta": sum(row["worst_group_delta"] for row in rows) / len(rows),
            }
        )
    summaries.sort(
        key=lambda item: (
            -item["all_group_win_count"],
            item["worst_group_delta"],
            item["mean_delta_sum"],
            cost_tuple(item["source"]),
        )
    )
    return summaries


def plot_source_robustness(results: dict, output_path: Path) -> None:
    top = results["source_summaries"][:10]
    labels = [
        f"S{row['source']['structured_violation_cost']:.1f}/F{row['source']['fallback_overbudget_cost']:.1f}\nE{row['source']['escalate_needed_cost']:.1f}/U{row['source']['escalate_unneeded_cost']:.1f}"
        for row in top
    ]
    wins = [row["all_group_win_count"] for row in top]
    worst = [row["worst_group_delta"] for row in top]

    figure, axis_left = plt.subplots(1, 1, figsize=(10.8, 4.8))
    xs = list(range(len(top)))
    axis_left.bar(xs, wins, color="#638f9c", label="target all-group wins")
    axis_left.set_ylabel("target configs won")
    axis_left.set_xticks(xs)
    axis_left.set_xticklabels(labels, fontsize=8)
    axis_left.grid(axis="y", alpha=0.25)
    axis_right = axis_left.twinx()
    axis_right.plot(xs, worst, color="#a05a4f", marker="o", linewidth=1.6, label="worst group delta")
    axis_right.axhline(0.0, color="black", linewidth=1.0)
    axis_right.set_ylabel("worst(policy - trivial)")
    axis_left.set_title("Cost-shift robustness by source policy cost")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_outputs(output_dir: Path, report_path: Path, followup_path: Path, results: dict) -> None:
    summary_lines = [
        "# Context Transfer Cost Shift v1",
        "",
        f"Source configs: `{results['source_config_count']}`",
        f"Unseen target configs: `{results['target_config_count']}`",
        f"All-target robust sources: `{results['all_target_robust_source_count']}`",
        f"Prior best target wins: `{results['prior_best_summary']['all_group_win_count']}/{results['target_config_count']}`",
        "",
        "| Rank | SVC | FOC | ENC | EUC | Target Wins | Mean Delta Sum | Worst Group Delta | Mean Worst Delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(results["source_summaries"][:10], start=1):
        source = row["source"]
        summary_lines.append(
            "| "
            + str(index)
            + f" | {source['structured_violation_cost']:.1f}"
            + f" | {source['fallback_overbudget_cost']:.1f}"
            + f" | {source['escalate_needed_cost']:.1f}"
            + f" | {source['escalate_unneeded_cost']:.1f}"
            + f" | {row['all_group_win_count']}/{row['target_count']}"
            + f" | {row['mean_delta_sum']:+.6f}"
            + f" | {row['worst_group_delta']:+.6f}"
            + f" | {row['mean_worst_group_delta']:+.6f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    (output_dir / "report.md").write_text(
        "\n".join(
            [
                "# Context Transfer Cost Shift v1",
                "",
                "## Plots",
                "",
                "![Source robustness](source_cost_shift_robustness.png)",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    best = results["source_summaries"][0]
    prior = results["prior_best_summary"]
    findings = [
        "# Context Transfer Cost Shift v1",
        "",
        "## Result",
        "",
        f"- Source policies tested: `{results['source_config_count']}` prior all-group winners.",
        f"- Unseen target cost configs: `{results['target_config_count']}`.",
        f"- All-target robust source policies: `{results['all_target_robust_source_count']}`.",
        f"- Prior best source wins `{prior['all_group_win_count']}/{results['target_config_count']}` target configs; worst group delta `{prior['worst_group_delta']:+.6f}`.",
        f"- Best robust source wins `{best['all_group_win_count']}/{best['target_count']}` target configs with worst group delta `{best['worst_group_delta']:+.6f}`.",
        "",
        "## Interpretation",
        "",
        "This probe separates cost selection from cost evaluation. A positive result means the explicit decision law is not only tuned to one deployment cost point; it retains useful behavior on unseen cost ratios.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(findings) + "\n", encoding="utf-8")

    followup = [
        "# Transfer Law Cost-Shift Follow-Up",
        "",
        "## Main Result",
        "",
        f"The policy law was evaluated under `{results['target_config_count']}` unseen cost-ratio settings after selecting thresholds with source costs from the previous policy sweep.",
        "",
        f"`{results['all_target_robust_source_count']}` source policies beat the best trivial baseline on every target setting and every `seed/world/family` split.",
        "",
        "## Interpretation",
        "",
        "If this count is nonzero, the decision law has a genuine robustness region. If it is zero, the policy remains useful only when the deployment cost model is calibrated closely to the training cost model.",
        "",
        "## Next Step",
        "",
        "The next strict test is a semi-real transition benchmark or a family of context shifts that changes visual style while preserving the same transfer-decision labels.",
    ]
    followup_path.parent.mkdir(parents=True, exist_ok=True)
    followup_path.write_text("\n".join(followup) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    operator_results = load_json(args.operator_results)
    adaptation_results = load_json(args.adaptation_results)
    source_sweep = load_json(args.source_sweep)
    rows = build_context_transfer_rows(operator_results, adaptation_results)
    annotated = annotate_transfer_tasks(rows, step_budgets=[args.budget], regret_tolerances=[args.regret_tolerance])
    diag_rows = [row for row in annotated if row["variant"] == "operator_diag_residual"]

    tolerance_key = format(args.regret_tolerance, ".0e") if args.regret_tolerance > 0 else "0"
    safe_label_key = f"task_safe_regret_{tolerance_key}"
    budget_label_key = f"task_within_budget_{args.budget}"
    source_configs = [
        source_config_payload(config) for config in source_sweep["all_configs"] if bool(config["all_group_win"])
    ]
    target_configs = build_target_configs()

    evaluations: list[dict] = []
    for source in source_configs:
        for target in target_configs:
            evaluations.append(
                evaluate_source_target(
                    diag_rows,
                    budget_label_key=budget_label_key,
                    safe_label_key=safe_label_key,
                    source=source,
                    target=target,
                )
            )
    source_summaries = summarize_sources(evaluations)
    prior_best_key = cost_tuple(source_config_payload(source_sweep["top_configs"][0]))
    prior_best_summary = next(row for row in source_summaries if cost_tuple(row["source"]) == prior_best_key)
    results = {
        "budget": args.budget,
        "regret_tolerance": args.regret_tolerance,
        "safe_label_key": safe_label_key,
        "budget_label_key": budget_label_key,
        "source_config_count": len(source_configs),
        "target_config_count": len(target_configs),
        "all_target_robust_source_count": sum(
            int(row["all_group_win_count"] == row["target_count"]) for row in source_summaries
        ),
        "prior_best_summary": prior_best_summary,
        "source_summaries": source_summaries,
        "target_configs": target_configs,
        "evaluations": evaluations,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_source_robustness(results, output_dir / "source_cost_shift_robustness.png")
    write_outputs(output_dir, Path(args.report_path), Path(args.followup_path), results)


if __name__ == "__main__":
    main()
