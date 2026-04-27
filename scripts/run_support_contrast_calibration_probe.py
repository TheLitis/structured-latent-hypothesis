from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from structured_latent_hypothesis.support_contrast import (
    cross_validate_rank_calibrated_transfer,
    evaluate_policy_against_trivial,
    evaluate_rank_external_transfer,
)
from structured_latent_hypothesis.transfer_criterion import cross_validate_transfer_decision_policy


SAFE_SCORE_KEYS = ["score_contrast", "score_gain_ratio_1", "score_gain_delta_1", "score_residual_normalized"]
BUDGET_SCORE_KEYS = ["score_contrast", "score_gain_ratio_8", "score_gain_delta_8", "score_instability"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate calibrated support-contrast transfer policies.")
    parser.add_argument("--source-results", default="results/support_contrast_transfer_probe_v2/results.json")
    parser.add_argument("--output-dir", default="results/support_contrast_calibration_probe_v1")
    parser.add_argument("--report-path", default="reports/2026-04-27_support_contrast_calibration_probe_v1_findings.md")
    parser.add_argument("--followup-path", default="reports/2026-04-27_support_contrast_calibration_followup.md")
    parser.add_argument("--structured-violation-cost", type=float, default=5.0)
    parser.add_argument("--fallback-overbudget-cost", type=float, default=3.0)
    parser.add_argument("--escalate-needed-cost", type=float, default=0.5)
    parser.add_argument("--escalate-unneeded-cost", type=float, default=1.0)
    return parser.parse_args()


def compact_cv(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key != "per_group"} | {
        "per_group": metrics.get("per_group", [])
    }


def compare_cv(metrics: dict) -> dict:
    comparison = evaluate_policy_against_trivial(metrics)
    return {
        "metrics": compact_cv(metrics),
        **comparison,
    }


def run_raw_semireal_cv(rows: list[dict], args: argparse.Namespace) -> dict:
    output = {}
    for group in ("seed", "world", "family"):
        metrics = cross_validate_transfer_decision_policy(
            rows,
            group_key=group,
            safe_score_keys=SAFE_SCORE_KEYS,
            budget_score_keys=BUDGET_SCORE_KEYS,
            safe_label_key="task_safe",
            budget_label_key="task_budget",
            structured_violation_cost=args.structured_violation_cost,
            fallback_overbudget_cost=args.fallback_overbudget_cost,
            escalate_needed_cost=args.escalate_needed_cost,
            escalate_unneeded_cost=args.escalate_unneeded_cost,
        )
        output[group] = compare_cv(metrics)
    return output


def run_rank_cv(rows: list[dict], args: argparse.Namespace, source_rows: list[dict] | None = None) -> dict:
    output = {}
    for group in ("seed", "world", "family"):
        metrics = cross_validate_rank_calibrated_transfer(
            rows,
            group_key=group,
            raw_safe_score_keys=SAFE_SCORE_KEYS,
            raw_budget_score_keys=BUDGET_SCORE_KEYS,
            structured_violation_cost=args.structured_violation_cost,
            fallback_overbudget_cost=args.fallback_overbudget_cost,
            escalate_needed_cost=args.escalate_needed_cost,
            escalate_unneeded_cost=args.escalate_unneeded_cost,
            source_rows=source_rows,
        )
        output[group] = compare_cv(metrics)
    return output


def group_win_count(rows: dict) -> int:
    return sum(int(row["wins_best_trivial"]) for row in rows.values())


def plot_modes(results: dict, output_path: Path) -> None:
    modes = ["raw_semireal_cv", "rank_external", "target_rank_cv", "hybrid_rank_cv"]
    labels = ["raw target CV", "rank external", "target rank CV", "hybrid rank CV"]
    deltas = [
        sum(row["delta_to_best_trivial"] for row in results[mode].values()) if mode != "rank_external" else results[mode]["delta_to_best_trivial"]
        for mode in modes
    ]
    figure, axis = plt.subplots(1, 1, figsize=(8.8, 4.8))
    axis.bar(range(len(modes)), deltas, color=["#9c6b5a", "#638f9c", "#6c9a72", "#8170a8"])
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.set_xticks(range(len(modes)))
    axis.set_xticklabels(labels, rotation=15, ha="right")
    axis.set_ylabel("delta to best trivial (sum for CV modes)")
    axis.set_title("Support contrast calibration modes")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_outputs(output_dir: Path, report_path: Path, followup_path: Path, results: dict) -> None:
    summary_lines = [
        "# Support Contrast Calibration Probe v1",
        "",
        "## External Policies",
        "",
        "| Mode | Cost | Best Trivial | Delta | Wins |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for mode in ("raw_external", "rank_external"):
        row = results[mode]
        summary_lines.append(
            "| "
            + mode
            + f" | {row['metrics']['average_cost']:.6f}"
            + f" | {row['best_trivial_cost']:.6f}"
            + f" | {row['delta_to_best_trivial']:+.6f}"
            + f" | {row['wins_best_trivial']} |"
        )
    summary_lines.extend(
        [
            "",
            "## Target CV Policies",
            "",
            "| Mode | Group | Cost | Best Trivial | Delta | Wins |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for mode in ("raw_semireal_cv", "target_rank_cv", "hybrid_rank_cv"):
        for group, row in results[mode].items():
            summary_lines.append(
                "| "
                + mode
                + " | "
                + group
                + f" | {row['metrics']['average_cost_mean']:.6f}"
                + f" | {row['best_trivial_cost']:.6f}"
                + f" | {row['delta_to_best_trivial']:+.6f}"
                + f" | {row['wins_best_trivial']} |"
            )
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(
        "# Support Contrast Calibration Probe v1\n\n![Calibration modes](calibration_modes.png)\n",
        encoding="utf-8",
    )

    findings = [
        "# Support Contrast Calibration Probe v1",
        "",
        "## Result",
        "",
        f"- Raw external delta: `{results['raw_external']['delta_to_best_trivial']:+.6f}`.",
        f"- Rank external delta: `{results['rank_external']['delta_to_best_trivial']:+.6f}`.",
        f"- Target-rank CV wins: `{group_win_count(results['target_rank_cv'])}/3`.",
        f"- Hybrid-rank CV wins: `{group_win_count(results['hybrid_rank_cv'])}/3`.",
        "",
        "## Interpretation",
        "",
        "This probe checks whether support-contrast failure is caused by score scale mismatch. Rank external is label-free target-domain calibration; target-rank CV is small labeled target calibration; hybrid-rank CV adds synthetic rows back into the calibrated target policy.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(findings) + "\n", encoding="utf-8")

    followup = [
        "# Support Contrast Calibration Follow-Up",
        "",
        "## Main Result",
        "",
        f"Label-free rank calibration {'wins' if results['rank_external']['wins_best_trivial'] else 'does not win'} on synthetic-to-semi-real transfer.",
        "",
        f"Small target rank calibration wins `{group_win_count(results['target_rank_cv'])}/3` semi-real CV splits.",
        "",
        "## Interpretation",
        "",
        "If rank external fails but target-rank CV wins, the criterion needs target-domain calibration labels. If both fail, support contrast is not enough for the current semi-real benchmark.",
    ]
    followup_path.parent.mkdir(parents=True, exist_ok=True)
    followup_path.write_text("\n".join(followup) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = json.loads(Path(args.source_results).read_text(encoding="utf-8"))
    synthetic_rows = source["synthetic_rows"]
    semireal_rows = source["semireal_rows"]
    raw_external = source["external_policy"]
    rank_external = evaluate_rank_external_transfer(
        synthetic_rows,
        semireal_rows,
        raw_safe_score_keys=SAFE_SCORE_KEYS,
        raw_budget_score_keys=BUDGET_SCORE_KEYS,
        structured_violation_cost=args.structured_violation_cost,
        fallback_overbudget_cost=args.fallback_overbudget_cost,
        escalate_needed_cost=args.escalate_needed_cost,
        escalate_unneeded_cost=args.escalate_unneeded_cost,
    )
    results = {
        "source_results": args.source_results,
        "raw_external": raw_external,
        "rank_external": rank_external,
        "raw_semireal_cv": run_raw_semireal_cv(semireal_rows, args),
        "target_rank_cv": run_rank_cv(semireal_rows, args),
        "hybrid_rank_cv": run_rank_cv(semireal_rows, args, source_rows=synthetic_rows),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_modes(results, output_dir / "calibration_modes.png")
    write_outputs(output_dir, Path(args.report_path), Path(args.followup_path), results)


if __name__ == "__main__":
    main()
