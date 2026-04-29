from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import fmean, median, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from structured_latent_hypothesis.support_contrast import (
    evaluate_rank_calibrated_policy,
    select_and_evaluate_transfer_policy,
)


SAFE_SCORE_KEYS = ["score_contrast", "score_gain_ratio_1", "score_gain_delta_1", "score_residual_normalized"]
BUDGET_SCORE_KEYS = ["score_contrast", "score_gain_ratio_8", "score_gain_delta_8", "score_instability"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep target-domain calibration size for support-contrast transfer.")
    parser.add_argument("--source-results", default="results/support_contrast_transfer_probe_v2/results.json")
    parser.add_argument("--output-dir", default="results/support_contrast_target_calibration_sweep_v1")
    parser.add_argument("--report-path", default="reports/2026-04-29_support_contrast_target_calibration_sweep_v1_findings.md")
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8])
    parser.add_argument("--trials", type=int, default=96)
    parser.add_argument("--sample-seed", type=int, default=20260429)
    parser.add_argument("--structured-violation-cost", type=float, default=5.0)
    parser.add_argument("--fallback-overbudget-cost", type=float, default=3.0)
    parser.add_argument("--escalate-needed-cost", type=float, default=0.5)
    parser.add_argument("--escalate-unneeded-cost", type=float, default=1.0)
    return parser.parse_args()


def mean_or_zero(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def std_or_zero(values: list[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def compact_selected(selected: dict) -> dict:
    return {
        "safe_score_key": selected["safe_score_key"],
        "budget_score_key": selected["budget_score_key"],
        "safe_threshold": selected["safe_classifier"]["threshold"],
        "safe_direction": selected["safe_classifier"]["direction"],
        "safe_band": selected["safe_classifier"]["band"],
        "budget_threshold": selected["budget_classifier"]["threshold"],
        "budget_direction": selected["budget_classifier"]["direction"],
        "budget_band": selected["budget_classifier"]["band"],
    }


def family_summary(per_row: list[dict], raw_test_rows: list[dict]) -> dict[str, dict]:
    lookup = {(row["world"], int(row["seed"])): row for row in raw_test_rows}
    grouped: dict[str, list[dict]] = {}
    for row in per_row:
        family = lookup[(row["world"], int(row["seed"]))]["family"]
        grouped.setdefault(family, []).append(row)
    return {
        family: {
            "count": len(rows),
            "average_cost": mean_or_zero([float(row["cost"]) for row in rows]),
            "oracle_average_cost": mean_or_zero([float(row["oracle_cost"]) for row in rows]),
            "regret": mean_or_zero([float(row["cost"]) - float(row["oracle_cost"]) for row in rows]),
        }
        for family, rows in grouped.items()
    }


def run_policy_mode(
    mode: str,
    *,
    calibration_rows: list[dict],
    test_rows: list[dict],
    synthetic_rows: list[dict],
    args: argparse.Namespace,
) -> dict:
    if mode == "raw_target":
        return select_and_evaluate_transfer_policy(
            calibration_rows,
            test_rows,
            safe_score_keys=SAFE_SCORE_KEYS,
            budget_score_keys=BUDGET_SCORE_KEYS,
            structured_violation_cost=args.structured_violation_cost,
            fallback_overbudget_cost=args.fallback_overbudget_cost,
            escalate_needed_cost=args.escalate_needed_cost,
            escalate_unneeded_cost=args.escalate_unneeded_cost,
        )
    if mode == "target_rank":
        return evaluate_rank_calibrated_policy(
            calibration_rows,
            test_rows,
            raw_safe_score_keys=SAFE_SCORE_KEYS,
            raw_budget_score_keys=BUDGET_SCORE_KEYS,
            structured_violation_cost=args.structured_violation_cost,
            fallback_overbudget_cost=args.fallback_overbudget_cost,
            escalate_needed_cost=args.escalate_needed_cost,
            escalate_unneeded_cost=args.escalate_unneeded_cost,
        )
    if mode == "hybrid_rank":
        return evaluate_rank_calibrated_policy(
            calibration_rows,
            test_rows,
            raw_safe_score_keys=SAFE_SCORE_KEYS,
            raw_budget_score_keys=BUDGET_SCORE_KEYS,
            structured_violation_cost=args.structured_violation_cost,
            fallback_overbudget_cost=args.fallback_overbudget_cost,
            escalate_needed_cost=args.escalate_needed_cost,
            escalate_unneeded_cost=args.escalate_unneeded_cost,
            source_rows=synthetic_rows,
        )
    raise ValueError(f"Unknown mode: {mode}")


def run_trial(
    *,
    mode: str,
    semireal_rows: list[dict],
    synthetic_rows: list[dict],
    calibration_worlds: list[str],
    budget: int,
    trial_index: int,
    args: argparse.Namespace,
) -> dict:
    calibration_set = set(calibration_worlds)
    calibration_rows = [row for row in semireal_rows if row["world"] in calibration_set]
    test_rows = [row for row in semireal_rows if row["world"] not in calibration_set]
    result = run_policy_mode(
        mode,
        calibration_rows=calibration_rows,
        test_rows=test_rows,
        synthetic_rows=synthetic_rows,
        args=args,
    )
    metrics = result["metrics"]
    return {
        "mode": mode,
        "calibration_world_count": budget,
        "trial_index": trial_index,
        "calibration_worlds": calibration_worlds,
        "test_world_count": len({row["world"] for row in test_rows}),
        "test_row_count": len(test_rows),
        "average_cost": metrics["average_cost"],
        "best_trivial_cost": result["best_trivial_cost"],
        "delta_to_best_trivial": result["delta_to_best_trivial"],
        "wins_best_trivial": result["wins_best_trivial"],
        "oracle_average_cost": metrics["oracle_average_cost"],
        "regret": metrics["regret"],
        "action_rates": metrics["action_rates"],
        "selected": compact_selected(result["selected"]),
        "family_metrics": family_summary(result["per_row"], test_rows),
    }


def summarize_trials(trials: list[dict]) -> dict:
    costs = [float(row["average_cost"]) for row in trials]
    deltas = [float(row["delta_to_best_trivial"]) for row in trials]
    regrets = [float(row["regret"]) for row in trials]
    structured_rates = [float(row["action_rates"]["structured"]) for row in trials]
    fallback_rates = [float(row["action_rates"]["fallback"]) for row in trials]
    escalate_rates = [float(row["action_rates"]["escalate"]) for row in trials]
    return {
        "trial_count": len(trials),
        "average_cost_mean": mean_or_zero(costs),
        "average_cost_std": std_or_zero(costs),
        "delta_mean": mean_or_zero(deltas),
        "delta_std": std_or_zero(deltas),
        "delta_median": median(deltas) if deltas else 0.0,
        "win_rate": mean_or_zero([float(row["wins_best_trivial"]) for row in trials]),
        "regret_mean": mean_or_zero(regrets),
        "structured_rate_mean": mean_or_zero(structured_rates),
        "fallback_rate_mean": mean_or_zero(fallback_rates),
        "escalate_rate_mean": mean_or_zero(escalate_rates),
    }


def summarize_by_mode_budget(trials: list[dict]) -> dict[str, dict[str, dict]]:
    summary: dict[str, dict[str, dict]] = {}
    for mode in sorted({row["mode"] for row in trials}):
        summary[mode] = {}
        for budget in sorted({int(row["calibration_world_count"]) for row in trials if row["mode"] == mode}):
            subset = [row for row in trials if row["mode"] == mode and int(row["calibration_world_count"]) == budget]
            summary[mode][str(budget)] = summarize_trials(subset)
    return summary


def summarize_family_trials(trials: list[dict]) -> dict[str, dict[str, dict[str, dict]]]:
    output: dict[str, dict[str, dict[str, dict]]] = {}
    for mode in sorted({row["mode"] for row in trials}):
        output[mode] = {}
        for budget in sorted({int(row["calibration_world_count"]) for row in trials if row["mode"] == mode}):
            subset = [row for row in trials if row["mode"] == mode and int(row["calibration_world_count"]) == budget]
            output[mode][str(budget)] = {}
            for family in sorted({family for row in subset for family in row["family_metrics"]}):
                family_costs = [
                    float(row["family_metrics"][family]["average_cost"])
                    for row in subset
                    if family in row["family_metrics"] and row["family_metrics"][family]["count"] > 0
                ]
                family_regrets = [
                    float(row["family_metrics"][family]["regret"])
                    for row in subset
                    if family in row["family_metrics"] and row["family_metrics"][family]["count"] > 0
                ]
                output[mode][str(budget)][family] = {
                    "average_cost_mean": mean_or_zero(family_costs),
                    "average_cost_std": std_or_zero(family_costs),
                    "regret_mean": mean_or_zero(family_regrets),
                }
    return output


def first_success_budget(summary: dict, mode: str) -> int | None:
    for budget_text, row in sorted(summary[mode].items(), key=lambda item: int(item[0])):
        if float(row["delta_mean"]) < 0.0 and float(row["win_rate"]) >= 0.55:
            return int(budget_text)
    return None


def plot_sweep(summary: dict, output_path: Path) -> None:
    colors = {"raw_target": "#9c6b5a", "target_rank": "#6c9a72", "hybrid_rank": "#8170a8"}
    labels = {"raw_target": "raw target", "target_rank": "target rank", "hybrid_rank": "hybrid rank"}
    figure, axis = plt.subplots(1, 1, figsize=(8.8, 5.2))
    for mode in ("raw_target", "target_rank", "hybrid_rank"):
        budgets = [int(key) for key in sorted(summary[mode], key=int)]
        means = [summary[mode][str(budget)]["delta_mean"] for budget in budgets]
        stds = [summary[mode][str(budget)]["delta_std"] for budget in budgets]
        axis.errorbar(budgets, means, yerr=stds, marker="o", linewidth=1.8, color=colors[mode], label=labels[mode])
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.set_xlabel("semi-real calibration worlds")
    axis.set_ylabel("delta to best trivial cost")
    axis.set_title("Target calibration sample-efficiency")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_outputs(output_dir: Path, report_path: Path, results: dict) -> None:
    summary = results["summary"]
    family_summary = results["family_summary"]
    lines = [
        "# Support Contrast Target Calibration Sweep v1",
        "",
        "## Aggregate",
        "",
        "| Mode | Calibration worlds | Cost mean | Delta mean | Delta std | Win rate | Structured | Fallback | Escalate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("raw_target", "target_rank", "hybrid_rank"):
        for budget_text in sorted(summary[mode], key=int):
            row = summary[mode][budget_text]
            lines.append(
                "| "
                + mode
                + f" | {int(budget_text)}"
                + f" | {row['average_cost_mean']:.6f}"
                + f" | {row['delta_mean']:+.6f}"
                + f" | {row['delta_std']:.6f}"
                + f" | {row['win_rate']:.3f}"
                + f" | {row['structured_rate_mean']:.3f}"
                + f" | {row['fallback_rate_mean']:.3f}"
                + f" | {row['escalate_rate_mean']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Family Costs",
            "",
            "| Mode | Calibration worlds | Family | Cost mean | Regret mean |",
            "| --- | ---: | --- | ---: | ---: |",
        ]
    )
    for mode in ("raw_target", "target_rank", "hybrid_rank"):
        for budget_text in sorted(family_summary[mode], key=int):
            for family, row in family_summary[mode][budget_text].items():
                lines.append(
                    "| "
                    + mode
                    + f" | {int(budget_text)}"
                    + f" | {family}"
                    + f" | {row['average_cost_mean']:.6f}"
                    + f" | {row['regret_mean']:.6f} |"
                )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(
        "# Support Contrast Target Calibration Sweep v1\n\n![Calibration sweep](target_calibration_sweep.png)\n",
        encoding="utf-8",
    )

    target_success = first_success_budget(summary, "target_rank")
    raw_success = first_success_budget(summary, "raw_target")
    hybrid_success = first_success_budget(summary, "hybrid_rank")
    target_result = "none" if target_success is None else str(target_success)
    raw_result = "none" if raw_success is None else str(raw_success)
    hybrid_result = "none" if hybrid_success is None else str(hybrid_success)
    target_k5 = summary["target_rank"].get("5")
    target_k6 = summary["target_rank"].get("6")
    interpretation = (
        "Target-rank calibration reaches the practical gate, but only after moderate labeled target calibration."
        if target_success is not None
        else "Target-rank calibration does not reach the practical gate on this small semi-real set."
    )
    findings = [
        "# Support Contrast Target Calibration Sweep v1",
        "",
        "## Result",
        "",
        f"- Raw-target first success budget: `{raw_result}`.",
        f"- Target-rank first success budget: `{target_result}`.",
        f"- Hybrid-rank first success budget: `{hybrid_result}`.",
        f"- Practical gate: mean delta < 0 and win rate >= 0.55 across random held-out semi-real world splits.",
        (
            f"- Target-rank at 5 worlds: delta `{target_k5['delta_mean']:+.6f}`, win rate `{target_k5['win_rate']:.3f}`."
            if target_k5 is not None
            else "- Target-rank at 5 worlds: `n/a`."
        ),
        (
            f"- Target-rank at 6 worlds: delta `{target_k6['delta_mean']:+.6f}`, win rate `{target_k6['win_rate']:.3f}`."
            if target_k6 is not None
            else "- Target-rank at 6 worlds: `n/a`."
        ),
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
        "The useful scope is therefore narrow: this is not a source-only transferable law and not a one-context detector. It becomes useful only when the deployment domain supplies enough labeled support worlds to calibrate the decision boundary.",
        "",
        "Raw-target and target-rank behave almost identically here, while hybrid-rank stays harmful. That means synthetic rows are not helping target calibration; they still miscalibrate the semi-real policy.",
        "",
        "The surviving claim is a target-domain context-adaptation diagnostic: given several calibrated contexts, support-contrast geometry can help decide structured transfer vs fallback/escalation on nearby held-out contexts.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(findings) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = json.loads(Path(args.source_results).read_text(encoding="utf-8"))
    synthetic_rows = source["synthetic_rows"]
    semireal_rows = source["semireal_rows"]
    worlds = sorted({row["world"] for row in semireal_rows})
    budgets = [budget for budget in args.budgets if 0 < budget < len(worlds)]
    rng = random.Random(args.sample_seed)
    trials: list[dict] = []
    for budget in budgets:
        for trial_index in range(args.trials):
            calibration_worlds = sorted(rng.sample(worlds, budget))
            for mode in ("raw_target", "target_rank", "hybrid_rank"):
                trials.append(
                    run_trial(
                        mode=mode,
                        semireal_rows=semireal_rows,
                        synthetic_rows=synthetic_rows,
                        calibration_worlds=calibration_worlds,
                        budget=budget,
                        trial_index=trial_index,
                        args=args,
                    )
                )
    results = {
        "label": "Support Contrast Target Calibration Sweep v1",
        "config": vars(args),
        "source_results": args.source_results,
        "world_count": len(worlds),
        "row_count": len(semireal_rows),
        "safe_score_keys": SAFE_SCORE_KEYS,
        "budget_score_keys": BUDGET_SCORE_KEYS,
        "summary": summarize_by_mode_budget(trials),
        "family_summary": summarize_family_trials(trials),
        "trials": trials,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_sweep(results["summary"], output_dir / "target_calibration_sweep.png")
    write_outputs(output_dir, Path(args.report_path), results)


if __name__ == "__main__":
    main()
