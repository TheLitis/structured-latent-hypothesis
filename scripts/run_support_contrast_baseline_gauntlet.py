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
    augment_support_diagnostic_scores,
    evaluate_rank_external_transfer,
    evaluate_rank_calibrated_policy,
    select_and_evaluate_transfer_policy,
)


SUPPORT_COMMUTATOR_SAFE = [
    "score_contrast",
    "score_gain_ratio_1",
    "score_gain_delta_1",
    "score_residual_normalized",
]
SUPPORT_COMMUTATOR_BUDGET = [
    "score_contrast",
    "score_gain_ratio_8",
    "score_gain_delta_8",
    "score_instability",
]

FAMILIES = {
    "support_commutator": {
        "protocol": "raw",
        "safe": SUPPORT_COMMUTATOR_SAFE,
        "budget": SUPPORT_COMMUTATOR_BUDGET,
        "description": "Our support-contrast commutator criterion.",
    },
    "validation_loss": {
        "protocol": "raw",
        "safe": ["score_validation_gap", "score_validation_ratio", "score_validation_structured"],
        "budget": ["score_validation_fallback", "score_fallback_gain_delta_8", "score_fallback_gain_ratio_8"],
        "description": "Plain support-set validation loss and adaptation gain.",
    },
    "conformal_validation_rank": {
        "protocol": "rank",
        "safe": ["score_validation_gap", "score_validation_ratio", "score_validation_structured"],
        "budget": ["score_validation_fallback", "score_fallback_gain_delta_8", "score_fallback_gain_ratio_8"],
        "description": "Conformal-style nonconformity ranks from support validation losses.",
    },
    "uncertainty_entropy_proxy": {
        "protocol": "raw",
        "safe": ["score_entropy_margin", "score_entropy_structured_proxy", "score_uncertainty_gap"],
        "budget": ["score_entropy_fallback_proxy", "score_uncertainty_fallback_curve_std", "score_fallback_support_slope_1"],
        "description": "Regression uncertainty proxy from support curve variance and log-loss scale.",
    },
    "router_margin": {
        "protocol": "raw",
        "safe": ["score_router_margin", "score_router_confidence", "score_router_abs_margin"],
        "budget": ["score_validation_fallback", "score_fallback_gain_delta_8", "score_fallback_support_slope_1"],
        "description": "LLM-router-style confidence margin between structured and fallback branches on support.",
    },
    "raw_commutator_only": {
        "protocol": "raw",
        "safe": ["score_residual_normalized", "score_interaction_support", "score_interaction_support_gap"],
        "budget": ["score_gain_delta_8", "score_fallback_gain_delta_8", "score_instability"],
        "description": "Raw interaction and residual scores without support-contrast composition.",
    },
    "metadata_alpha_control": {
        "protocol": "raw",
        "safe": ["score_alpha_metadata"],
        "budget": ["score_alpha_metadata"],
        "description": "Non-deployable metadata control using true alpha.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare support-commutator criterion against strong routing baselines.")
    parser.add_argument("--source-results", default="results/support_contrast_transfer_probe_v2/results.json")
    parser.add_argument("--synthetic-suite", default="results/support_contrast_transfer_probe_v2/synthetic_results.json")
    parser.add_argument("--semireal-suite", default="results/support_contrast_transfer_probe_v2/semireal_results.json")
    parser.add_argument("--output-dir", default="results/support_contrast_baseline_gauntlet_v1")
    parser.add_argument("--report-path", default="reports/2026-04-29_support_contrast_baseline_gauntlet_v1_findings.md")
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8])
    parser.add_argument("--trials", type=int, default=64)
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


def compact_result(result: dict) -> dict:
    return {
        "average_cost": result["metrics"]["average_cost"],
        "best_trivial_cost": result["best_trivial_cost"],
        "delta_to_best_trivial": result["delta_to_best_trivial"],
        "wins_best_trivial": result["wins_best_trivial"],
        "oracle_average_cost": result["metrics"]["oracle_average_cost"],
        "regret": result["metrics"]["regret"],
        "action_rates": result["metrics"]["action_rates"],
        "selected": compact_selected(result["selected"]),
    }


def evaluate_family(
    family_name: str,
    train_rows: list[dict],
    test_rows: list[dict],
    args: argparse.Namespace,
) -> dict:
    family = FAMILIES[family_name]
    if family["protocol"] == "rank":
        return evaluate_rank_calibrated_policy(
            train_rows,
            test_rows,
            raw_safe_score_keys=family["safe"],
            raw_budget_score_keys=family["budget"],
            structured_violation_cost=args.structured_violation_cost,
            fallback_overbudget_cost=args.fallback_overbudget_cost,
            escalate_needed_cost=args.escalate_needed_cost,
            escalate_unneeded_cost=args.escalate_unneeded_cost,
        )
    return select_and_evaluate_transfer_policy(
        train_rows,
        test_rows,
        safe_score_keys=family["safe"],
        budget_score_keys=family["budget"],
        structured_violation_cost=args.structured_violation_cost,
        fallback_overbudget_cost=args.fallback_overbudget_cost,
        escalate_needed_cost=args.escalate_needed_cost,
        escalate_unneeded_cost=args.escalate_unneeded_cost,
    )


def evaluate_external_family(
    family_name: str,
    synthetic_rows: list[dict],
    semireal_rows: list[dict],
    args: argparse.Namespace,
) -> dict:
    family = FAMILIES[family_name]
    if family["protocol"] == "rank":
        return evaluate_rank_external_transfer(
            synthetic_rows,
            semireal_rows,
            raw_safe_score_keys=family["safe"],
            raw_budget_score_keys=family["budget"],
            structured_violation_cost=args.structured_violation_cost,
            fallback_overbudget_cost=args.fallback_overbudget_cost,
            escalate_needed_cost=args.escalate_needed_cost,
            escalate_unneeded_cost=args.escalate_unneeded_cost,
        )
    return evaluate_family(family_name, synthetic_rows, semireal_rows, args)


def run_trial(
    *,
    family_name: str,
    semireal_rows: list[dict],
    calibration_worlds: list[str],
    budget: int,
    trial_index: int,
    args: argparse.Namespace,
) -> dict:
    calibration_set = set(calibration_worlds)
    calibration_rows = [row for row in semireal_rows if row["world"] in calibration_set]
    test_rows = [row for row in semireal_rows if row["world"] not in calibration_set]
    result = compact_result(evaluate_family(family_name, calibration_rows, test_rows, args))
    result.update(
        {
            "family_name": family_name,
            "calibration_world_count": budget,
            "trial_index": trial_index,
            "calibration_worlds": calibration_worlds,
            "test_world_count": len({row["world"] for row in test_rows}),
            "test_row_count": len(test_rows),
        }
    )
    return result


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


def summarize_by_family_budget(trials: list[dict]) -> dict[str, dict[str, dict]]:
    summary: dict[str, dict[str, dict]] = {}
    for family_name in FAMILIES:
        summary[family_name] = {}
        budgets = sorted({int(row["calibration_world_count"]) for row in trials if row["family_name"] == family_name})
        for budget in budgets:
            subset = [
                row
                for row in trials
                if row["family_name"] == family_name and int(row["calibration_world_count"]) == budget
            ]
            summary[family_name][str(budget)] = summarize_trials(subset)
    return summary


def first_success_budget(summary: dict, family_name: str) -> int | None:
    for budget_text, row in sorted(summary[family_name].items(), key=lambda item: int(item[0])):
        if float(row["delta_mean"]) < 0.0 and float(row["win_rate"]) >= 0.55:
            return int(budget_text)
    return None


def leader_at_budget(summary: dict, budget: int) -> tuple[str, dict]:
    rows = {
        family_name: family_summary[str(budget)]
        for family_name, family_summary in summary.items()
        if str(budget) in family_summary
    }
    return min(rows.items(), key=lambda item: (item[1]["delta_mean"], -item[1]["win_rate"]))


def plot_gauntlet(summary: dict, output_path: Path) -> None:
    colors = {
        "support_commutator": "#2f6f73",
        "validation_loss": "#9c6b5a",
        "conformal_validation_rank": "#b89445",
        "uncertainty_entropy_proxy": "#6f789c",
        "router_margin": "#6c9a72",
        "raw_commutator_only": "#8170a8",
        "metadata_alpha_control": "#777777",
    }
    figure, axis = plt.subplots(1, 1, figsize=(10.2, 5.8))
    for family_name, rows in summary.items():
        budgets = [int(key) for key in sorted(rows, key=int)]
        means = [rows[str(budget)]["delta_mean"] for budget in budgets]
        axis.plot(budgets, means, marker="o", linewidth=1.7, color=colors[family_name], label=family_name)
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.set_xlabel("semi-real calibration worlds")
    axis.set_ylabel("delta to best trivial cost")
    axis.set_title("Baseline gauntlet under target calibration")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_outputs(output_dir: Path, report_path: Path, results: dict) -> None:
    summary = results["summary"]
    external = results["external"]
    lines = [
        "# Support Contrast Baseline Gauntlet v1",
        "",
        "## Source-Only External Transfer",
        "",
        "| Family | Cost | Best trivial | Delta | Wins |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for family_name, row in external.items():
        lines.append(
            "| "
            + family_name
            + f" | {row['average_cost']:.6f}"
            + f" | {row['best_trivial_cost']:.6f}"
            + f" | {row['delta_to_best_trivial']:+.6f}"
            + f" | {row['wins_best_trivial']} |"
        )
    lines.extend(
        [
            "",
            "## Target Calibration Sweep",
            "",
            "| Family | Calibration worlds | Cost mean | Delta mean | Delta std | Win rate | Structured | Fallback | Escalate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for family_name in FAMILIES:
        for budget_text in sorted(summary[family_name], key=int):
            row = summary[family_name][budget_text]
            lines.append(
                "| "
                + family_name
                + f" | {int(budget_text)}"
                + f" | {row['average_cost_mean']:.6f}"
                + f" | {row['delta_mean']:+.6f}"
                + f" | {row['delta_std']:.6f}"
                + f" | {row['win_rate']:.3f}"
                + f" | {row['structured_rate_mean']:.3f}"
                + f" | {row['fallback_rate_mean']:.3f}"
                + f" | {row['escalate_rate_mean']:.3f} |"
            )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(
        "# Support Contrast Baseline Gauntlet v1\n\n![Baseline gauntlet](baseline_gauntlet.png)\n",
        encoding="utf-8",
    )

    k5_leader, k5_row = leader_at_budget(summary, 5)
    k6_leader, k6_row = leader_at_budget(summary, 6)
    support_success = first_success_budget(summary, "support_commutator")
    validation_success = first_success_budget(summary, "validation_loss")
    router_success = first_success_budget(summary, "router_margin")
    conformal_success = first_success_budget(summary, "conformal_validation_rank")
    support_k5 = summary["support_commutator"]["5"]
    validation_k5 = summary["validation_loss"]["5"]
    router_k5 = summary["router_margin"]["5"]
    external_leader, external_row = min(
        results["external"].items(),
        key=lambda item: item[1]["delta_to_best_trivial"],
    )
    unique_claim_survives = (
        support_success is not None
        and (validation_success is None or support_success < validation_success)
        and (router_success is None or support_success < router_success)
        and support_k5["delta_mean"] <= validation_k5["delta_mean"]
        and support_k5["delta_mean"] <= router_k5["delta_mean"]
    )
    findings = [
        "# Support Contrast Baseline Gauntlet v1",
        "",
        "## Result",
        "",
        f"- Support-commutator first success budget: `{support_success if support_success is not None else 'none'}`.",
        f"- Validation-loss first success budget: `{validation_success if validation_success is not None else 'none'}`.",
        f"- Conformal-rank validation first success budget: `{conformal_success if conformal_success is not None else 'none'}`.",
        f"- Router-margin first success budget: `{router_success if router_success is not None else 'none'}`.",
        f"- Leader at 5 calibration worlds: `{k5_leader}` with delta `{k5_row['delta_mean']:+.6f}` and win rate `{k5_row['win_rate']:.3f}`.",
        f"- Leader at 6 calibration worlds: `{k6_leader}` with delta `{k6_row['delta_mean']:+.6f}` and win rate `{k6_row['win_rate']:.3f}`.",
        f"- Best source-only external family: `{external_leader}` with delta `{external_row['delta_to_best_trivial']:+.6f}`.",
        f"- Unique support-commutator claim survives: `{unique_claim_survives}`.",
        "",
        "## Interpretation",
        "",
        "This is the strict comparison cycle. A claim survives only if the support-commutator family beats validation-loss, uncertainty/conformal, and router-margin baselines at small target calibration budgets.",
        "",
        "The unique support-commutator claim does not survive this gauntlet. The support-commutator policy becomes useful with target calibration, but simpler support validation and router-margin baselines reach the practical gate earlier and with lower cost.",
        "",
        "The practical lesson is narrower: the useful mechanism is support-set model selection/routing under context shift. The triple-derived commutator scores can remain diagnostic features, but they are not currently the best decision criterion.",
        "",
        "The entropy baseline here is a regression proxy from support-loss scale and adaptation-curve variance, not a classifier softmax entropy.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(findings) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = json.loads(Path(args.source_results).read_text(encoding="utf-8"))
    synthetic_suite = json.loads(Path(args.synthetic_suite).read_text(encoding="utf-8"))
    semireal_suite = json.loads(Path(args.semireal_suite).read_text(encoding="utf-8"))
    synthetic_rows = augment_support_diagnostic_scores(source["synthetic_rows"], synthetic_suite)
    semireal_rows = augment_support_diagnostic_scores(source["semireal_rows"], semireal_suite)
    worlds = sorted({row["world"] for row in semireal_rows})
    budgets = [budget for budget in args.budgets if 0 < budget < len(worlds)]
    external = {
        family_name: compact_result(evaluate_external_family(family_name, synthetic_rows, semireal_rows, args))
        for family_name in FAMILIES
    }
    rng = random.Random(args.sample_seed)
    trials: list[dict] = []
    for budget in budgets:
        for trial_index in range(args.trials):
            calibration_worlds = sorted(rng.sample(worlds, budget))
            for family_name in FAMILIES:
                trials.append(
                    run_trial(
                        family_name=family_name,
                        semireal_rows=semireal_rows,
                        calibration_worlds=calibration_worlds,
                        budget=budget,
                        trial_index=trial_index,
                        args=args,
                    )
                )
    results = {
        "label": "Support Contrast Baseline Gauntlet v1",
        "config": vars(args),
        "family_specs": FAMILIES,
        "world_count": len(worlds),
        "row_count": len(semireal_rows),
        "external": external,
        "summary": summarize_by_family_budget(trials),
        "trials": trials,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_gauntlet(results["summary"], output_dir / "baseline_gauntlet.png")
    write_outputs(output_dir, Path(args.report_path), results)


if __name__ == "__main__":
    main()
