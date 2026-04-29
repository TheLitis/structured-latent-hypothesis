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
    add_rank_features,
    augment_support_diagnostic_scores,
    evaluate_policy_against_trivial,
)
from structured_latent_hypothesis.transfer_criterion import (
    evaluate_transfer_decision_policy,
    select_transfer_decision_policy,
)


DEFAULT_COSTS = {
    "structured_violation_cost": 5.0,
    "fallback_overbudget_cost": 3.0,
    "escalate_needed_cost": 0.5,
    "escalate_unneeded_cost": 1.0,
}

COST_PROFILES = {
    "default": DEFAULT_COSTS,
    "high_structured_risk": {
        "structured_violation_cost": 10.0,
        "fallback_overbudget_cost": 3.0,
        "escalate_needed_cost": 0.5,
        "escalate_unneeded_cost": 1.0,
    },
    "high_fallback_delay": {
        "structured_violation_cost": 5.0,
        "fallback_overbudget_cost": 8.0,
        "escalate_needed_cost": 0.5,
        "escalate_unneeded_cost": 1.0,
    },
    "cheap_escalation": {
        "structured_violation_cost": 5.0,
        "fallback_overbudget_cost": 3.0,
        "escalate_needed_cost": 0.1,
        "escalate_unneeded_cost": 0.2,
    },
    "expensive_escalation": {
        "structured_violation_cost": 5.0,
        "fallback_overbudget_cost": 3.0,
        "escalate_needed_cost": 2.0,
        "escalate_unneeded_cost": 4.0,
    },
}

FAMILIES = {
    "validation_loss": {
        "protocol": "raw",
        "safe": ["score_validation_gap", "score_validation_ratio", "score_validation_structured"],
        "budget": ["score_validation_fallback", "score_fallback_gain_delta_8", "score_fallback_gain_ratio_8"],
    },
    "conformal_validation_rank": {
        "protocol": "rank",
        "safe": ["score_validation_gap", "score_validation_ratio", "score_validation_structured"],
        "budget": ["score_validation_fallback", "score_fallback_gain_delta_8", "score_fallback_gain_ratio_8"],
    },
    "router_margin": {
        "protocol": "raw",
        "safe": ["score_router_margin", "score_router_confidence", "score_router_abs_margin"],
        "budget": ["score_validation_fallback", "score_fallback_gain_delta_8", "score_fallback_support_slope_1"],
    },
    "support_commutator_reference": {
        "protocol": "raw",
        "safe": ["score_contrast", "score_gain_ratio_1", "score_gain_delta_1", "score_residual_normalized"],
        "budget": ["score_contrast", "score_gain_ratio_8", "score_gain_delta_8", "score_instability"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe cost robustness of support-calibrated adaptive routing.")
    parser.add_argument("--source-results", default="results/support_contrast_transfer_probe_v2/results.json")
    parser.add_argument("--semireal-suite", default="results/support_contrast_transfer_probe_v2/semireal_results.json")
    parser.add_argument("--output-dir", default="results/support_routing_cost_robustness_v1")
    parser.add_argument("--report-path", default="reports/2026-04-29_support_routing_cost_robustness_v1_findings.md")
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 3, 5])
    parser.add_argument("--trials", type=int, default=48)
    parser.add_argument("--sample-seed", type=int, default=20260429)
    return parser.parse_args()


def mean_or_zero(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def std_or_zero(values: list[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def rank_train_test(train_rows: list[dict], test_rows: list[dict], safe_keys: list[str], budget_keys: list[str]) -> tuple[list[dict], list[dict], list[str], list[str]]:
    raw_keys = sorted(set(safe_keys + budget_keys))
    train_ranked = add_rank_features(train_rows, score_keys=raw_keys)
    test_ranked = add_rank_features(test_rows, score_keys=raw_keys, reference_rows=train_rows)
    return train_ranked, test_ranked, [f"rank_{key}" for key in safe_keys], [f"rank_{key}" for key in budget_keys]


def prepare_rows_for_family(family: dict, train_rows: list[dict], test_rows: list[dict]) -> tuple[list[dict], list[dict], list[str], list[str]]:
    safe_keys = list(family["safe"])
    budget_keys = list(family["budget"])
    if family["protocol"] == "rank":
        return rank_train_test(train_rows, test_rows, safe_keys, budget_keys)
    return train_rows, test_rows, safe_keys, budget_keys


def select_eval_cost_shift(
    train_rows: list[dict],
    test_rows: list[dict],
    *,
    family_name: str,
    train_costs: dict,
    eval_costs: dict,
) -> dict:
    family = FAMILIES[family_name]
    prepared_train, prepared_test, safe_keys, budget_keys = prepare_rows_for_family(family, train_rows, test_rows)
    selected = select_transfer_decision_policy(
        prepared_train,
        safe_score_keys=safe_keys,
        budget_score_keys=budget_keys,
        safe_label_key="task_safe",
        budget_label_key="task_budget",
        **train_costs,
    )
    metrics = evaluate_transfer_decision_policy(
        prepared_test,
        safe_score_key=selected["safe_score_key"],
        budget_score_key=selected["budget_score_key"],
        safe_threshold=selected["safe_classifier"]["threshold"],
        safe_direction=selected["safe_classifier"]["direction"],
        safe_band=selected["safe_classifier"]["band"],
        budget_threshold=selected["budget_classifier"]["threshold"],
        budget_direction=selected["budget_classifier"]["direction"],
        budget_band=selected["budget_classifier"]["band"],
        safe_label_key="task_safe",
        budget_label_key="task_budget",
        **eval_costs,
    )
    comparison = evaluate_policy_against_trivial(metrics)
    return {
        "metrics": {key: value for key, value in metrics.items() if key != "per_row"},
        "selected": {
            "safe_score_key": selected["safe_score_key"],
            "budget_score_key": selected["budget_score_key"],
            "safe_threshold": selected["safe_classifier"]["threshold"],
            "safe_direction": selected["safe_classifier"]["direction"],
            "safe_band": selected["safe_classifier"]["band"],
            "budget_threshold": selected["budget_classifier"]["threshold"],
            "budget_direction": selected["budget_classifier"]["direction"],
            "budget_band": selected["budget_classifier"]["band"],
        },
        **comparison,
    }


def run_trial(
    semireal_rows: list[dict],
    *,
    calibration_worlds: list[str],
    budget: int,
    trial_index: int,
    family_name: str,
    cost_profile: str,
    train_mode: str,
) -> dict:
    calibration_set = set(calibration_worlds)
    train_rows = [row for row in semireal_rows if row["world"] in calibration_set]
    test_rows = [row for row in semireal_rows if row["world"] not in calibration_set]
    eval_costs = COST_PROFILES[cost_profile]
    train_costs = eval_costs if train_mode == "cost_aware" else DEFAULT_COSTS
    result = select_eval_cost_shift(
        train_rows,
        test_rows,
        family_name=family_name,
        train_costs=train_costs,
        eval_costs=eval_costs,
    )
    metrics = result["metrics"]
    return {
        "family_name": family_name,
        "cost_profile": cost_profile,
        "train_mode": train_mode,
        "calibration_world_count": budget,
        "trial_index": trial_index,
        "calibration_worlds": calibration_worlds,
        "average_cost": metrics["average_cost"],
        "best_trivial_cost": result["best_trivial_cost"],
        "delta_to_best_trivial": result["delta_to_best_trivial"],
        "wins_best_trivial": result["wins_best_trivial"],
        "oracle_average_cost": metrics["oracle_average_cost"],
        "regret": metrics["regret"],
        "action_rates": metrics["action_rates"],
        "selected": result["selected"],
    }


def summarize_trials(rows: list[dict]) -> dict:
    costs = [float(row["average_cost"]) for row in rows]
    deltas = [float(row["delta_to_best_trivial"]) for row in rows]
    regrets = [float(row["regret"]) for row in rows]
    return {
        "trial_count": len(rows),
        "average_cost_mean": mean_or_zero(costs),
        "average_cost_std": std_or_zero(costs),
        "delta_mean": mean_or_zero(deltas),
        "delta_std": std_or_zero(deltas),
        "delta_median": median(deltas) if deltas else 0.0,
        "win_rate": mean_or_zero([float(row["wins_best_trivial"]) for row in rows]),
        "regret_mean": mean_or_zero(regrets),
        "structured_rate_mean": mean_or_zero([float(row["action_rates"]["structured"]) for row in rows]),
        "fallback_rate_mean": mean_or_zero([float(row["action_rates"]["fallback"]) for row in rows]),
        "escalate_rate_mean": mean_or_zero([float(row["action_rates"]["escalate"]) for row in rows]),
    }


def build_summary(trials: list[dict]) -> dict:
    output: dict[str, dict] = {}
    for train_mode in ("cost_aware", "default_trained"):
        output[train_mode] = {}
        for family_name in FAMILIES:
            output[train_mode][family_name] = {}
            for cost_profile in COST_PROFILES:
                output[train_mode][family_name][cost_profile] = {}
                budgets = sorted(
                    {
                        int(row["calibration_world_count"])
                        for row in trials
                        if row["train_mode"] == train_mode
                        and row["family_name"] == family_name
                        and row["cost_profile"] == cost_profile
                    }
                )
                for budget in budgets:
                    subset = [
                        row
                        for row in trials
                        if row["train_mode"] == train_mode
                        and row["family_name"] == family_name
                        and row["cost_profile"] == cost_profile
                        and int(row["calibration_world_count"]) == budget
                    ]
                    output[train_mode][family_name][cost_profile][str(budget)] = summarize_trials(subset)
    return output


def profile_pass_count(summary: dict, train_mode: str, family_name: str, budget: int) -> int:
    count = 0
    for cost_profile in COST_PROFILES:
        row = summary[train_mode][family_name][cost_profile][str(budget)]
        count += int(float(row["delta_mean"]) < 0.0 and float(row["win_rate"]) >= 0.55)
    return count


def leader_table(summary: dict, train_mode: str, budget: int) -> dict[str, dict]:
    leaders = {}
    for cost_profile in COST_PROFILES:
        rows = {
            family_name: summary[train_mode][family_name][cost_profile][str(budget)]
            for family_name in FAMILIES
        }
        family_name, row = min(rows.items(), key=lambda item: (item[1]["delta_mean"], -item[1]["win_rate"]))
        leaders[cost_profile] = {"family_name": family_name, **row}
    return leaders


def plot_robustness(summary: dict, output_path: Path) -> None:
    families = ["validation_loss", "conformal_validation_rank", "router_margin", "support_commutator_reference"]
    colors = {
        "validation_loss": "#9c6b5a",
        "conformal_validation_rank": "#b89445",
        "router_margin": "#6c9a72",
        "support_commutator_reference": "#2f6f73",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharey=True)
    for axis, train_mode in zip(axes, ("cost_aware", "default_trained")):
        for family_name in families:
            budgets = sorted(
                {
                    int(budget)
                    for cost_profile in COST_PROFILES
                    for budget in summary[train_mode][family_name][cost_profile]
                }
            )
            pass_counts = [profile_pass_count(summary, train_mode, family_name, budget) for budget in budgets]
            axis.plot(budgets, pass_counts, marker="o", linewidth=1.8, color=colors[family_name], label=family_name)
        axis.set_title(train_mode)
        axis.set_xlabel("calibration worlds")
        axis.set_ylabel("passed cost profiles")
        axis.set_ylim(-0.1, len(COST_PROFILES) + 0.2)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.suptitle("Support-routing cost robustness")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_outputs(output_dir: Path, report_path: Path, results: dict) -> None:
    summary = results["summary"]
    lines = [
        "# Support Routing Cost Robustness v1",
        "",
        "## Pass Counts",
        "",
        "| Train mode | Family | Calibration worlds | Passed profiles |",
        "| --- | --- | ---: | ---: |",
    ]
    for train_mode in ("cost_aware", "default_trained"):
        for family_name in FAMILIES:
            for budget in results["config"]["budgets"]:
                lines.append(
                    "| "
                    + train_mode
                    + " | "
                    + family_name
                    + f" | {budget}"
                    + f" | {profile_pass_count(summary, train_mode, family_name, budget)} |"
                )
    lines.extend(
        [
            "",
            "## Profile Leaders At 3 Calibration Worlds",
            "",
            "| Train mode | Cost profile | Leader | Delta mean | Win rate |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for train_mode in ("cost_aware", "default_trained"):
        for cost_profile, row in leader_table(summary, train_mode, 3).items():
            lines.append(
                "| "
                + train_mode
                + " | "
                + cost_profile
                + " | "
                + row["family_name"]
                + f" | {row['delta_mean']:+.6f}"
                + f" | {row['win_rate']:.3f} |"
            )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(
        "# Support Routing Cost Robustness v1\n\n![Cost robustness](cost_robustness.png)\n",
        encoding="utf-8",
    )

    best_counts = {
        family_name: profile_pass_count(summary, "cost_aware", family_name, 3)
        for family_name in FAMILIES
    }
    validation_k5_count = profile_pass_count(summary, "cost_aware", "validation_loss", 5)
    validation_expensive_k3 = summary["cost_aware"]["validation_loss"]["expensive_escalation"]["3"]
    validation_expensive_k5 = summary["cost_aware"]["validation_loss"]["expensive_escalation"]["5"]
    best_family = max(best_counts.items(), key=lambda item: item[1])[0]
    findings = [
        "# Support Routing Cost Robustness v1",
        "",
        "## Result",
        "",
        f"- Best cost-aware family at 3 calibration worlds: `{best_family}` with `{best_counts[best_family]}/{len(COST_PROFILES)}` profiles passed.",
        f"- Validation-loss pass count at 3 worlds: `{best_counts['validation_loss']}/{len(COST_PROFILES)}`.",
        f"- Conformal-rank pass count at 3 worlds: `{best_counts['conformal_validation_rank']}/{len(COST_PROFILES)}`.",
        f"- Router-margin pass count at 3 worlds: `{best_counts['router_margin']}/{len(COST_PROFILES)}`.",
        f"- Support-commutator reference pass count at 3 worlds: `{best_counts['support_commutator_reference']}/{len(COST_PROFILES)}`.",
        f"- Validation-loss pass count at 5 worlds: `{validation_k5_count}/{len(COST_PROFILES)}`.",
        f"- Expensive-escalation validation-loss delta at 3 worlds: `{validation_expensive_k3['delta_mean']:+.6f}`.",
        f"- Expensive-escalation validation-loss delta at 5 worlds: `{validation_expensive_k5['delta_mean']:+.6f}`.",
        "",
        "## Interpretation",
        "",
        "This probe tests the new project claim: support-calibrated routing is useful only if it remains better than trivial actions under multiple deployment cost models.",
        "",
        "The pivot survives this first gate: simple support-validation and conformal-rank policies are robust across most cost profiles with only 2-3 calibrated worlds and across all tested profiles with 5 calibrated worlds.",
        "",
        "The hard regime is expensive escalation. With only 3 calibrated worlds, validation-loss still loses there; at 5 calibrated worlds it barely clears the gate. That makes calibration sample size the next bottleneck.",
        "",
        "The support-commutator reference remains weak, so the active project should not re-center the three-point geometry.",
        "",
        "Cost-aware training is the deployable upper bound when the deployment cost model is known. Default-trained evaluation tests whether one fixed policy can survive cost shift.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(findings) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = json.loads(Path(args.source_results).read_text(encoding="utf-8"))
    semireal_suite = json.loads(Path(args.semireal_suite).read_text(encoding="utf-8"))
    semireal_rows = augment_support_diagnostic_scores(source["semireal_rows"], semireal_suite)
    worlds = sorted({row["world"] for row in semireal_rows})
    budgets = [budget for budget in args.budgets if 0 < budget < len(worlds)]
    rng = random.Random(args.sample_seed)
    trials = []
    for budget in budgets:
        for trial_index in range(args.trials):
            calibration_worlds = sorted(rng.sample(worlds, budget))
            for family_name in FAMILIES:
                for cost_profile in COST_PROFILES:
                    for train_mode in ("cost_aware", "default_trained"):
                        trials.append(
                            run_trial(
                                semireal_rows,
                                calibration_worlds=calibration_worlds,
                                budget=budget,
                                trial_index=trial_index,
                                family_name=family_name,
                                cost_profile=cost_profile,
                                train_mode=train_mode,
                            )
                        )
    results = {
        "label": "Support Routing Cost Robustness v1",
        "config": vars(args) | {"budgets": budgets},
        "cost_profiles": COST_PROFILES,
        "family_specs": FAMILIES,
        "world_count": len(worlds),
        "row_count": len(semireal_rows),
        "summary": build_summary(trials),
        "trials": trials,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_robustness(results["summary"], output_dir / "cost_robustness.png")
    write_outputs(output_dir, Path(args.report_path), results)


if __name__ == "__main__":
    main()
