from __future__ import annotations

from statistics import mean

from .transfer_criterion import (
    evaluate_transfer_decision_policy,
    select_transfer_decision_policy,
    spearman_correlation,
)


def run_index(results: dict) -> dict[tuple[str, int, str], dict]:
    return {
        (run["config"]["world"], int(run["config"]["seed"]), run["config"]["variant"]): run
        for run in results["runs"]
    }


def curve_value(curve: list[float], step: int) -> float:
    if not curve:
        return 0.0
    return float(curve[min(step, len(curve) - 1)])


def adaptation_gain(run: dict, *, step: int, curve_key: str) -> float:
    curve = [float(value) for value in run["adaptation"][curve_key]]
    return curve[0] - curve_value(curve, step)


def safe_ratio(numerator: float, denominator: float, eps: float = 1e-12) -> float:
    return float(numerator) / max(abs(float(denominator)), eps)


def build_support_contrast_rows(
    results: dict,
    *,
    structured_variant: str = "operator_diag_residual",
    fallback_variant: str = "full_transition",
    steps: tuple[int, ...] = (1, 2, 4, 8),
    regret_tolerance: float = 1e-5,
    budget: int = 8,
) -> list[dict]:
    indexed = run_index(results)
    rows: list[dict] = []
    for world in results["worlds"]:
        family = "commuting" if "commuting" in world else "coupled"
        alpha = float(world.rsplit("_", maxsplit=1)[-1])
        for seed in results["seeds"]:
            fallback = indexed[(world, int(seed), fallback_variant)]
            structured = indexed[(world, int(seed), structured_variant)]
            fallback_adapt = fallback["adaptation"]
            structured_adapt = structured["adaptation"]
            fallback_gain_early = adaptation_gain(fallback, step=steps[0], curve_key="support_curve")
            structured_gain_early = adaptation_gain(structured, step=steps[0], curve_key="support_curve")
            fallback_gain_late = adaptation_gain(fallback, step=steps[-1], curve_key="support_curve")
            structured_gain_late = adaptation_gain(structured, step=steps[-1], curve_key="support_curve")
            structured_query_regret = float(structured_adapt["best_query_mse"]) - float(fallback_adapt["best_query_mse"])
            residual_curve = [float(value) for value in structured_adapt.get("support_residual_curve", [0.0])]
            support_curve = [float(value) for value in structured_adapt["support_curve"]]
            instability = max(0.0, max(support_curve[: min(steps[-1] + 1, len(support_curve))]) - support_curve[0])
            residual_final = curve_value(residual_curve, steps[-1])
            residual_norm = safe_ratio(residual_final, curve_value(support_curve, 0))
            rows.append(
                {
                    "world": world,
                    "family": family,
                    "alpha": alpha,
                    "seed": int(seed),
                    "variant": structured_variant,
                    "score_gain_ratio_1": safe_ratio(structured_gain_early, fallback_gain_early),
                    "score_gain_ratio_8": safe_ratio(structured_gain_late, fallback_gain_late),
                    "score_gain_delta_1": structured_gain_early - fallback_gain_early,
                    "score_gain_delta_8": structured_gain_late - fallback_gain_late,
                    "score_residual_normalized": residual_norm,
                    "score_instability": instability,
                    "score_contrast": safe_ratio(structured_gain_late, fallback_gain_late)
                    - residual_norm
                    - safe_ratio(instability, curve_value(support_curve, 0)),
                    "structured_query_regret": structured_query_regret,
                    "fallback_best_query_mse": float(fallback_adapt["best_query_mse"]),
                    "structured_best_query_mse": float(structured_adapt["best_query_mse"]),
                    "fallback_adaptation_steps": int(fallback_adapt["steps_to_target"]),
                    "structured_adaptation_steps": int(structured_adapt["steps_to_target"]),
                    "task_safe": structured_query_regret <= regret_tolerance,
                    "task_budget": int(fallback_adapt["steps_to_target"]) <= budget,
                }
            )
    return rows


def evaluate_support_contrast_transfer(
    synthetic_rows: list[dict],
    semireal_rows: list[dict],
    *,
    safe_score_keys: list[str],
    budget_score_keys: list[str],
    structured_violation_cost: float,
    fallback_overbudget_cost: float,
    escalate_needed_cost: float,
    escalate_unneeded_cost: float,
) -> dict:
    selected = select_transfer_decision_policy(
        synthetic_rows,
        safe_score_keys=safe_score_keys,
        budget_score_keys=budget_score_keys,
        safe_label_key="task_safe",
        budget_label_key="task_budget",
        structured_violation_cost=structured_violation_cost,
        fallback_overbudget_cost=fallback_overbudget_cost,
        escalate_needed_cost=escalate_needed_cost,
        escalate_unneeded_cost=escalate_unneeded_cost,
    )
    metrics = evaluate_transfer_decision_policy(
        semireal_rows,
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
        structured_violation_cost=structured_violation_cost,
        fallback_overbudget_cost=fallback_overbudget_cost,
        escalate_needed_cost=escalate_needed_cost,
        escalate_unneeded_cost=escalate_unneeded_cost,
    )
    best_trivial = min(
        metrics["always_structured_cost"],
        metrics["always_fallback_cost"],
        metrics["always_escalate_cost"],
    )
    return {
        "selected": selected,
        "metrics": {key: value for key, value in metrics.items() if key != "per_row"},
        "best_trivial_cost": best_trivial,
        "delta_to_best_trivial": metrics["average_cost"] - best_trivial,
        "wins_best_trivial": metrics["average_cost"] < best_trivial - 1e-12,
        "per_row": metrics["per_row"],
    }


def score_correlations(rows: list[dict], score_keys: list[str]) -> dict[str, dict[str, float]]:
    targets = {
        "structured_query_regret": [float(row["structured_query_regret"]) for row in rows],
        "fallback_adaptation_steps": [float(row["fallback_adaptation_steps"]) for row in rows],
        "structured_adaptation_steps": [float(row["structured_adaptation_steps"]) for row in rows],
    }
    return {
        score_key: {
            target: spearman_correlation([float(row[score_key]) for row in rows], values)
            for target, values in targets.items()
        }
        for score_key in score_keys
    }


def summarize_binary_labels(rows: list[dict]) -> dict[str, float]:
    return {
        "safe_rate": mean([float(row["task_safe"]) for row in rows]) if rows else 0.0,
        "budget_rate": mean([float(row["task_budget"]) for row in rows]) if rows else 0.0,
    }


def percentile_rank(value: float, reference_values: list[float]) -> float:
    if not reference_values:
        return 0.5
    less = sum(1 for ref in reference_values if ref < value)
    equal = sum(1 for ref in reference_values if ref == value)
    return (less + 0.5 * equal) / len(reference_values)


def add_rank_features(
    rows: list[dict],
    *,
    score_keys: list[str],
    reference_rows: list[dict] | None = None,
    prefix: str = "rank_",
) -> list[dict]:
    reference = reference_rows if reference_rows is not None else rows
    reference_values = {
        score_key: [float(row[score_key]) for row in reference]
        for score_key in score_keys
    }
    ranked: list[dict] = []
    for row in rows:
        updated = dict(row)
        for score_key in score_keys:
            updated[f"{prefix}{score_key}"] = percentile_rank(float(row[score_key]), reference_values[score_key])
        ranked.append(updated)
    return ranked


def best_trivial_policy_cost(metrics: dict) -> float:
    keys = ("always_structured_cost", "always_fallback_cost", "always_escalate_cost")
    if all(key in metrics for key in keys):
        return min(float(metrics[key]) for key in keys)
    mean_keys = ("always_structured_cost_mean", "always_fallback_cost_mean", "always_escalate_cost_mean")
    return min(float(metrics[key]) for key in mean_keys)


def evaluate_policy_against_trivial(metrics: dict) -> dict:
    best_trivial = best_trivial_policy_cost(metrics)
    average_cost = float(metrics.get("average_cost", metrics.get("average_cost_mean")))
    return {
        "best_trivial_cost": best_trivial,
        "delta_to_best_trivial": average_cost - best_trivial,
        "wins_best_trivial": average_cost < best_trivial - 1e-12,
    }


def evaluate_rank_external_transfer(
    synthetic_rows: list[dict],
    target_rows: list[dict],
    *,
    raw_safe_score_keys: list[str],
    raw_budget_score_keys: list[str],
    structured_violation_cost: float,
    fallback_overbudget_cost: float,
    escalate_needed_cost: float,
    escalate_unneeded_cost: float,
) -> dict:
    raw_keys = sorted(set(raw_safe_score_keys + raw_budget_score_keys))
    synthetic_ranked = add_rank_features(synthetic_rows, score_keys=raw_keys)
    target_ranked = add_rank_features(target_rows, score_keys=raw_keys)
    safe_score_keys = [f"rank_{key}" for key in raw_safe_score_keys]
    budget_score_keys = [f"rank_{key}" for key in raw_budget_score_keys]
    selected = select_transfer_decision_policy(
        synthetic_ranked,
        safe_score_keys=safe_score_keys,
        budget_score_keys=budget_score_keys,
        safe_label_key="task_safe",
        budget_label_key="task_budget",
        structured_violation_cost=structured_violation_cost,
        fallback_overbudget_cost=fallback_overbudget_cost,
        escalate_needed_cost=escalate_needed_cost,
        escalate_unneeded_cost=escalate_unneeded_cost,
    )
    metrics = evaluate_transfer_decision_policy(
        target_ranked,
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
        structured_violation_cost=structured_violation_cost,
        fallback_overbudget_cost=fallback_overbudget_cost,
        escalate_needed_cost=escalate_needed_cost,
        escalate_unneeded_cost=escalate_unneeded_cost,
    )
    comparison = evaluate_policy_against_trivial(metrics)
    return {
        "selected": selected,
        "metrics": {key: value for key, value in metrics.items() if key != "per_row"},
        "per_row": metrics["per_row"],
        **comparison,
    }


def cross_validate_rank_calibrated_transfer(
    target_rows: list[dict],
    *,
    group_key: str,
    raw_safe_score_keys: list[str],
    raw_budget_score_keys: list[str],
    structured_violation_cost: float,
    fallback_overbudget_cost: float,
    escalate_needed_cost: float,
    escalate_unneeded_cost: float,
    source_rows: list[dict] | None = None,
) -> dict:
    raw_keys = sorted(set(raw_safe_score_keys + raw_budget_score_keys))
    safe_score_keys = [f"rank_{key}" for key in raw_safe_score_keys]
    budget_score_keys = [f"rank_{key}" for key in raw_budget_score_keys]
    per_group: list[dict] = []
    for group_value in sorted({row[group_key] for row in target_rows}):
        calibration_rows = [row for row in target_rows if row[group_key] != group_value]
        test_rows = [row for row in target_rows if row[group_key] == group_value]
        calibration_ranked = add_rank_features(calibration_rows, score_keys=raw_keys)
        test_ranked = add_rank_features(test_rows, score_keys=raw_keys, reference_rows=calibration_rows)
        if source_rows is not None:
            source_ranked = add_rank_features(source_rows, score_keys=raw_keys)
            train_rows = source_ranked + calibration_ranked
        else:
            train_rows = calibration_ranked
        selected = select_transfer_decision_policy(
            train_rows,
            safe_score_keys=safe_score_keys,
            budget_score_keys=budget_score_keys,
            safe_label_key="task_safe",
            budget_label_key="task_budget",
            structured_violation_cost=structured_violation_cost,
            fallback_overbudget_cost=fallback_overbudget_cost,
            escalate_needed_cost=escalate_needed_cost,
            escalate_unneeded_cost=escalate_unneeded_cost,
        )
        metrics = evaluate_transfer_decision_policy(
            test_ranked,
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
            structured_violation_cost=structured_violation_cost,
            fallback_overbudget_cost=fallback_overbudget_cost,
            escalate_needed_cost=escalate_needed_cost,
            escalate_unneeded_cost=escalate_unneeded_cost,
        )
        metrics[group_key] = group_value
        metrics["safe_score_key"] = selected["safe_score_key"]
        metrics["budget_score_key"] = selected["budget_score_key"]
        per_group.append(metrics)
    return {
        "per_group": [{key: value for key, value in row.items() if key != "per_row"} for row in per_group],
        "average_cost_mean": mean([float(row["average_cost"]) for row in per_group]),
        "oracle_average_cost_mean": mean([float(row["oracle_average_cost"]) for row in per_group]),
        "regret_mean": mean([float(row["regret"]) for row in per_group]),
        "always_structured_cost_mean": mean([float(row["always_structured_cost"]) for row in per_group]),
        "always_fallback_cost_mean": mean([float(row["always_fallback_cost"]) for row in per_group]),
        "always_escalate_cost_mean": mean([float(row["always_escalate_cost"]) for row in per_group]),
        "structured_rate_mean": mean([float(row["action_rates"]["structured"]) for row in per_group]),
        "fallback_rate_mean": mean([float(row["action_rates"]["fallback"]) for row in per_group]),
        "escalate_rate_mean": mean([float(row["action_rates"]["escalate"]) for row in per_group]),
    }
