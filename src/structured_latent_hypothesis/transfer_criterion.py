from __future__ import annotations

import json
import math
from pathlib import Path


CANDIDATE_SPECS: dict[str, tuple[str, str]] = {
    "plus_interaction": ("operator_plus_residual", "interaction"),
    "plus_residual": ("operator_plus_residual", "residual"),
    "plus_joint_sum": ("operator_plus_residual", "joint_sum"),
    "plus_joint_prod": ("operator_plus_residual", "joint_prod"),
    "diag_interaction": ("operator_diag_residual", "interaction"),
    "diag_residual": ("operator_diag_residual", "residual"),
    "diag_joint_sum": ("operator_diag_residual", "joint_sum"),
    "diag_joint_prod": ("operator_diag_residual", "joint_prod"),
}


def parse_world(world: str) -> tuple[str, float]:
    return ("commuting" if "commuting" in world else "coupled", float(world.rsplit("_", maxsplit=1)[-1]))


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_context_transfer_rows(operator_results: dict, adaptation_results: dict) -> list[dict]:
    operator_runs = {
        (run["config"]["world"], run["config"]["seed"], run["config"]["variant"]): run for run in operator_results["runs"]
    }
    adaptation_runs = {
        (run["config"]["world"], run["config"]["seed"], run["config"]["variant"]): run for run in adaptation_results["runs"]
    }

    worlds = operator_results["worlds"]
    seeds = operator_results["seeds"]
    rows: list[dict] = []
    for world in worlds:
        family, alpha = parse_world(world)
        for seed in seeds:
            full_operator = operator_runs[(world, seed, "full_transition")]
            full_adaptation = adaptation_runs[(world, seed, "full_transition")]
            for variant in ("operator_plus_residual", "operator_diag_residual"):
                operator_run = operator_runs[(world, seed, variant)]
                adaptation_run = adaptation_runs[(world, seed, variant)]
                interaction = float(operator_run["interaction_norm_holdout"])
                residual = float(adaptation_run["adaptation"]["residual_norm_final"])
                rows.append(
                    {
                        "world": world,
                        "family": family,
                        "alpha": alpha,
                        "seed": int(seed),
                        "variant": variant,
                        "score_interaction": interaction,
                        "score_residual": residual,
                        "score_joint_sum": interaction + residual,
                        "score_joint_prod": interaction * max(residual, 1e-12),
                        "full_zero_shot_rollout5_mse": float(full_operator["zero_shot_rollout5_mse"]),
                        "structured_zero_shot_rollout5_mse": float(operator_run["zero_shot_rollout5_mse"]),
                        "full_zero_shot_one_step_mse": float(full_operator["zero_shot_one_step_mse"]),
                        "structured_zero_shot_one_step_mse": float(operator_run["zero_shot_one_step_mse"]),
                        "full_adaptation_steps": int(full_adaptation["adaptation"]["steps_to_target"]),
                        "structured_adaptation_steps": int(adaptation_run["adaptation"]["steps_to_target"]),
                        "full_adaptation_gain": float(full_adaptation["adaptation"]["adaptation_gain"]),
                        "structured_adaptation_gain": float(adaptation_run["adaptation"]["adaptation_gain"]),
                        "structured_adaptation_residual_norm": residual,
                    }
                )
    return rows


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(values):
        upper = index
        while upper + 1 < len(values) and values[order[upper + 1]] == values[order[index]]:
            upper += 1
        rank = (index + upper) / 2.0 + 1.0
        for cursor in range(index, upper + 1):
            ranks[order[cursor]] = rank
        index = upper + 1
    return ranks


def spearman_correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    rank_x = rankdata(xs)
    rank_y = rankdata(ys)
    mean_x = sum(rank_x) / len(rank_x)
    mean_y = sum(rank_y) / len(rank_y)
    numerator = sum((left - mean_x) * (right - mean_y) for left, right in zip(rank_x, rank_y))
    denominator = math.sqrt(sum((value - mean_x) ** 2 for value in rank_x) * sum((value - mean_y) ** 2 for value in rank_y))
    return numerator / denominator if denominator > 0.0 else float("nan")


def subset_rows(rows: list[dict], subset: str) -> list[dict]:
    if subset == "all":
        return list(rows)
    if subset == "coupled":
        return [row for row in rows if row["family"] == "coupled"]
    if subset == "commuting":
        return [row for row in rows if row["family"] == "commuting"]
    if subset == "low_alpha":
        return [row for row in rows if row["alpha"] <= 0.35]
    if subset == "high_alpha":
        return [row for row in rows if row["alpha"] >= 0.75]
    raise ValueError(f"Unknown subset: {subset}")


def select_threshold_router(rows: list[dict], score_key: str) -> dict:
    unique_scores = sorted({float(row[score_key]) for row in rows})
    best: tuple[float, float, float, float, float, str] | None = None
    for threshold in unique_scores:
        for direction in ("low", "high"):
            routed_values: list[float] = []
            oracle_values: list[float] = []
            correct = 0
            route_count = 0
            for row in rows:
                choose_structured = row[score_key] <= threshold if direction == "low" else row[score_key] >= threshold
                routed_values.append(
                    row["structured_zero_shot_rollout5_mse"] if choose_structured else row["full_zero_shot_rollout5_mse"]
                )
                oracle_values.append(min(row["structured_zero_shot_rollout5_mse"], row["full_zero_shot_rollout5_mse"]))
                structured_is_better = row["structured_zero_shot_rollout5_mse"] <= row["full_zero_shot_rollout5_mse"]
                correct += int(choose_structured == structured_is_better)
                route_count += int(choose_structured)
            routed_mean = sum(routed_values) / len(routed_values)
            regret_mean = sum(value - oracle for value, oracle in zip(routed_values, oracle_values)) / len(routed_values)
            accuracy = correct / len(rows)
            route_rate = route_count / len(rows)
            candidate = (routed_mean, regret_mean, -accuracy, abs(route_rate - 0.5), threshold, direction)
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return {
        "threshold": best[4],
        "direction": best[5],
        "train_routed_rollout5_mse": best[0],
        "train_regret": best[1],
        "train_accuracy": -best[2],
    }


def evaluate_router(rows: list[dict], score_key: str, threshold: float, direction: str) -> dict:
    routed_values: list[float] = []
    oracle_values: list[float] = []
    full_values: list[float] = []
    structured_values: list[float] = []
    correct = 0
    route_count = 0
    for row in rows:
        choose_structured = row[score_key] <= threshold if direction == "low" else row[score_key] >= threshold
        routed_values.append(row["structured_zero_shot_rollout5_mse"] if choose_structured else row["full_zero_shot_rollout5_mse"])
        oracle_values.append(min(row["structured_zero_shot_rollout5_mse"], row["full_zero_shot_rollout5_mse"]))
        full_values.append(row["full_zero_shot_rollout5_mse"])
        structured_values.append(row["structured_zero_shot_rollout5_mse"])
        structured_is_better = row["structured_zero_shot_rollout5_mse"] <= row["full_zero_shot_rollout5_mse"]
        correct += int(choose_structured == structured_is_better)
        route_count += int(choose_structured)
    return {
        "routed_rollout5_mse": sum(routed_values) / len(routed_values),
        "oracle_rollout5_mse": sum(oracle_values) / len(oracle_values),
        "always_full_rollout5_mse": sum(full_values) / len(full_values),
        "always_structured_rollout5_mse": sum(structured_values) / len(structured_values),
        "regret": sum(value - oracle for value, oracle in zip(routed_values, oracle_values)) / len(routed_values),
        "accuracy": correct / len(rows),
        "structured_route_rate": route_count / len(rows),
    }


def leave_one_seed_out_router(rows: list[dict], score_key: str) -> dict:
    per_seed: list[dict] = []
    for seed in sorted({row["seed"] for row in rows}):
        train_rows = [row for row in rows if row["seed"] != seed]
        test_rows = [row for row in rows if row["seed"] == seed]
        selected = select_threshold_router(train_rows, score_key)
        metrics = evaluate_router(test_rows, score_key, selected["threshold"], selected["direction"])
        metrics["seed"] = seed
        metrics["threshold"] = selected["threshold"]
        metrics["direction"] = selected["direction"]
        per_seed.append(metrics)
    return {
        "per_seed": per_seed,
        "routed_rollout5_mse_mean": sum(row["routed_rollout5_mse"] for row in per_seed) / len(per_seed),
        "always_full_rollout5_mse_mean": sum(row["always_full_rollout5_mse"] for row in per_seed) / len(per_seed),
        "always_structured_rollout5_mse_mean": sum(row["always_structured_rollout5_mse"] for row in per_seed) / len(per_seed),
        "regret_mean": sum(row["regret"] for row in per_seed) / len(per_seed),
        "accuracy_mean": sum(row["accuracy"] for row in per_seed) / len(per_seed),
        "structured_route_rate_mean": sum(row["structured_route_rate"] for row in per_seed) / len(per_seed),
    }


def confusion_counts(labels: list[bool], predictions: list[bool]) -> dict[str, int]:
    tp = sum(int(label and prediction) for label, prediction in zip(labels, predictions))
    tn = sum(int((not label) and (not prediction)) for label, prediction in zip(labels, predictions))
    fp = sum(int((not label) and prediction) for label, prediction in zip(labels, predictions))
    fn = sum(int(label and (not prediction)) for label, prediction in zip(labels, predictions))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def balanced_accuracy_from_counts(counts: dict[str, int]) -> float:
    tpr_denom = counts["tp"] + counts["fn"]
    tnr_denom = counts["tn"] + counts["fp"]
    true_positive_rate = counts["tp"] / tpr_denom if tpr_denom > 0 else 0.5
    true_negative_rate = counts["tn"] / tnr_denom if tnr_denom > 0 else 0.5
    return 0.5 * (true_positive_rate + true_negative_rate)


def average_cost_from_counts(counts: dict[str, int], false_positive_cost: float, false_negative_cost: float) -> float:
    total = counts["tp"] + counts["tn"] + counts["fp"] + counts["fn"]
    if total == 0:
        return 0.0
    return (counts["fp"] * false_positive_cost + counts["fn"] * false_negative_cost) / total


def select_cost_sensitive_threshold(
    rows: list[dict],
    score_key: str,
    label_key: str,
    *,
    false_positive_cost: float,
    false_negative_cost: float,
) -> dict:
    unique_scores = sorted({float(row[score_key]) for row in rows})
    best: tuple[float, float, float, float, float, str] | None = None
    for threshold in unique_scores:
        for direction in ("low", "high"):
            labels = [bool(row[label_key]) for row in rows]
            predictions = [
                (row[score_key] <= threshold if direction == "low" else row[score_key] >= threshold) for row in rows
            ]
            counts = confusion_counts(labels, predictions)
            average_cost = average_cost_from_counts(counts, false_positive_cost, false_negative_cost)
            balanced_accuracy = balanced_accuracy_from_counts(counts)
            accuracy = (counts["tp"] + counts["tn"]) / len(rows)
            positive_rate = sum(predictions) / len(predictions)
            candidate = (average_cost, -balanced_accuracy, -accuracy, abs(positive_rate - 0.5), threshold, direction)
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return {
        "threshold": best[4],
        "direction": best[5],
        "train_cost": best[0],
        "train_balanced_accuracy": -best[1],
        "train_accuracy": -best[2],
    }


def evaluate_cost_sensitive_threshold(
    rows: list[dict],
    score_key: str,
    label_key: str,
    *,
    threshold: float,
    direction: str,
    false_positive_cost: float,
    false_negative_cost: float,
) -> dict:
    labels = [bool(row[label_key]) for row in rows]
    predictions = [
        (row[score_key] <= threshold if direction == "low" else row[score_key] >= threshold) for row in rows
    ]
    counts = confusion_counts(labels, predictions)
    positive_rate = sum(predictions) / len(predictions) if predictions else 0.0
    always_positive_counts = confusion_counts(labels, [True] * len(labels))
    always_negative_counts = confusion_counts(labels, [False] * len(labels))
    return {
        "average_cost": average_cost_from_counts(counts, false_positive_cost, false_negative_cost),
        "accuracy": (counts["tp"] + counts["tn"]) / len(rows) if rows else 0.0,
        "balanced_accuracy": balanced_accuracy_from_counts(counts),
        "positive_rate": positive_rate,
        "counts": counts,
        "always_positive_cost": average_cost_from_counts(
            always_positive_counts, false_positive_cost, false_negative_cost
        ),
        "always_negative_cost": average_cost_from_counts(
            always_negative_counts, false_positive_cost, false_negative_cost
        ),
    }


def leave_one_seed_out_classifier(
    rows: list[dict],
    score_key: str,
    label_key: str,
    *,
    false_positive_cost: float,
    false_negative_cost: float,
) -> dict:
    per_seed: list[dict] = []
    for seed in sorted({row["seed"] for row in rows}):
        train_rows = [row for row in rows if row["seed"] != seed]
        test_rows = [row for row in rows if row["seed"] == seed]
        selected = select_cost_sensitive_threshold(
            train_rows,
            score_key,
            label_key,
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
        )
        metrics = evaluate_cost_sensitive_threshold(
            test_rows,
            score_key,
            label_key,
            threshold=selected["threshold"],
            direction=selected["direction"],
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
        )
        metrics["seed"] = seed
        metrics["threshold"] = selected["threshold"]
        metrics["direction"] = selected["direction"]
        per_seed.append(metrics)
    return {
        "per_seed": per_seed,
        "average_cost_mean": sum(row["average_cost"] for row in per_seed) / len(per_seed),
        "accuracy_mean": sum(row["accuracy"] for row in per_seed) / len(per_seed),
        "balanced_accuracy_mean": sum(row["balanced_accuracy"] for row in per_seed) / len(per_seed),
        "positive_rate_mean": sum(row["positive_rate"] for row in per_seed) / len(per_seed),
        "always_positive_cost_mean": sum(row["always_positive_cost"] for row in per_seed) / len(per_seed),
        "always_negative_cost_mean": sum(row["always_negative_cost"] for row in per_seed) / len(per_seed),
    }


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * q))
    return ordered[max(0, min(index, len(ordered) - 1))]


def abstain_confusion_counts(labels: list[bool], predictions: list[bool | None]) -> dict[str, int]:
    tp = sum(int(label and prediction is True) for label, prediction in zip(labels, predictions))
    tn = sum(int((not label) and prediction is False) for label, prediction in zip(labels, predictions))
    fp = sum(int((not label) and prediction is True) for label, prediction in zip(labels, predictions))
    fn = sum(int(label and prediction is False) for label, prediction in zip(labels, predictions))
    ap = sum(int(label and prediction is None) for label, prediction in zip(labels, predictions))
    an = sum(int((not label) and prediction is None) for label, prediction in zip(labels, predictions))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "ap": ap, "an": an}


def average_abstain_cost(
    counts: dict[str, int],
    *,
    false_positive_cost: float,
    false_negative_cost: float,
    abstain_positive_cost: float,
    abstain_negative_cost: float,
) -> float:
    total = counts["tp"] + counts["tn"] + counts["fp"] + counts["fn"] + counts["ap"] + counts["an"]
    if total == 0:
        return 0.0
    numerator = (
        counts["fp"] * false_positive_cost
        + counts["fn"] * false_negative_cost
        + counts["ap"] * abstain_positive_cost
        + counts["an"] * abstain_negative_cost
    )
    return numerator / total


def select_cost_sensitive_abstain(
    rows: list[dict],
    score_key: str,
    label_key: str,
    *,
    false_positive_cost: float,
    false_negative_cost: float,
    abstain_positive_cost: float,
    abstain_negative_cost: float,
) -> dict:
    unique_scores = sorted({float(row[score_key]) for row in rows})
    best: tuple[float, float, float, float, float, str, float] | None = None
    for threshold in unique_scores:
        margins = [abs(float(row[score_key]) - threshold) for row in rows]
        band_candidates = sorted({0.0, quantile(margins, 0.20), quantile(margins, 0.35), quantile(margins, 0.50)})
        for direction in ("low", "high"):
            for band in band_candidates:
                labels = [bool(row[label_key]) for row in rows]
                predictions: list[bool | None] = []
                for row in rows:
                    score = float(row[score_key])
                    if band > 0.0 and abs(score - threshold) <= band:
                        predictions.append(None)
                    elif direction == "low":
                        predictions.append(score <= threshold)
                    else:
                        predictions.append(score >= threshold)
                counts = abstain_confusion_counts(labels, predictions)
                average_cost = average_abstain_cost(
                    counts,
                    false_positive_cost=false_positive_cost,
                    false_negative_cost=false_negative_cost,
                    abstain_positive_cost=abstain_positive_cost,
                    abstain_negative_cost=abstain_negative_cost,
                )
                resolved = counts["tp"] + counts["tn"] + counts["fp"] + counts["fn"]
                coverage = resolved / len(rows)
                balanced_accuracy = balanced_accuracy_from_counts(counts)
                candidate = (average_cost, -coverage, -balanced_accuracy, band, threshold, direction, coverage)
                if best is None or candidate < best:
                    best = candidate
    assert best is not None
    return {
        "threshold": best[4],
        "direction": best[5],
        "band": best[3],
        "train_cost": best[0],
        "train_coverage": best[6],
        "train_balanced_accuracy": -best[2],
    }


def evaluate_cost_sensitive_abstain(
    rows: list[dict],
    score_key: str,
    label_key: str,
    *,
    threshold: float,
    direction: str,
    band: float,
    false_positive_cost: float,
    false_negative_cost: float,
    abstain_positive_cost: float,
    abstain_negative_cost: float,
) -> dict:
    labels = [bool(row[label_key]) for row in rows]
    predictions: list[bool | None] = []
    for row in rows:
        score = float(row[score_key])
        if band > 0.0 and abs(score - threshold) <= band:
            predictions.append(None)
        elif direction == "low":
            predictions.append(score <= threshold)
        else:
            predictions.append(score >= threshold)
    counts = abstain_confusion_counts(labels, predictions)
    total = len(rows)
    resolved = counts["tp"] + counts["tn"] + counts["fp"] + counts["fn"]
    always_positive_counts = abstain_confusion_counts(labels, [True] * total)
    always_negative_counts = abstain_confusion_counts(labels, [False] * total)
    always_abstain_counts = abstain_confusion_counts(labels, [None] * total)
    return {
        "average_cost": average_abstain_cost(
            counts,
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
            abstain_positive_cost=abstain_positive_cost,
            abstain_negative_cost=abstain_negative_cost,
        ),
        "coverage": resolved / total if total else 0.0,
        "abstain_rate": (counts["ap"] + counts["an"]) / total if total else 0.0,
        "balanced_accuracy": balanced_accuracy_from_counts(counts),
        "counts": counts,
        "always_positive_cost": average_abstain_cost(
            always_positive_counts,
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
            abstain_positive_cost=abstain_positive_cost,
            abstain_negative_cost=abstain_negative_cost,
        ),
        "always_negative_cost": average_abstain_cost(
            always_negative_counts,
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
            abstain_positive_cost=abstain_positive_cost,
            abstain_negative_cost=abstain_negative_cost,
        ),
        "always_abstain_cost": average_abstain_cost(
            always_abstain_counts,
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
            abstain_positive_cost=abstain_positive_cost,
            abstain_negative_cost=abstain_negative_cost,
        ),
    }


def cross_validate_classifier_by_group(
    rows: list[dict],
    score_key: str,
    label_key: str,
    *,
    group_key: str,
    false_positive_cost: float,
    false_negative_cost: float,
) -> dict:
    per_group: list[dict] = []
    for group_value in sorted({row[group_key] for row in rows}):
        train_rows = [row for row in rows if row[group_key] != group_value]
        test_rows = [row for row in rows if row[group_key] == group_value]
        selected = select_cost_sensitive_threshold(
            train_rows,
            score_key,
            label_key,
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
        )
        metrics = evaluate_cost_sensitive_threshold(
            test_rows,
            score_key,
            label_key,
            threshold=selected["threshold"],
            direction=selected["direction"],
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
        )
        metrics[group_key] = group_value
        metrics["threshold"] = selected["threshold"]
        metrics["direction"] = selected["direction"]
        per_group.append(metrics)
    return {
        "per_group": per_group,
        "average_cost_mean": sum(row["average_cost"] for row in per_group) / len(per_group),
        "accuracy_mean": sum(row["accuracy"] for row in per_group) / len(per_group),
        "balanced_accuracy_mean": sum(row["balanced_accuracy"] for row in per_group) / len(per_group),
        "positive_rate_mean": sum(row["positive_rate"] for row in per_group) / len(per_group),
        "always_positive_cost_mean": sum(row["always_positive_cost"] for row in per_group) / len(per_group),
        "always_negative_cost_mean": sum(row["always_negative_cost"] for row in per_group) / len(per_group),
    }


def cross_validate_abstain_by_group(
    rows: list[dict],
    score_key: str,
    label_key: str,
    *,
    group_key: str,
    false_positive_cost: float,
    false_negative_cost: float,
    abstain_positive_cost: float,
    abstain_negative_cost: float,
) -> dict:
    per_group: list[dict] = []
    for group_value in sorted({row[group_key] for row in rows}):
        train_rows = [row for row in rows if row[group_key] != group_value]
        test_rows = [row for row in rows if row[group_key] == group_value]
        selected = select_cost_sensitive_abstain(
            train_rows,
            score_key,
            label_key,
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
            abstain_positive_cost=abstain_positive_cost,
            abstain_negative_cost=abstain_negative_cost,
        )
        metrics = evaluate_cost_sensitive_abstain(
            test_rows,
            score_key,
            label_key,
            threshold=selected["threshold"],
            direction=selected["direction"],
            band=selected["band"],
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
            abstain_positive_cost=abstain_positive_cost,
            abstain_negative_cost=abstain_negative_cost,
        )
        metrics[group_key] = group_value
        metrics["threshold"] = selected["threshold"]
        metrics["direction"] = selected["direction"]
        metrics["band"] = selected["band"]
        per_group.append(metrics)
    return {
        "per_group": per_group,
        "average_cost_mean": sum(row["average_cost"] for row in per_group) / len(per_group),
        "coverage_mean": sum(row["coverage"] for row in per_group) / len(per_group),
        "abstain_rate_mean": sum(row["abstain_rate"] for row in per_group) / len(per_group),
        "balanced_accuracy_mean": sum(row["balanced_accuracy"] for row in per_group) / len(per_group),
        "always_positive_cost_mean": sum(row["always_positive_cost"] for row in per_group) / len(per_group),
        "always_negative_cost_mean": sum(row["always_negative_cost"] for row in per_group) / len(per_group),
        "always_abstain_cost_mean": sum(row["always_abstain_cost"] for row in per_group) / len(per_group),
    }


def analyze_context_transfer_criterion(rows: list[dict]) -> dict:
    subsets = ["all", "coupled", "commuting", "low_alpha", "high_alpha"]
    targets = {
        "structured_rollout5_regret": lambda row: row["structured_zero_shot_rollout5_mse"] - row["full_zero_shot_rollout5_mse"],
        "structured_one_step_regret": lambda row: row["structured_zero_shot_one_step_mse"] - row["full_zero_shot_one_step_mse"],
        "full_adaptation_steps": lambda row: float(row["full_adaptation_steps"]),
        "structured_adaptation_steps": lambda row: float(row["structured_adaptation_steps"]),
        "structured_adaptation_gain": lambda row: row["structured_adaptation_gain"],
    }
    analyses: dict[str, dict] = {}
    for candidate_name, (variant, score_suffix) in CANDIDATE_SPECS.items():
        score_key = f"score_{score_suffix}"
        variant_rows = [row for row in rows if row["variant"] == variant]
        correlations: dict[str, dict] = {}
        for subset in subsets:
            subset_view = subset_rows(variant_rows, subset)
            correlations[subset] = {}
            if not subset_view:
                continue
            xs = [float(row[score_key]) for row in subset_view]
            for target_name, getter in targets.items():
                ys = [float(getter(row)) for row in subset_view]
                correlations[subset][target_name] = spearman_correlation(xs, ys)
        analyses[candidate_name] = {
            "variant": variant,
            "score_key": score_key,
            "correlations": correlations,
            "router": leave_one_seed_out_router(variant_rows, score_key),
        }
    return analyses


def annotate_transfer_tasks(
    rows: list[dict],
    *,
    step_budgets: list[int],
    regret_tolerances: list[float],
) -> list[dict]:
    annotated: list[dict] = []
    for row in rows:
        updated = dict(row)
        regret = row["structured_zero_shot_rollout5_mse"] - row["full_zero_shot_rollout5_mse"]
        for budget in step_budgets:
            updated[f"task_within_budget_{budget}"] = row["full_adaptation_steps"] <= budget
        for tolerance in regret_tolerances:
            key = format(tolerance, ".0e") if tolerance > 0 else "0"
            updated[f"task_safe_regret_{key}"] = regret <= tolerance
        annotated.append(updated)
    return annotated


def analyze_context_transfer_budget(
    rows: list[dict],
    *,
    step_budgets: list[int],
    regret_tolerances: list[float],
    false_positive_cost: float = 3.0,
    false_negative_cost: float = 1.0,
) -> dict:
    annotated_rows = annotate_transfer_tasks(rows, step_budgets=step_budgets, regret_tolerances=regret_tolerances)
    task_specs: dict[str, dict] = {}
    for budget in step_budgets:
        task_specs[f"within_budget_{budget}"] = {
            "label_key": f"task_within_budget_{budget}",
            "false_positive_cost": false_positive_cost,
            "false_negative_cost": false_negative_cost,
        }
    for tolerance in regret_tolerances:
        suffix = format(tolerance, ".0e") if tolerance > 0 else "0"
        task_specs[f"safe_regret_{suffix}"] = {
            "label_key": f"task_safe_regret_{suffix}",
            "false_positive_cost": false_positive_cost,
            "false_negative_cost": false_negative_cost,
        }

    analyses: dict[str, dict] = {}
    for candidate_name, (variant, score_suffix) in CANDIDATE_SPECS.items():
        score_key = f"score_{score_suffix}"
        variant_rows = [row for row in annotated_rows if row["variant"] == variant]
        task_results: dict[str, dict] = {}
        for task_name, task_config in task_specs.items():
            task_results[task_name] = leave_one_seed_out_classifier(
                variant_rows,
                score_key,
                task_config["label_key"],
                false_positive_cost=task_config["false_positive_cost"],
                false_negative_cost=task_config["false_negative_cost"],
            )
        analyses[candidate_name] = {
            "variant": variant,
            "score_key": score_key,
            "tasks": task_results,
        }
    return {"rows": annotated_rows, "analyses": analyses, "task_specs": task_specs}


def render_budget_markdown(results: dict) -> str:
    task_names = list(results["task_specs"].keys())
    lines = ["# Context Transfer Budget Benchmark", ""]
    for task_name in task_names:
        lines.extend(
            [
                f"## Task: {task_name}",
                "",
                "| Candidate | LOO Cost | Always Pos Cost | Always Neg Cost | LOO Balanced Acc | LOO Acc |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for candidate_name, analysis in results["analyses"].items():
            task = analysis["tasks"][task_name]
            lines.append(
                "| "
                + candidate_name
                + f" | {task['average_cost_mean']:.6f}"
                + f" | {task['always_positive_cost_mean']:.6f}"
                + f" | {task['always_negative_cost_mean']:.6f}"
                + f" | {task['balanced_accuracy_mean']:.3f}"
                + f" | {task['accuracy_mean']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def render_criterion_markdown(results: dict) -> str:
    lines = [
        "# Context Transfer Criterion Benchmark",
        "",
        "## Candidate Scores",
        "",
        "| Candidate | Regret Spearman (All) | Full Steps Spearman (All) | LOO Routed Rollout5 | Always Full | LOO Regret | LOO Accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate_name, analysis in results["analyses"].items():
        correlations = analysis["correlations"]["all"]
        router = analysis["router"]
        lines.append(
            "| "
            + candidate_name
            + f" | {correlations.get('structured_rollout5_regret', float('nan')):+.3f}"
            + f" | {correlations.get('full_adaptation_steps', float('nan')):+.3f}"
            + f" | {router['routed_rollout5_mse_mean']:.6f}"
            + f" | {router['always_full_rollout5_mse_mean']:.6f}"
            + f" | {router['regret_mean']:.6f}"
            + f" | {router['accuracy_mean']:.3f} |"
        )
    return "\n".join(lines) + "\n"
