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
