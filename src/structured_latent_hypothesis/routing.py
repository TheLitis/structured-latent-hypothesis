from __future__ import annotations

import math
import random
from statistics import mean, pstdev


def metric(results: dict, world: str, variant: str, name: str = "test_recon_mse") -> float:
    return float(results["summary"][world][variant][name]["mean"])


def available_structured_variants(results: dict, world: str, base_variant: str = "coord_latent") -> list[str]:
    return [variant for variant in results["summary"][world].keys() if variant != base_variant]


def structured_advantage(
    results: dict,
    world: str,
    structured_variants: list[str] | None = None,
    base_variant: str = "coord_latent",
    metric_name: str = "test_recon_mse",
) -> float:
    structured = structured_variants or available_structured_variants(results, world, base_variant=base_variant)
    base_value = metric(results, world, base_variant, metric_name)
    best_structured = min(metric(results, world, variant, metric_name) for variant in structured)
    return base_value - best_structured


def threshold_candidates(scores: list[float]) -> list[float]:
    unique_scores = sorted(set(float(score) for score in scores))
    if not unique_scores:
        return [0.0]
    thresholds = [unique_scores[0] - 1e-6]
    for left, right in zip(unique_scores[:-1], unique_scores[1:]):
        thresholds.append(0.5 * (left + right))
    thresholds.append(unique_scores[-1] + 1e-6)
    return thresholds


def prediction_side(score: float, threshold: float, direction: str, abstain_band: float = 0.0) -> str:
    if abstain_band < 0.0:
        raise ValueError("abstain_band must be non-negative.")
    score = float(score)
    threshold = float(threshold)
    if direction == "high_positive":
        if score >= threshold + abstain_band:
            return "positive"
        if score <= threshold - abstain_band:
            return "negative"
        return "abstain"
    if direction == "low_positive":
        if score <= threshold - abstain_band:
            return "positive"
        if score >= threshold + abstain_band:
            return "negative"
        return "abstain"
    raise ValueError(f"Unsupported direction: {direction}")


def predict_positive(score: float, threshold: float, direction: str) -> bool:
    return prediction_side(score, threshold, direction, abstain_band=0.0) == "positive"


def route_decision_label(
    score: float,
    threshold: float,
    direction: str,
    abstain_band: float = 0.0,
    invert: bool = False,
) -> str:
    side = prediction_side(score, threshold, direction, abstain_band=abstain_band)
    if side == "abstain":
        return "abstain"
    positive_is_structured = not invert
    if side == "positive":
        return "structured" if positive_is_structured else "coord"
    return "coord" if positive_is_structured else "structured"


def route_variant(
    score: float,
    threshold: float,
    direction: str,
    structured_variant: str,
    base_variant: str = "coord_latent",
    invert: bool = False,
    abstain_band: float = 0.0,
) -> str:
    label = route_decision_label(
        score,
        threshold,
        direction,
        abstain_band=abstain_band,
        invert=invert,
    )
    return structured_variant if label == "structured" else base_variant


def oracle_variant(
    results: dict,
    world: str,
    structured_variants: list[str] | None = None,
    base_variant: str = "coord_latent",
    metric_name: str = "test_recon_mse",
) -> str:
    candidates = [base_variant]
    candidates.extend(structured_variants or available_structured_variants(results, world, base_variant=base_variant))
    return min(candidates, key=lambda variant: metric(results, world, variant, metric_name))


def mean_metric_for_assignments(
    results: dict,
    assignments: dict[str, str],
    metric_name: str = "test_recon_mse",
) -> float:
    return mean(metric(results, world, variant, metric_name) for world, variant in assignments.items())


def mean_regret_to_oracle(
    results: dict,
    assignments: dict[str, str],
    structured_variants: list[str] | None = None,
    base_variant: str = "coord_latent",
    metric_name: str = "test_recon_mse",
) -> float:
    regrets = []
    for world, variant in assignments.items():
        chosen = metric(results, world, variant, metric_name)
        oracle = metric(
            results,
            world,
            oracle_variant(results, world, structured_variants=structured_variants, base_variant=base_variant, metric_name=metric_name),
            metric_name,
        )
        regrets.append(chosen - oracle)
    return mean(regrets)


def balanced_accuracy(labels: list[bool], predictions: list[bool]) -> float:
    positives = [index for index, label in enumerate(labels) if label]
    negatives = [index for index, label in enumerate(labels) if not label]
    if not positives or not negatives:
        return mean(float(pred == label) for pred, label in zip(predictions, labels))
    tpr = mean(float(predictions[index]) for index in positives)
    tnr = mean(float(not predictions[index]) for index in negatives)
    return 0.5 * (tpr + tnr)


def sign_accuracy(scores: list[float], advantages: list[float], threshold: float, direction: str) -> float:
    labels = [value > 0.0 for value in advantages]
    predictions = [predict_positive(score, threshold, direction) for score in scores]
    return mean(float(pred == label) for pred, label in zip(predictions, labels))


def filtered_sign_accuracy(
    labels: list[bool],
    predictions: list[bool],
    distances_to_zero: list[float],
    minimum_distance: float = 1e-5,
) -> float | None:
    keep = [index for index, distance in enumerate(distances_to_zero) if abs(float(distance)) >= minimum_distance]
    if not keep:
        return None
    return mean(float(predictions[index] == labels[index]) for index in keep)


def decisions_to_predictions(decisions: list[str]) -> list[bool]:
    return [decision == "structured" for decision in decisions]


def structured_route_rate(decisions: list[str]) -> float:
    return mean(float(decision == "structured") for decision in decisions)


def coverage_rate(decisions: list[str]) -> float:
    return mean(float(decision != "abstain") for decision in decisions)


def routed_values_from_decisions(
    decisions: list[str],
    coord_values: list[float],
    structured_values: list[float],
) -> list[float]:
    routed = []
    for decision, coord, structured in zip(decisions, coord_values, structured_values):
        routed.append(structured if decision == "structured" else coord)
    return routed


def calibrate_threshold(scores: list[float], advantages: list[float]) -> tuple[float, str, float]:
    labels = [value > 0.0 for value in advantages]
    best_threshold = threshold_candidates(scores)[0]
    best_direction = "low_positive"
    best_accuracy = -1.0
    for threshold in threshold_candidates(scores):
        for direction in ("low_positive", "high_positive"):
            predictions = [predict_positive(score, threshold, direction) for score in scores]
            accuracy = mean(float(pred == label) for pred, label in zip(predictions, labels))
            if accuracy > best_accuracy + 1e-12:
                best_accuracy = accuracy
                best_threshold = threshold
                best_direction = direction
    return best_threshold, best_direction, best_accuracy


def calibrate_pair_threshold(
    scores: list[float],
    labels: list[bool],
    coord_values: list[float],
    structured_values: list[float],
    direction: str,
) -> tuple[float, float, float, float, float]:
    best_threshold = threshold_candidates(scores)[0]
    best_balanced = -1.0
    best_regret = float("inf")
    best_margin = -1.0
    best_accuracy = -1.0
    for threshold in threshold_candidates(scores):
        decisions = [route_decision_label(score, threshold, direction) for score in scores]
        predictions = decisions_to_predictions(decisions)
        balanced = balanced_accuracy(labels, predictions)
        routed_values = routed_values_from_decisions(decisions, coord_values, structured_values)
        regret = mean(
            routed - min(coord, structured)
            for routed, coord, structured in zip(routed_values, coord_values, structured_values)
        )
        margin = min(abs(float(score) - threshold) for score in scores)
        accuracy = mean(float(pred == label) for pred, label in zip(predictions, labels))
        if (
            balanced > best_balanced + 1e-12
            or (
                abs(balanced - best_balanced) <= 1e-12
                and (
                    regret < best_regret - 1e-12
                    or (
                        abs(regret - best_regret) <= 1e-12
                        and (
                            margin > best_margin + 1e-12
                            or (
                                abs(margin - best_margin) <= 1e-12
                                and accuracy > best_accuracy + 1e-12
                            )
                        )
                    )
                )
            )
        ):
            best_threshold = threshold
            best_balanced = balanced
            best_regret = regret
            best_margin = margin
            best_accuracy = accuracy
    return best_threshold, best_balanced, best_regret, best_margin, best_accuracy


def choose_best_directional_threshold(
    scores: list[float],
    labels: list[bool],
    coord_values: list[float],
    structured_values: list[float],
) -> tuple[float, str, float, float, float, float]:
    candidates = []
    for direction in ("high_positive", "low_positive"):
        threshold, balanced, regret, margin, accuracy = calibrate_pair_threshold(
            scores,
            labels,
            coord_values,
            structured_values,
            direction=direction,
        )
        candidates.append((balanced, -regret, margin, accuracy, threshold, direction))
    best = max(candidates, key=lambda item: (item[0], item[1], item[2], item[3]))
    balanced = best[0]
    regret = -best[1]
    margin = best[2]
    accuracy = best[3]
    return best[4], best[5], balanced, regret, margin, accuracy


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0.0:
        return min(values)
    if q >= 1.0:
        return max(values)
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def abstain_band_candidates(
    scores: list[float],
    threshold: float,
    quantiles: tuple[float, ...] = (0.2, 0.35, 0.5),
) -> list[float]:
    margins = [abs(float(score) - float(threshold)) for score in scores]
    candidates = [0.0]
    candidates.extend(quantile(margins, q) for q in quantiles)
    return sorted(set(float(candidate) for candidate in candidates))


def choose_abstain_band(
    scores: list[float],
    labels: list[bool],
    coord_values: list[float],
    structured_values: list[float],
    threshold: float,
    direction: str,
    invert: bool = False,
) -> tuple[float, float, float, float, float]:
    best_band = 0.0
    best_balanced = -1.0
    best_regret = float("inf")
    best_coverage = -1.0
    best_accuracy = -1.0
    for band in abstain_band_candidates(scores, threshold):
        decisions = [
            route_decision_label(score, threshold, direction, abstain_band=band, invert=invert)
            for score in scores
        ]
        predictions = decisions_to_predictions(decisions)
        balanced = balanced_accuracy(labels, predictions)
        routed_values = routed_values_from_decisions(decisions, coord_values, structured_values)
        regret = mean(
            routed - min(coord, structured)
            for routed, coord, structured in zip(routed_values, coord_values, structured_values)
        )
        coverage = coverage_rate(decisions)
        accuracy = mean(float(pred == label) for pred, label in zip(predictions, labels))
        if (
            balanced > best_balanced + 1e-12
            or (
                abs(balanced - best_balanced) <= 1e-12
                and (
                    regret < best_regret - 1e-12
                    or (
                        abs(regret - best_regret) <= 1e-12
                        and (
                            coverage > best_coverage + 1e-12
                            or (
                                abs(coverage - best_coverage) <= 1e-12
                                and accuracy > best_accuracy + 1e-12
                            )
                        )
                    )
                )
            )
        ):
            best_band = band
            best_balanced = balanced
            best_regret = regret
            best_coverage = coverage
            best_accuracy = accuracy
    return best_band, best_balanced, best_regret, best_coverage, best_accuracy


def random_router_decisions(
    worlds: list[str],
    structured_rate: float,
    seed: int,
) -> dict[str, str]:
    rng = random.Random(seed)
    return {
        world: ("structured" if rng.random() < structured_rate else "coord")
        for world in worlds
    }


def mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "count": 0.0}
    numeric = [float(value) for value in values]
    return {
        "mean": mean(numeric),
        "std": pstdev(numeric),
        "count": float(len(numeric)),
    }


def aggregate_metric_records(records: list[dict[str, float | None]]) -> dict[str, dict[str, float] | None]:
    if not records:
        return {}
    keys = records[0].keys()
    aggregated: dict[str, dict[str, float] | None] = {}
    for key in keys:
        values = [record[key] for record in records if record[key] is not None]
        if not values:
            aggregated[key] = None
            continue
        numeric = [float(value) for value in values]
        aggregated[key] = {
            "mean": mean(numeric),
            "std": pstdev(numeric),
            "count": float(len(numeric)),
        }
    return aggregated


def family_from_world(world: str) -> str:
    if world.startswith("semireal_"):
        return "semi-real"
    if world.startswith("stepcurve_coupled_"):
        return "synthetic"
    if world.startswith("stepcurve_") or world.startswith("matched_"):
        return "synthetic"
    return "other"


def family_conditioned_values(worlds: list[str], values_by_world: dict[str, float]) -> dict[str, dict[str, float]]:
    families: dict[str, list[float]] = {}
    for world in worlds:
        families.setdefault(family_from_world(world), []).append(float(values_by_world[world]))
    return {family: mean_std(values) for family, values in families.items()}
