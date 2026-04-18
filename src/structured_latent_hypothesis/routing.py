from __future__ import annotations

from statistics import mean


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


def calibrate_threshold(scores: list[float], advantages: list[float]) -> tuple[float, str, float]:
    labels = [value > 0.0 for value in advantages]
    unique_scores = sorted(set(float(score) for score in scores))
    thresholds = []
    if unique_scores:
        thresholds.append(unique_scores[0] - 1e-6)
        for left, right in zip(unique_scores[:-1], unique_scores[1:]):
            thresholds.append(0.5 * (left + right))
        thresholds.append(unique_scores[-1] + 1e-6)
    else:
        thresholds = [0.0]

    best_threshold = thresholds[0]
    best_direction = "low_positive"
    best_accuracy = -1.0
    for threshold in thresholds:
        for direction in ("low_positive", "high_positive"):
            predictions = [predict_positive(score, threshold, direction) for score in scores]
            accuracy = mean(float(pred == label) for pred, label in zip(predictions, labels))
            if accuracy > best_accuracy + 1e-12:
                best_accuracy = accuracy
                best_threshold = threshold
                best_direction = direction
    return best_threshold, best_direction, best_accuracy


def predict_positive(score: float, threshold: float, direction: str) -> bool:
    if direction == "low_positive":
        return float(score) <= float(threshold)
    if direction == "high_positive":
        return float(score) >= float(threshold)
    raise ValueError(f"Unsupported direction: {direction}")


def sign_accuracy(scores: list[float], advantages: list[float], threshold: float, direction: str) -> float:
    labels = [value > 0.0 for value in advantages]
    predictions = [predict_positive(score, threshold, direction) for score in scores]
    return mean(float(pred == label) for pred, label in zip(predictions, labels))


def route_variant(
    score: float,
    threshold: float,
    direction: str,
    structured_variant: str,
    base_variant: str = "coord_latent",
    invert: bool = False,
) -> str:
    choose_structured = predict_positive(score, threshold, direction)
    if invert:
        choose_structured = not choose_structured
    return structured_variant if choose_structured else base_variant


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
