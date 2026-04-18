from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from structured_latent_hypothesis.synthetic import generate_world, sample_train_mask


ROOT = Path("D:/Experiment")
SYNTHETIC_RESULTS = ROOT / "results" / "structured_hybrid_probe_v1" / "results.json"
SEMIREAL_RESULTS = ROOT / "results" / "semireal_transfer_probe_v1" / "results.json"
OUTPUT_DIR = ROOT / "results" / "interaction_score_validation_v1"

TARGET_STRUCTURED = [
    "operator_diag_r2_selected",
    "curv_hankel_r4_selected",
    "curvature_field_r4_selected",
    "additive_resid_selected",
]


@dataclass
class ScoreEvaluation:
    name: str
    synthetic_spearman: float
    semireal_spearman: float
    synthetic_sign_accuracy: float
    semireal_sign_accuracy: float
    threshold: float
    direction: str


def load_results(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ordered_worlds(results: dict) -> list[str]:
    return sorted(
        results["worlds"],
        key=lambda world: float(results["world_metadata"][world]["ground_truth_coupling_strength"] or 0.0),
    )


def metric(results: dict, world: str, variant: str, name: str) -> float:
    return float(results["summary"][world][variant][name]["mean"])


def structured_advantage(results: dict, world: str) -> float:
    coord = metric(results, world, "coord_latent", "test_recon_mse")
    best_structured = min(metric(results, world, variant, "test_recon_mse") for variant in TARGET_STRUCTURED)
    return coord - best_structured


def world_config(results: dict, world: str) -> tuple[int, int, str, float]:
    for run in results["runs"]:
        if run["config"]["world"] == world:
            config = run["config"]
            return int(config["grid_size"]), int(config["image_size"]), str(config["split_strategy"]), float(config["train_fraction"])
    raise ValueError(f"No config found for world {world}.")


def score_coupling(results: dict, world: str, eps: float = 1e-8) -> float:
    meta = results["world_metadata"][world]
    comm = float(meta["ground_truth_commutator"] or 0.0)
    drift = meta["ground_truth_step_drift"]
    if drift is None:
        return comm
    return comm / (eps + float(drift))


def score_irreducible_structured(results: dict, world: str, eps: float = 1e-8) -> float:
    values = []
    for variant in TARGET_STRUCTURED:
        residual = metric(results, world, variant, "residual_norm_all")
        latent = metric(results, world, variant, "latent_norm_mean")
        values.append(residual / (eps + latent))
    return min(values)


def train_cell_mask(mask: torch.Tensor) -> torch.Tensor:
    return mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]


def score_diagonal_defect(results: dict, world: str, eps: float = 1e-8) -> float:
    grid_size, image_size, split_strategy, train_fraction = world_config(results, world)
    mask = sample_train_mask(grid_size, train_fraction, seed=0, split_strategy=split_strategy)
    world_tensor = generate_world(world, grid_size, image_size).reshape(grid_size, grid_size, -1)
    kappa = world_tensor[1:, 1:] - world_tensor[1:, :-1] - world_tensor[:-1, 1:] + world_tensor[:-1, :-1]
    cell_mask = train_cell_mask(mask)

    if not torch.any(cell_mask):
        return 0.0

    total_energy = float(kappa[cell_mask].pow(2).sum().item())
    if total_energy < eps:
        return 0.0

    diag_energy = 0.0
    for diag in range((grid_size - 1) * 2 - 1):
        entries = []
        for row in range(grid_size - 1):
            col = diag - row
            if col < 0 or col >= grid_size - 1:
                continue
            if not bool(cell_mask[row, col]):
                continue
            entries.append(kappa[row, col])
        if not entries:
            continue
        stack = torch.stack(entries, dim=0)
        diag_mean = stack.mean(dim=0)
        diag_energy += float(stack.shape[0]) * float(diag_mean.pow(2).sum().item())
    return diag_energy / (total_energy + eps)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and sorted_values[end] == sorted_values[index]:
            end += 1
        average_rank = 0.5 * (index + end - 1) + 1.0
        ranks[order[index:end]] = average_rank
        index = end
    return ranks


def spearman(scores: list[float], targets: list[float]) -> float:
    if len(scores) < 2:
        return 0.0
    score_ranks = rankdata(np.asarray(scores, dtype=float))
    target_ranks = rankdata(np.asarray(targets, dtype=float))
    score_centered = score_ranks - score_ranks.mean()
    target_centered = target_ranks - target_ranks.mean()
    denom = math.sqrt(float(np.sum(score_centered**2) * np.sum(target_centered**2)))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(score_centered * target_centered) / denom)


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
            if direction == "low_positive":
                predictions = [score <= threshold for score in scores]
            else:
                predictions = [score >= threshold for score in scores]
            accuracy = mean(float(pred == label) for pred, label in zip(predictions, labels))
            if accuracy > best_accuracy + 1e-12:
                best_accuracy = accuracy
                best_threshold = threshold
                best_direction = direction
    return best_threshold, best_direction, best_accuracy


def sign_accuracy(scores: list[float], advantages: list[float], threshold: float, direction: str) -> float:
    labels = [value > 0.0 for value in advantages]
    if direction == "low_positive":
        predictions = [score <= threshold for score in scores]
    else:
        predictions = [score >= threshold for score in scores]
    return mean(float(pred == label) for pred, label in zip(predictions, labels))


def evaluate_score(
    name: str,
    score_fn: Callable[[dict, str], float],
    synthetic: dict,
    semireal: dict,
) -> tuple[ScoreEvaluation, dict[str, dict[str, float]]]:
    synthetic_worlds = ordered_worlds(synthetic)
    semireal_worlds = ordered_worlds(semireal)

    synthetic_scores = [score_fn(synthetic, world) for world in synthetic_worlds]
    synthetic_advantages = [structured_advantage(synthetic, world) for world in synthetic_worlds]
    semireal_scores = [score_fn(semireal, world) for world in semireal_worlds]
    semireal_advantages = [structured_advantage(semireal, world) for world in semireal_worlds]

    threshold, direction, synthetic_acc = calibrate_threshold(synthetic_scores, synthetic_advantages)
    semireal_acc = sign_accuracy(semireal_scores, semireal_advantages, threshold, direction)

    evaluation = ScoreEvaluation(
        name=name,
        synthetic_spearman=spearman(synthetic_scores, synthetic_advantages),
        semireal_spearman=spearman(semireal_scores, semireal_advantages),
        synthetic_sign_accuracy=synthetic_acc,
        semireal_sign_accuracy=semireal_acc,
        threshold=threshold,
        direction=direction,
    )
    detail = {
        "synthetic": {world: score for world, score in zip(synthetic_worlds, synthetic_scores)},
        "semireal": {world: score for world, score in zip(semireal_worlds, semireal_scores)},
    }
    return evaluation, detail


def plot_score_scatter(
    synthetic: dict,
    semireal: dict,
    detail: dict[str, dict[str, float]],
    output_path: Path,
    title: str,
    threshold: float,
    direction: str,
) -> None:
    figure, axis = plt.subplots(1, 1, figsize=(7.2, 4.8))
    synthetic_points = [(score, structured_advantage(synthetic, world)) for world, score in detail["synthetic"].items()]
    semireal_points = [(score, structured_advantage(semireal, world)) for world, score in detail["semireal"].items()]
    axis.scatter([x for x, _ in synthetic_points], [y for _, y in synthetic_points], label="synthetic", color="#1f77b4", s=64)
    axis.scatter([x for x, _ in semireal_points], [y for _, y in semireal_points], label="semi-real", color="#d62728", s=64)
    axis.axhline(0.0, color="#444444", linestyle="--", linewidth=1.2)
    axis.axvline(threshold, color="#2ca02c", linestyle=":", linewidth=1.4, label=f"threshold ({direction})")
    axis.set_xlabel("score")
    axis.set_ylabel("structured advantage A(w)")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_accuracy(evaluations: list[ScoreEvaluation], output_path: Path) -> None:
    labels = [evaluation.name for evaluation in evaluations]
    synthetic_values = [evaluation.synthetic_sign_accuracy for evaluation in evaluations]
    semireal_values = [evaluation.semireal_sign_accuracy for evaluation in evaluations]
    x = np.arange(len(labels))
    width = 0.34
    figure, axis = plt.subplots(1, 1, figsize=(8.4, 4.8))
    axis.bar(x - width / 2, synthetic_values, width=width, label="synthetic calibration")
    axis.bar(x + width / 2, semireal_values, width=width, label="semi-real zero-shot")
    axis.set_ylim(0.0, 1.05)
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylabel("sign accuracy")
    axis.set_title("Zero-shot sign prediction accuracy")
    axis.grid(alpha=0.2, axis="y")
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    synthetic = load_results(SYNTHETIC_RESULTS)
    semireal = load_results(SEMIREAL_RESULTS)

    score_fns = {
        "S_cpl": score_coupling,
        "S_irr": score_irreducible_structured,
        "S_diag": score_diagonal_defect,
    }

    evaluations: list[ScoreEvaluation] = []
    details: dict[str, dict[str, dict[str, float]]] = {}
    for name, score_fn in score_fns.items():
        evaluation, detail = evaluate_score(name, score_fn, synthetic, semireal)
        evaluations.append(evaluation)
        details[name] = detail
        plot_score_scatter(
            synthetic,
            semireal,
            detail,
            OUTPUT_DIR / f"{name.lower()}_scatter.png",
            f"{name}: score vs structured advantage",
            evaluation.threshold,
            evaluation.direction,
        )

    plot_accuracy(evaluations, OUTPUT_DIR / "sign_accuracy.png")

    lines = [
        "# Interaction Score Validation",
        "",
        "Structured advantage is defined as:",
        "",
        "`A(w) = MSE_coord(w) - min_s MSE_structured_s(w)`",
        "",
        "Positive `A(w)` means the best structured model beats `coord_latent`.",
        "",
        "## Summary",
        "",
    ]
    for evaluation in evaluations:
        lines.append(
            f"- `{evaluation.name}`: synthetic Spearman `{evaluation.synthetic_spearman:+.3f}`, "
            f"semi-real Spearman `{evaluation.semireal_spearman:+.3f}`, "
            f"synthetic sign accuracy `{evaluation.synthetic_sign_accuracy:.2f}`, "
            f"semi-real zero-shot sign accuracy `{evaluation.semireal_sign_accuracy:.2f}`, "
            f"threshold `{evaluation.threshold:.6f}`, direction `{evaluation.direction}`."
        )
    lines.extend(
        [
            "",
            "## Plots",
            "",
            "![Sign accuracy](sign_accuracy.png)",
            "",
            "![S_cpl scatter](s_cpl_scatter.png)",
            "",
            "![S_irr scatter](s_irr_scatter.png)",
            "",
            "![S_diag scatter](s_diag_scatter.png)",
            "",
        ]
    )
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
