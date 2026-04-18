from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from structured_latent_hypothesis.shared_representation import (
    SharedRepresentationConfig,
    WorldBatch,
    compute_latent_scores,
    encode_world_grid,
    fit_shared_autoencoder,
    prepare_world_images,
)
from structured_latent_hypothesis.synthetic import generate_world, ground_truth_coupling_strength, sample_train_mask


ROOT = Path("D:/Experiment")
SYNTHETIC_RESULTS = ROOT / "results" / "structured_hybrid_probe_v1" / "results.json"
SEMIREAL_RESULTS = ROOT / "results" / "semireal_transfer_probe_v1" / "results.json"
RAW_SCORE_REPORT = ROOT / "results" / "interaction_score_validation_v1" / "report.md"
OUTPUT_DIR = ROOT / "results" / "shared_latent_score_validation_v1"
REPORT_PATH = ROOT / "reports" / "2026-04-18_shared_latent_score_validation_v1_findings.md"

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
    pooled_spearman: float
    synthetic_sign_accuracy: float
    semireal_sign_accuracy: float
    pooled_sign_accuracy: float
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
    available = [variant for variant in TARGET_STRUCTURED if variant in results["summary"][world]]
    best_structured = min(metric(results, world, variant, "test_recon_mse") for variant in available)
    return coord - best_structured


def world_config(results: dict, world: str) -> tuple[int, str, float]:
    for run in results["runs"]:
        if run["config"]["world"] == world:
            config = run["config"]
            return int(config["grid_size"]), str(config["split_strategy"]), float(config["train_fraction"])
    raise ValueError(f"No config found for world {world}.")


def build_world_batches(results: dict, image_size: int) -> list[WorldBatch]:
    batches = []
    for world in ordered_worlds(results):
        grid_size, split_strategy, train_fraction = world_config(results, world)
        train_mask = sample_train_mask(grid_size, train_fraction, seed=0, split_strategy=split_strategy)
        images = prepare_world_images(generate_world(world, grid_size, image_size), image_size=image_size)
        batches.append(WorldBatch(name=world, images=images, train_mask=train_mask))
    return batches


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
            predictions = [score <= threshold for score in scores] if direction == "low_positive" else [score >= threshold for score in scores]
            accuracy = mean(float(pred == label) for pred, label in zip(predictions, labels))
            if accuracy > best_accuracy + 1e-12:
                best_accuracy = accuracy
                best_threshold = threshold
                best_direction = direction
    return best_threshold, best_direction, best_accuracy


def sign_accuracy(scores: list[float], advantages: list[float], threshold: float, direction: str) -> float:
    labels = [value > 0.0 for value in advantages]
    predictions = [score <= threshold for score in scores] if direction == "low_positive" else [score >= threshold for score in scores]
    return mean(float(pred == label) for pred, label in zip(predictions, labels))


def evaluate_score(
    name: str,
    synthetic_worlds: list[str],
    semireal_worlds: list[str],
    score_values: dict[str, dict[str, float]],
    synthetic_advantages: dict[str, float],
    semireal_advantages: dict[str, float],
) -> ScoreEvaluation:
    synthetic_scores = [score_values[world][name] for world in synthetic_worlds]
    semireal_scores = [score_values[world][name] for world in semireal_worlds]
    synthetic_targets = [synthetic_advantages[world] for world in synthetic_worlds]
    semireal_targets = [semireal_advantages[world] for world in semireal_worlds]
    pooled_scores = synthetic_scores + semireal_scores
    pooled_targets = synthetic_targets + semireal_targets
    threshold, direction, synthetic_acc = calibrate_threshold(synthetic_scores, synthetic_targets)
    return ScoreEvaluation(
        name=name,
        synthetic_spearman=spearman(synthetic_scores, synthetic_targets),
        semireal_spearman=spearman(semireal_scores, semireal_targets),
        pooled_spearman=spearman(pooled_scores, pooled_targets),
        synthetic_sign_accuracy=synthetic_acc,
        semireal_sign_accuracy=sign_accuracy(semireal_scores, semireal_targets, threshold, direction),
        pooled_sign_accuracy=sign_accuracy(pooled_scores, pooled_targets, threshold, direction),
        threshold=threshold,
        direction=direction,
    )


def plot_training(history: list[float], output_path: Path) -> None:
    figure, axis = plt.subplots(1, 1, figsize=(7.2, 4.6))
    axis.plot(range(1, len(history) + 1), history, color="#1f77b4", linewidth=1.8)
    axis.set_xlabel("epoch")
    axis.set_ylabel("training loss")
    axis.set_title("Shared Encoder Training Loss")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_score_scatter(
    name: str,
    synthetic_worlds: list[str],
    semireal_worlds: list[str],
    score_values: dict[str, dict[str, float]],
    synthetic_advantages: dict[str, float],
    semireal_advantages: dict[str, float],
    threshold: float,
    direction: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(1, 1, figsize=(7.2, 4.8))
    synthetic_points = [(score_values[world][name], synthetic_advantages[world]) for world in synthetic_worlds]
    semireal_points = [(score_values[world][name], semireal_advantages[world]) for world in semireal_worlds]
    axis.scatter([x for x, _ in synthetic_points], [y for _, y in synthetic_points], label="synthetic", color="#1f77b4", s=64)
    axis.scatter([x for x, _ in semireal_points], [y for _, y in semireal_points], label="semi-real", color="#d62728", s=64)
    axis.axhline(0.0, color="#444444", linestyle="--", linewidth=1.2)
    axis.axvline(threshold, color="#2ca02c", linestyle=":", linewidth=1.4, label=f"threshold ({direction})")
    axis.set_xlabel(name)
    axis.set_ylabel("structured advantage A(w)")
    axis.set_title(f"{name} in shared latent")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_accuracy(evaluations: list[ScoreEvaluation], output_path: Path) -> None:
    labels = [evaluation.name for evaluation in evaluations]
    synthetic_values = [evaluation.synthetic_sign_accuracy for evaluation in evaluations]
    semireal_values = [evaluation.semireal_sign_accuracy for evaluation in evaluations]
    x = np.arange(len(labels))
    width = 0.34
    figure, axis = plt.subplots(1, 1, figsize=(9.2, 4.8))
    axis.bar(x - width / 2, synthetic_values, width=width, label="synthetic calibration")
    axis.bar(x + width / 2, semireal_values, width=width, label="semi-real zero-shot")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("sign accuracy")
    axis.set_title("Shared-Latent Sign Accuracy")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def best_raw_baseline_accuracy() -> float | None:
    if not RAW_SCORE_REPORT.exists():
        return None
    text = RAW_SCORE_REPORT.read_text(encoding="utf-8")
    values = [float(value) for value in re.findall(r"semi-real zero-shot sign accuracy `([0-9.]+)`", text)]
    if not values:
        return None
    return max(values)


def write_summary(evaluations: list[ScoreEvaluation]) -> None:
    lines = [
        "# Shared Latent Score Validation v1",
        "",
        "| score | synthetic Spearman | semi-real Spearman | pooled Spearman | synthetic sign acc | semi-real sign acc | pooled sign acc | threshold | direction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for evaluation in evaluations:
        lines.append(
            f"| {evaluation.name} | {evaluation.synthetic_spearman:+.3f} | {evaluation.semireal_spearman:+.3f} | "
            f"{evaluation.pooled_spearman:+.3f} | {evaluation.synthetic_sign_accuracy:.2f} | {evaluation.semireal_sign_accuracy:.2f} | "
            f"{evaluation.pooled_sign_accuracy:.2f} | "
            f"{evaluation.threshold:.6f} | {evaluation.direction} |"
        )
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    bundle_config: SharedRepresentationConfig,
    best_loss: float,
    evaluations: list[ScoreEvaluation],
    world_scores: dict[str, dict[str, float]],
    recon_mse: dict[str, float],
    synthetic_worlds: list[str],
    semireal_worlds: list[str],
    synthetic_advantages: dict[str, float],
    semireal_advantages: dict[str, float],
) -> None:
    best_eval = max(
        evaluations,
        key=lambda evaluation: (evaluation.pooled_sign_accuracy, evaluation.semireal_sign_accuracy, evaluation.pooled_spearman),
    )
    raw_best = best_raw_baseline_accuracy()
    raw_line = ""
    if raw_best is not None:
        delta = best_eval.semireal_sign_accuracy - raw_best
        raw_line = f"- Relative to raw-world scores, best semi-real zero-shot sign accuracy changed by {delta:+.2f} (raw best `{raw_best:.2f}` -> shared best `{best_eval.semireal_sign_accuracy:.2f}`)."

    report_lines = [
        "# Shared Latent Score Validation v1",
        "",
        "Shared encoder setup:",
        f"- image size `{bundle_config.image_size}`",
        f"- latent dim `{bundle_config.latent_dim}`",
        f"- epochs `{bundle_config.epochs}`",
        f"- batch size `{bundle_config.batch_size}`",
        f"- best train loss `{best_loss:.6f}`",
        "",
        "Best latent score:",
        f"- `{best_eval.name}` with synthetic sign accuracy `{best_eval.synthetic_sign_accuracy:.2f}`, semi-real zero-shot sign accuracy `{best_eval.semireal_sign_accuracy:.2f}`, pooled sign accuracy `{best_eval.pooled_sign_accuracy:.2f}`, synthetic Spearman `{best_eval.synthetic_spearman:+.3f}`, semi-real Spearman `{best_eval.semireal_spearman:+.3f}`, pooled Spearman `{best_eval.pooled_spearman:+.3f}`.",
    ]
    if raw_line:
        report_lines.extend(["", raw_line])

    report_lines.extend(
        [
            "",
            "Per-world shared-latent scores and reconstruction:",
            "| world | coupling | A(w) | shared recon mse | S_add | S_curv | S_diag | S_combo | S_joint |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for world in synthetic_worlds + semireal_worlds:
        target = synthetic_advantages.get(world, semireal_advantages.get(world, 0.0))
        scores = world_scores[world]
        report_lines.append(
            f"| {world} | {float(ground_truth_coupling_strength(world) or 0.0):.2f} | {target:+.6f} | {recon_mse[world]:.6f} | "
            f"{scores['S_add']:.6f} | {scores['S_curv']:.6f} | {scores['S_diag']:.6f} | {scores['S_combo']:.6f} | {scores['S_joint']:.6f} |"
        )

    report_lines.extend(
        [
            "",
            "Plots:",
            "![Training loss](training_loss.png)",
            "![Sign accuracy](sign_accuracy.png)",
        ]
    )
    for evaluation in evaluations:
        report_lines.append(f"![{evaluation.name} scatter]({evaluation.name.lower()}_scatter.png)")

    (OUTPUT_DIR / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def write_findings(evaluations: list[ScoreEvaluation]) -> None:
    best_eval = max(
        evaluations,
        key=lambda evaluation: (evaluation.pooled_sign_accuracy, evaluation.semireal_sign_accuracy, evaluation.pooled_spearman),
    )
    lines = [
        "# Shared-Latent Score Validation v1",
        "",
        "## Result",
        "",
        f"- Best score: `{best_eval.name}`.",
        f"- Synthetic sign accuracy: `{best_eval.synthetic_sign_accuracy:.2f}`.",
        f"- Semi-real zero-shot sign accuracy: `{best_eval.semireal_sign_accuracy:.2f}`.",
        f"- Pooled sign accuracy: `{best_eval.pooled_sign_accuracy:.2f}`.",
        f"- Synthetic Spearman: `{best_eval.synthetic_spearman:+.3f}`.",
        f"- Semi-real Spearman: `{best_eval.semireal_spearman:+.3f}`.",
        f"- Pooled Spearman: `{best_eval.pooled_spearman:+.3f}`.",
        "",
    ]
    if best_eval.semireal_sign_accuracy >= 0.70 and best_eval.semireal_spearman >= 0.50:
        lines.extend(
            [
                "## Interpretation",
                "",
                "Shared latent scoring cleared the transfer bar: the interaction score now looks world-transferable rather than synthetic-only.",
            ]
        )
    elif best_eval.semireal_sign_accuracy >= 0.80 and best_eval.pooled_sign_accuracy >= 0.90:
        lines.extend(
            [
                "## Interpretation",
                "",
                "Shared latent scoring recovered a coarse transfer detector. It separates helpful-vs-unhelpful worlds across domains, but it still does not rank interaction strength inside the semi-real family.",
            ]
        )
    else:
        lines.extend(
            [
                "## Interpretation",
                "",
                "Shared latent scoring did not clear the transfer bar. The score may explain synthetic regime structure inside a common encoder, but it still does not become a reliable cross-world detector on semi-real worlds.",
            ]
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    synthetic_results = load_results(SYNTHETIC_RESULTS)
    semireal_results = load_results(SEMIREAL_RESULTS)
    synthetic_worlds = ordered_worlds(synthetic_results)
    semireal_worlds = ordered_worlds(semireal_results)

    config = SharedRepresentationConfig(
        image_size=24,
        latent_dim=16,
        batch_size=64,
        epochs=140,
        lr=1e-3,
        weight_decay=1e-5,
        lambda_var=1e-3,
        sigma_min=0.25,
        seed=17,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    world_batches = build_world_batches(synthetic_results, image_size=config.image_size)
    world_batches.extend(build_world_batches(semireal_results, image_size=config.image_size))
    bundle = fit_shared_autoencoder(world_batches, config)

    score_values: dict[str, dict[str, float]] = {}
    recon_mse: dict[str, float] = {}
    for batch in world_batches:
        latent_grid, mse = encode_world_grid(bundle, batch)
        score_values[batch.name] = compute_latent_scores(latent_grid)
        recon_mse[batch.name] = mse

    synthetic_advantages = {world: structured_advantage(synthetic_results, world) for world in synthetic_worlds}
    semireal_advantages = {world: structured_advantage(semireal_results, world) for world in semireal_worlds}

    score_names = list(next(iter(score_values.values())).keys())
    evaluations = [
        evaluate_score(
            score_name,
            synthetic_worlds,
            semireal_worlds,
            score_values,
            synthetic_advantages,
            semireal_advantages,
        )
        for score_name in score_names
    ]

    plot_training(bundle.history, OUTPUT_DIR / "training_loss.png")
    plot_accuracy(evaluations, OUTPUT_DIR / "sign_accuracy.png")
    for evaluation in evaluations:
        plot_score_scatter(
            evaluation.name,
            synthetic_worlds,
            semireal_worlds,
            score_values,
            synthetic_advantages,
            semireal_advantages,
            evaluation.threshold,
            evaluation.direction,
            OUTPUT_DIR / f"{evaluation.name.lower()}_scatter.png",
        )

    payload = {
        "config": asdict(config),
        "history": bundle.history,
        "best_loss": bundle.best_loss,
        "synthetic_worlds": synthetic_worlds,
        "semireal_worlds": semireal_worlds,
        "synthetic_advantages": synthetic_advantages,
        "semireal_advantages": semireal_advantages,
        "world_scores": score_values,
        "world_recon_mse": recon_mse,
        "evaluations": [asdict(evaluation) for evaluation in evaluations],
    }
    (OUTPUT_DIR / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary(evaluations)
    write_report(
        bundle_config=config,
        best_loss=bundle.best_loss,
        evaluations=evaluations,
        world_scores=score_values,
        recon_mse=recon_mse,
        synthetic_worlds=synthetic_worlds,
        semireal_worlds=semireal_worlds,
        synthetic_advantages=synthetic_advantages,
        semireal_advantages=semireal_advantages,
    )
    write_findings(evaluations)


if __name__ == "__main__":
    main()
