from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from structured_latent_hypothesis.routing import aggregate_metric_records, family_from_world, mean_std
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
DEFAULT_OUTPUT_DIR = ROOT / "results" / "shared_latent_score_stability_v1"
DEFAULT_REPORT_PATH = ROOT / "reports" / "2026-04-18_shared_latent_score_stability_v1_findings.md"

TARGET_STRUCTURED = [
    "operator_diag_r2_selected",
    "curv_hankel_r4_selected",
    "curvature_field_r4_selected",
    "additive_resid_selected",
]
MODES = ("shared_train", "synthetic_only_train")
SCORE_NAMES = ("S_add", "S_curv", "S_diag", "S_combo", "S_joint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--image-size", type=int, default=24)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-var", type=float, default=1e-3)
    parser.add_argument("--sigma-min", type=float, default=0.25)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seeds", type=str, default="17,29,43,71,101")
    return parser.parse_args()


def parse_seeds(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


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
    thresholds = [0.0] if not unique_scores else [unique_scores[0] - 1e-6]
    if unique_scores:
        for left, right in zip(unique_scores[:-1], unique_scores[1:]):
            thresholds.append(0.5 * (left + right))
        thresholds.append(unique_scores[-1] + 1e-6)
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


def evaluate_scores(
    score_values: dict[str, dict[str, float]],
    synthetic_worlds: list[str],
    semireal_worlds: list[str],
    synthetic_advantages: dict[str, float],
    semireal_advantages: dict[str, float],
) -> dict[str, dict[str, float | str]]:
    evaluations: dict[str, dict[str, float | str]] = {}
    for score_name in SCORE_NAMES:
        synthetic_scores = [score_values[world][score_name] for world in synthetic_worlds]
        semireal_scores = [score_values[world][score_name] for world in semireal_worlds]
        synthetic_targets = [synthetic_advantages[world] for world in synthetic_worlds]
        semireal_targets = [semireal_advantages[world] for world in semireal_worlds]
        pooled_scores = synthetic_scores + semireal_scores
        pooled_targets = synthetic_targets + semireal_targets
        threshold, direction, synthetic_acc = calibrate_threshold(synthetic_scores, synthetic_targets)
        evaluations[score_name] = {
            "threshold": threshold,
            "direction": direction,
            "synthetic_sign_accuracy": synthetic_acc,
            "semireal_sign_accuracy": sign_accuracy(semireal_scores, semireal_targets, threshold, direction),
            "pooled_sign_accuracy": sign_accuracy(pooled_scores, pooled_targets, threshold, direction),
            "synthetic_spearman": spearman(synthetic_scores, synthetic_targets),
            "semireal_spearman": spearman(semireal_scores, semireal_targets),
            "pooled_spearman": spearman(pooled_scores, pooled_targets),
        }
    return evaluations


def aggregate_evaluation_records(records: list[dict[str, float | str]]) -> dict[str, dict | str]:
    numeric_keys = [key for key, value in records[0].items() if not isinstance(value, str)]
    aggregated = {
        key: aggregate_metric_records([{key: record[key]} for record in records])[key]
        for key in numeric_keys
    }
    if "direction" in records[0]:
        counts: dict[str, int] = {}
        for record in records:
            counts[str(record["direction"])] = counts.get(str(record["direction"]), 0) + 1
        aggregated["direction_mode"] = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
        aggregated["direction_counts"] = counts
    return aggregated


def run_mode(
    mode: str,
    seeds: list[int],
    config_template: SharedRepresentationConfig,
    synthetic_batches: list[WorldBatch],
    semireal_batches: list[WorldBatch],
    synthetic_advantages: dict[str, float],
    semireal_advantages: dict[str, float],
) -> dict:
    if mode not in MODES:
        raise ValueError(f"Unsupported mode: {mode}")
    train_batches = synthetic_batches + semireal_batches if mode == "shared_train" else synthetic_batches
    eval_batches = synthetic_batches + semireal_batches
    synthetic_worlds = [batch.name for batch in synthetic_batches]
    semireal_worlds = [batch.name for batch in semireal_batches]

    seed_runs = []
    for seed in seeds:
        bundle = fit_shared_autoencoder(train_batches, SharedRepresentationConfig(**{**asdict(config_template), "seed": seed}))
        world_scores: dict[str, dict[str, float]] = {}
        recon_mse: dict[str, float] = {}
        for batch in eval_batches:
            latent_grid, mse = encode_world_grid(bundle, batch)
            world_scores[batch.name] = compute_latent_scores(latent_grid)
            recon_mse[batch.name] = mse
        evaluations = evaluate_scores(world_scores, synthetic_worlds, semireal_worlds, synthetic_advantages, semireal_advantages)
        seed_runs.append(
            {
                "seed": seed,
                "best_loss": bundle.best_loss,
                "history": bundle.history,
                "world_scores": world_scores,
                "world_recon_mse": recon_mse,
                "evaluations": evaluations,
            }
        )

    per_world_score_stats: dict[str, dict[str, dict[str, float]]] = {}
    per_world_recon_stats: dict[str, dict[str, float]] = {}
    for world in synthetic_worlds + semireal_worlds:
        per_world_score_stats[world] = {
            score_name: mean_std([run["world_scores"][world][score_name] for run in seed_runs])
            for score_name in SCORE_NAMES
        }
        per_world_recon_stats[world] = mean_std([run["world_recon_mse"][world] for run in seed_runs])

    evaluation_stats = {
        score_name: aggregate_evaluation_records([run["evaluations"][score_name] for run in seed_runs])
        for score_name in SCORE_NAMES
    }
    return {
        "mode": mode,
        "seed_runs": seed_runs,
        "per_world_score_stats": per_world_score_stats,
        "per_world_recon_stats": per_world_recon_stats,
        "evaluation_stats": evaluation_stats,
    }


def plot_s_joint_by_world(mode_runs: dict[str, dict], ordered_world_list: list[str], output_path: Path) -> None:
    figure, axis = plt.subplots(1, 1, figsize=(10.5, 4.8))
    for mode, color in (("shared_train", "#1f77b4"), ("synthetic_only_train", "#d62728")):
        means = [mode_runs[mode]["per_world_score_stats"][world]["S_joint"]["mean"] for world in ordered_world_list]
        stds = [mode_runs[mode]["per_world_score_stats"][world]["S_joint"]["std"] for world in ordered_world_list]
        axis.plot(range(len(ordered_world_list)), means, marker="o", linewidth=1.6, label=mode, color=color)
        axis.fill_between(
            range(len(ordered_world_list)),
            np.asarray(means) - np.asarray(stds),
            np.asarray(means) + np.asarray(stds),
            alpha=0.15,
            color=color,
        )
    axis.set_xticks(range(len(ordered_world_list)))
    axis.set_xticklabels(ordered_world_list, rotation=26, ha="right")
    axis.set_ylabel("S_joint")
    axis.set_title("S_joint Across Worlds")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_s_joint_family(mode_runs: dict[str, dict], synthetic_worlds: list[str], semireal_worlds: list[str], output_path: Path) -> None:
    figure, axis = plt.subplots(1, 1, figsize=(7.8, 4.8))
    for mode, color in (("shared_train", "#1f77b4"), ("synthetic_only_train", "#d62728")):
        synthetic_values = [mode_runs[mode]["per_world_score_stats"][world]["S_joint"]["mean"] for world in synthetic_worlds]
        semireal_values = [mode_runs[mode]["per_world_score_stats"][world]["S_joint"]["mean"] for world in semireal_worlds]
        axis.plot(
            [ground_truth_coupling_strength(world) or 0.0 for world in synthetic_worlds],
            synthetic_values,
            marker="o",
            linewidth=1.8,
            label=f"{mode} synthetic",
            color=color,
        )
        axis.plot(
            [ground_truth_coupling_strength(world) or 0.0 for world in semireal_worlds],
            semireal_values,
            marker="s",
            linewidth=1.5,
            linestyle="--",
            label=f"{mode} semi-real",
            color=color,
        )
    axis.set_xlabel("coupling strength")
    axis.set_ylabel("S_joint")
    axis.set_title("S_joint by Family and Training Mode")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_summary(output_dir: Path, mode_runs: dict[str, dict]) -> None:
    lines = [
        "# Shared Latent Score Stability v1",
        "",
        "| mode | score | synthetic sign acc | semi-real sign acc | pooled sign acc | synthetic Spearman | semi-real Spearman | pooled Spearman |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in MODES:
        for score_name in SCORE_NAMES:
            stats = mode_runs[mode]["evaluation_stats"][score_name]
            lines.append(
                f"| {mode} | {score_name} | {stats['synthetic_sign_accuracy']['mean']:.2f}+/-{stats['synthetic_sign_accuracy']['std']:.2f} | "
                f"{stats['semireal_sign_accuracy']['mean']:.2f}+/-{stats['semireal_sign_accuracy']['std']:.2f} | "
                f"{stats['pooled_sign_accuracy']['mean']:.2f}+/-{stats['pooled_sign_accuracy']['std']:.2f} | "
                f"{stats['synthetic_spearman']['mean']:+.3f}+/-{stats['synthetic_spearman']['std']:.3f} | "
                f"{stats['semireal_spearman']['mean']:+.3f}+/-{stats['semireal_spearman']['std']:.3f} | "
                f"{stats['pooled_spearman']['mean']:+.3f}+/-{stats['pooled_spearman']['std']:.3f} |"
            )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    output_dir: Path,
    report_path: Path,
    mode_runs: dict[str, dict],
    ordered_world_list: list[str],
    config: SharedRepresentationConfig,
    seeds: list[int],
) -> None:
    shared_joint = mode_runs["shared_train"]["evaluation_stats"]["S_joint"]
    synthetic_only_joint = mode_runs["synthetic_only_train"]["evaluation_stats"]["S_joint"]
    lines = [
        "# Shared Latent Score Stability v1",
        "",
        "Configuration:",
        f"- seeds `{', '.join(str(seed) for seed in seeds)}`",
        f"- image size `{config.image_size}`",
        f"- latent dim `{config.latent_dim}`",
        f"- epochs `{config.epochs}`",
        f"- batch size `{config.batch_size}`",
        "",
        "Main comparison on `S_joint`:",
        f"- `shared_train` semi-real sign accuracy `{shared_joint['semireal_sign_accuracy']['mean']:.2f}+/-{shared_joint['semireal_sign_accuracy']['std']:.2f}`, pooled sign accuracy `{shared_joint['pooled_sign_accuracy']['mean']:.2f}+/-{shared_joint['pooled_sign_accuracy']['std']:.2f}`.",
        f"- `synthetic_only_train` semi-real sign accuracy `{synthetic_only_joint['semireal_sign_accuracy']['mean']:.2f}+/-{synthetic_only_joint['semireal_sign_accuracy']['std']:.2f}`, pooled sign accuracy `{synthetic_only_joint['pooled_sign_accuracy']['mean']:.2f}+/-{synthetic_only_joint['pooled_sign_accuracy']['std']:.2f}`.",
        "",
        "Plots:",
        "![S_joint across worlds](s_joint_by_world.png)",
        "![S_joint by family](s_joint_by_family.png)",
        "",
        "Per-world `S_joint` mean/std:",
        "| world | family | coupling | shared_train | synthetic_only_train | shared recon mse | synthetic-only recon mse |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for world in ordered_world_list:
        lines.append(
            f"| {world} | {family_from_world(world)} | {float(ground_truth_coupling_strength(world) or 0.0):.2f} | "
            f"{mode_runs['shared_train']['per_world_score_stats'][world]['S_joint']['mean']:.6f}+/-{mode_runs['shared_train']['per_world_score_stats'][world]['S_joint']['std']:.6f} | "
            f"{mode_runs['synthetic_only_train']['per_world_score_stats'][world]['S_joint']['mean']:.6f}+/-{mode_runs['synthetic_only_train']['per_world_score_stats'][world]['S_joint']['std']:.6f} | "
            f"{mode_runs['shared_train']['per_world_recon_stats'][world]['mean']:.6f} | "
            f"{mode_runs['synthetic_only_train']['per_world_recon_stats'][world]['mean']:.6f} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    interpretation = (
        "Shared training retained the stronger cross-world detector."
        if shared_joint["semireal_sign_accuracy"]["mean"] >= synthetic_only_joint["semireal_sign_accuracy"]["mean"]
        else "Synthetic-only training did not underperform on semi-real, so the current transfer claim is weaker than expected."
    )
    findings_lines = [
        "# Shared-Latent Score Stability v1",
        "",
        "## Result",
        "",
        f"- `shared_train` semi-real `S_joint` sign accuracy: `{shared_joint['semireal_sign_accuracy']['mean']:.2f}+/-{shared_joint['semireal_sign_accuracy']['std']:.2f}`.",
        f"- `synthetic_only_train` semi-real `S_joint` sign accuracy: `{synthetic_only_joint['semireal_sign_accuracy']['mean']:.2f}+/-{synthetic_only_joint['semireal_sign_accuracy']['std']:.2f}`.",
        f"- `shared_train` pooled `S_joint` sign accuracy: `{shared_joint['pooled_sign_accuracy']['mean']:.2f}+/-{shared_joint['pooled_sign_accuracy']['std']:.2f}`.",
        f"- `synthetic_only_train` pooled `S_joint` sign accuracy: `{synthetic_only_joint['pooled_sign_accuracy']['mean']:.2f}+/-{synthetic_only_joint['pooled_sign_accuracy']['std']:.2f}`.",
        "",
        "## Interpretation",
        "",
        interpretation,
    ]
    report_path.write_text("\n".join(findings_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    output_dir: Path = args.output_dir
    report_path: Path = args.report_path
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    synthetic_results = load_results(SYNTHETIC_RESULTS)
    semireal_results = load_results(SEMIREAL_RESULTS)
    synthetic_worlds = ordered_worlds(synthetic_results)
    semireal_worlds = ordered_worlds(semireal_results)
    ordered_world_list = synthetic_worlds + semireal_worlds
    synthetic_advantages = {world: structured_advantage(synthetic_results, world) for world in synthetic_worlds}
    semireal_advantages = {world: structured_advantage(semireal_results, world) for world in semireal_worlds}

    config = SharedRepresentationConfig(
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_var=args.lambda_var,
        sigma_min=args.sigma_min,
        seed=seeds[0],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    synthetic_batches = build_world_batches(synthetic_results, image_size=config.image_size)
    semireal_batches = build_world_batches(semireal_results, image_size=config.image_size)

    mode_runs = {
        mode: run_mode(
            mode,
            seeds,
            config,
            synthetic_batches,
            semireal_batches,
            synthetic_advantages,
            semireal_advantages,
        )
        for mode in MODES
    }

    plot_s_joint_by_world(mode_runs, ordered_world_list, output_dir / "s_joint_by_world.png")
    plot_s_joint_family(mode_runs, synthetic_worlds, semireal_worlds, output_dir / "s_joint_by_family.png")

    payload = {
        "config": asdict(config),
        "seeds": seeds,
        "synthetic_worlds": synthetic_worlds,
        "semireal_worlds": semireal_worlds,
        "modes": mode_runs,
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary(output_dir, mode_runs)
    write_report(output_dir, report_path, mode_runs, ordered_world_list, config, seeds)


if __name__ == "__main__":
    main()
