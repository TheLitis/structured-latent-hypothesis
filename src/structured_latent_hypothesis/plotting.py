from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cfp_variants(results: dict[str, Any]) -> list[str]:
    variants = []
    for variant in results["variants"]:
        recipe = results["variant_recipes"][variant]
        if recipe.get("lambda_comm", 0.0) > 0.0 and recipe.get("lambda_smooth", 0.0) == 0.0 and recipe.get("lambda_step", 0.0) == 0.0:
            variants.append(variant)
    return sorted(variants, key=lambda name: results["variant_recipes"][name]["lambda_comm"])


def metric_mean(results: dict[str, Any], world: str, variant: str, metric: str) -> float:
    return float(results["summary"][world][variant][metric]["mean"])


def metric_std(results: dict[str, Any], world: str, variant: str, metric: str) -> float:
    return float(results["summary"][world][variant][metric]["std"])


def plot_metric_vs_lambda(results: dict[str, Any], metric: str, title: str, output_path: str | Path) -> None:
    variants = cfp_variants(results)
    lambdas = [results["variant_recipes"][variant]["lambda_comm"] for variant in variants]
    worlds = results["worlds"]

    figure, axes = plt.subplots(1, len(worlds), figsize=(6.5 * len(worlds), 4.5), sharex=True)
    if len(worlds) == 1:
        axes = [axes]

    for axis, world in zip(axes, worlds):
        means = [metric_mean(results, world, variant, metric) for variant in variants]
        stds = [metric_std(results, world, variant, metric) for variant in variants]
        axis.errorbar(lambdas, means, yerr=stds, marker="o", linewidth=2.0, capsize=4, color="#1f77b4", label="CFP sweep")

        baseline = metric_mean(results, world, "baseline", metric)
        smooth = metric_mean(results, world, "smooth", metric) if "smooth" in results["variants"] else None
        axis.axhline(baseline, linestyle="--", linewidth=1.5, color="#444444", label="baseline")
        if smooth is not None:
            axis.axhline(smooth, linestyle=":", linewidth=1.8, color="#b56576", label="smooth")

        axis.set_title(world)
        axis.set_xlabel("lambda_comm")
        axis.set_ylabel(metric)
        axis.grid(alpha=0.25)
        axis.legend()

    figure.suptitle(title)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def average_error_grid(results: dict[str, Any], world: str, variant: str) -> np.ndarray:
    grids = []
    for run in results["runs"]:
        config = run["config"]
        if config["world"] == world and config["variant"] == variant:
            grids.append(np.asarray(run["recon_error_grid"], dtype=float))
    return np.mean(np.stack(grids, axis=0), axis=0)


def train_mask_grid(results: dict[str, Any], world: str, variant: str) -> np.ndarray:
    for run in results["runs"]:
        config = run["config"]
        if config["world"] == world and config["variant"] == variant:
            return np.asarray(run["train_mask"], dtype=float)
    raise ValueError(f"No run found for {world}/{variant}.")


def best_cfp_variant(results: dict[str, Any], world: str) -> str:
    variants = cfp_variants(results)
    return min(variants, key=lambda variant: metric_mean(results, world, variant, "test_recon_mse"))


def plot_error_heatmaps(results: dict[str, Any], world: str, output_path: str | Path) -> str:
    best_variant = best_cfp_variant(results, world)
    variants = ["baseline", "smooth", best_variant]
    mask = train_mask_grid(results, world, "baseline")
    grids = [average_error_grid(results, world, variant) for variant in variants]
    delta = grids[-1] - grids[0]
    vmax = max(float(grid.max()) for grid in grids)
    delta_abs_max = max(abs(float(delta.min())), abs(float(delta.max())))

    figure, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(mask, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("train mask")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    for axis, variant, grid in zip(axes[1:4], variants, grids):
        image = axis.imshow(grid, cmap="magma", vmin=0.0, vmax=vmax)
        axis.set_title(variant)
        axis.set_xticks([])
        axis.set_yticks([])

    delta_image = axes[4].imshow(delta, cmap="coolwarm", vmin=-delta_abs_max, vmax=delta_abs_max)
    axes[4].set_title(f"{best_variant} - baseline")
    axes[4].set_xticks([])
    axes[4].set_yticks([])

    figure.colorbar(image, ax=axes[:4], fraction=0.03, pad=0.04)
    figure.colorbar(delta_image, ax=axes[4], fraction=0.046, pad=0.04)
    figure.suptitle(f"Reconstruction error heatmaps: {world}")
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return best_variant


def plot_pareto(results: dict[str, Any], output_path: str | Path) -> None:
    worlds = results["worlds"]
    figure, axes = plt.subplots(1, len(worlds), figsize=(6 * len(worlds), 4.5), sharex=False, sharey=False)
    if len(worlds) == 1:
        axes = [axes]

    for axis, world in zip(axes, worlds):
        for variant in results["variants"]:
            xs = []
            ys = []
            for run in results["runs"]:
                config = run["config"]
                if config["world"] == world and config["variant"] == variant:
                    xs.append(float(run["test_recon_mse"]))
                    ys.append(float(run["comm_error_holdout_cells"]))
            axis.scatter(xs, ys, label=variant, alpha=0.8)
        axis.set_title(world)
        axis.set_xlabel("holdout test recon mse")
        axis.set_ylabel("holdout-cell comm error")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    figure.suptitle("Pareto view: held-out quality vs held-out flatness")
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_train_vs_holdout_comm(results: dict[str, Any], output_path: str | Path) -> None:
    worlds = results["worlds"]
    figure, axes = plt.subplots(1, len(worlds), figsize=(6.5 * len(worlds), 4.5), sharey=True)
    if len(worlds) == 1:
        axes = [axes]

    for axis, world in zip(axes, worlds):
        variants = results["variants"]
        x_positions = np.arange(len(variants))
        train_values = [metric_mean(results, world, variant, "comm_error_train_cells") for variant in variants]
        holdout_values = [metric_mean(results, world, variant, "comm_error_holdout_cells") for variant in variants]

        axis.plot(x_positions, train_values, marker="o", linewidth=2.0, color="#1f77b4", label="train cells")
        axis.plot(x_positions, holdout_values, marker="s", linewidth=2.0, color="#d62728", label="holdout cells")
        axis.set_title(world)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(variants, rotation=30, ha="right")
        axis.set_ylabel("comm error")
        axis.grid(alpha=0.25)
        axis.legend()

    figure.suptitle("Train vs holdout cell commutativity error")
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_gain_vs_commutator(results: dict[str, Any], output_path: str | Path) -> None:
    worlds = results["worlds"]
    variants = [variant for variant in results["variants"] if variant != "baseline"]

    x_values = [results["world_metadata"][world]["ground_truth_commutator"] for world in worlds]
    figure, axis = plt.subplots(1, 1, figsize=(7, 4.5))

    for variant in variants:
        gains = []
        for world in worlds:
            baseline = metric_mean(results, world, "baseline", "test_recon_mse")
            variant_value = metric_mean(results, world, variant, "test_recon_mse")
            gains.append(baseline - variant_value)
        axis.plot(x_values, gains, marker="o", linewidth=2.0, label=variant)

    axis.axhline(0.0, color="#444444", linestyle="--", linewidth=1.2)
    axis.set_xlabel("ground-truth commutator magnitude")
    axis.set_ylabel("baseline - variant test recon mse")
    axis.set_title("CFP gain vs ground-truth commutator magnitude")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
