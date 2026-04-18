from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("D:/Experiment")
SYNTHETIC_RESULTS = ROOT / "results" / "structured_hybrid_probe_v1" / "results.json"
SEMIREAL_RESULTS = ROOT / "results" / "semireal_transfer_probe_v1" / "results.json"
OUTPUT_DIR = ROOT / "results" / "crossworld_diagnostic_v1"

TARGET_VARIANTS = [
    "coord_latent",
    "operator_diag_r2_selected",
    "curv_hankel_r4_selected",
    "curvature_field_r4_selected",
    "additive_resid_selected",
]


def load_results(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def coupling_from_world_meta(results: dict, world: str) -> float:
    value = results["world_metadata"][world]["ground_truth_coupling_strength"]
    return float(value if value is not None else 0.0)


def ordered_worlds(results: dict) -> list[str]:
    return sorted(results["worlds"], key=lambda world: coupling_from_world_meta(results, world))


def metric(results: dict, world: str, variant: str, name: str) -> float:
    return float(results["summary"][world][variant][name]["mean"])


def ranks_for_world(results: dict, world: str) -> list[str]:
    values = []
    for variant in TARGET_VARIANTS:
        values.append((metric(results, world, variant, "test_recon_mse"), variant))
    return [variant for _, variant in sorted(values)]


def rank_positions(ordering: list[str]) -> dict[str, int]:
    return {variant: index for index, variant in enumerate(ordering)}


def average_rank_shift(synth: dict, semireal: dict) -> dict[str, float]:
    synth_worlds = ordered_worlds(synth)
    semi_worlds = ordered_worlds(semireal)
    world_pairs = list(zip(synth_worlds[: len(semi_worlds)], semi_worlds))
    shifts: dict[str, list[float]] = {variant: [] for variant in TARGET_VARIANTS}
    for synth_world, semi_world in world_pairs:
        synth_positions = rank_positions(ranks_for_world(synth, synth_world))
        semi_positions = rank_positions(ranks_for_world(semireal, semi_world))
        for variant in TARGET_VARIANTS:
            shifts[variant].append(float(semi_positions[variant] - synth_positions[variant]))
    return {variant: mean(values) for variant, values in shifts.items()}


def best_structured_variant(results: dict, world: str) -> tuple[str, float]:
    structured = [variant for variant in TARGET_VARIANTS if variant != "coord_latent"]
    scored = [(metric(results, world, variant, "test_recon_mse"), variant) for variant in structured]
    value, variant = min(scored)
    return variant, float(value)


def plot_best_structured_gap(synth: dict, semireal: dict, output_path: Path) -> None:
    figure, axis = plt.subplots(1, 1, figsize=(8.6, 4.8))
    for label, results, color in [
        ("synthetic", synth, "#1f77b4"),
        ("semi-real", semireal, "#d62728"),
    ]:
        worlds = ordered_worlds(results)
        couplings = [coupling_from_world_meta(results, world) for world in worlds]
        gaps = []
        for world in worlds:
            coord = metric(results, world, "coord_latent", "test_recon_mse")
            _, best_value = best_structured_variant(results, world)
            gaps.append(coord - best_value)
        axis.plot(couplings, gaps, marker="o", linewidth=2.0, label=label, color=color)
    axis.axhline(0.0, color="#444444", linestyle="--", linewidth=1.2)
    axis.set_xlabel("coupling strength")
    axis.set_ylabel("coord_latent - best structured")
    axis.set_title("Best structured advantage across worlds")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_variant_gap_to_coord(synth: dict, semireal: dict, output_path: Path) -> None:
    variants = [variant for variant in TARGET_VARIANTS if variant != "coord_latent"]
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    for axis, title, results in [
        (axes[0], "Synthetic ladder", synth),
        (axes[1], "Semi-real transfer", semireal),
    ]:
        worlds = ordered_worlds(results)
        couplings = [coupling_from_world_meta(results, world) for world in worlds]
        coord_values = [metric(results, world, "coord_latent", "test_recon_mse") for world in worlds]
        for variant in variants:
            gaps = []
            for index, world in enumerate(worlds):
                gaps.append(coord_values[index] - metric(results, world, variant, "test_recon_mse"))
            axis.plot(couplings, gaps, marker="o", linewidth=2.0, label=variant)
        axis.axhline(0.0, color="#444444", linestyle="--", linewidth=1.2)
        axis.set_title(title)
        axis.set_xlabel("coupling strength")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("coord_latent - variant")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_rank_shift(shift_by_variant: dict[str, float], output_path: Path) -> None:
    variants = [variant for variant in TARGET_VARIANTS if variant != "coord_latent"]
    values = [shift_by_variant[variant] for variant in variants]
    figure, axis = plt.subplots(1, 1, figsize=(8.2, 4.6))
    axis.bar(variants, values, color=["#6baed6", "#fd8d3c", "#74c476", "#9e9ac8"])
    axis.axhline(0.0, color="#444444", linestyle="--", linewidth=1.2)
    axis.set_ylabel("average semi-real rank - synthetic rank")
    axis.set_title("Cross-world rank shift of structured variants")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(alpha=0.2, axis="y")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    synthetic = load_results(SYNTHETIC_RESULTS)
    semireal = load_results(SEMIREAL_RESULTS)

    shift_by_variant = average_rank_shift(synthetic, semireal)
    plot_best_structured_gap(synthetic, semireal, OUTPUT_DIR / "best_structured_gap_vs_coupling.png")
    plot_variant_gap_to_coord(synthetic, semireal, OUTPUT_DIR / "variant_gap_vs_coord.png")
    plot_rank_shift(shift_by_variant, OUTPUT_DIR / "rank_shift.png")

    synthetic_worlds = ordered_worlds(synthetic)
    semireal_worlds = ordered_worlds(semireal)
    paired = list(zip(synthetic_worlds[: len(semireal_worlds)], semireal_worlds))

    lines = [
        "# Cross-World Diagnostic Analysis",
        "",
        "This report compares the same structured variants across the strongest synthetic ladder and the semi-real transfer ladder.",
        "",
        "## Best Structured Gap",
        "",
    ]
    for synth_world, semi_world in paired:
        synth_variant, synth_value = best_structured_variant(synthetic, synth_world)
        semi_variant, semi_value = best_structured_variant(semireal, semi_world)
        synth_coord = metric(synthetic, synth_world, "coord_latent", "test_recon_mse")
        semi_coord = metric(semireal, semi_world, "coord_latent", "test_recon_mse")
        lines.append(
            f"- Synthetic `{synth_world}`: best structured `{synth_variant}` = `{synth_value:.6f}`, "
            f"`coord_latent` = `{synth_coord:.6f}`, gap = `{(synth_coord - synth_value):.6f}`."
        )
        lines.append(
            f"- Semi-real `{semi_world}`: best structured `{semi_variant}` = `{semi_value:.6f}`, "
            f"`coord_latent` = `{semi_coord:.6f}`, gap = `{(semi_coord - semi_value):.6f}`."
        )
    lines.extend(
        [
            "",
            "## Average Rank Shift",
            "",
        ]
    )
    for variant in TARGET_VARIANTS:
        if variant == "coord_latent":
            continue
        lines.append(f"- `{variant}`: average rank shift `{shift_by_variant[variant]:+.2f}` (positive means worse on semi-real).")
    lines.extend(
        [
            "",
            "## Plots",
            "",
            "![Best structured gap](best_structured_gap_vs_coupling.png)",
            "",
            "![Variant gap vs coord](variant_gap_vs_coord.png)",
            "",
            "![Rank shift](rank_shift.png)",
            "",
        ]
    )

    (OUTPUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
