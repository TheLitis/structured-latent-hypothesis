from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from structured_latent_hypothesis.routing import (
    aggregate_metric_records,
    choose_abstain_band,
    choose_best_directional_threshold,
    coverage_rate,
    decisions_to_predictions,
    family_from_world,
    filtered_sign_accuracy,
    metric,
    random_router_decisions,
    route_decision_label,
    structured_route_rate,
)
from structured_latent_hypothesis.synthetic import ground_truth_coupling_strength


ROOT = Path("D:/Experiment")
STABILITY_RESULTS = ROOT / "results" / "shared_latent_score_stability_v1" / "results.json"
SYNTHETIC_RESULTS = ROOT / "results" / "structured_hybrid_probe_v1" / "results.json"
SEMIREAL_RESULTS = ROOT / "results" / "semireal_transfer_probe_v1" / "results.json"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "regime_gated_routing_v2"
DEFAULT_REPORT_PATH = ROOT / "reports" / "2026-04-18_regime_gated_routing_v2_findings.md"

BASE_VARIANT = "coord_latent"
PAIR_VARIANTS = {
    "pairA_curv_hankel": "curv_hankel_r4_selected",
    "pairB_operator_diag": "operator_diag_r2_selected",
}
SPLITS = ("synthetic", "semi-real", "pooled")
CONTROL_NAMES = (
    "always_coord",
    "always_structured",
    "inverted_router",
    "metadata_control_router",
    "random_router",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stability-results", type=Path, default=STABILITY_RESULTS)
    parser.add_argument("--synthetic-results", type=Path, default=SYNTHETIC_RESULTS)
    parser.add_argument("--semireal-results", type=Path, default=SEMIREAL_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--score-name", type=str, default="S_joint")
    parser.add_argument("--random-trials", type=int, default=512)
    return parser.parse_args()


def load_results(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ordered_worlds(results: dict) -> list[str]:
    return sorted(
        results["worlds"],
        key=lambda world: float(results["world_metadata"][world]["ground_truth_coupling_strength"] or 0.0),
    )


def metadata_s_cpl(results: dict, world: str, eps: float = 1e-8) -> float:
    meta = results["world_metadata"][world]
    commutator = float(meta["ground_truth_commutator"] or 0.0)
    step_drift = meta["ground_truth_step_drift"]
    if step_drift is None:
        return commutator
    return commutator / (eps + float(step_drift))


def pair_delta(results: dict, world: str, structured_variant: str) -> float:
    return metric(results, world, BASE_VARIANT) - metric(results, world, structured_variant)


def pair_label(results: dict, world: str, structured_variant: str) -> bool:
    return pair_delta(results, world, structured_variant) > 0.0


def evaluate_decisions(
    worlds: list[str],
    decisions: dict[str, str],
    results_by_world: dict[str, dict],
    structured_variant: str,
    filtered_delta_floor: float = 1e-5,
) -> dict[str, float | None]:
    labels = [pair_label(results_by_world[world], world, structured_variant) for world in worlds]
    predictions = decisions_to_predictions([decisions[world] for world in worlds])
    coord_values = [metric(results_by_world[world], world, BASE_VARIANT) for world in worlds]
    structured_values = [metric(results_by_world[world], world, structured_variant) for world in worlds]
    routed_values = [
        structured if decisions[world] == "structured" else coord
        for world, coord, structured in zip(worlds, coord_values, structured_values)
    ]
    oracle_values = [min(coord, structured) for coord, structured in zip(coord_values, structured_values)]
    distances = [abs(pair_delta(results_by_world[world], world, structured_variant)) for world in worlds]
    positives = [index for index, label in enumerate(labels) if label]
    negatives = [index for index, label in enumerate(labels) if not label]
    if positives and negatives:
        tpr = float(sum(predictions[index] for index in positives)) / float(len(positives))
        tnr = float(sum(not predictions[index] for index in negatives)) / float(len(negatives))
        balanced = 0.5 * (tpr + tnr)
    else:
        balanced = float(sum(pred == label for pred, label in zip(predictions, labels))) / float(len(labels))
    possible_gain = sum(max(0.0, coord - oracle) for coord, oracle in zip(coord_values, oracle_values))
    realized_gain = sum(coord - routed for coord, routed in zip(coord_values, routed_values))
    return {
        "route_sign_accuracy": float(sum(pred == label for pred, label in zip(predictions, labels))) / float(len(labels)),
        "route_balanced_accuracy": balanced,
        "margin_filtered_sign_accuracy": filtered_sign_accuracy(labels, predictions, distances, minimum_distance=filtered_delta_floor),
        "mean_routed_mse": float(sum(routed_values)) / float(len(routed_values)),
        "gain_over_coord": float(sum(coord - routed for coord, routed in zip(coord_values, routed_values))) / float(len(routed_values)),
        "gain_over_always_structured": float(sum(structured - routed for structured, routed in zip(structured_values, routed_values))) / float(len(routed_values)),
        "regret_to_pair_oracle": float(sum(routed - oracle for routed, oracle in zip(routed_values, oracle_values))) / float(len(routed_values)),
        "capture_ratio": (realized_gain / possible_gain) if possible_gain > 1e-12 else 0.0,
        "structured_route_rate": structured_route_rate([decisions[world] for world in worlds]),
        "coverage": coverage_rate([decisions[world] for world in worlds]),
        "abstain_rate": 1.0 - coverage_rate([decisions[world] for world in worlds]),
    }


def family_conditioned_metrics(
    worlds: list[str],
    decisions: dict[str, str],
    results_by_world: dict[str, dict],
    structured_variant: str,
) -> dict[str, dict[str, float | None]]:
    families: dict[str, list[str]] = {}
    for world in worlds:
        families.setdefault(family_from_world(world), []).append(world)
    return {
        family: evaluate_decisions(group_worlds, decisions, results_by_world, structured_variant)
        for family, group_worlds in families.items()
    }


def per_world_row(
    world: str,
    score: float,
    threshold: float,
    direction: str,
    abstain_band: float,
    decision: str,
    results_by_world: dict[str, dict],
    structured_variant: str,
) -> dict[str, float | str]:
    coord = metric(results_by_world[world], world, BASE_VARIANT)
    structured = metric(results_by_world[world], world, structured_variant)
    routed = structured if decision == "structured" else coord
    oracle = min(coord, structured)
    return {
        "world": world,
        "family": family_from_world(world),
        "coupling": float(ground_truth_coupling_strength(world) or 0.0),
        "score": score,
        "threshold": threshold,
        "direction": direction,
        "abstain_band": abstain_band,
        "score_margin": abs(score - threshold),
        "delta_pair": coord - structured,
        "distance_to_zero": abs(coord - structured),
        "route": decision,
        "coord_mse": coord,
        "structured_mse": structured,
        "routed_mse": routed,
        "oracle_mse": oracle,
        "regret": routed - oracle,
    }


def random_control_metrics(
    worlds: list[str],
    structured_rate: float,
    results_by_world: dict[str, dict],
    structured_variant: str,
    trials: int,
    seed_prefix: int,
) -> dict[str, float | None]:
    runs = []
    for trial_index in range(trials):
        sampled = random_router_decisions(
            worlds,
            structured_rate=structured_rate,
            seed=seed_prefix * 10000 + trial_index,
        )
        runs.append(evaluate_decisions(worlds, sampled, results_by_world, structured_variant))
    metrics: dict[str, float | None] = {}
    for key in runs[0]:
        values = [run[key] for run in runs if run[key] is not None]
        if not values:
            metrics[key] = None
            metrics[f"{key}_trial_std"] = None
            continue
        numeric = [float(value) for value in values]
        metrics[key] = float(np.mean(numeric))
        metrics[f"{key}_trial_std"] = float(np.std(numeric))
    return metrics


def aggregate_world_rows(seed_rows: list[list[dict]]) -> list[dict]:
    by_world: dict[str, list[dict]] = {}
    for rows in seed_rows:
        for row in rows:
            by_world.setdefault(str(row["world"]), []).append(row)
    aggregated = []
    for world, rows in by_world.items():
        structured_count = sum(1 for row in rows if row["route"] == "structured")
        coord_count = sum(1 for row in rows if row["route"] == "coord")
        abstain_count = sum(1 for row in rows if row["route"] == "abstain")
        if abstain_count > 0:
            route_label = f"mixed structured {structured_count}/{len(rows)}, abstain {abstain_count}/{len(rows)}"
        elif structured_count == len(rows):
            route_label = f"structured {structured_count}/{len(rows)}"
        elif coord_count == len(rows):
            route_label = f"coord {coord_count}/{len(rows)}"
        else:
            route_label = f"mixed structured {structured_count}/{len(rows)}"
        aggregated.append(
            {
                "world": world,
                "family": rows[0]["family"],
                "coupling": rows[0]["coupling"],
                "score_mean": float(np.mean([row["score"] for row in rows])),
                "score_std": float(np.std([row["score"] for row in rows])),
                "threshold_mean": float(np.mean([row["threshold"] for row in rows])),
                "threshold_std": float(np.std([row["threshold"] for row in rows])),
                "abstain_band_mean": float(np.mean([row["abstain_band"] for row in rows])),
                "abstain_band_std": float(np.std([row["abstain_band"] for row in rows])),
                "score_margin_mean": float(np.mean([row["score_margin"] for row in rows])),
                "score_margin_std": float(np.std([row["score_margin"] for row in rows])),
                "delta_pair": rows[0]["delta_pair"],
                "distance_to_zero": rows[0]["distance_to_zero"],
                "route": route_label,
                "coord_mse": rows[0]["coord_mse"],
                "structured_mse": rows[0]["structured_mse"],
                "routed_mse_mean": float(np.mean([row["routed_mse"] for row in rows])),
                "routed_mse_std": float(np.std([row["routed_mse"] for row in rows])),
                "oracle_mse": rows[0]["oracle_mse"],
                "regret_mean": float(np.mean([row["regret"] for row in rows])),
                "regret_std": float(np.std([row["regret"] for row in rows])),
            }
        )
    return sorted(aggregated, key=lambda row: (row["family"], row["coupling"]))


def aggregate_controls(seed_runs: list[dict]) -> dict:
    controls = {}
    for control_name in seed_runs[0]["control_rows"].keys():
        controls[control_name] = {
            split: aggregate_metric_records([run["control_rows"][control_name][split] for run in seed_runs])
            for split in SPLITS
        }
    return controls


def aggregate_seed_runs(seed_runs: list[dict]) -> dict:
    families = sorted({family for run in seed_runs for family in run["family_conditioned_metrics"].keys()})
    return {
        "threshold": aggregate_metric_records([{"threshold": run["threshold"]} for run in seed_runs])["threshold"],
        "abstain_band": aggregate_metric_records([{"abstain_band": run["abstain_band"]} for run in seed_runs])["abstain_band"],
        "synthetic_metrics": aggregate_metric_records([run["synthetic_metrics"] for run in seed_runs]),
        "semireal_metrics": aggregate_metric_records([run["semireal_metrics"] for run in seed_runs]),
        "pooled_metrics": aggregate_metric_records([run["pooled_metrics"] for run in seed_runs]),
        "family_conditioned_metrics": {
            family: aggregate_metric_records([run["family_conditioned_metrics"][family] for run in seed_runs])
            for family in families
        },
        "control_rows": aggregate_controls(seed_runs),
        "per_world_rows": aggregate_world_rows([run["per_world_rows"] for run in seed_runs]),
    }


def build_seed_run(
    encoder_mode: str,
    seed_run: dict,
    synthetic_worlds: list[str],
    semireal_worlds: list[str],
    results_by_world: dict[str, dict],
    structured_variant: str,
    score_name: str,
    random_trials: int,
) -> dict:
    pooled_worlds = synthetic_worlds + semireal_worlds
    score_values = {world: float(seed_run["world_scores"][world][score_name]) for world in pooled_worlds}
    control_scores = {world: metadata_s_cpl(results_by_world[world], world) for world in pooled_worlds}

    labels_synth = [pair_label(results_by_world[world], world, structured_variant) for world in synthetic_worlds]
    coord_synth = [metric(results_by_world[world], world, BASE_VARIANT) for world in synthetic_worlds]
    structured_synth = [metric(results_by_world[world], world, structured_variant) for world in synthetic_worlds]
    seed = int(seed_run["seed"])

    threshold, direction, _, _, _, _ = choose_best_directional_threshold(
        [score_values[world] for world in synthetic_worlds],
        labels_synth,
        coord_synth,
        structured_synth,
    )
    abstain_band, _, _, _, _ = choose_abstain_band(
        [score_values[world] for world in synthetic_worlds],
        labels_synth,
        coord_synth,
        structured_synth,
        threshold=threshold,
        direction=direction,
    )

    control_threshold, control_direction, _, _, _, _ = choose_best_directional_threshold(
        [control_scores[world] for world in synthetic_worlds],
        labels_synth,
        coord_synth,
        structured_synth,
    )
    control_band, _, _, _, _ = choose_abstain_band(
        [control_scores[world] for world in synthetic_worlds],
        labels_synth,
        coord_synth,
        structured_synth,
        threshold=control_threshold,
        direction=control_direction,
    )

    learned_decisions = {
        world: route_decision_label(score_values[world], threshold, direction, abstain_band=abstain_band)
        for world in pooled_worlds
    }
    inverted_decisions = {
        world: route_decision_label(score_values[world], threshold, direction, abstain_band=abstain_band, invert=True)
        for world in pooled_worlds
    }
    control_decisions = {
        world: route_decision_label(control_scores[world], control_threshold, control_direction, abstain_band=control_band)
        for world in pooled_worlds
    }
    always_coord = {world: "coord" for world in pooled_worlds}
    always_structured = {world: "structured" for world in pooled_worlds}

    learned_rate_pooled = structured_route_rate([learned_decisions[world] for world in pooled_worlds])
    learned_rate_synthetic = structured_route_rate([learned_decisions[world] for world in synthetic_worlds])
    learned_rate_semireal = structured_route_rate([learned_decisions[world] for world in semireal_worlds])

    return {
        "encoder_mode": encoder_mode,
        "seed": seed,
        "score_name": score_name,
        "threshold": threshold,
        "direction": direction,
        "abstain_band": abstain_band,
        "coverage": coverage_rate([learned_decisions[world] for world in pooled_worlds]),
        "structured_route_rate": learned_rate_pooled,
        "synthetic_metrics": evaluate_decisions(synthetic_worlds, learned_decisions, results_by_world, structured_variant),
        "semireal_metrics": evaluate_decisions(semireal_worlds, learned_decisions, results_by_world, structured_variant),
        "pooled_metrics": evaluate_decisions(pooled_worlds, learned_decisions, results_by_world, structured_variant),
        "family_conditioned_metrics": family_conditioned_metrics(pooled_worlds, learned_decisions, results_by_world, structured_variant),
        "per_world_rows": [
            per_world_row(
                world,
                score_values[world],
                threshold,
                direction,
                abstain_band,
                learned_decisions[world],
                results_by_world,
                structured_variant,
            )
            for world in pooled_worlds
        ],
        "control_rows": {
            "inverted_router": {
                "synthetic": evaluate_decisions(synthetic_worlds, inverted_decisions, results_by_world, structured_variant),
                "semi-real": evaluate_decisions(semireal_worlds, inverted_decisions, results_by_world, structured_variant),
                "pooled": evaluate_decisions(pooled_worlds, inverted_decisions, results_by_world, structured_variant),
            },
            "metadata_control_router": {
                "synthetic": evaluate_decisions(synthetic_worlds, control_decisions, results_by_world, structured_variant),
                "semi-real": evaluate_decisions(semireal_worlds, control_decisions, results_by_world, structured_variant),
                "pooled": evaluate_decisions(pooled_worlds, control_decisions, results_by_world, structured_variant),
            },
            "always_coord": {
                "synthetic": evaluate_decisions(synthetic_worlds, always_coord, results_by_world, structured_variant),
                "semi-real": evaluate_decisions(semireal_worlds, always_coord, results_by_world, structured_variant),
                "pooled": evaluate_decisions(pooled_worlds, always_coord, results_by_world, structured_variant),
            },
            "always_structured": {
                "synthetic": evaluate_decisions(synthetic_worlds, always_structured, results_by_world, structured_variant),
                "semi-real": evaluate_decisions(semireal_worlds, always_structured, results_by_world, structured_variant),
                "pooled": evaluate_decisions(pooled_worlds, always_structured, results_by_world, structured_variant),
            },
            "random_router": {
                "synthetic": random_control_metrics(
                    synthetic_worlds,
                    learned_rate_synthetic,
                    results_by_world,
                    structured_variant,
                    random_trials,
                    seed + 1000,
                ),
                "semi-real": random_control_metrics(
                    semireal_worlds,
                    learned_rate_semireal,
                    results_by_world,
                    structured_variant,
                    random_trials,
                    seed + 2000,
                ),
                "pooled": random_control_metrics(
                    pooled_worlds,
                    learned_rate_pooled,
                    results_by_world,
                    structured_variant,
                    random_trials,
                    seed,
                ),
            },
        },
    }


def plot_pooled_router_mse(rows: list[dict], output_path: Path) -> None:
    labels = [row["label"] for row in rows]
    values = [row["pooled_metrics"]["mean_routed_mse"]["mean"] for row in rows]
    figure, axis = plt.subplots(1, 1, figsize=(11.0, 4.8))
    axis.bar(np.arange(len(labels)), values, color="#1f77b4")
    axis.set_xticks(np.arange(len(labels)))
    axis.set_xticklabels(labels, rotation=26, ha="right")
    axis.set_ylabel("pooled routed mse")
    axis.set_title("Routing v2: Pooled Routed MSE")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_threshold_band(rows: list[dict], output_path: Path) -> None:
    figure, axis = plt.subplots(1, 1, figsize=(9.5, 4.8))
    for index, row in enumerate(rows):
        axis.errorbar(index, row["threshold"]["mean"], yerr=row["threshold"]["std"], fmt="o", color="#1f77b4")
        axis.errorbar(index, row["abstain_band"]["mean"], yerr=row["abstain_band"]["std"], fmt="s", color="#d62728")
    axis.set_xticks(range(len(rows)))
    axis.set_xticklabels([row["label"] for row in rows], rotation=24, ha="right")
    axis.set_ylabel("value")
    axis.set_title("Threshold and Abstain-Band Stability")
    axis.grid(alpha=0.25)
    axis.legend(["threshold", "abstain band"], fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_summary(output_dir: Path, summary_rows: list[dict]) -> None:
    lines = [
        "# Regime-Gated Routing v2",
        "",
        "| router | pooled mse | pooled gain vs coord | pooled regret | semi-real mse | semi-real gain vs coord | semi-real route rate | coverage | filtered sign acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        pooled = row["pooled_metrics"]
        semireal = row["semireal_metrics"]
        filtered = pooled["margin_filtered_sign_accuracy"]
        filtered_str = "n/a" if filtered is None else f"{filtered['mean']:.2f}+/-{filtered['std']:.2f}"
        lines.append(
            f"| {row['label']} | {pooled['mean_routed_mse']['mean']:.6f} | {pooled['gain_over_coord']['mean']:+.6f} | "
            f"{pooled['regret_to_pair_oracle']['mean']:.6f} | {semireal['mean_routed_mse']['mean']:.6f} | "
            f"{semireal['gain_over_coord']['mean']:+.6f} | {semireal['structured_route_rate']['mean']:.2f} | "
            f"{pooled['coverage']['mean']:.2f} | {filtered_str} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(output_dir: Path, report_path: Path, summary_rows: list[dict]) -> None:
    best = min(
        [row for row in summary_rows if row["router_type"] == "learned"],
        key=lambda row: (row["pooled_metrics"]["mean_routed_mse"]["mean"], row["pooled_metrics"]["regret_to_pair_oracle"]["mean"]),
    )
    lines = [
        "# Regime-Gated Routing v2",
        "",
        "Best deployable router:",
        f"- `{best['label']}` with pooled routed mse `{best['pooled_metrics']['mean_routed_mse']['mean']:.6f}`, pooled gain over coord `{best['pooled_metrics']['gain_over_coord']['mean']:+.6f}`, semi-real gain over coord `{best['semireal_metrics']['gain_over_coord']['mean']:+.6f}`, semi-real route rate `{best['semireal_metrics']['structured_route_rate']['mean']:.2f}`.",
        "",
        "Plots:",
        "![Pooled routed mse](pooled_router_mse.png)",
        "![Threshold and band stability](threshold_band_stability.png)",
    ]
    for row in summary_rows:
        if row["router_type"] != "learned":
            continue
        lines.extend(
            [
                "",
                f"## {row['label']}",
                "",
                f"- encoder mode: `{row['encoder_mode']}`",
                f"- pair: `{row['pair_name']}`",
                f"- threshold: `{row['threshold']['mean']:.6f}+/-{row['threshold']['std']:.6f}`",
                f"- abstain band: `{row['abstain_band']['mean']:.6f}+/-{row['abstain_band']['std']:.6f}`",
                "",
                "| world | family | score_mean | score_std | |score-tau| | delta_pair | route | coord_mse | structured_mse | routed_mse | oracle_mse | regret |",
                "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for world_row in row["per_world_rows"]:
            lines.append(
                f"| {world_row['world']} | {world_row['family']} | {world_row['score_mean']:.6f} | {world_row['score_std']:.6f} | "
                f"{world_row['score_margin_mean']:.6f} | {world_row['delta_pair']:+.6f} | {world_row['route']} | "
                f"{world_row['coord_mse']:.6f} | {world_row['structured_mse']:.6f} | {world_row['routed_mse_mean']:.6f} | "
                f"{world_row['oracle_mse']:.6f} | {world_row['regret_mean']:.6f} |"
            )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    shared_pair_a = next(
        row
        for row in summary_rows
        if row["router_type"] == "learned" and row["encoder_mode"] == "shared_train" and row["pair_name"] == "pairA_curv_hankel"
    )
    synthetic_only_pair_a = next(
        row
        for row in summary_rows
        if row["router_type"] == "learned" and row["encoder_mode"] == "synthetic_only_train" and row["pair_name"] == "pairA_curv_hankel"
    )
    best_beats_coord = (
        best["pooled_metrics"]["mean_routed_mse"]["mean"] < best["controls"]["always_coord"]["pooled"]["mean_routed_mse"]["mean"]
        and best["semireal_metrics"]["mean_routed_mse"]["mean"]
        <= best["controls"]["always_coord"]["semi-real"]["mean_routed_mse"]["mean"] + 1e-12
    )
    shared_transfer_pass = (
        shared_pair_a["semireal_metrics"]["mean_routed_mse"]["mean"] <= synthetic_only_pair_a["semireal_metrics"]["mean_routed_mse"]["mean"] + 1e-12
        or shared_pair_a["semireal_metrics"]["route_sign_accuracy"]["mean"]
        >= synthetic_only_pair_a["semireal_metrics"]["route_sign_accuracy"]["mean"] - 1e-12
    )
    if best_beats_coord and shared_transfer_pass:
        verdict = (
            "Routing v2 preserves the coarse-gate claim under multi-seed aggregation and the shared-latent transfer check remains competitive."
        )
    elif best_beats_coord:
        verdict = (
            "Routing v2 still yields a deployable coarse gate, but the transfer-separation check failed: "
            "synthetic-only training beats shared training on the semi-real side, so the shared-latent portability claim is weakened."
        )
    else:
        verdict = (
            "Routing v2 does not preserve the practical coarse-gate claim after multi-seed aggregation; "
            "end-to-end gating should be deferred."
        )
    findings_lines = [
        "# Regime-Gated Routing v2",
        "",
        "## Result",
        "",
        f"- best deployable router: `{best['label']}` with pooled routed mse `{best['pooled_metrics']['mean_routed_mse']['mean']:.6f}` and semi-real gain over coord `{best['semireal_metrics']['gain_over_coord']['mean']:+.6f}`.",
        f"- `shared_train` pair A pooled routed mse: `{shared_pair_a['pooled_metrics']['mean_routed_mse']['mean']:.6f}`.",
        f"- `shared_train` pair A semi-real gain over always-coord: `{shared_pair_a['semireal_metrics']['gain_over_coord']['mean']:+.6f}`.",
        f"- `synthetic_only_train` pair A pooled routed mse: `{synthetic_only_pair_a['pooled_metrics']['mean_routed_mse']['mean']:.6f}`.",
        f"- `synthetic_only_train` pair A semi-real gain over always-coord: `{synthetic_only_pair_a['semireal_metrics']['gain_over_coord']['mean']:+.6f}`.",
        "",
        "## Interpretation",
        "",
        verdict,
        "",
        "This should still be treated as a coarse gate only. The result does not upgrade the score into a continuous regime estimator.",
    ]
    report_path.write_text("\n".join(findings_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    report_path: Path = args.report_path
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    stability_results = load_results(args.stability_results)
    synthetic_results = load_results(args.synthetic_results)
    semireal_results = load_results(args.semireal_results)
    synthetic_worlds = ordered_worlds(synthetic_results)
    semireal_worlds = ordered_worlds(semireal_results)
    results_by_world = {world: synthetic_results for world in synthetic_worlds} | {world: semireal_results for world in semireal_worlds}

    summary_rows: list[dict] = []
    payload_rows: list[dict] = []

    for encoder_mode in ("shared_train", "synthetic_only_train"):
        mode_seed_runs = stability_results["modes"][encoder_mode]["seed_runs"]
        for pair_name, structured_variant in PAIR_VARIANTS.items():
            seed_runs = [
                build_seed_run(
                    encoder_mode=encoder_mode,
                    seed_run=seed_run,
                    synthetic_worlds=synthetic_worlds,
                    semireal_worlds=semireal_worlds,
                    results_by_world=results_by_world,
                    structured_variant=structured_variant,
                    score_name=args.score_name,
                    random_trials=args.random_trials,
                )
                for seed_run in mode_seed_runs
            ]
            aggregate = aggregate_seed_runs(seed_runs)
            learned_row = {
                "label": f"{encoder_mode}_{pair_name}",
                "encoder_mode": encoder_mode,
                "pair_name": pair_name,
                "router_type": "learned",
                "threshold": aggregate["threshold"],
                "abstain_band": aggregate["abstain_band"],
                "synthetic_metrics": aggregate["synthetic_metrics"],
                "semireal_metrics": aggregate["semireal_metrics"],
                "pooled_metrics": aggregate["pooled_metrics"],
                "family_conditioned_metrics": aggregate["family_conditioned_metrics"],
                "per_world_rows": aggregate["per_world_rows"],
                "controls": aggregate["control_rows"],
            }
            summary_rows.append(learned_row)
            payload_rows.append(
                {
                    "encoder_mode": encoder_mode,
                    "pair_name": pair_name,
                    "structured_variant": structured_variant,
                    "seed_runs": seed_runs,
                    "aggregate": aggregate,
                }
            )
            for control_name in CONTROL_NAMES:
                summary_rows.append(
                    {
                        "label": f"{encoder_mode}_{pair_name}_{control_name}",
                        "encoder_mode": encoder_mode,
                        "pair_name": pair_name,
                        "router_type": "control",
                        "threshold": aggregate["threshold"],
                        "abstain_band": aggregate["abstain_band"],
                        "synthetic_metrics": aggregate["control_rows"][control_name]["synthetic"],
                        "semireal_metrics": aggregate["control_rows"][control_name]["semi-real"],
                        "pooled_metrics": aggregate["control_rows"][control_name]["pooled"],
                        "family_conditioned_metrics": {},
                        "per_world_rows": [],
                        "controls": {},
                    }
                )

    plot_pooled_router_mse(summary_rows, output_dir / "pooled_router_mse.png")
    plot_threshold_band(
        [row for row in summary_rows if row["router_type"] == "learned"],
        output_dir / "threshold_band_stability.png",
    )

    payload = {
        "score_name": args.score_name,
        "synthetic_worlds": synthetic_worlds,
        "semireal_worlds": semireal_worlds,
        "rows": payload_rows,
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary(output_dir, summary_rows)
    write_report(output_dir, report_path, summary_rows)


if __name__ == "__main__":
    main()
