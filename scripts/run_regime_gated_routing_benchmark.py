from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from structured_latent_hypothesis.routing import (
    mean_metric_for_assignments,
    mean_regret_to_oracle,
    metric,
    oracle_variant,
    route_variant,
)
from structured_latent_hypothesis.synthetic import ground_truth_coupling_strength


ROOT = Path("D:/Experiment")
SHARED_RESULTS = ROOT / "results" / "shared_latent_score_validation_v1" / "results.json"
SYNTHETIC_RESULTS = ROOT / "results" / "structured_hybrid_probe_v1" / "results.json"
SEMIREAL_RESULTS = ROOT / "results" / "semireal_transfer_probe_v1" / "results.json"
OUTPUT_DIR = ROOT / "results" / "regime_gated_routing_v1"
REPORT_PATH = ROOT / "reports" / "2026-04-18_regime_gated_routing_v1_findings.md"

PAIR_VARIANTS = {
    "pairA_curv_hankel": "curv_hankel_r4_selected",
    "pairB_operator_diag": "operator_diag_r2_selected",
}
BASE_VARIANT = "coord_latent"
ORACLE_STRUCTURED = [
    "additive_resid_selected",
    "curvature_field_r4_selected",
    "curv_hankel_r4_selected",
    "operator_diag_r2_selected",
]


@dataclass
class RouterMetrics:
    route_sign_accuracy: float
    route_balanced_accuracy: float
    mean_routed_mse: float
    gain_over_coord: float
    gain_over_always_structured: float
    regret_to_pair_oracle: float
    capture_ratio: float
    structured_route_rate: float


def load_results(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ordered_worlds(results: dict) -> list[str]:
    return sorted(
        results["worlds"],
        key=lambda world: float(results["world_metadata"][world]["ground_truth_coupling_strength"] or 0.0),
    )


def world_score(shared_results: dict, world: str, score_name: str) -> float:
    return float(shared_results["world_scores"][world][score_name])


def metadata_cpl(results: dict, world: str, eps: float = 1e-8) -> float:
    meta = results["world_metadata"][world]
    comm = float(meta["ground_truth_commutator"] or 0.0)
    drift = meta["ground_truth_step_drift"]
    if drift is None:
        return comm
    return comm / (eps + float(drift))


def branch_delta(results: dict, world: str, structured_variant: str) -> float:
    return metric(results, world, BASE_VARIANT) - metric(results, world, structured_variant)


def branch_label(results: dict, world: str, structured_variant: str) -> bool:
    return branch_delta(results, world, structured_variant) > 0.0


def balanced_accuracy(labels: list[bool], predictions: list[bool]) -> float:
    positives = [index for index, label in enumerate(labels) if label]
    negatives = [index for index, label in enumerate(labels) if not label]
    if not positives or not negatives:
        return mean(float(pred == label) for pred, label in zip(predictions, labels))
    tpr = mean(float(predictions[index]) for index in positives)
    tnr = mean(float(not predictions[index]) for index in negatives)
    return 0.5 * (tpr + tnr)


def threshold_candidates(scores: list[float]) -> list[float]:
    unique_scores = sorted(set(float(score) for score in scores))
    if not unique_scores:
        return [0.0]
    thresholds = [unique_scores[0] - 1e-6]
    for left, right in zip(unique_scores[:-1], unique_scores[1:]):
        thresholds.append(0.5 * (left + right))
    thresholds.append(unique_scores[-1] + 1e-6)
    return thresholds


def calibrate_pair_threshold(
    scores: list[float],
    labels: list[bool],
    coord_values: list[float],
    structured_values: list[float],
    direction: str = "high_positive",
) -> tuple[float, float, float, float]:
    best_threshold = threshold_candidates(scores)[0]
    best_balanced = -1.0
    best_regret = float("inf")
    best_margin = -1.0
    best_accuracy = -1.0
    for threshold in threshold_candidates(scores):
        predictions = [
            (score >= threshold if direction == "high_positive" else score <= threshold)
            for score in scores
        ]
        balanced = balanced_accuracy(labels, predictions)
        routed_values = [
            structured if choose_structured else coord
            for choose_structured, coord, structured in zip(predictions, coord_values, structured_values)
        ]
        regrets = [routed - min(coord, structured) for routed, coord, structured in zip(routed_values, coord_values, structured_values)]
        regret = mean(regrets)
        margin = min(abs(score - threshold) for score in scores)
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
                            or (abs(margin - best_margin) <= 1e-12 and accuracy > best_accuracy + 1e-12)
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
    return best_threshold, best_balanced, best_regret, best_margin


def choose_best_directional_threshold(
    scores: list[float],
    labels: list[bool],
    coord_values: list[float],
    structured_values: list[float],
) -> tuple[float, str]:
    candidates = []
    for direction in ("high_positive", "low_positive"):
        threshold, balanced, regret, margin = calibrate_pair_threshold(
            scores,
            labels,
            coord_values,
            structured_values,
            direction=direction,
        )
        candidates.append((balanced, -regret, margin, threshold, direction))
    best = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    return best[3], best[4]


def build_assignments(
    worlds: list[str],
    scores: dict[str, float],
    threshold: float,
    direction: str,
    structured_variant: str,
    invert: bool = False,
) -> dict[str, str]:
    return {
        world: route_variant(scores[world], threshold, direction, structured_variant=structured_variant, base_variant=BASE_VARIANT, invert=invert)
        for world in worlds
    }


def build_oracle_structured_assignments(
    results_by_world: dict[str, dict],
    worlds: list[str],
    scores: dict[str, float],
    threshold: float,
    direction: str,
) -> dict[str, str]:
    assignments = {}
    for world in worlds:
        choose_structured = scores[world] >= threshold if direction == "high_positive" else scores[world] <= threshold
        if choose_structured:
            assignments[world] = oracle_variant(
                results_by_world[world],
                world,
                structured_variants=[variant for variant in ORACLE_STRUCTURED if variant in results_by_world[world]["summary"][world]],
                base_variant=BASE_VARIANT,
            )
        else:
            assignments[world] = BASE_VARIANT
    return assignments


def evaluate_assignments(
    results_by_world: dict[str, dict],
    worlds: list[str],
    assignments: dict[str, str],
    structured_reference: dict[str, str],
    oracle_variants: dict[str, str],
) -> RouterMetrics:
    labels = [branch_label(results_by_world[world], world, structured_reference[world]) for world in worlds]
    predictions = [assignments[world] != BASE_VARIANT for world in worlds]
    routed_values = [metric(results_by_world[world], world, assignments[world]) for world in worlds]
    coord_values = [metric(results_by_world[world], world, BASE_VARIANT) for world in worlds]
    structured_values = [metric(results_by_world[world], world, structured_reference[world]) for world in worlds]
    oracle_values = [metric(results_by_world[world], world, oracle_variants[world]) for world in worlds]

    possible_gain = sum(max(0.0, coord - oracle) for coord, oracle in zip(coord_values, oracle_values))
    realized_gain = sum(coord - routed for coord, routed in zip(coord_values, routed_values))

    return RouterMetrics(
        route_sign_accuracy=mean(float(pred == label) for pred, label in zip(predictions, labels)),
        route_balanced_accuracy=balanced_accuracy(labels, predictions),
        mean_routed_mse=mean(routed_values),
        gain_over_coord=mean(coord - routed for coord, routed in zip(coord_values, routed_values)),
        gain_over_always_structured=mean(structured - routed for structured, routed in zip(structured_values, routed_values)),
        regret_to_pair_oracle=mean(routed - oracle for routed, oracle in zip(routed_values, oracle_values)),
        capture_ratio=(realized_gain / possible_gain) if possible_gain > 1e-12 else 0.0,
        structured_route_rate=mean(float(pred) for pred in predictions),
    )


def router_world_rows(
    results_by_world: dict[str, dict],
    worlds: list[str],
    scores: dict[str, float],
    threshold: float,
    assignments: dict[str, str],
    structured_reference: dict[str, str],
    oracle_variants: dict[str, str],
) -> list[dict[str, float | str | bool]]:
    rows = []
    for world in worlds:
        coord = metric(results_by_world[world], world, BASE_VARIANT)
        structured = metric(results_by_world[world], world, structured_reference[world])
        routed = metric(results_by_world[world], world, assignments[world])
        oracle = metric(results_by_world[world], world, oracle_variants[world])
        delta = coord - structured
        rows.append(
            {
                "world": world,
                "coupling": float(ground_truth_coupling_strength(world) or 0.0),
                "score": scores[world],
                "score_margin": abs(scores[world] - threshold),
                "branch_delta": delta,
                "distance_to_zero": abs(delta),
                "label_positive": delta > 0.0,
                "route": assignments[world],
                "coord_mse": coord,
                "structured_variant": structured_reference[world],
                "structured_mse": structured,
                "routed_mse": routed,
                "oracle_variant": oracle_variants[world],
                "oracle_mse": oracle,
                "regret": routed - oracle,
            }
        )
    return rows


def plot_router_bars(summary_rows: list[dict], output_path: Path) -> None:
    labels = [row["router"] for row in summary_rows]
    pooled_values = [row["pooled"]["mean_routed_mse"] for row in summary_rows]
    regret_values = [row["pooled"]["regret_to_pair_oracle"] for row in summary_rows]
    x = np.arange(len(labels))
    width = 0.38
    figure, axis = plt.subplots(1, 1, figsize=(11.5, 5.2))
    axis.bar(x - width / 2, pooled_values, width=width, label="pooled routed mse")
    axis.bar(x + width / 2, regret_values, width=width, label="pooled regret")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=24, ha="right")
    axis.set_title("Regime-Gated Routing Benchmark")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_score_by_family(shared_results: dict, output_path: Path) -> None:
    synthetic_worlds = shared_results["synthetic_worlds"]
    semireal_worlds = shared_results["semireal_worlds"]
    figure, axis = plt.subplots(1, 1, figsize=(7.5, 4.8))
    axis.plot(
        [ground_truth_coupling_strength(world) or 0.0 for world in synthetic_worlds],
        [world_score(shared_results, world, "S_joint") for world in synthetic_worlds],
        marker="o",
        linewidth=1.8,
        label="synthetic",
    )
    axis.plot(
        [ground_truth_coupling_strength(world) or 0.0 for world in semireal_worlds],
        [world_score(shared_results, world, "S_joint") for world in semireal_worlds],
        marker="o",
        linewidth=1.8,
        label="semi-real",
    )
    axis.set_xlabel("coupling strength")
    axis.set_ylabel("S_joint")
    axis.set_title("S_joint by World Family")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def metrics_dict(metrics: RouterMetrics) -> dict[str, float]:
    return asdict(metrics)


def add_summary_row(
    rows: list[dict],
    name: str,
    synthetic: RouterMetrics,
    semireal: RouterMetrics,
    pooled: RouterMetrics,
    threshold: float | None,
    direction: str | None,
) -> None:
    rows.append(
        {
            "router": name,
            "threshold": threshold,
            "direction": direction,
            "synthetic": metrics_dict(synthetic),
            "semireal": metrics_dict(semireal),
            "pooled": metrics_dict(pooled),
        }
    )


def oracle_variant_for_world(results_by_world: dict[str, dict], world: str) -> str:
    return oracle_variant(
        results_by_world[world],
        world,
        structured_variants=[variant for variant in ORACLE_STRUCTURED if variant in results_by_world[world]["summary"][world]],
        base_variant=BASE_VARIANT,
    )


def write_summary(summary_rows: list[dict]) -> None:
    lines = [
        "# Regime-Gated Routing v1",
        "",
        "| router | threshold | synthetic mse | semi-real mse | pooled mse | pooled gain vs coord | pooled regret | pooled capture ratio | semi-real route rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        threshold = "n/a" if row["threshold"] is None else f"{row['threshold']:.6f}"
        lines.append(
            f"| {row['router']} | {threshold} | {row['synthetic']['mean_routed_mse']:.6f} | {row['semireal']['mean_routed_mse']:.6f} | "
            f"{row['pooled']['mean_routed_mse']:.6f} | {row['pooled']['gain_over_coord']:+.6f} | {row['pooled']['regret_to_pair_oracle']:.6f} | "
            f"{row['pooled']['capture_ratio']:.3f} | {row['semireal']['structured_route_rate']:.2f} |"
        )
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(summary_rows: list[dict], world_rows: dict[str, list[dict]]) -> None:
    best_row = min(
        [row for row in summary_rows if row["router"].startswith("router_")],
        key=lambda row: (row["pooled"]["mean_routed_mse"], row["pooled"]["regret_to_pair_oracle"]),
    )
    lines = [
        "# Regime-Gated Routing v1",
        "",
        "Best deployable router:",
        f"- `{best_row['router']}` with pooled routed mse `{best_row['pooled']['mean_routed_mse']:.6f}`, pooled gain over coord `{best_row['pooled']['gain_over_coord']:+.6f}`, pooled regret `{best_row['pooled']['regret_to_pair_oracle']:.6f}`, semi-real route rate `{best_row['semireal']['structured_route_rate']:.2f}`.",
        "",
        "This benchmark uses synthetic-only threshold calibration and then freezes the router before semi-real evaluation.",
        "",
        "Plots:",
        "![Routing benchmark](routing_bars.png)",
        "![S_joint by family](s_joint_by_family.png)",
    ]
    for router_name, rows in world_rows.items():
        lines.extend(
            [
                "",
                f"## {router_name}",
                "",
                "| world | coupling | score | |S-tau| | delta_pair | route | coord mse | structured variant | structured mse | routed mse | oracle variant | oracle mse | regret |",
                "| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | --- | ---: | ---: |",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['world']} | {row['coupling']:.2f} | {row['score']:.6f} | {row['score_margin']:.6f} | {row['branch_delta']:+.6f} | "
                f"{row['route']} | {row['coord_mse']:.6f} | {row['structured_variant']} | {row['structured_mse']:.6f} | "
                f"{row['routed_mse']:.6f} | {row['oracle_variant']} | {row['oracle_mse']:.6f} | {row['regret']:.6f} |"
            )
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_findings(summary_rows: list[dict]) -> None:
    deployable_rows = [row for row in summary_rows if row["router"].startswith("router_")]
    best = min(deployable_rows, key=lambda row: (row["pooled"]["mean_routed_mse"], row["pooled"]["regret_to_pair_oracle"]))
    lines = [
        "# Regime-Gated Routing v1",
        "",
        "## Result",
        "",
        f"- Best deployable router: `{best['router']}`.",
        f"- Pooled routed mse: `{best['pooled']['mean_routed_mse']:.6f}`.",
        f"- Pooled gain over always-coord: `{best['pooled']['gain_over_coord']:+.6f}`.",
        f"- Pooled regret to pair oracle: `{best['pooled']['regret_to_pair_oracle']:.6f}`.",
        f"- Semi-real structured route rate: `{best['semireal']['structured_route_rate']:.2f}`.",
        "",
        "## Interpretation",
        "",
    ]
    if best["pooled"]["gain_over_coord"] > 0.0 and best["semireal"]["structured_route_rate"] <= 0.25:
        lines.append(
            "Shared-latent gating now has practical value: it preserves coord-like behavior on semi-real while recovering structured gains on the synthetic boundary regime."
        )
    else:
        lines.append(
            "Shared-latent gating is still not a reliable practical router. It may describe regimes, but it does not yet improve routed performance enough over simple fixed baselines."
        )
    lines.append("")
    lines.append(
        "The result should still be treated as a coarse gate, not a continuous regime estimator, because within-family semi-real ranking remains unresolved."
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    random.seed(0)
    shared_results = load_results(SHARED_RESULTS)
    synthetic_results = load_results(SYNTHETIC_RESULTS)
    semireal_results = load_results(SEMIREAL_RESULTS)
    synthetic_worlds = ordered_worlds(synthetic_results)
    semireal_worlds = ordered_worlds(semireal_results)
    pooled_worlds = synthetic_worlds + semireal_worlds
    results_by_world = {world: synthetic_results for world in synthetic_worlds} | {world: semireal_results for world in semireal_worlds}
    shared_scores = {world: world_score(shared_results, world, "S_joint") for world in pooled_worlds}
    control_scores = {world: metadata_cpl(results_by_world[world], world) for world in pooled_worlds}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    world_tables: dict[str, list[dict]] = {}

    oracle_pair_reference = {world: oracle_variant_for_world(results_by_world, world) for world in pooled_worlds}
    oracle_assignments = dict(oracle_pair_reference)
    oracle_metrics_synth = evaluate_assignments(results_by_world, synthetic_worlds, oracle_assignments, oracle_pair_reference, oracle_pair_reference)
    oracle_metrics_semi = evaluate_assignments(results_by_world, semireal_worlds, oracle_assignments, oracle_pair_reference, oracle_pair_reference)
    oracle_metrics_pool = evaluate_assignments(results_by_world, pooled_worlds, oracle_assignments, oracle_pair_reference, oracle_pair_reference)
    add_summary_row(summary_rows, "oracle_world_best", oracle_metrics_synth, oracle_metrics_semi, oracle_metrics_pool, None, None)

    for router_name, structured_variant in PAIR_VARIANTS.items():
        structured_reference = {world: structured_variant for world in pooled_worlds}
        labels_synth = [branch_label(synthetic_results, world, structured_variant) for world in synthetic_worlds]
        coord_synth = [metric(synthetic_results, world, BASE_VARIANT) for world in synthetic_worlds]
        struct_synth = [metric(synthetic_results, world, structured_variant) for world in synthetic_worlds]
        synth_scores = [shared_scores[world] for world in synthetic_worlds]
        threshold, _, _, _ = calibrate_pair_threshold(synth_scores, labels_synth, coord_synth, struct_synth, direction="high_positive")
        control_threshold, control_direction = choose_best_directional_threshold(
            [control_scores[world] for world in synthetic_worlds],
            labels_synth,
            coord_synth,
            struct_synth,
        )

        learned_assignments = build_assignments(pooled_worlds, shared_scores, threshold, "high_positive", structured_variant)
        inverted_assignments = build_assignments(pooled_worlds, shared_scores, threshold, "high_positive", structured_variant, invert=True)
        control_assignments = build_assignments(pooled_worlds, control_scores, control_threshold, control_direction, structured_variant)
        always_coord = {world: BASE_VARIANT for world in pooled_worlds}
        always_structured = {world: structured_variant for world in pooled_worlds}

        pair_oracle = {
            world: oracle_variant(
                results_by_world[world],
                world,
                structured_variants=[structured_variant],
                base_variant=BASE_VARIANT,
            )
            for world in pooled_worlds
        }
        structured_oracle_router = build_oracle_structured_assignments(results_by_world, pooled_worlds, shared_scores, threshold, "high_positive")

        configurations = {
            f"router_{router_name}_shared_joint": learned_assignments,
            f"router_{router_name}_inverted": inverted_assignments,
            f"router_{router_name}_meta_cpl_control": control_assignments,
            f"always_coord_{router_name}": always_coord,
            f"always_structured_{router_name}": always_structured,
            f"oracle_threshold_{router_name}": structured_oracle_router,
        }

        for name, assignments in configurations.items():
            synth_metrics = evaluate_assignments(results_by_world, synthetic_worlds, assignments, structured_reference, pair_oracle)
            semi_metrics = evaluate_assignments(results_by_world, semireal_worlds, assignments, structured_reference, pair_oracle)
            pooled_metrics = evaluate_assignments(results_by_world, pooled_worlds, assignments, structured_reference, pair_oracle)
            add_summary_row(
                summary_rows,
                name,
                synth_metrics,
                semi_metrics,
                pooled_metrics,
                (
                    None
                    if "always_" in name
                    else control_threshold if "meta_cpl_control" in name else threshold
                ),
                (
                    None
                    if "always_" in name
                    else control_direction if "meta_cpl_control" in name else "high_positive"
                ),
            )
            if name.startswith("router_"):
                world_tables[name] = router_world_rows(
                    results_by_world,
                    pooled_worlds,
                    shared_scores if "meta_cpl_control" not in name else control_scores,
                    control_threshold if "meta_cpl_control" in name else threshold,
                    assignments,
                    structured_reference,
                    pair_oracle,
                )

    plot_router_bars(summary_rows, OUTPUT_DIR / "routing_bars.png")
    plot_score_by_family(shared_results, OUTPUT_DIR / "s_joint_by_family.png")

    payload = {
        "shared_score_name": "S_joint",
        "routers": summary_rows,
        "world_tables": world_tables,
    }
    (OUTPUT_DIR / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary(summary_rows)
    write_report(summary_rows, world_tables)
    write_findings(summary_rows)


if __name__ == "__main__":
    main()
