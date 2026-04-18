from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from structured_latent_hypothesis.transfer_criterion import (
    analyze_context_transfer_criterion,
    build_context_transfer_rows,
    load_json,
    render_criterion_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze diagnostic transfer criteria on top of context-transfer v1 results.")
    parser.add_argument("--operator-results", default="results/context_operator_probe_v1/results.json")
    parser.add_argument("--adaptation-results", default="results/context_adaptation_probe_v1/results.json")
    parser.add_argument("--output-dir", default="results/context_transfer_criterion_v1")
    parser.add_argument("--report-path", default="reports/2026-04-19_context_transfer_criterion_v1_findings.md")
    return parser.parse_args()


def plot_correlations(results: dict, output_path: Path, target: str, title: str) -> None:
    names = list(results["analyses"].keys())
    values = [results["analyses"][name]["correlations"]["all"][target] for name in names]
    figure, axis = plt.subplots(1, 1, figsize=(10.5, 4.8))
    axis.bar(range(len(names)), values, color="#5b7fff")
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels(names, rotation=24, ha="right")
    axis.set_ylabel("Spearman")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_router(results: dict, output_path: Path) -> None:
    names = list(results["analyses"].keys())
    routed = [results["analyses"][name]["router"]["routed_rollout5_mse_mean"] for name in names]
    always_full = [results["analyses"][name]["router"]["always_full_rollout5_mse_mean"] for name in names]
    deltas = [left - right for left, right in zip(routed, always_full)]

    figure, axis = plt.subplots(1, 1, figsize=(10.5, 4.8))
    axis.bar(range(len(names)), deltas, color="#f08a5d")
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels(names, rotation=24, ha="right")
    axis.set_ylabel("routed - always_full")
    axis.set_title("Leave-one-seed-out routing delta vs always_full")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_report(output_dir: Path, report_path: Path, results: dict) -> None:
    analyses = results["analyses"]
    best_regret = max(
        analyses.items(),
        key=lambda item: item[1]["correlations"]["all"]["structured_rollout5_regret"],
    )
    best_steps = max(
        analyses.items(),
        key=lambda item: item[1]["correlations"]["all"]["full_adaptation_steps"],
    )
    best_router = min(
        analyses.items(),
        key=lambda item: item[1]["router"]["routed_rollout5_mse_mean"],
    )

    report_lines = [
        "# Context Transfer Criterion v1",
        "",
        "## Plots",
        "",
        "![Regret correlation](criterion_regret_correlation.png)",
        "",
        "![Difficulty correlation](criterion_steps_correlation.png)",
        "",
        "![Routing delta](criterion_router_delta.png)",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    always_full = best_router[1]["router"]["always_full_rollout5_mse_mean"]
    routed = best_router[1]["router"]["routed_rollout5_mse_mean"]
    router_delta = routed - always_full
    router_note = "no improvement over always_full" if router_delta >= -1e-12 else "improves over always_full"
    findings_lines = [
        "# Context Transfer Criterion v1",
        "",
        "## Result",
        "",
        f"- Best regret-tracking score: `{best_regret[0]}` with all-row Spearman `{best_regret[1]['correlations']['all']['structured_rollout5_regret']:+.3f}`.",
        f"- Best transfer-difficulty score: `{best_steps[0]}` with all-row Spearman `{best_steps[1]['correlations']['all']['full_adaptation_steps']:+.3f}`.",
        f"- Best leave-one-seed-out router: `{best_router[0]}` with routed rollout@5 `{routed:.6f}` versus `always_full` `{always_full:.6f}`; {router_note}.",
        "",
        "## Interpretation",
        "",
        "This probe asks a narrower question than the earlier architecture claims: whether triple-derived interaction scores can serve as a usable transfer criterion. A candidate counts as practical only if it both tracks regret/difficulty and beats `always_full` in leave-one-seed-out routing.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(findings_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    operator_results = load_json(args.operator_results)
    adaptation_results = load_json(args.adaptation_results)
    rows = build_context_transfer_rows(operator_results, adaptation_results)
    analyses = analyze_context_transfer_criterion(rows)
    results = {
        "operator_results": str(args.operator_results),
        "adaptation_results": str(args.adaptation_results),
        "rows": rows,
        "analyses": analyses,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(render_criterion_markdown(results), encoding="utf-8")
    plot_correlations(results, output_dir / "criterion_regret_correlation.png", "structured_rollout5_regret", "Score vs structured rollout regret")
    plot_correlations(results, output_dir / "criterion_steps_correlation.png", "full_adaptation_steps", "Score vs full adaptation steps")
    plot_router(results, output_dir / "criterion_router_delta.png")
    write_report(output_dir, Path(args.report_path), results)


if __name__ == "__main__":
    main()
