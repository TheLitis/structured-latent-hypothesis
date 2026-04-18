from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from structured_latent_hypothesis.optimizer_geometry import run_optimizer_geometry_suite
from structured_latent_hypothesis.plotting import load_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the optimizer-geometry quadratic coupling ladder.")
    parser.add_argument("--output-dir", default="results/optimizer_geometry_probe_v1")
    parser.add_argument("--report-path", default="reports/2026-04-19_optimizer_geometry_probe_v1_findings.md")
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--basis-refresh", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 11, 17, 29, 43, 71, 89, 101])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.00, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00])
    return parser.parse_args()


VARIANTS = [
    "adam_full",
    "random_subspace_diag",
    "oja_subspace_diag",
    "oja_subspace_full",
    "low_mixed_curvature_basis",
]


def plot_metric(results: dict, output_path: Path, metric: str, title: str, ylabel: str) -> None:
    alphas = [f"{float(alpha):0.2f}" for alpha in results["alphas"]]
    figure, axis = plt.subplots(1, 1, figsize=(8.6, 4.8))
    for variant in results["variants"]:
        values = [results["summary"][alpha][variant][metric]["mean"] for alpha in alphas]
        axis.plot([float(alpha) for alpha in alphas], values, marker="o", linewidth=1.9, label=variant)
    axis.set_xlabel("alpha")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_report(output_dir: Path, report_path: Path, results: dict) -> None:
    summary = results["summary"]
    low_alpha = "0.20"
    high_alpha = "0.75"
    low_diag = summary[low_alpha]["oja_subspace_diag"]["final_loss"]["median"]
    low_proposed = summary[low_alpha]["low_mixed_curvature_basis"]["final_loss"]["median"]
    relative_gain = (low_diag - low_proposed) / max(low_diag, 1e-12)

    report_lines = [
        "# Optimizer Geometry Probe v1",
        "",
        "## Plots",
        "",
        "![Final loss](final_loss.png)",
        "",
        "![Loss AUC](loss_auc.png)",
        "",
        "![Off-diagonal curvature](off_diagonal_curvature.png)",
        "",
        "![Order sensitivity](order_sensitivity.png)",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    findings_lines = [
        "# Optimizer Geometry Probe v1",
        "",
        "## Result",
        "",
        f"- Low-coupling check `alpha={low_alpha}`: `oja_subspace_diag` median final loss `{low_diag:.6f}`, `low_mixed_curvature_basis` median final loss `{low_proposed:.6f}`, relative gain `{relative_gain:+.2%}`.",
        f"- Stronger-coupling check `alpha={high_alpha}`: `oja_subspace_diag` mean final loss `{summary[high_alpha]['oja_subspace_diag']['final_loss']['mean']:.6f}`, `low_mixed_curvature_basis` mean final loss `{summary[high_alpha]['low_mixed_curvature_basis']['final_loss']['mean']:.6f}`.",
        "",
        "## Interpretation",
        "",
        "The branch continues only if low-mixed-curvature basis wins in the low-coupling regime and that win coincides with lower off-diagonal curvature.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(findings_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    summary_path = output_dir / "summary.md"

    run_optimizer_geometry_suite(
        seeds=args.seeds,
        alphas=args.alphas,
        variants=VARIANTS,
        output_json=str(results_path),
        output_markdown=str(summary_path),
        dimension=args.dimension,
        rank=args.rank,
        steps=args.steps,
        basis_refresh=args.basis_refresh,
    )

    results = load_results(results_path)
    plot_metric(results, output_dir / "final_loss.png", "final_loss", "Optimizer geometry final loss", "final loss")
    plot_metric(results, output_dir / "loss_auc.png", "loss_auc", "Optimizer geometry loss AUC", "loss AUC")
    plot_metric(results, output_dir / "off_diagonal_curvature.png", "off_diagonal_curvature", "Off-diagonal curvature", "C_off")
    plot_metric(results, output_dir / "order_sensitivity.png", "order_sensitivity", "Order sensitivity", "order sensitivity")
    write_report(output_dir, Path(args.report_path), results)


if __name__ == "__main__":
    main()
