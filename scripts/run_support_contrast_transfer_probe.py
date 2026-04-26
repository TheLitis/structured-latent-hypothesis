from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from structured_latent_hypothesis.context_transfer import run_context_transfer_suite
from structured_latent_hypothesis.support_contrast import (
    build_support_contrast_rows,
    evaluate_support_contrast_transfer,
    score_correlations,
    summarize_binary_labels,
)
from structured_latent_hypothesis.transfer_criterion import cross_validate_transfer_decision_policy


SAFE_SCORE_KEYS = ["score_contrast", "score_gain_ratio_1", "score_gain_delta_1", "score_residual_normalized"]
BUDGET_SCORE_KEYS = ["score_contrast", "score_gain_ratio_8", "score_gain_delta_8", "score_instability"]
ALL_SCORE_KEYS = sorted(set(SAFE_SCORE_KEYS + BUDGET_SCORE_KEYS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run support-contrast transfer criterion benchmark.")
    parser.add_argument("--label", default="Support Contrast Transfer Probe v1")
    parser.add_argument("--output-dir", default="results/support_contrast_transfer_probe_v1")
    parser.add_argument("--report-path", default="reports/2026-04-27_support_contrast_transfer_probe_v1_findings.md")
    parser.add_argument("--followup-path", default="reports/2026-04-27_support_contrast_transfer_followup.md")
    parser.add_argument("--context-count", type=int, default=5)
    parser.add_argument("--state-count", type=int, default=10)
    parser.add_argument("--action-count", type=int, default=4)
    parser.add_argument("--rollout-length", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=18)
    parser.add_argument("--latent-dim", type=int, default=20)
    parser.add_argument("--context-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--adapt-steps", type=int, default=30)
    parser.add_argument("--adapt-lr", type=float, default=6e-2)
    parser.add_argument("--support-fraction", type=float, default=0.35)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 11])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.00, 0.20, 0.35, 0.75, 1.00])
    parser.add_argument("--synthetic-alphas", type=float, nargs="+", default=None)
    parser.add_argument("--semireal-alphas", type=float, nargs="+", default=None)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--regret-tolerance", type=float, default=1e-5)
    parser.add_argument("--structured-violation-cost", type=float, default=5.0)
    parser.add_argument("--fallback-overbudget-cost", type=float, default=3.0)
    parser.add_argument("--escalate-needed-cost", type=float, default=0.5)
    parser.add_argument("--escalate-unneeded-cost", type=float, default=1.0)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def world_name(prefix: str, family: str, alpha: float) -> str:
    return f"{prefix}_{family}_{alpha:0.2f}"


def build_variant_recipes(args: argparse.Namespace) -> dict[str, dict]:
    shared = {
        "adapt_steps": args.adapt_steps,
        "adapt_lr": args.adapt_lr,
        "support_fraction": args.support_fraction,
    }
    return {
        "full_transition": {"model_type": "full_transition", "lambda_residual": 0.0, **shared},
        "commuting_operator": {"model_type": "commuting_operator", "lambda_residual": 0.0, **shared},
        "operator_plus_residual": {
            "model_type": "operator_plus_residual",
            "interaction_rank": 4,
            "lambda_residual": 2e-3,
            **shared,
        },
        "operator_diag_residual": {
            "model_type": "operator_diag_residual",
            "interaction_rank": 4,
            "lambda_residual": 2e-3,
            **shared,
        },
    }


def has_support_curves(path: Path) -> bool:
    if not path.exists():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    return bool(data.get("runs")) and "support_curve" in data["runs"][0].get("adaptation", {})


def run_or_load_suite(args: argparse.Namespace, output_dir: Path, label: str, worlds: list[str]) -> dict:
    results_path = output_dir / f"{label}_results.json"
    summary_path = output_dir / f"{label}_summary.md"
    if args.reuse_existing and has_support_curves(results_path):
        return json.loads(results_path.read_text(encoding="utf-8"))
    return run_context_transfer_suite(
        seeds=args.seeds,
        worlds=worlds,
        variants=list(build_variant_recipes(args).keys()),
        variant_recipes=build_variant_recipes(args),
        output_json=str(results_path),
        output_markdown=str(summary_path),
        context_count=args.context_count,
        state_count=args.state_count,
        action_count=args.action_count,
        rollout_length=args.rollout_length,
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        context_dim=args.context_dim,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        evaluate_adaptation=True,
    )


def compact_policy(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key != "per_row"}


def semireal_cv(rows: list[dict], args: argparse.Namespace) -> dict:
    output = {}
    for group in ("seed", "world", "family"):
        metrics = cross_validate_transfer_decision_policy(
            rows,
            group_key=group,
            safe_score_keys=SAFE_SCORE_KEYS,
            budget_score_keys=BUDGET_SCORE_KEYS,
            safe_label_key="task_safe",
            budget_label_key="task_budget",
            structured_violation_cost=args.structured_violation_cost,
            fallback_overbudget_cost=args.fallback_overbudget_cost,
            escalate_needed_cost=args.escalate_needed_cost,
            escalate_unneeded_cost=args.escalate_unneeded_cost,
        )
        best_trivial = min(
            metrics["always_structured_cost_mean"],
            metrics["always_fallback_cost_mean"],
            metrics["always_escalate_cost_mean"],
        )
        output[group] = {
            "metrics": {key: value for key, value in metrics.items() if key != "per_group"},
            "best_trivial_cost": best_trivial,
            "delta_to_best_trivial": metrics["average_cost_mean"] - best_trivial,
            "wins": metrics["average_cost_mean"] < best_trivial - 1e-12,
        }
    return output


def plot_score_scatter(results: dict, output_path: Path) -> None:
    figure, axis = plt.subplots(1, 1, figsize=(7.8, 5.2))
    for label, color in (("synthetic", "#5078a0"), ("semireal", "#b15f4a")):
        rows = results[f"{label}_rows"]
        axis.scatter(
            [row["score_contrast"] for row in rows],
            [row["structured_query_regret"] for row in rows],
            s=42,
            alpha=0.78,
            label=label,
            color=color,
        )
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.set_xlabel("support contrast score")
    axis.set_ylabel("structured query regret")
    axis.set_title("Support contrast vs held-out regret")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_outputs(output_dir: Path, report_path: Path, followup_path: Path, results: dict) -> None:
    external = results["external_policy"]
    cv = results["semireal_cv"]
    summary_lines = [
        f"# {results['label']}",
        "",
        "## Synthetic-Trained External Policy",
        "",
        f"Policy cost: `{external['metrics']['average_cost']:.6f}`",
        f"Best trivial cost: `{external['best_trivial_cost']:.6f}`",
        f"Delta: `{external['delta_to_best_trivial']:+.6f}`",
        f"Wins best trivial: `{external['wins_best_trivial']}`",
        "",
        "## Semi-Real In-Domain CV",
        "",
        "| Group | Cost | Best Trivial | Delta | Wins |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for group, row in cv.items():
        summary_lines.append(
            "| "
            + group
            + f" | {row['metrics']['average_cost_mean']:.6f}"
            + f" | {row['best_trivial_cost']:.6f}"
            + f" | {row['delta_to_best_trivial']:+.6f}"
            + f" | {row['wins']} |"
        )
    summary_lines.extend(
        [
            "",
            "## Score Correlations",
            "",
            "| Domain | Score | Regret Spearman | Fallback Steps Spearman | Structured Steps Spearman |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for domain, correlations in (
        ("synthetic", results["synthetic_correlations"]),
        ("semireal", results["semireal_correlations"]),
    ):
        for score, values in correlations.items():
            summary_lines.append(
                "| "
                + domain
                + " | "
                + score
                + f" | {values['structured_query_regret']:+.3f}"
                + f" | {values['fallback_adaptation_steps']:+.3f}"
                + f" | {values['structured_adaptation_steps']:+.3f} |"
            )
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(
        f"# {results['label']}\n\n![Score scatter](support_contrast_scatter.png)\n",
        encoding="utf-8",
    )

    semireal_wins = sum(int(row["wins"]) for row in cv.values())
    findings = [
        f"# {results['label']}",
        "",
        "## Result",
        "",
        f"- External synthetic-to-semi-real policy cost `{external['metrics']['average_cost']:.6f}` vs best trivial `{external['best_trivial_cost']:.6f}`; delta `{external['delta_to_best_trivial']:+.6f}`.",
        f"- Semi-real in-domain CV wins: `{semireal_wins}/3`.",
        f"- Synthetic label rates: safe `{results['synthetic_label_summary']['safe_rate']:.3f}`, budget `{results['synthetic_label_summary']['budget_rate']:.3f}`.",
        f"- Semi-real label rates: safe `{results['semireal_label_summary']['safe_rate']:.3f}`, budget `{results['semireal_label_summary']['budget_rate']:.3f}`.",
        "",
        "## Interpretation",
        "",
        "This probe checks whether support-set contrast fixes the representation-scale failure of raw residual scores. The result should be read against the prior semi-real context failure: a win here would mean small support adaptation gives a more portable transfer criterion.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(findings) + "\n", encoding="utf-8")

    followup = [
        f"# {results['label']} Follow-Up",
        "",
        "## Main Result",
        "",
        f"The support-contrast criterion {'beats' if external['wins_best_trivial'] else 'does not beat'} the best trivial baseline when trained on synthetic context-transfer and tested on semi-real context-transfer.",
        "",
        "## Interpretation",
        "",
        "If this fails while in-domain semi-real CV succeeds, the issue is domain transfer. If both fail, support contrast is not enough and the next step must change the probing protocol or target label.",
    ]
    followup_path.parent.mkdir(parents=True, exist_ok=True)
    followup_path.write_text("\n".join(followup) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic_alphas = args.synthetic_alphas if args.synthetic_alphas is not None else args.alphas
    semireal_alphas = args.semireal_alphas if args.semireal_alphas is not None else args.alphas
    synthetic_worlds = [world_name("context", family, value) for family in ("commuting", "coupled") for value in synthetic_alphas]
    semireal_worlds = [
        world_name("semireal_context", family, value) for family in ("commuting", "coupled") for value in semireal_alphas
    ]
    synthetic_results = run_or_load_suite(args, output_dir, "synthetic", synthetic_worlds)
    semireal_results = run_or_load_suite(args, output_dir, "semireal", semireal_worlds)
    synthetic_rows = build_support_contrast_rows(
        synthetic_results,
        regret_tolerance=args.regret_tolerance,
        budget=args.budget,
    )
    semireal_rows = build_support_contrast_rows(
        semireal_results,
        regret_tolerance=args.regret_tolerance,
        budget=args.budget,
    )
    external = evaluate_support_contrast_transfer(
        synthetic_rows,
        semireal_rows,
        safe_score_keys=SAFE_SCORE_KEYS,
        budget_score_keys=BUDGET_SCORE_KEYS,
        structured_violation_cost=args.structured_violation_cost,
        fallback_overbudget_cost=args.fallback_overbudget_cost,
        escalate_needed_cost=args.escalate_needed_cost,
        escalate_unneeded_cost=args.escalate_unneeded_cost,
    )
    results = {
        "label": args.label,
        "config": vars(args),
        "safe_score_keys": SAFE_SCORE_KEYS,
        "budget_score_keys": BUDGET_SCORE_KEYS,
        "synthetic_rows": synthetic_rows,
        "semireal_rows": semireal_rows,
        "synthetic_label_summary": summarize_binary_labels(synthetic_rows),
        "semireal_label_summary": summarize_binary_labels(semireal_rows),
        "synthetic_correlations": score_correlations(synthetic_rows, ALL_SCORE_KEYS),
        "semireal_correlations": score_correlations(semireal_rows, ALL_SCORE_KEYS),
        "external_policy": external,
        "semireal_cv": semireal_cv(semireal_rows, args),
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_score_scatter(results, output_dir / "support_contrast_scatter.png")
    write_outputs(output_dir, Path(args.report_path), Path(args.followup_path), results)


if __name__ == "__main__":
    main()
