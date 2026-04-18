from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import torch

from .synthetic import set_seed


@dataclass
class OptimizerGeometryConfig:
    alpha: float
    variant: str
    seed: int
    dimension: int = 128
    rank: int = 8
    steps: int = 300
    basis_refresh: int = 10
    lr_full: float = 4e-2
    lr_subspace: float = 0.22
    oja_lr: float = 0.18
    lambda_perp: float = 24.0


@dataclass
class QuadraticObjective:
    q_matrix: torch.Tensor
    active_basis: torch.Tensor
    coupling_matrix: torch.Tensor
    theta0: torch.Tensor
    eigenvalues: torch.Tensor


def random_orthonormal(rows: int, cols: int) -> torch.Tensor:
    matrix = torch.randn(rows, cols)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q[:, :cols]


def make_quadratic_objective(config: OptimizerGeometryConfig) -> QuadraticObjective:
    set_seed(config.seed)
    basis = random_orthonormal(config.dimension, config.rank)
    spectrum = torch.linspace(1.0, 16.0, config.rank)
    coupling = torch.randn(config.rank, config.rank)
    coupling = 0.5 * (coupling + coupling.T)
    coupling.fill_diagonal_(0.0)
    coupling = coupling / coupling.norm().clamp_min(1e-8)
    active_hessian = torch.diag(spectrum) + float(config.alpha) * coupling
    projector = basis @ basis.T
    q_matrix = basis @ active_hessian @ basis.T + float(config.lambda_perp) * (torch.eye(config.dimension) - projector)
    theta0 = torch.randn(config.dimension) / (config.dimension**0.5)
    return QuadraticObjective(
        q_matrix=q_matrix,
        active_basis=basis,
        coupling_matrix=coupling,
        theta0=theta0,
        eigenvalues=spectrum,
    )


def loss_value(q_matrix: torch.Tensor, theta: torch.Tensor) -> float:
    return float(0.5 * theta @ (q_matrix @ theta))


def gradient(q_matrix: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    return q_matrix @ theta


def off_diagonal_curvature_score(q_matrix: torch.Tensor, basis: torch.Tensor) -> float:
    reduced = basis.T @ q_matrix @ basis
    off_diag = reduced - torch.diag(torch.diagonal(reduced))
    denom = torch.linalg.matrix_norm(reduced).item()
    if denom < 1e-12:
        return 0.0
    return float((torch.linalg.matrix_norm(off_diag).item() ** 2) / (denom**2))


def order_sensitivity(q_matrix: torch.Tensor, basis: torch.Tensor, theta: torch.Tensor) -> float:
    reduced = basis.T @ q_matrix @ basis
    coordinates = basis.T @ theta
    gradient0 = reduced @ coordinates
    diagonal = torch.diagonal(reduced).clamp_min(1e-6)

    def sweep(order: range) -> torch.Tensor:
        coord = coordinates.clone()
        grad = gradient0.clone()
        for index in order:
            step = grad[index] / diagonal[index]
            coord[index] = coord[index] - step
            grad = reduced @ coord
        return basis @ coord

    forward = sweep(range(reduced.shape[0]))
    reverse = sweep(range(reduced.shape[0] - 1, -1, -1))
    return float(torch.linalg.vector_norm(forward - reverse).item())


def oja_update(basis: torch.Tensor, grad_vector: torch.Tensor, lr: float) -> torch.Tensor:
    updated = basis + float(lr) * torch.outer(grad_vector, grad_vector @ basis)
    q, _ = torch.linalg.qr(updated, mode="reduced")
    return q[:, : basis.shape[1]]


def subspace_step(
    theta: torch.Tensor,
    q_matrix: torch.Tensor,
    basis: torch.Tensor,
    lr: float,
    full_preconditioner: bool,
) -> torch.Tensor:
    grad = gradient(q_matrix, theta)
    reduced = basis.T @ q_matrix @ basis
    reduced_grad = basis.T @ grad
    if full_preconditioner:
        step = torch.linalg.solve(reduced + 1e-4 * torch.eye(reduced.shape[0]), reduced_grad)
    else:
        diag = torch.diagonal(reduced).clamp_min(1e-4)
        step = reduced_grad / diag
    return theta - float(lr) * (basis @ step)


def run_full_adam(objective: QuadraticObjective, config: OptimizerGeometryConfig) -> dict[str, Any]:
    theta = objective.theta0.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([theta], lr=config.lr_full)
    losses = []
    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = 0.5 * theta @ (objective.q_matrix @ theta)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    basis = torch.eye(config.dimension)[:, : config.rank]
    return {
        "losses": losses,
        "final_theta": theta.detach(),
        "basis": basis,
        "off_diagonal_curvature": off_diagonal_curvature_score(objective.q_matrix, basis),
        "order_sensitivity": order_sensitivity(objective.q_matrix, basis, objective.theta0),
    }


def run_random_subspace_diag(objective: QuadraticObjective, config: OptimizerGeometryConfig) -> dict[str, Any]:
    set_seed(config.seed + 1000)
    basis = random_orthonormal(config.dimension, config.rank)
    theta = objective.theta0.clone()
    losses = []
    for _ in range(config.steps):
        losses.append(loss_value(objective.q_matrix, theta))
        theta = subspace_step(theta, objective.q_matrix, basis, config.lr_subspace, full_preconditioner=False)
    return {
        "losses": losses,
        "final_theta": theta,
        "basis": basis,
        "off_diagonal_curvature": off_diagonal_curvature_score(objective.q_matrix, basis),
        "order_sensitivity": order_sensitivity(objective.q_matrix, basis, objective.theta0),
    }


def run_oja_subspace(objective: QuadraticObjective, config: OptimizerGeometryConfig, full_preconditioner: bool) -> dict[str, Any]:
    set_seed(config.seed + 2000 + (1 if full_preconditioner else 0))
    basis = random_orthonormal(config.dimension, config.rank)
    theta = objective.theta0.clone()
    losses = []
    for step in range(config.steps):
        losses.append(loss_value(objective.q_matrix, theta))
        if step % config.basis_refresh == 0:
            basis = oja_update(basis, gradient(objective.q_matrix, theta), config.oja_lr)
        theta = subspace_step(theta, objective.q_matrix, basis, config.lr_subspace, full_preconditioner=full_preconditioner)
    return {
        "losses": losses,
        "final_theta": theta,
        "basis": basis,
        "off_diagonal_curvature": off_diagonal_curvature_score(objective.q_matrix, basis),
        "order_sensitivity": order_sensitivity(objective.q_matrix, basis, objective.theta0),
    }


def run_low_mixed_curvature_basis(objective: QuadraticObjective, config: OptimizerGeometryConfig) -> dict[str, Any]:
    set_seed(config.seed + 3000)
    basis = random_orthonormal(config.dimension, config.rank)
    theta = objective.theta0.clone()
    losses = []
    for step in range(config.steps):
        losses.append(loss_value(objective.q_matrix, theta))
        if step % config.basis_refresh == 0:
            basis = oja_update(basis, gradient(objective.q_matrix, theta), config.oja_lr)
            reduced = basis.T @ objective.q_matrix @ basis
            _, rotation = torch.linalg.eigh(reduced)
            basis = basis @ rotation
        theta = subspace_step(theta, objective.q_matrix, basis, config.lr_subspace, full_preconditioner=False)
    return {
        "losses": losses,
        "final_theta": theta,
        "basis": basis,
        "off_diagonal_curvature": off_diagonal_curvature_score(objective.q_matrix, basis),
        "order_sensitivity": order_sensitivity(objective.q_matrix, basis, objective.theta0),
    }


def auc(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def run_optimizer_geometry_once(config: OptimizerGeometryConfig) -> dict[str, Any]:
    objective = make_quadratic_objective(config)
    if config.variant == "adam_full":
        result = run_full_adam(objective, config)
    elif config.variant == "random_subspace_diag":
        result = run_random_subspace_diag(objective, config)
    elif config.variant == "oja_subspace_diag":
        result = run_oja_subspace(objective, config, full_preconditioner=False)
    elif config.variant == "oja_subspace_full":
        result = run_oja_subspace(objective, config, full_preconditioner=True)
    elif config.variant == "low_mixed_curvature_basis":
        result = run_low_mixed_curvature_basis(objective, config)
    else:
        raise ValueError(f"Unsupported optimizer variant: {config.variant}")
    return {
        "config": asdict(config),
        "final_loss": loss_value(objective.q_matrix, result["final_theta"]),
        "loss_auc": auc(result["losses"]),
        "off_diagonal_curvature": result["off_diagonal_curvature"],
        "order_sensitivity": result["order_sensitivity"],
        "loss_curve": result["losses"],
    }


def aggregate_optimizer_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ["final_loss", "loss_auc", "off_diagonal_curvature", "order_sensitivity"]
    summary: dict[str, Any] = {}
    for metric_name in metrics:
        values = [float(run[metric_name]) for run in runs]
        summary[metric_name] = {
            "mean": mean(values),
            "median": median(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
        }
    return summary


def render_optimizer_markdown_report(results: dict[str, Any]) -> str:
    lines = [
        "# Optimizer Geometry Benchmark",
        "",
        f"Steps: `{results['steps']}`",
        f"Rank: `{results['rank']}`",
        "",
    ]
    for alpha, variants in results["summary"].items():
        lines.append(f"## Alpha: {alpha}")
        lines.append("")
        lines.append("| Variant | Final Loss | Loss AUC | C_off | Order Sensitivity |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for variant, metrics in variants.items():
            lines.append(
                f"| {variant} | {metrics['final_loss']['mean']:.6f} +/- {metrics['final_loss']['std']:.6f} | "
                f"{metrics['loss_auc']['mean']:.6f} +/- {metrics['loss_auc']['std']:.6f} | "
                f"{metrics['off_diagonal_curvature']['mean']:.6f} +/- {metrics['off_diagonal_curvature']['std']:.6f} | "
                f"{metrics['order_sensitivity']['mean']:.6f} +/- {metrics['order_sensitivity']['std']:.6f} |"
            )
        lines.append("")
    return "\n".join(lines)


def run_optimizer_geometry_suite(
    seeds: list[int],
    alphas: list[float],
    variants: list[str],
    output_json: str | None = None,
    output_markdown: str | None = None,
    dimension: int = 128,
    rank: int = 8,
    steps: int = 300,
    basis_refresh: int = 10,
) -> dict[str, Any]:
    raw_runs: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for alpha in alphas:
        alpha_key = f"{float(alpha):0.2f}"
        summary[alpha_key] = {}
        for variant in variants:
            variant_runs = []
            for seed in seeds:
                config = OptimizerGeometryConfig(
                    alpha=float(alpha),
                    variant=variant,
                    seed=seed,
                    dimension=dimension,
                    rank=rank,
                    steps=steps,
                    basis_refresh=basis_refresh,
                )
                run = run_optimizer_geometry_once(config)
                raw_runs.append(run)
                variant_runs.append(run)
            summary[alpha_key][variant] = aggregate_optimizer_runs(variant_runs)

    output = {
        "seeds": seeds,
        "alphas": alphas,
        "variants": variants,
        "dimension": dimension,
        "rank": rank,
        "steps": steps,
        "basis_refresh": basis_refresh,
        "summary": summary,
        "runs": raw_runs,
    }

    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    if output_markdown:
        path = Path(output_markdown)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_optimizer_markdown_report(output), encoding="utf-8")

    return output
