from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
from torch import nn

from .synthetic import (
    generate_world,
    ground_truth_commutator_magnitude,
    ground_truth_coupling_strength,
    ground_truth_step_drift_magnitude,
    sample_nested_train_mask,
    sample_train_mask,
    set_seed,
)


@dataclass
class DirectBenchmarkConfig:
    world: str
    variant: str
    seed: int
    model_type: str
    split_strategy: str = "cartesian_blocks"
    grid_size: int = 8
    image_size: int = 20
    latent_dim: int = 8
    hidden_dim: int = 96
    train_fraction: float = 0.78
    epochs: int = 280
    lr: float = 3e-3
    lambda_residual: float = 0.0
    interaction_rank: int = 0


class SharedDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.net(latents)


class CoordLatentDecoder(nn.Module):
    def __init__(self, grid_size: int, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        axis = torch.linspace(-1.0, 1.0, steps=grid_size)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("coords", torch.stack([yy, xx], dim=-1))
        self.coord_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = SharedDecoder(latent_dim, hidden_dim, output_dim)

    def latent_grid(self) -> torch.Tensor:
        coords = self.coords.reshape(-1, 2)
        return self.coord_net(coords).reshape(self.coords.shape[0], self.coords.shape[1], -1)

    def residual_grid(self) -> torch.Tensor:
        latent = self.latent_grid()
        return torch.zeros_like(latent)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latents = self.latent_grid()
        recon = self.decoder(latents.reshape(-1, latents.shape[-1])).reshape(latents.shape[0], latents.shape[1], -1)
        return recon, latents, self.residual_grid()


class CellLatentDecoder(nn.Module):
    def __init__(self, grid_size: int, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.cell = nn.Parameter(torch.empty(grid_size, grid_size, latent_dim))
        nn.init.normal_(self.cell, mean=0.0, std=0.02)
        self.decoder = SharedDecoder(latent_dim, hidden_dim, output_dim)

    def latent_grid(self) -> torch.Tensor:
        return self.cell

    def residual_grid(self) -> torch.Tensor:
        return torch.zeros_like(self.cell)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latents = self.latent_grid()
        recon = self.decoder(latents.reshape(-1, latents.shape[-1])).reshape(latents.shape[0], latents.shape[1], -1)
        return recon, latents, self.residual_grid()


class AdditiveLatentDecoder(nn.Module):
    def __init__(self, grid_size: int, latent_dim: int, hidden_dim: int, output_dim: int, use_residual: bool) -> None:
        super().__init__()
        self.row = nn.Parameter(torch.empty(grid_size, latent_dim))
        self.col = nn.Parameter(torch.empty(grid_size, latent_dim))
        nn.init.normal_(self.row, mean=0.0, std=0.02)
        nn.init.normal_(self.col, mean=0.0, std=0.02)
        self.residual = nn.Parameter(torch.empty(grid_size, grid_size, latent_dim)) if use_residual else None
        if self.residual is not None:
            nn.init.normal_(self.residual, mean=0.0, std=0.02)
        self.decoder = SharedDecoder(latent_dim, hidden_dim, output_dim)

    def latent_grid(self) -> torch.Tensor:
        base = self.row[:, None, :] + self.col[None, :, :]
        if self.residual is None:
            return base
        return base + self.residual

    def residual_grid(self) -> torch.Tensor:
        if self.residual is None:
            return torch.zeros(self.row.shape[0], self.col.shape[0], self.row.shape[1], device=self.row.device)
        return self.residual

    def forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latents = self.latent_grid()
        recon = self.decoder(latents.reshape(-1, latents.shape[-1])).reshape(latents.shape[0], latents.shape[1], -1)
        return recon, latents, self.residual_grid()


class AdditiveLowRankDecoder(nn.Module):
    def __init__(self, grid_size: int, latent_dim: int, hidden_dim: int, output_dim: int, interaction_rank: int) -> None:
        super().__init__()
        if interaction_rank <= 0:
            raise ValueError("interaction_rank must be positive for AdditiveLowRankDecoder.")
        self.row = nn.Parameter(torch.empty(grid_size, latent_dim))
        self.col = nn.Parameter(torch.empty(grid_size, latent_dim))
        self.row_scores = nn.Parameter(torch.empty(grid_size, interaction_rank))
        self.col_scores = nn.Parameter(torch.empty(grid_size, interaction_rank))
        self.basis = nn.Parameter(torch.empty(interaction_rank, latent_dim))
        nn.init.normal_(self.row, mean=0.0, std=0.02)
        nn.init.normal_(self.col, mean=0.0, std=0.02)
        nn.init.normal_(self.row_scores, mean=0.0, std=0.02)
        nn.init.normal_(self.col_scores, mean=0.0, std=0.02)
        nn.init.normal_(self.basis, mean=0.0, std=0.02)
        self.decoder = SharedDecoder(latent_dim, hidden_dim, output_dim)

    def low_rank_residual(self) -> torch.Tensor:
        row_scores = self.row_scores - self.row_scores.mean(dim=0, keepdim=True)
        col_scores = self.col_scores - self.col_scores.mean(dim=0, keepdim=True)
        return torch.einsum("ir,jr,rd->ijd", row_scores, col_scores, self.basis)

    def latent_grid(self) -> torch.Tensor:
        base = self.row[:, None, :] + self.col[None, :, :]
        return base + self.low_rank_residual()

    def residual_grid(self) -> torch.Tensor:
        return self.low_rank_residual()

    def forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latents = self.latent_grid()
        recon = self.decoder(latents.reshape(-1, latents.shape[-1])).reshape(latents.shape[0], latents.shape[1], -1)
        return recon, latents, self.residual_grid()


class AdditiveInteractionMLPDecoder(nn.Module):
    def __init__(self, grid_size: int, latent_dim: int, hidden_dim: int, output_dim: int, interaction_rank: int) -> None:
        super().__init__()
        if interaction_rank <= 0:
            raise ValueError("interaction_rank must be positive for AdditiveInteractionMLPDecoder.")
        self.row = nn.Parameter(torch.empty(grid_size, latent_dim))
        self.col = nn.Parameter(torch.empty(grid_size, latent_dim))
        self.row_interaction = nn.Parameter(torch.empty(grid_size, interaction_rank))
        self.col_interaction = nn.Parameter(torch.empty(grid_size, interaction_rank))
        nn.init.normal_(self.row, mean=0.0, std=0.02)
        nn.init.normal_(self.col, mean=0.0, std=0.02)
        nn.init.normal_(self.row_interaction, mean=0.0, std=0.02)
        nn.init.normal_(self.col_interaction, mean=0.0, std=0.02)
        interaction_hidden = max(16, hidden_dim // 2)
        self.interaction_net = nn.Sequential(
            nn.Linear(interaction_rank * 2, interaction_hidden),
            nn.ReLU(),
            nn.Linear(interaction_hidden, latent_dim),
        )
        self.decoder = SharedDecoder(latent_dim, hidden_dim, output_dim)

    def interaction_residual(self) -> torch.Tensor:
        row_codes = self.row_interaction[:, None, :].expand(-1, self.col_interaction.shape[0], -1)
        col_codes = self.col_interaction[None, :, :].expand(self.row_interaction.shape[0], -1, -1)
        raw = self.interaction_net(torch.cat([row_codes, col_codes], dim=-1))
        row_mean = raw.mean(dim=1, keepdim=True)
        col_mean = raw.mean(dim=0, keepdim=True)
        grand_mean = raw.mean(dim=(0, 1), keepdim=True)
        return raw - row_mean - col_mean + grand_mean

    def latent_grid(self) -> torch.Tensor:
        base = self.row[:, None, :] + self.col[None, :, :]
        return base + self.interaction_residual()

    def residual_grid(self) -> torch.Tensor:
        return self.interaction_residual()

    def forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latents = self.latent_grid()
        recon = self.decoder(latents.reshape(-1, latents.shape[-1])).reshape(latents.shape[0], latents.shape[1], -1)
        return recon, latents, self.residual_grid()


class AdditiveOperatorDecoder(nn.Module):
    def __init__(self, grid_size: int, latent_dim: int, hidden_dim: int, output_dim: int, interaction_rank: int) -> None:
        super().__init__()
        if interaction_rank <= 0:
            raise ValueError("interaction_rank must be positive for AdditiveOperatorDecoder.")
        self.row = nn.Parameter(torch.empty(grid_size, latent_dim))
        self.col = nn.Parameter(torch.empty(grid_size, latent_dim))
        self.col_scores = nn.Parameter(torch.empty(grid_size, interaction_rank))
        self.operator_basis = nn.Parameter(torch.empty(interaction_rank, latent_dim, latent_dim))
        nn.init.normal_(self.row, mean=0.0, std=0.02)
        nn.init.normal_(self.col, mean=0.0, std=0.02)
        nn.init.normal_(self.col_scores, mean=0.0, std=0.02)
        nn.init.normal_(self.operator_basis, mean=0.0, std=0.02)
        self.decoder = SharedDecoder(latent_dim, hidden_dim, output_dim)

    def operator_residual(self) -> torch.Tensor:
        row_centered = self.row - self.row.mean(dim=0, keepdim=True)
        col_scores_centered = self.col_scores - self.col_scores.mean(dim=0, keepdim=True)
        return torch.einsum("jq,qde,ie->ijd", col_scores_centered, self.operator_basis, row_centered)

    def latent_grid(self) -> torch.Tensor:
        base = self.row[:, None, :] + self.col[None, :, :]
        return base + self.operator_residual()

    def residual_grid(self) -> torch.Tensor:
        return self.operator_residual()

    def forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latents = self.latent_grid()
        recon = self.decoder(latents.reshape(-1, latents.shape[-1])).reshape(latents.shape[0], latents.shape[1], -1)
        return recon, latents, self.residual_grid()


class AdditiveOperatorDiagDecoder(nn.Module):
    def __init__(self, grid_size: int, latent_dim: int, hidden_dim: int, output_dim: int, interaction_rank: int) -> None:
        super().__init__()
        if interaction_rank <= 0:
            raise ValueError("interaction_rank must be positive for AdditiveOperatorDiagDecoder.")
        self.row = nn.Parameter(torch.empty(grid_size, latent_dim))
        self.col = nn.Parameter(torch.empty(grid_size, latent_dim))
        self.col_scores = nn.Parameter(torch.empty(grid_size, interaction_rank))
        self.diag_basis = nn.Parameter(torch.empty(interaction_rank, latent_dim))
        nn.init.normal_(self.row, mean=0.0, std=0.02)
        nn.init.normal_(self.col, mean=0.0, std=0.02)
        nn.init.normal_(self.col_scores, mean=0.0, std=0.02)
        nn.init.normal_(self.diag_basis, mean=0.0, std=0.02)
        self.decoder = SharedDecoder(latent_dim, hidden_dim, output_dim)

    def operator_residual(self) -> torch.Tensor:
        row_centered = self.row - self.row.mean(dim=0, keepdim=True)
        col_scores_centered = self.col_scores - self.col_scores.mean(dim=0, keepdim=True)
        return torch.einsum("jq,qd,id->ijd", col_scores_centered, self.diag_basis, row_centered)

    def latent_grid(self) -> torch.Tensor:
        base = self.row[:, None, :] + self.col[None, :, :]
        return base + self.operator_residual()

    def residual_grid(self) -> torch.Tensor:
        return self.operator_residual()

    def forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latents = self.latent_grid()
        recon = self.decoder(latents.reshape(-1, latents.shape[-1])).reshape(latents.shape[0], latents.shape[1], -1)
        return recon, latents, self.residual_grid()


def build_model(config: DirectBenchmarkConfig, output_dim: int) -> nn.Module:
    if config.model_type == "coord":
        return CoordLatentDecoder(config.grid_size, config.latent_dim, config.hidden_dim, output_dim)
    if config.model_type == "cell":
        return CellLatentDecoder(config.grid_size, config.latent_dim, config.hidden_dim, output_dim)
    if config.model_type == "additive":
        return AdditiveLatentDecoder(config.grid_size, config.latent_dim, config.hidden_dim, output_dim, use_residual=False)
    if config.model_type == "additive_residual":
        return AdditiveLatentDecoder(config.grid_size, config.latent_dim, config.hidden_dim, output_dim, use_residual=True)
    if config.model_type == "additive_low_rank":
        return AdditiveLowRankDecoder(
            config.grid_size,
            config.latent_dim,
            config.hidden_dim,
            output_dim,
            interaction_rank=config.interaction_rank,
        )
    if config.model_type == "additive_interaction_mlp":
        return AdditiveInteractionMLPDecoder(
            config.grid_size,
            config.latent_dim,
            config.hidden_dim,
            output_dim,
            interaction_rank=config.interaction_rank,
        )
    if config.model_type == "additive_operator":
        return AdditiveOperatorDecoder(
            config.grid_size,
            config.latent_dim,
            config.hidden_dim,
            output_dim,
            interaction_rank=config.interaction_rank,
        )
    if config.model_type == "additive_operator_diag":
        return AdditiveOperatorDiagDecoder(
            config.grid_size,
            config.latent_dim,
            config.hidden_dim,
            output_dim,
            interaction_rank=config.interaction_rank,
        )
    raise ValueError(f"Unsupported model_type: {config.model_type}")


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(nn.functional.mse_loss(pred, target).item())


def train_one_direct(
    config: DirectBenchmarkConfig,
    train_mask_override: torch.Tensor | None = None,
    eval_mask_override: torch.Tensor | None = None,
) -> dict[str, Any]:
    set_seed(config.seed)
    images = generate_world(config.world, config.grid_size, config.image_size)
    mask = (
        train_mask_override.clone().to(torch.bool)
        if train_mask_override is not None
        else sample_train_mask(config.grid_size, config.train_fraction, config.seed, config.split_strategy)
    )
    eval_mask = eval_mask_override.clone().to(torch.bool) if eval_mask_override is not None else ~mask

    flat = images.reshape(config.grid_size, config.grid_size, -1)
    train_idx = mask.reshape(-1)
    test_idx = eval_mask.reshape(-1)

    model = build_model(config, flat.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    last_losses: dict[str, float] = {}
    for epoch in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        recon_grid, latent_grid, residual_grid = model()
        recon_flat = recon_grid.reshape(-1, recon_grid.shape[-1])
        target_flat = flat.reshape(-1, flat.shape[-1])

        recon_loss = nn.functional.mse_loss(recon_flat[train_idx], target_flat[train_idx])
        residual_loss = residual_grid.pow(2).mean()
        total_loss = recon_loss + config.lambda_residual * residual_loss
        total_loss.backward()
        optimizer.step()

        if epoch == config.epochs - 1:
            last_losses = {
                "recon_loss": float(recon_loss.item()),
                "residual_loss": float(residual_loss.item()),
                "total_loss": float(total_loss.item()),
            }

    with torch.no_grad():
        recon_grid, latent_grid, residual_grid = model()
        recon_flat = recon_grid.reshape(-1, recon_grid.shape[-1])
        target_flat = flat.reshape(-1, flat.shape[-1])
        residual_norm_grid = residual_grid.pow(2).mean(dim=-1)

        result = {
            "config": asdict(config),
            "train_points": int(train_idx.sum().item()),
            "test_points": int(test_idx.sum().item()),
            "train_recon_mse": mse(recon_flat[train_idx], target_flat[train_idx]),
            "test_recon_mse": mse(recon_flat[test_idx], target_flat[test_idx]),
            "generalization_gap": mse(recon_flat[test_idx], target_flat[test_idx]) - mse(recon_flat[train_idx], target_flat[train_idx]),
            "latent_norm_mean": float(latent_grid.pow(2).mean(dim=-1).sqrt().mean().item()),
            "residual_norm_train": float(residual_norm_grid[mask].mean().item()),
            "residual_norm_holdout": float(residual_norm_grid[eval_mask].mean().item()),
            "residual_norm_all": float(residual_norm_grid.mean().item()),
            "train_mask": mask.to(torch.int64).tolist(),
            "eval_mask": eval_mask.to(torch.int64).tolist(),
            "recon_error_grid": ((recon_grid - flat).pow(2).mean(dim=-1)).tolist(),
            "residual_norm_grid": residual_norm_grid.tolist(),
            "final_losses": last_losses,
        }
    return result


def train_direct_with_nested_selection(
    config: DirectBenchmarkConfig,
    candidate_recipes: dict[str, dict[str, Any]],
    selection_metric: str = "test_recon_mse",
    inner_train_fraction: float = 0.72,
) -> dict[str, Any]:
    outer_train_mask = sample_train_mask(config.grid_size, config.train_fraction, config.seed, config.split_strategy)
    inner_train_mask = sample_nested_train_mask(outer_train_mask, config.seed, keep_fraction=inner_train_fraction)
    inner_eval_mask = outer_train_mask & ~inner_train_mask

    if not torch.any(inner_eval_mask):
        raise RuntimeError("Nested selection produced an empty inner validation set.")

    selection_scores = []
    best_name = ""
    best_recipe: dict[str, Any] = {}
    best_score = float("inf")

    for candidate_name, candidate_recipe in candidate_recipes.items():
        candidate_config = replace(config, variant=candidate_name, **candidate_recipe)
        selection_run = train_one_direct(
            candidate_config,
            train_mask_override=inner_train_mask,
            eval_mask_override=inner_eval_mask,
        )
        score = float(selection_run[selection_metric])
        selection_scores.append(
            {
                "candidate": candidate_name,
                "recipe": candidate_recipe,
                "selection_metric": selection_metric,
                "score": score,
            }
        )
        if score < best_score - 1e-12:
            best_name = candidate_name
            best_recipe = dict(candidate_recipe)
            best_score = score

    final_config = replace(config, **best_recipe)
    final_run = train_one_direct(final_config, train_mask_override=outer_train_mask)
    realized_fraction = float(inner_train_mask.sum().item()) / float(max(1, outer_train_mask.sum().item()))
    final_run["selection"] = {
        "mode": "nested",
        "metric": selection_metric,
        "inner_train_fraction": inner_train_fraction,
        "realized_inner_train_fraction": realized_fraction,
        "chosen_candidate": best_name,
        "chosen_recipe": best_recipe,
        "scores": selection_scores,
    }
    return final_run


def aggregate_direct_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "train_recon_mse",
        "test_recon_mse",
        "generalization_gap",
        "latent_norm_mean",
        "residual_norm_train",
        "residual_norm_holdout",
        "residual_norm_all",
    ]
    summary: dict[str, Any] = {}
    for metric in metrics:
        values = [float(run[metric]) for run in runs]
        summary[metric] = {
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
        }
    return summary


def render_direct_markdown_report(results: dict[str, Any]) -> str:
    lines = [
        "# Direct Separable Benchmark",
        "",
        f"Split strategy: `{results['split_strategy']}`",
        "",
    ]
    for world, variants in results["summary"].items():
        lines.append(f"## World: {world}")
        lines.append("")
        lines.append("| Variant | Test Recon MSE | Gap | Train Residual | Holdout Residual |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for variant, metrics in variants.items():
            lines.append(
                "| "
                + variant
                + " | "
                + f"{metrics['test_recon_mse']['mean']:.6f} +/- {metrics['test_recon_mse']['std']:.6f}"
                + " | "
                + f"{metrics['generalization_gap']['mean']:.6f} +/- {metrics['generalization_gap']['std']:.6f}"
                + " | "
                + f"{metrics['residual_norm_train']['mean']:.6f} +/- {metrics['residual_norm_train']['std']:.6f}"
                + " | "
                + f"{metrics['residual_norm_holdout']['mean']:.6f} +/- {metrics['residual_norm_holdout']['std']:.6f}"
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def run_direct_benchmark_suite(
    seeds: list[int],
    worlds: list[str],
    variants: list[str],
    variant_recipes: dict[str, dict[str, Any]],
    output_json: str | None = None,
    output_markdown: str | None = None,
    split_strategy: str = "cartesian_blocks",
    grid_size: int = 8,
    image_size: int = 20,
    latent_dim: int = 8,
    hidden_dim: int = 96,
    train_fraction: float = 0.78,
    epochs: int = 280,
) -> dict[str, Any]:
    raw_runs: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    for world in worlds:
        summary[world] = {}
        for variant in variants:
            variant_runs = []
            for seed in seeds:
                recipe = dict(variant_recipes[variant])
                selection = recipe.pop("selection", None)
                config = DirectBenchmarkConfig(
                    world=world,
                    variant=variant,
                    seed=seed,
                    split_strategy=split_strategy,
                    grid_size=grid_size,
                    image_size=image_size,
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    train_fraction=train_fraction,
                    epochs=epochs,
                    **recipe,
                )
                if selection is None:
                    run = train_one_direct(config)
                else:
                    run = train_direct_with_nested_selection(
                        config,
                        candidate_recipes=selection["candidates"],
                        selection_metric=selection.get("metric", "test_recon_mse"),
                        inner_train_fraction=float(selection.get("inner_train_fraction", 0.72)),
                    )
                raw_runs.append(run)
                variant_runs.append(run)
            summary[world][variant] = aggregate_direct_runs(variant_runs)

    output = {
        "seeds": seeds,
        "worlds": worlds,
        "variants": variants,
        "variant_recipes": variant_recipes,
        "split_strategy": split_strategy,
        "world_metadata": {
            world: {
                "ground_truth_commutator": ground_truth_commutator_magnitude(world, grid_size, image_size),
                "ground_truth_step_drift": ground_truth_step_drift_magnitude(world, grid_size),
                "ground_truth_coupling_strength": ground_truth_coupling_strength(world),
            }
            for world in worlds
        },
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
        path.write_text(render_direct_markdown_report(output), encoding="utf-8")

    return output
