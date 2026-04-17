from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
from torch import nn


@dataclass
class BenchmarkConfig:
    world: str
    variant: str
    seed: int
    split_strategy: str = "random"
    grid_size: int = 8
    image_size: int = 20
    latent_dim: int = 8
    hidden_dim: int = 96
    train_fraction: float = 0.78
    epochs: int = 280
    lr: float = 3e-3
    lambda_comm: float = 0.0
    lambda_step: float = 0.0
    lambda_smooth: float = 0.0
    lambda_var: float = 1e-3
    sigma_min: float = 0.35
    warmup_epochs: int = 40


VARIANT_DEFAULTS: dict[str, dict[str, float]] = {
    "baseline": {},
    "smooth": {"lambda_smooth": 0.05},
    "step_only": {"lambda_step": 0.08},
    "comm": {"lambda_comm": 0.18},
    "comm_step": {"lambda_comm": 0.18, "lambda_step": 0.08},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def build_base_pattern(size: int) -> torch.Tensor:
    coords = torch.linspace(-1.0, 1.0, size)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    blob_a = torch.exp(-((xx + 0.35) ** 2 + (yy + 0.20) ** 2) * 10.0)
    blob_b = 0.85 * torch.exp(-((xx - 0.20) ** 2 / 0.05 + (yy - 0.15) ** 2 / 0.16))
    ridge = 0.35 * torch.exp(-((yy + 0.40) ** 2) * 35.0) * (xx > -0.2).float()
    gradient = 0.15 * (xx + 1.0) + 0.08 * (yy + 1.0)
    img = blob_a + blob_b + ridge + gradient
    return (img / img.max()).clamp(0.0, 1.0)


def horizontal_ramp(size: int) -> torch.Tensor:
    coords = torch.linspace(-1.0, 1.0, size)
    return coords.unsqueeze(0).repeat(size, 1)


def apply_shift(image: torch.Tensor, shift: int) -> torch.Tensor:
    return torch.roll(image, shifts=(int(shift),), dims=(1,))


def apply_modulation(image: torch.Tensor, scale: float, noncomm_strength: float) -> torch.Tensor:
    ramp = horizontal_ramp(image.shape[-1])
    modulator = float(scale) * (1.0 + float(noncomm_strength) * ramp)
    return (image * modulator).clamp(0.0, 1.0)


def apply_center_scale(image: torch.Tensor, noncomm_strength: float) -> torch.Tensor:
    scale_factor = 1.0 + float(noncomm_strength)
    theta = torch.tensor(
        [
            [scale_factor, 0.0, 0.0],
            [0.0, scale_factor, 0.0],
        ],
        dtype=image.dtype,
    )
    sample = image.unsqueeze(0).unsqueeze(0)
    grid = nn.functional.affine_grid(theta.unsqueeze(0), sample.size(), align_corners=False)
    scaled = nn.functional.grid_sample(
        sample,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return scaled.squeeze(0).squeeze(0).clamp(0.0, 1.0)


def shift_brightness_world(base: torch.Tensor, grid_size: int) -> torch.Tensor:
    shifts = torch.round(torch.linspace(-3.0, 3.0, grid_size)).to(torch.int64)
    brightness = torch.linspace(0.55, 1.45, grid_size)
    world = []
    for shift in shifts.tolist():
        row = []
        shifted = apply_shift(base, int(shift))
        for scale in brightness.tolist():
            row.append((shifted * float(scale)).clamp(0.0, 1.0))
        world.append(torch.stack(row))
    return torch.stack(world)


def make_asymmetric_window(size: int, scale: float) -> torch.Tensor:
    coords = torch.linspace(-1.0, 1.0, size)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    x_extent = 0.95 * scale
    y_extent = 0.72 * scale
    mask = ((xx + 0.18).abs() <= x_extent) & ((yy - 0.12).abs() <= y_extent)
    softened = mask.float()
    return softened


def rotate_image(image: torch.Tensor, degrees: float) -> torch.Tensor:
    radians = math.radians(degrees)
    theta = torch.tensor(
        [
            [math.cos(radians), -math.sin(radians), 0.0],
            [math.sin(radians), math.cos(radians), 0.0],
        ],
        dtype=image.dtype,
    )
    sample = image.unsqueeze(0).unsqueeze(0)
    grid = nn.functional.affine_grid(theta.unsqueeze(0), sample.size(), align_corners=False)
    rotated = nn.functional.grid_sample(
        sample,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return rotated.squeeze(0).squeeze(0)


def window_rotation_world(base: torch.Tensor, grid_size: int) -> torch.Tensor:
    scales = torch.linspace(0.52, 1.0, grid_size)
    angles = torch.linspace(-38.0, 38.0, grid_size)
    world = []
    for scale in scales.tolist():
        row = []
        window = make_asymmetric_window(base.shape[-1], float(scale))
        windowed = base * window
        for angle in angles.tolist():
            row.append(rotate_image(windowed, float(angle)).clamp(0.0, 1.0))
        world.append(torch.stack(row))
    return torch.stack(world)


def matched_shift_modulation_world(base: torch.Tensor, grid_size: int, noncomm_strength: float) -> torch.Tensor:
    shifts = torch.round(torch.linspace(-3.0, 3.0, grid_size)).to(torch.int64)
    _, scales = matched_column_coordinates(grid_size)
    world = []
    for shift in shifts.tolist():
        row = []
        shifted = apply_shift(base, int(shift))
        for scale in scales.tolist():
            row.append(apply_modulation(shifted, float(scale), noncomm_strength))
        world.append(torch.stack(row))
    return torch.stack(world)


def matched_column_coordinates(grid_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    phases = torch.linspace(-1.0, 1.0, grid_size)
    brightness = torch.linspace(0.55, 1.45, grid_size)
    return phases, brightness


def apply_scale_brightness_operator(
    image: torch.Tensor,
    brightness: float,
    phase: float,
    noncomm_strength: float,
) -> torch.Tensor:
    delta = float(noncomm_strength) * float(phase)
    transformed = image if abs(delta) < 1e-9 else apply_center_scale(image, delta)
    return (transformed * float(brightness)).clamp(0.0, 1.0)


def apply_rotation_brightness_operator(
    image: torch.Tensor,
    brightness: float,
    phase: float,
    max_degrees: float,
) -> torch.Tensor:
    angle = float(max_degrees) * float(phase)
    transformed = image if abs(angle) < 1e-9 else rotate_image(image, angle)
    return (transformed * float(brightness)).clamp(0.0, 1.0)


def matched_shift_scale_world(base: torch.Tensor, grid_size: int, noncomm_strength: float) -> torch.Tensor:
    shifts = torch.round(torch.linspace(-3.0, 3.0, grid_size)).to(torch.int64)
    phases, brightness = matched_column_coordinates(grid_size)
    world = []
    for shift in shifts.tolist():
        row = []
        shifted = apply_shift(base, int(shift))
        for phase, scale in zip(phases.tolist(), brightness.tolist()):
            row.append(apply_scale_brightness_operator(shifted, float(scale), float(phase), noncomm_strength))
        world.append(torch.stack(row))
    return torch.stack(world)


def matched_shift_rotation_world(base: torch.Tensor, grid_size: int, noncomm_strength: float) -> torch.Tensor:
    shifts = torch.round(torch.linspace(-3.0, 3.0, grid_size)).to(torch.int64)
    phases, brightness = matched_column_coordinates(grid_size)
    world = []
    for shift in shifts.tolist():
        row = []
        shifted = apply_shift(base, int(shift))
        for phase, scale in zip(phases.tolist(), brightness.tolist()):
            row.append(apply_rotation_brightness_operator(shifted, float(scale), float(phase), noncomm_strength))
        world.append(torch.stack(row))
    return torch.stack(world)


def parse_matched_world(world: str) -> tuple[str, float] | None:
    prefixes = {
        "matched_comm_": "ramp",
        "matched_ramp_": "ramp",
        "matched_scale_": "scale",
        "matched_rotate_": "rotation",
    }
    for prefix, family in prefixes.items():
        if world.startswith(prefix):
            return family, float(world[len(prefix) :])
    return None


def generate_world(world: str, grid_size: int, image_size: int) -> torch.Tensor:
    base = build_base_pattern(image_size)
    matched_world = parse_matched_world(world)
    if matched_world is not None:
        family, matched_strength = matched_world
        if family == "ramp":
            return matched_shift_modulation_world(base, grid_size, matched_strength)
        if family == "scale":
            return matched_shift_scale_world(base, grid_size, matched_strength)
        if family == "rotation":
            return matched_shift_rotation_world(base, grid_size, matched_strength)
    if world == "commutative":
        return shift_brightness_world(base, grid_size)
    if world == "noncommutative":
        return window_rotation_world(base, grid_size)
    raise ValueError(f"Unsupported world: {world}")


def ground_truth_commutator_magnitude(world: str, grid_size: int, image_size: int) -> float | None:
    matched_world = parse_matched_world(world)
    if matched_world is None:
        return None

    family, matched_strength = matched_world
    base = build_base_pattern(image_size)
    shifts = torch.round(torch.linspace(-3.0, 3.0, grid_size)).to(torch.int64)
    diffs = []
    if family == "ramp":
        _, scales = matched_column_coordinates(grid_size)
        for shift in shifts.tolist():
            for scale in scales.tolist():
                ab = apply_modulation(apply_shift(base, int(shift)), float(scale), matched_strength)
                ba = apply_shift(apply_modulation(base, float(scale), matched_strength), int(shift))
                diffs.append(float((ab - ba).pow(2).mean().item()))
    elif family == "scale":
        phases, brightness = matched_column_coordinates(grid_size)
        for shift in shifts.tolist():
            for phase, scale in zip(phases.tolist(), brightness.tolist()):
                ab = apply_scale_brightness_operator(
                    apply_shift(base, int(shift)),
                    float(scale),
                    float(phase),
                    matched_strength,
                )
                ba = apply_shift(
                    apply_scale_brightness_operator(base, float(scale), float(phase), matched_strength),
                    int(shift),
                )
                diffs.append(float((ab - ba).pow(2).mean().item()))
    elif family == "rotation":
        phases, brightness = matched_column_coordinates(grid_size)
        for shift in shifts.tolist():
            for phase, scale in zip(phases.tolist(), brightness.tolist()):
                ab = apply_rotation_brightness_operator(
                    apply_shift(base, int(shift)),
                    float(scale),
                    float(phase),
                    matched_strength,
                )
                ba = apply_shift(
                    apply_rotation_brightness_operator(base, float(scale), float(phase), matched_strength),
                    int(shift),
                )
                diffs.append(float((ab - ba).pow(2).mean().item()))
    return mean(diffs)


def cartesian_block_train_mask(grid_size: int, row_phase: int = 0, col_phase: int = 0) -> torch.Tensor:
    if grid_size < 6:
        raise ValueError("cartesian_blocks split requires grid_size >= 6.")

    mask = torch.ones((grid_size, grid_size), dtype=torch.bool)
    holdout_rows = [index for index in range(grid_size) if (index + row_phase) % 4 in (1, 2)]
    holdout_cols = [index for index in range(grid_size) if (index + col_phase) % 4 in (1, 2)]
    for row in holdout_rows:
        for col in holdout_cols:
            mask[row, col] = False
    return mask


def sample_random_train_mask(grid_size: int, train_fraction: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + 1234)
    total = grid_size * grid_size
    for _ in range(512):
        mask = torch.rand((grid_size, grid_size), generator=generator) < train_fraction
        if mask.sum().item() < total * 0.6 or (~mask).sum().item() < total * 0.1:
            continue
        if (mask.sum(dim=0) >= 3).all() and (mask.sum(dim=1) >= 3).all():
            cells = mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]
            if cells.sum().item() >= max(4, int(0.28 * (grid_size - 1) ** 2)):
                return mask
    raise RuntimeError("Failed to sample a usable train/test split.")


def sample_train_mask(grid_size: int, train_fraction: float, seed: int, split_strategy: str) -> torch.Tensor:
    if split_strategy == "random":
        return sample_random_train_mask(grid_size, train_fraction, seed)
    if split_strategy == "cartesian_blocks":
        return cartesian_block_train_mask(grid_size)
    raise ValueError(f"Unsupported split strategy: {split_strategy}")


def sample_nested_train_mask(base_mask: torch.Tensor, seed: int, keep_fraction: float = 0.72) -> torch.Tensor:
    if keep_fraction <= 0.0 or keep_fraction >= 1.0:
        raise ValueError("keep_fraction must be between 0 and 1.")

    base_mask = base_mask.to(torch.bool)
    grid_size = base_mask.shape[0]
    base_rows = base_mask.sum(dim=1)
    base_cols = base_mask.sum(dim=0)
    total_base = int(base_mask.sum().item())
    min_validation = max(4, int(round(total_base * (1.0 - keep_fraction))))
    min_cells = max(4, int(0.18 * max(1, int(cell_mask(base_mask).sum().item()))))

    phases = [(row_phase, col_phase) for row_phase in range(4) for col_phase in range(4)]
    offset = seed % len(phases)
    for index in range(len(phases)):
        row_phase, col_phase = phases[(index + offset) % len(phases)]
        if row_phase == 0 and col_phase == 0:
            continue

        selector = cartesian_block_train_mask(grid_size, row_phase=row_phase, col_phase=col_phase)
        proposal = base_mask & selector
        validation_points = int((base_mask & ~proposal).sum().item())
        if validation_points < min_validation:
            continue
        if torch.any(proposal.sum(dim=1)[base_rows > 0] < 2):
            continue
        if torch.any(proposal.sum(dim=0)[base_cols > 0] < 2):
            continue
        if int(cell_mask(proposal).sum().item()) < min_cells:
            continue
        return proposal

    raise RuntimeError("Failed to sample a usable nested inner-train mask.")


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.numel() == 0 or not torch.any(mask):
        return values.new_tensor(0.0)
    expanded = mask.unsqueeze(-1) if values.ndim == mask.ndim + 1 else mask
    selected = values[expanded.expand_as(values)]
    if selected.numel() == 0:
        return values.new_tensor(0.0)
    return selected.mean()


def cell_mask(mask: torch.Tensor) -> torch.Tensor:
    return mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]


def mixed_difference_loss(z_grid: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mixed = z_grid[1:, 1:] - z_grid[1:, :-1] - z_grid[:-1, 1:] + z_grid[:-1, :-1]
    per_cell = mixed.pow(2).mean(dim=-1)
    return masked_mean(per_cell, mask)


def step_consistency_loss(z_grid: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vertical_mask = mask[1:, :] & mask[:-1, :]
    horizontal_mask = mask[:, 1:] & mask[:, :-1]

    vdiff = z_grid[1:, :, :] - z_grid[:-1, :, :]
    hdiff = z_grid[:, 1:, :] - z_grid[:, :-1, :]

    v_steps = vdiff[vertical_mask]
    h_steps = hdiff[horizontal_mask]

    zero = z_grid.new_tensor(0.0)
    if v_steps.numel() == 0 or h_steps.numel() == 0:
        return zero, zero, zero

    v_mean = v_steps.mean(dim=0)
    h_mean = h_steps.mean(dim=0)
    loss = (v_steps - v_mean).pow(2).mean() + (h_steps - h_mean).pow(2).mean()
    orth = (v_mean * h_mean).sum().pow(2)
    return loss, v_mean, orth


def smoothness_loss(z_grid: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    zero = z_grid.new_tensor(0.0)

    if z_grid.shape[0] < 3 or z_grid.shape[1] < 3:
        return zero

    v_mask = mask[2:, :] & mask[1:-1, :] & mask[:-2, :]
    h_mask = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]

    v_second = z_grid[2:, :, :] - 2.0 * z_grid[1:-1, :, :] + z_grid[:-2, :, :]
    h_second = z_grid[:, 2:, :] - 2.0 * z_grid[:, 1:-1, :] + z_grid[:, :-2, :]

    v_loss = masked_mean(v_second.pow(2).mean(dim=-1), v_mask)
    h_loss = masked_mean(h_second.pow(2).mean(dim=-1), h_mask)
    return v_loss + h_loss


def variance_floor_loss(z_train: torch.Tensor, sigma_min: float) -> torch.Tensor:
    std = z_train.std(dim=0, unbiased=False)
    return torch.relu(z_train.new_tensor(sigma_min) - std).pow(2).mean()


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(nn.functional.mse_loss(pred, target).item())


def train_one(
    config: BenchmarkConfig,
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

    flat = images.reshape(config.grid_size * config.grid_size, -1)
    train_idx = mask.reshape(-1)
    test_idx = eval_mask.reshape(-1)

    model = AutoEncoder(flat.shape[-1], config.hidden_dim, config.latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    train_cell_mask = cell_mask(mask)
    holdout_cell_mask = cell_mask(eval_mask)

    last_losses: dict[str, float] = {}
    for epoch in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        recon_all, latent_all = model(flat)
        z_grid = latent_all.reshape(config.grid_size, config.grid_size, config.latent_dim)

        recon_loss = nn.functional.mse_loss(recon_all[train_idx], flat[train_idx])
        comm_loss = mixed_difference_loss(z_grid, train_cell_mask)
        step_loss, _, orth_penalty = step_consistency_loss(z_grid, mask)
        smooth_loss = smoothness_loss(z_grid, mask)
        var_loss = variance_floor_loss(latent_all[train_idx], config.sigma_min)

        warmup = min(1.0, float(epoch + 1) / float(max(1, config.warmup_epochs)))
        total_loss = (
            recon_loss
            + warmup * config.lambda_comm * comm_loss
            + warmup * config.lambda_step * step_loss
            + config.lambda_smooth * smooth_loss
            + config.lambda_var * var_loss
            + 1e-4 * config.lambda_step * orth_penalty
        )
        total_loss.backward()
        optimizer.step()

        if epoch == config.epochs - 1:
            last_losses = {
                "recon_loss": float(recon_loss.item()),
                "comm_loss": float(comm_loss.item()),
                "step_loss": float(step_loss.item()),
                "smooth_loss": float(smooth_loss.item()),
                "var_loss": float(var_loss.item()),
                "total_loss": float(total_loss.item()),
            }

    with torch.no_grad():
        recon_all, latent_all = model(flat)
        z_grid = latent_all.reshape(config.grid_size, config.grid_size, config.latent_dim)
        full_cell_mask = torch.ones((config.grid_size - 1, config.grid_size - 1), dtype=torch.bool)

        result = {
            "config": asdict(config),
            "train_points": int(train_idx.sum().item()),
            "test_points": int(test_idx.sum().item()),
            "train_recon_mse": mse(recon_all[train_idx], flat[train_idx]),
            "test_recon_mse": mse(recon_all[test_idx], flat[test_idx]),
            "generalization_gap": mse(recon_all[test_idx], flat[test_idx]) - mse(recon_all[train_idx], flat[train_idx]),
            "comm_error_all": float(mixed_difference_loss(z_grid, full_cell_mask).item()),
            "comm_error_train_cells": float(mixed_difference_loss(z_grid, train_cell_mask).item()),
            "comm_error_holdout_cells": float(mixed_difference_loss(z_grid, holdout_cell_mask).item()),
            "latent_std_mean": float(latent_all[train_idx].std(dim=0, unbiased=False).mean().item()),
            "train_mask": mask.to(torch.int64).tolist(),
            "eval_mask": eval_mask.to(torch.int64).tolist(),
            "recon_error_grid": ((recon_all - flat).pow(2).mean(dim=-1).reshape(config.grid_size, config.grid_size)).tolist(),
            "final_losses": last_losses,
        }
    return result


def train_with_nested_selection(
    config: BenchmarkConfig,
    candidate_recipes: dict[str, dict[str, float]],
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
    best_recipe: dict[str, float] = {}
    best_score = float("inf")

    for candidate_name, candidate_recipe in candidate_recipes.items():
        candidate_config = replace(config, variant=candidate_name, **candidate_recipe)
        selection_run = train_one(
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
    final_run = train_one(final_config, train_mask_override=outer_train_mask)
    final_run["selection"] = {
        "mode": "nested",
        "metric": selection_metric,
        "inner_train_fraction": inner_train_fraction,
        "chosen_candidate": best_name,
        "chosen_recipe": best_recipe,
        "scores": selection_scores,
    }
    return final_run


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "train_recon_mse",
        "test_recon_mse",
        "generalization_gap",
        "comm_error_all",
        "comm_error_train_cells",
        "comm_error_holdout_cells",
        "latent_std_mean",
    ]
    summary: dict[str, Any] = {}
    for metric in metrics:
        values = [float(run[metric]) for run in runs]
        summary[metric] = {
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
        }
    return summary


def render_markdown_report(results: dict[str, Any]) -> str:
    lines = [
        "# Initial Synthetic Benchmark",
        "",
        "This report compares baseline, generic smoothness, CFP-only, and CFP-plus-step regularization.",
        "",
        f"Split strategy: `{results['split_strategy']}`",
        "",
    ]
    for world, variants in results["summary"].items():
        lines.append(f"## World: {world}")
        lines.append("")
        lines.append("| Variant | Test Recon MSE | Generalization Gap | Train Comm | Holdout Comm | Latent Std |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for variant, metrics in variants.items():
            lines.append(
                "| "
                + variant
                + " | "
                + f"{metrics['test_recon_mse']['mean']:.6f} +/- {metrics['test_recon_mse']['std']:.6f}"
                + " | "
                + f"{metrics['generalization_gap']['mean']:.6f} +/- {metrics['generalization_gap']['std']:.6f}"
                + " | "
                + f"{metrics['comm_error_train_cells']['mean']:.6f} +/- {metrics['comm_error_train_cells']['std']:.6f}"
                + " | "
                + f"{metrics['comm_error_holdout_cells']['mean']:.6f} +/- {metrics['comm_error_holdout_cells']['std']:.6f}"
                + " | "
                + f"{metrics['latent_std_mean']['mean']:.6f} +/- {metrics['latent_std_mean']['std']:.6f}"
                + " |"
            )
        lines.append("")

    lines.extend(
        [
            "## Reading Guide",
            "",
            "- A good CFP outcome is lower held-out reconstruction error in the commutative world.",
            "- The non-commutative control should not show the same pattern of gain.",
            "- If `smooth` matches `comm`, the effect is probably not specific to CFP.",
            "",
        ]
    )
    return "\n".join(lines)


def run_benchmark_suite(
    seeds: list[int],
    worlds: list[str],
    variants: list[str],
    output_json: str | None = None,
    output_markdown: str | None = None,
    variant_recipes: dict[str, dict[str, float]] | None = None,
    split_strategy: str = "random",
    grid_size: int = 8,
    image_size: int = 20,
    latent_dim: int = 8,
    hidden_dim: int = 96,
    train_fraction: float = 0.78,
    warmup_epochs: int = 40,
    epochs: int = 280,
) -> dict[str, Any]:
    raw_runs: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    recipes = variant_recipes or VARIANT_DEFAULTS

    for world in worlds:
        summary[world] = {}
        for variant in variants:
            variant_runs = []
            for seed in seeds:
                recipe = dict(recipes[variant])
                selection = recipe.pop("selection", None)
                config = BenchmarkConfig(
                    world=world,
                    variant=variant,
                    seed=seed,
                    split_strategy=split_strategy,
                    grid_size=grid_size,
                    image_size=image_size,
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    train_fraction=train_fraction,
                    warmup_epochs=warmup_epochs,
                    epochs=epochs,
                    **recipe,
                )
                if selection is None:
                    run = train_one(config)
                else:
                    run = train_with_nested_selection(
                        config,
                        candidate_recipes=selection["candidates"],
                        selection_metric=selection.get("metric", "test_recon_mse"),
                        inner_train_fraction=float(selection.get("inner_train_fraction", 0.72)),
                    )
                raw_runs.append(run)
                variant_runs.append(run)
            summary[world][variant] = aggregate_runs(variant_runs)

    output = {
        "seeds": seeds,
        "worlds": worlds,
        "variants": variants,
        "variant_recipes": recipes,
        "split_strategy": split_strategy,
        "world_metadata": {
            world: {"ground_truth_commutator": ground_truth_commutator_magnitude(world, grid_size, image_size)}
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
        path.write_text(render_markdown_report(output), encoding="utf-8")

    return output
