from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
from torch import nn

from .synthetic import set_seed


@dataclass
class ContextTransferConfig:
    world: str
    variant: str
    seed: int
    model_type: str
    split_strategy: str = "holdout_context"
    context_count: int = 5
    state_count: int = 12
    action_count: int = 4
    rollout_length: int = 5
    image_size: int = 20
    latent_dim: int = 24
    context_dim: int = 8
    hidden_dim: int = 96
    interaction_rank: int = 4
    epochs: int = 220
    lr: float = 2e-3
    lambda_residual: float = 2e-3
    autoencode_weight: float = 0.20
    adapt_steps: int = 40
    adapt_lr: float = 6e-2
    support_fraction: float = 0.35


def parse_context_world(world: str) -> tuple[str, float]:
    if world.startswith("context_commuting_"):
        return "commuting", float(world[len("context_commuting_") :])
    if world.startswith("context_coupled_"):
        return "coupled", float(world[len("context_coupled_") :])
    raise ValueError(f"Unsupported context-transfer world: {world}")


def holdout_context_index(context_count: int) -> int:
    return context_count - 1


def context_phases(context_count: int) -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, context_count)


def support_state_indices(state_count: int, support_fraction: float) -> torch.Tensor:
    count = max(1, int(math.ceil(float(state_count) * float(support_fraction))))
    return torch.arange(count, dtype=torch.long)


def scene_coords(size: int) -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.linspace(-1.0, 1.0, size)
    return torch.meshgrid(coords, coords, indexing="ij")


def state_positions(state_count: int) -> torch.Tensor:
    angles = torch.linspace(0.0, 2.0 * math.pi, state_count + 1)[:-1]
    radius_x = 0.55
    radius_y = 0.42
    x = radius_x * torch.cos(angles)
    y = radius_y * torch.sin(angles)
    return torch.stack([x, y], dim=-1)


def action_vectors(action_count: int) -> torch.Tensor:
    if action_count != 4:
        raise ValueError("This benchmark expects exactly 4 actions.")
    return torch.tensor(
        [
            [-0.22, 0.00],
            [0.22, 0.00],
            [0.00, -0.18],
            [0.00, 0.18],
        ],
        dtype=torch.float32,
    )


def bounce_position(position: torch.Tensor) -> torch.Tensor:
    limit = torch.tensor([0.72, 0.62], dtype=position.dtype, device=position.device)
    reflected = torch.where(position.abs() <= limit, position, torch.sign(position) * (2.0 * limit - position.abs()))
    return reflected.clamp(-limit, limit)


def style_parameters(phase: float, alpha: float) -> dict[str, float]:
    return {
        "background_shift": float(alpha) * float(phase),
        "sensor_strength": 0.03 + 0.05 * float(alpha) * (0.5 + 0.5 * abs(float(phase))),
        "occluder_shift": 0.20 * float(alpha) * float(phase),
        "palette_mix": 0.5 + 0.35 * float(alpha) * float(phase),
    }


def action_matrix(phase: float, alpha: float, family: str) -> torch.Tensor:
    if family == "commuting":
        return torch.eye(2, dtype=torch.float32)
    shear = 0.18 * float(alpha) * float(phase)
    scale_x = 1.0 + 0.35 * float(alpha) * float(phase)
    scale_y = 1.0 - 0.28 * float(alpha) * float(phase)
    return torch.tensor(
        [
            [scale_x, shear],
            [0.10 * float(alpha) * float(phase), scale_y],
        ],
        dtype=torch.float32,
    )


def canonical_step(position: torch.Tensor, action: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    delta = transform @ action
    return bounce_position(position + delta)


def render_state(position: torch.Tensor, phase: float, alpha: float, image_size: int) -> torch.Tensor:
    yy, xx = scene_coords(image_size)
    params = style_parameters(phase, alpha)

    bg_r = 0.26 + 0.12 * (xx + 1.0) + 0.04 * torch.sin(2.0 * math.pi * yy + params["background_shift"])
    bg_g = 0.24 + 0.10 * (yy + 1.0) + 0.05 * torch.cos(2.6 * math.pi * xx - 0.7 * params["background_shift"])
    bg_b = 0.28 + 0.04 * torch.sin(3.2 * math.pi * (xx + yy))
    background = torch.stack([bg_r, bg_g, bg_b], dim=0).clamp(0.0, 1.0)

    object_mask = torch.exp(-(((xx - float(position[0])) / 0.18) ** 2 + ((yy - float(position[1])) / 0.14) ** 2) * 3.5)
    highlight = torch.exp(-(((xx - float(position[0]) + 0.06) / 0.12) ** 2 + ((yy - float(position[1]) - 0.04) / 0.10) ** 2) * 5.0)
    stripe = 0.5 + 0.5 * torch.sin(6.0 * math.pi * (xx - float(position[0])) + 1.5 * phase)
    palette_mix = params["palette_mix"]
    warm = torch.tensor([0.80, 0.50, 0.28], dtype=torch.float32).view(3, 1, 1)
    cool = torch.tensor([0.30, 0.58, 0.82], dtype=torch.float32).view(3, 1, 1)
    obj_color = palette_mix * warm + (1.0 - palette_mix) * cool
    object_rgb = obj_color * (0.60 + 0.40 * stripe).unsqueeze(0) + 0.16 * highlight.unsqueeze(0)

    occluder_center = 0.25 + params["occluder_shift"]
    occluder = (torch.sigmoid((xx - occluder_center) * 26.0) - torch.sigmoid((xx - occluder_center - 0.14) * 26.0)).clamp(0.0, 1.0)
    sensor = torch.stack(
        [
            params["sensor_strength"] * torch.sin(4.0 * math.pi * xx + 0.8 * yy),
            params["sensor_strength"] * torch.cos(3.0 * math.pi * yy - 0.5 * xx),
            params["sensor_strength"] * torch.sin(2.5 * math.pi * (xx + yy)),
        ],
        dim=0,
    )

    scene = background * (1.0 - object_mask.unsqueeze(0)) + object_rgb * object_mask.unsqueeze(0)
    scene = scene * (1.0 - 0.18 * occluder.unsqueeze(0)) + 0.06 * occluder.unsqueeze(0)
    return (scene + sensor).clamp(0.0, 1.0)


@dataclass
class ContextTransferWorld:
    frames: torch.Tensor
    positions: torch.Tensor
    holdout_context: int
    support_states: torch.Tensor
    transition_train_mask: torch.Tensor
    transition_eval_mask: torch.Tensor
    support_mask: torch.Tensor
    query_mask: torch.Tensor


def generate_context_transfer_world(config: ContextTransferConfig) -> ContextTransferWorld:
    family, alpha = parse_context_world(config.world)
    contexts = context_phases(config.context_count)
    states = state_positions(config.state_count)
    actions = action_vectors(config.action_count)

    frames = torch.zeros(
        config.context_count,
        config.state_count,
        config.action_count,
        config.rollout_length + 1,
        3,
        config.image_size,
        config.image_size,
        dtype=torch.float32,
    )
    positions = torch.zeros(
        config.context_count,
        config.state_count,
        config.action_count,
        config.rollout_length + 1,
        2,
        dtype=torch.float32,
    )

    for context_index, phase in enumerate(contexts.tolist()):
        transform = action_matrix(phase, alpha, family=family)
        for state_index, start in enumerate(states):
            for action_index, action in enumerate(actions):
                position = start.clone()
                positions[context_index, state_index, action_index, 0] = position
                frames[context_index, state_index, action_index, 0] = render_state(position, phase, alpha, config.image_size)
                for step in range(config.rollout_length):
                    position = canonical_step(position, action, transform)
                    positions[context_index, state_index, action_index, step + 1] = position
                    frames[context_index, state_index, action_index, step + 1] = render_state(position, phase, alpha, config.image_size)

    holdout_context = holdout_context_index(config.context_count)
    support_states = support_state_indices(config.state_count, config.support_fraction)
    context_grid = torch.arange(config.context_count, dtype=torch.long)[:, None, None]
    state_grid = torch.arange(config.state_count, dtype=torch.long)[None, :, None]
    transition_train_mask = context_grid != holdout_context
    transition_train_mask = transition_train_mask.expand(-1, config.state_count, config.action_count)
    transition_eval_mask = ~transition_train_mask
    support_mask = transition_eval_mask & torch.isin(state_grid.expand(config.context_count, -1, config.action_count), support_states)
    query_mask = transition_eval_mask & ~support_mask
    return ContextTransferWorld(
        frames=frames,
        positions=positions,
        holdout_context=holdout_context,
        support_states=support_states,
        transition_train_mask=transition_train_mask,
        transition_eval_mask=transition_eval_mask,
        support_mask=support_mask,
        query_mask=query_mask,
    )


def flattened_transition_samples(world: ContextTransferWorld) -> dict[str, torch.Tensor]:
    current = world.frames[:, :, :, :-1].reshape(-1, 3, world.frames.shape[-2], world.frames.shape[-1])
    target = world.frames[:, :, :, 1:].reshape(-1, 3, world.frames.shape[-2], world.frames.shape[-1])

    context_ids = (
        torch.arange(world.frames.shape[0], dtype=torch.long)[:, None, None, None]
        .expand(world.frames.shape[0], world.frames.shape[1], world.frames.shape[2], world.frames.shape[3] - 1)
        .reshape(-1)
    )
    state_ids = (
        torch.arange(world.frames.shape[1], dtype=torch.long)[None, :, None, None]
        .expand(world.frames.shape[0], world.frames.shape[1], world.frames.shape[2], world.frames.shape[3] - 1)
        .reshape(-1)
    )
    action_ids = (
        torch.arange(world.frames.shape[2], dtype=torch.long)[None, None, :, None]
        .expand(world.frames.shape[0], world.frames.shape[1], world.frames.shape[2], world.frames.shape[3] - 1)
        .reshape(-1)
    )
    time_ids = (
        torch.arange(world.frames.shape[3] - 1, dtype=torch.long)[None, None, None, :]
        .expand(world.frames.shape[0], world.frames.shape[1], world.frames.shape[2], world.frames.shape[3] - 1)
        .reshape(-1)
    )
    sample_shape = (
        world.frames.shape[0],
        world.frames.shape[1],
        world.frames.shape[2],
        world.frames.shape[3] - 1,
    )
    combo_train = world.transition_train_mask[:, :, :, None].expand(sample_shape).reshape(-1)
    combo_eval = world.transition_eval_mask[:, :, :, None].expand(sample_shape).reshape(-1)
    support_mask = world.support_mask[:, :, :, None].expand(sample_shape).reshape(-1)
    query_mask = world.query_mask[:, :, :, None].expand(sample_shape).reshape(-1)
    return {
        "current": current,
        "target": target,
        "contexts": context_ids,
        "states": state_ids,
        "actions": action_ids,
        "times": time_ids,
        "train_mask": combo_train.to(torch.bool),
        "eval_mask": combo_eval.to(torch.bool),
        "support_mask": support_mask.to(torch.bool),
        "query_mask": query_mask.to(torch.bool),
    }


def ground_truth_context_commutator(world: str, context_count: int) -> float:
    family, alpha = parse_context_world(world)
    if family == "commuting":
        return 0.0
    phases = context_phases(context_count)
    values = [torch.linalg.vector_norm(action_matrix(float(phase), alpha, family) - torch.eye(2)).item() for phase in phases]
    return float(mean(values))


def ground_truth_transfer_coupling(world: str, context_count: int) -> float:
    family, alpha = parse_context_world(world)
    if family == "commuting":
        return 0.0
    phases = context_phases(context_count)
    values = []
    for phase in phases:
        matrix = action_matrix(float(phase), alpha, family)
        symmetric = matrix.T @ matrix - torch.eye(2)
        values.append(float(torch.linalg.vector_norm(symmetric).item()))
    return float(mean(values))


def ground_truth_adaptation_cost_proxy(world: str, context_count: int) -> float:
    family, alpha = parse_context_world(world)
    if family == "commuting":
        return 0.0
    phases = context_phases(context_count)
    values = []
    for phase in phases:
        matrix = action_matrix(float(phase), alpha, family)
        values.append(float(torch.linalg.matrix_norm(matrix - torch.eye(2), ord=2).item()))
    return float(mean(values))


class TransitionBaseModel(nn.Module):
    def __init__(
        self,
        image_size: int,
        latent_dim: int,
        context_dim: int,
        hidden_dim: int,
        context_count: int,
        action_count: int,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.context_count = context_count
        self.action_count = action_count
        pixel_dim = 3 * image_size * image_size
        self.context_embeddings = nn.Parameter(torch.empty(context_count, context_dim))
        nn.init.normal_(self.context_embeddings, mean=0.0, std=0.02)
        self.encoder = nn.Sequential(
            nn.Linear(pixel_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pixel_dim),
            nn.Sigmoid(),
        )

    def context_embed(self, contexts: torch.Tensor) -> torch.Tensor:
        return self.context_embeddings[contexts]

    def encode(self, images: torch.Tensor, contexts: torch.Tensor) -> torch.Tensor:
        flat = images.reshape(images.shape[0], -1)
        ctx = self.context_embed(contexts)
        return self.encoder(torch.cat([flat, ctx], dim=-1))

    def decode(self, latent: torch.Tensor, contexts: torch.Tensor) -> torch.Tensor:
        ctx = self.context_embed(contexts)
        flat = self.decoder(torch.cat([latent, ctx], dim=-1))
        return flat.reshape(latent.shape[0], 3, self.image_size, self.image_size)

    def autoencode(self, images: torch.Tensor, contexts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(images, contexts)
        return self.decode(latent, contexts), latent

    def residual_delta(self, latent: torch.Tensor, contexts: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(latent)

    def action_delta(self, latent: torch.Tensor, contexts: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predict_next(self, images: torch.Tensor, contexts: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(images, contexts)
        next_latent = latent + self.action_delta(latent, contexts, actions) + self.residual_delta(latent, contexts, actions)
        return self.decode(next_latent, contexts), next_latent

    def residual_penalty(self, latent: torch.Tensor, contexts: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        residual = self.residual_delta(latent, contexts, actions)
        return residual.pow(2).mean()

    def interaction_norm(self, latent: torch.Tensor, contexts: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.residual_delta(latent, contexts, actions).pow(2).mean(dim=-1).sqrt()

    def adaptation_parameter_prefixes(self) -> tuple[str, ...]:
        return ("context_embeddings",)


class FullTransitionModel(TransitionBaseModel):
    def __init__(
        self,
        image_size: int,
        latent_dim: int,
        context_dim: int,
        hidden_dim: int,
        context_count: int,
        action_count: int,
    ) -> None:
        super().__init__(image_size, latent_dim, context_dim, hidden_dim, context_count, action_count)
        self.action_embeddings = nn.Parameter(torch.empty(action_count, latent_dim))
        nn.init.normal_(self.action_embeddings, mean=0.0, std=0.02)
        self.transition = nn.Sequential(
            nn.Linear((latent_dim * 2) + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def action_delta(self, latent: torch.Tensor, contexts: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        action = self.action_embeddings[actions]
        ctx = self.context_embed(contexts)
        return self.transition(torch.cat([latent, action, ctx], dim=-1))


class CommutingOperatorModel(TransitionBaseModel):
    def __init__(
        self,
        image_size: int,
        latent_dim: int,
        context_dim: int,
        hidden_dim: int,
        context_count: int,
        action_count: int,
    ) -> None:
        super().__init__(image_size, latent_dim, context_dim, hidden_dim, context_count, action_count)
        self.action_shift = nn.Parameter(torch.empty(action_count, latent_dim))
        nn.init.normal_(self.action_shift, mean=0.0, std=0.02)

    def action_delta(self, latent: torch.Tensor, contexts: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        del latent, contexts
        return self.action_shift[actions]


class OperatorPlusResidualModel(CommutingOperatorModel):
    def __init__(
        self,
        image_size: int,
        latent_dim: int,
        context_dim: int,
        hidden_dim: int,
        context_count: int,
        action_count: int,
        interaction_rank: int,
    ) -> None:
        super().__init__(image_size, latent_dim, context_dim, hidden_dim, context_count, action_count)
        self.context_scores = nn.Parameter(torch.empty(context_count, interaction_rank))
        self.action_scores = nn.Parameter(torch.empty(action_count, interaction_rank))
        self.operator_basis = nn.Parameter(torch.empty(interaction_rank, latent_dim, latent_dim))
        nn.init.normal_(self.context_scores, mean=0.0, std=0.02)
        nn.init.normal_(self.action_scores, mean=0.0, std=0.02)
        nn.init.normal_(self.operator_basis, mean=0.0, std=0.02)

    def residual_delta(self, latent: torch.Tensor, contexts: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        context_scores = self.context_scores[contexts] - self.context_scores.mean(dim=0, keepdim=True)
        action_scores = self.action_scores[actions] - self.action_scores.mean(dim=0, keepdim=True)
        weights = context_scores * action_scores
        return torch.einsum("br,rde,be->bd", weights, self.operator_basis, latent)

    def adaptation_parameter_prefixes(self) -> tuple[str, ...]:
        return ("context_embeddings", "context_scores")


class OperatorDiagResidualModel(CommutingOperatorModel):
    def __init__(
        self,
        image_size: int,
        latent_dim: int,
        context_dim: int,
        hidden_dim: int,
        context_count: int,
        action_count: int,
        interaction_rank: int,
    ) -> None:
        super().__init__(image_size, latent_dim, context_dim, hidden_dim, context_count, action_count)
        self.context_scores = nn.Parameter(torch.empty(context_count, interaction_rank))
        self.action_scores = nn.Parameter(torch.empty(action_count, interaction_rank))
        self.diag_basis = nn.Parameter(torch.empty(interaction_rank, latent_dim))
        nn.init.normal_(self.context_scores, mean=0.0, std=0.02)
        nn.init.normal_(self.action_scores, mean=0.0, std=0.02)
        nn.init.normal_(self.diag_basis, mean=0.0, std=0.02)

    def residual_delta(self, latent: torch.Tensor, contexts: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        context_scores = self.context_scores[contexts] - self.context_scores.mean(dim=0, keepdim=True)
        action_scores = self.action_scores[actions] - self.action_scores.mean(dim=0, keepdim=True)
        weights = context_scores * action_scores
        return torch.einsum("br,rd,bd->bd", weights, self.diag_basis, latent)

    def adaptation_parameter_prefixes(self) -> tuple[str, ...]:
        return ("context_embeddings", "context_scores")


def build_model(config: ContextTransferConfig) -> TransitionBaseModel:
    shared = dict(
        image_size=config.image_size,
        latent_dim=config.latent_dim,
        context_dim=config.context_dim,
        hidden_dim=config.hidden_dim,
        context_count=config.context_count,
        action_count=config.action_count,
    )
    if config.model_type == "full_transition":
        return FullTransitionModel(**shared)
    if config.model_type == "commuting_operator":
        return CommutingOperatorModel(**shared)
    if config.model_type == "operator_plus_residual":
        return OperatorPlusResidualModel(**shared, interaction_rank=config.interaction_rank)
    if config.model_type == "operator_diag_residual":
        return OperatorDiagResidualModel(**shared, interaction_rank=config.interaction_rank)
    raise ValueError(f"Unsupported model_type: {config.model_type}")


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(nn.functional.mse_loss(pred, target).item())


def configure_adaptation_parameters(model: TransitionBaseModel) -> list[str]:
    allowed = model.adaptation_parameter_prefixes()
    trainable = []
    for name, parameter in model.named_parameters():
        enabled = any(name.startswith(prefix) for prefix in allowed)
        parameter.requires_grad = enabled
        if enabled:
            trainable.append(name)
    return trainable


def evaluate_rollout(
    model: TransitionBaseModel,
    world: ContextTransferWorld,
    context_index: int,
    horizon: int,
    state_mask: torch.Tensor | None = None,
) -> float:
    device = next(model.parameters()).device
    errors = []
    with torch.no_grad():
        for state_index in range(world.frames.shape[1]):
            if state_mask is not None and not bool(state_mask[state_index]):
                continue
            for action_index in range(world.frames.shape[2]):
                current = world.frames[context_index, state_index, action_index, 0].unsqueeze(0).to(device)
                context = torch.tensor([context_index], dtype=torch.long, device=device)
                action = torch.tensor([action_index], dtype=torch.long, device=device)
                for _ in range(horizon):
                    current, _ = model.predict_next(current, context, action)
                target = world.frames[context_index, state_index, action_index, horizon].unsqueeze(0).to(device)
                errors.append(float(nn.functional.mse_loss(current, target).item()))
    return float(mean(errors)) if errors else 0.0


def evaluate_one_step(
    model: TransitionBaseModel,
    samples: dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> tuple[float, float]:
    device = next(model.parameters()).device
    with torch.no_grad():
        images = samples["current"][mask].to(device)
        targets = samples["target"][mask].to(device)
        contexts = samples["contexts"][mask].to(device)
        actions = samples["actions"][mask].to(device)
        prediction, latent = model.predict_next(images, contexts, actions)
        return (
            mse(prediction, targets),
            float(model.interaction_norm(latent, contexts, actions).mean().item()),
        )


def adapt_to_holdout_context(
    model: TransitionBaseModel,
    samples: dict[str, torch.Tensor],
    config: ContextTransferConfig,
) -> dict[str, Any]:
    adapted = copy.deepcopy(model)
    device = next(model.parameters()).device
    trainable_names = configure_adaptation_parameters(adapted)
    optimizer = torch.optim.Adam([parameter for parameter in adapted.parameters() if parameter.requires_grad], lr=config.adapt_lr)

    support_mask = samples["support_mask"]
    query_mask = samples["query_mask"]
    support_images = samples["current"][support_mask].to(device)
    support_targets = samples["target"][support_mask].to(device)
    support_contexts = samples["contexts"][support_mask].to(device)
    support_actions = samples["actions"][support_mask].to(device)
    query_images = samples["current"][query_mask].to(device)
    query_targets = samples["target"][query_mask].to(device)
    query_contexts = samples["contexts"][query_mask].to(device)
    query_actions = samples["actions"][query_mask].to(device)

    with torch.no_grad():
        zero_pred, zero_latent = adapted.predict_next(query_images, query_contexts, query_actions)
        zero_query_mse = float(nn.functional.mse_loss(zero_pred, query_targets).item())
        zero_residual = float(adapted.interaction_norm(zero_latent, query_contexts, query_actions).mean().item())

    query_curve = [zero_query_mse]
    best_query_mse = zero_query_mse
    best_residual_norm = zero_residual
    support_final = zero_query_mse

    for _ in range(config.adapt_steps):
        optimizer.zero_grad(set_to_none=True)
        support_recon, _ = adapted.autoencode(support_images, support_contexts)
        predicted, latent = adapted.predict_next(support_images, support_contexts, support_actions)
        recon_loss = nn.functional.mse_loss(support_recon, support_images)
        transition_loss = nn.functional.mse_loss(predicted, support_targets)
        residual_loss = adapted.residual_penalty(latent, support_contexts, support_actions)
        total_loss = transition_loss + config.autoencode_weight * recon_loss + config.lambda_residual * residual_loss
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            query_pred, query_latent = adapted.predict_next(query_images, query_contexts, query_actions)
            query_mse = float(nn.functional.mse_loss(query_pred, query_targets).item())
            residual_norm = float(adapted.interaction_norm(query_latent, query_contexts, query_actions).mean().item())
            query_curve.append(query_mse)
            if query_mse < best_query_mse - 1e-12:
                best_query_mse = query_mse
                best_residual_norm = residual_norm
            support_final = float(total_loss.item())

    possible_gain = max(0.0, zero_query_mse - best_query_mse)
    target = zero_query_mse - 0.8 * possible_gain if possible_gain > 1e-12 else zero_query_mse
    steps_to_target = 0
    if possible_gain > 1e-12:
        steps_to_target = config.adapt_steps + 1
        for index, value in enumerate(query_curve[1:], start=1):
            if value <= target + 1e-12:
                steps_to_target = index
                break

    return {
        "trainable_parameter_names": trainable_names,
        "zero_shot_query_mse": zero_query_mse,
        "best_query_mse": best_query_mse,
        "adaptation_gain": zero_query_mse - best_query_mse,
        "steps_to_target": int(steps_to_target),
        "residual_norm_final": best_residual_norm,
        "support_final_objective": support_final,
        "query_curve": query_curve,
    }


def train_one_context_transfer(config: ContextTransferConfig, evaluate_adaptation: bool = False) -> dict[str, Any]:
    set_seed(config.seed)
    world = generate_context_transfer_world(config)
    samples = flattened_transition_samples(world)

    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_mask = samples["train_mask"]
    current_all = samples["current"].to(device)
    contexts_all = samples["contexts"].to(device)
    current_train = samples["current"][train_mask].to(device)
    target_train = samples["target"][train_mask].to(device)
    context_train = samples["contexts"][train_mask].to(device)
    actions_train = samples["actions"][train_mask].to(device)

    last_losses: dict[str, float] = {}
    for epoch in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        recon_all, _ = model.autoencode(current_all, contexts_all)
        predicted, latent = model.predict_next(current_train, context_train, actions_train)
        recon_loss = nn.functional.mse_loss(recon_all, current_all)
        transition_loss = nn.functional.mse_loss(predicted, target_train)
        residual_loss = model.residual_penalty(latent, context_train, actions_train)
        total_loss = transition_loss + config.autoencode_weight * recon_loss + config.lambda_residual * residual_loss
        total_loss.backward()
        optimizer.step()
        if epoch == config.epochs - 1:
            last_losses = {
                "recon_loss": float(recon_loss.item()),
                "transition_loss": float(transition_loss.item()),
                "residual_loss": float(residual_loss.item()),
                "total_loss": float(total_loss.item()),
            }

    train_one_step, train_interaction = evaluate_one_step(model, samples, samples["train_mask"])
    zero_shot_one_step, holdout_interaction = evaluate_one_step(model, samples, samples["eval_mask"])
    support_one_step, support_interaction = evaluate_one_step(model, samples, samples["support_mask"])
    query_one_step, query_interaction = evaluate_one_step(model, samples, samples["query_mask"])
    query_state_mask = torch.zeros(config.state_count, dtype=torch.bool)
    query_state_mask[~torch.isin(torch.arange(config.state_count), world.support_states)] = True
    zero_rollout3 = evaluate_rollout(model, world, world.holdout_context, horizon=min(3, config.rollout_length), state_mask=query_state_mask)
    zero_rollout5 = evaluate_rollout(model, world, world.holdout_context, horizon=min(5, config.rollout_length), state_mask=query_state_mask)

    adaptation = None
    if evaluate_adaptation:
        adaptation = adapt_to_holdout_context(model, samples, config)

    result: dict[str, Any] = {
        "config": asdict(config),
        "heldout_context": int(world.holdout_context),
        "support_states": world.support_states.tolist(),
        "train_transition_pairs": int(samples["train_mask"].sum().item()),
        "eval_transition_pairs": int(samples["eval_mask"].sum().item()),
        "support_transition_pairs": int(samples["support_mask"].sum().item()),
        "query_transition_pairs": int(samples["query_mask"].sum().item()),
        "autoencode_mse": mse(model.autoencode(current_all, contexts_all)[0], current_all),
        "train_one_step_mse": train_one_step,
        "zero_shot_one_step_mse": zero_shot_one_step,
        "support_one_step_mse": support_one_step,
        "query_one_step_mse": query_one_step,
        "zero_shot_rollout3_mse": zero_rollout3,
        "zero_shot_rollout5_mse": zero_rollout5,
        "interaction_norm_train": train_interaction,
        "interaction_norm_holdout": holdout_interaction,
        "interaction_norm_support": support_interaction,
        "interaction_norm_query": query_interaction,
        "transition_train_mask": world.transition_train_mask.to(torch.int64).tolist(),
        "support_mask": world.support_mask.to(torch.int64).tolist(),
        "query_mask": world.query_mask.to(torch.int64).tolist(),
        "final_losses": last_losses,
    }
    if adaptation is not None:
        result["adaptation"] = adaptation
    return result


def aggregate_context_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "autoencode_mse",
        "train_one_step_mse",
        "zero_shot_one_step_mse",
        "support_one_step_mse",
        "query_one_step_mse",
        "zero_shot_rollout3_mse",
        "zero_shot_rollout5_mse",
        "interaction_norm_train",
        "interaction_norm_holdout",
        "interaction_norm_support",
        "interaction_norm_query",
    ]
    if "adaptation" in runs[0]:
        metrics.extend(
            [
                "adaptation.zero_shot_query_mse",
                "adaptation.best_query_mse",
                "adaptation.adaptation_gain",
                "adaptation.steps_to_target",
                "adaptation.residual_norm_final",
            ]
        )
    summary: dict[str, Any] = {}
    for metric_name in metrics:
        if "." in metric_name:
            head, tail = metric_name.split(".", maxsplit=1)
            values = [float(run[head][tail]) for run in runs]
        else:
            values = [float(run[metric_name]) for run in runs]
        key = metric_name.replace(".", "_")
        summary[key] = {
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
        }
    return summary


def render_context_markdown_report(results: dict[str, Any], include_adaptation: bool) -> str:
    lines = [
        "# Context Transfer Benchmark",
        "",
        f"Split strategy: `{results['split_strategy']}`",
        "",
    ]
    for world, variants in results["summary"].items():
        lines.append(f"## World: {world}")
        lines.append("")
        headers = "| Variant | Zero-Shot 1-Step | Rollout@3 | Rollout@5 | Holdout Interaction |"
        separator = "| --- | ---: | ---: | ---: | ---: |"
        if include_adaptation:
            headers = "| Variant | Zero-Shot 1-Step | Rollout@3 | Rollout@5 | Adapt Gain | Steps To Target | Holdout Interaction |"
            separator = "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
        lines.append(headers)
        lines.append(separator)
        for variant, metrics in variants.items():
            if include_adaptation:
                lines.append(
                    f"| {variant} | {metrics['zero_shot_one_step_mse']['mean']:.6f} +/- {metrics['zero_shot_one_step_mse']['std']:.6f} | "
                    f"{metrics['zero_shot_rollout3_mse']['mean']:.6f} +/- {metrics['zero_shot_rollout3_mse']['std']:.6f} | "
                    f"{metrics['zero_shot_rollout5_mse']['mean']:.6f} +/- {metrics['zero_shot_rollout5_mse']['std']:.6f} | "
                    f"{metrics['adaptation_adaptation_gain']['mean']:+.6f} +/- {metrics['adaptation_adaptation_gain']['std']:.6f} | "
                    f"{metrics['adaptation_steps_to_target']['mean']:.2f} +/- {metrics['adaptation_steps_to_target']['std']:.2f} | "
                    f"{metrics['interaction_norm_holdout']['mean']:.6f} +/- {metrics['interaction_norm_holdout']['std']:.6f} |"
                )
            else:
                lines.append(
                    f"| {variant} | {metrics['zero_shot_one_step_mse']['mean']:.6f} +/- {metrics['zero_shot_one_step_mse']['std']:.6f} | "
                    f"{metrics['zero_shot_rollout3_mse']['mean']:.6f} +/- {metrics['zero_shot_rollout3_mse']['std']:.6f} | "
                    f"{metrics['zero_shot_rollout5_mse']['mean']:.6f} +/- {metrics['zero_shot_rollout5_mse']['std']:.6f} | "
                    f"{metrics['interaction_norm_holdout']['mean']:.6f} +/- {metrics['interaction_norm_holdout']['std']:.6f} |"
                )
        lines.append("")
    return "\n".join(lines)


def run_context_transfer_suite(
    seeds: list[int],
    worlds: list[str],
    variants: list[str],
    variant_recipes: dict[str, dict[str, Any]],
    output_json: str | None = None,
    output_markdown: str | None = None,
    split_strategy: str = "holdout_context",
    context_count: int = 5,
    state_count: int = 12,
    action_count: int = 4,
    rollout_length: int = 5,
    image_size: int = 20,
    latent_dim: int = 24,
    context_dim: int = 8,
    hidden_dim: int = 96,
    epochs: int = 220,
    evaluate_adaptation: bool = False,
) -> dict[str, Any]:
    raw_runs: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    for world in worlds:
        summary[world] = {}
        for variant in variants:
            variant_runs = []
            for seed in seeds:
                config = ContextTransferConfig(
                    world=world,
                    variant=variant,
                    seed=seed,
                    split_strategy=split_strategy,
                    context_count=context_count,
                    state_count=state_count,
                    action_count=action_count,
                    rollout_length=rollout_length,
                    image_size=image_size,
                    latent_dim=latent_dim,
                    context_dim=context_dim,
                    hidden_dim=hidden_dim,
                    epochs=epochs,
                    **variant_recipes[variant],
                )
                run = train_one_context_transfer(config, evaluate_adaptation=evaluate_adaptation)
                raw_runs.append(run)
                variant_runs.append(run)
            summary[world][variant] = aggregate_context_runs(variant_runs)

    output = {
        "seeds": seeds,
        "worlds": worlds,
        "variants": variants,
        "variant_recipes": variant_recipes,
        "split_strategy": split_strategy,
        "evaluate_adaptation": evaluate_adaptation,
        "world_metadata": {
            world: {
                "ground_truth_context_commutator": ground_truth_context_commutator(world, context_count),
                "ground_truth_transfer_coupling": ground_truth_transfer_coupling(world, context_count),
                "ground_truth_adaptation_cost_proxy": ground_truth_adaptation_cost_proxy(world, context_count),
                "heldout_context": holdout_context_index(context_count),
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
        path.write_text(render_context_markdown_report(output, include_adaptation=evaluate_adaptation), encoding="utf-8")

    return output
