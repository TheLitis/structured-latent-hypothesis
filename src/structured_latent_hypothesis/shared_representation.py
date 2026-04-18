from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn

from .direct_separable import mixed_difference


@dataclass
class WorldBatch:
    name: str
    images: torch.Tensor
    train_mask: torch.Tensor


@dataclass
class SharedRepresentationConfig:
    image_size: int = 24
    latent_dim: int = 16
    batch_size: int = 64
    epochs: int = 120
    lr: float = 1e-3
    weight_decay: float = 1e-5
    lambda_var: float = 1e-3
    sigma_min: float = 0.25
    seed: int = 17
    device: str = "cpu"


@dataclass
class SharedRepresentationBundle:
    model: nn.Module
    config: SharedRepresentationConfig
    pixel_mean: torch.Tensor
    pixel_std: torch.Tensor
    latent_mean: torch.Tensor
    latent_std: torch.Tensor
    history: list[float]
    best_loss: float


class SharedConvAutoencoder(nn.Module):
    def __init__(self, image_size: int, latent_dim: int) -> None:
        super().__init__()
        self.image_size = image_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((3, 3))
        self.to_latent = nn.Linear(64 * 3 * 3, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 64 * 3 * 3)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        pooled = self.pool(features).reshape(images.shape[0], -1)
        return self.to_latent(pooled)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        features = self.from_latent(latent).reshape(latent.shape[0], 64, 3, 3)
        recon = self.decoder(features)
        if recon.shape[-1] != self.image_size:
            recon = nn.functional.interpolate(
                recon,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return recon

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(images)
        return self.decode(latent), latent


def prepare_world_images(world_tensor: torch.Tensor, image_size: int) -> torch.Tensor:
    world_tensor = world_tensor.to(torch.float32)
    if world_tensor.ndim == 4:
        world_tensor = world_tensor.unsqueeze(2)
    if world_tensor.ndim != 5:
        raise ValueError(f"Expected world tensor with 4 or 5 dimensions, got {world_tensor.shape}.")
    if world_tensor.shape[2] == 1:
        world_tensor = world_tensor.repeat(1, 1, 3, 1, 1)
    if world_tensor.shape[2] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {world_tensor.shape[2]}.")

    grid_size = world_tensor.shape[0]
    flattened = world_tensor.reshape(-1, world_tensor.shape[2], world_tensor.shape[3], world_tensor.shape[4])
    if flattened.shape[-1] != image_size or flattened.shape[-2] != image_size:
        flattened = nn.functional.interpolate(
            flattened,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
    return flattened.reshape(grid_size, grid_size, 3, image_size, image_size)


def train_images(world_batches: list[WorldBatch]) -> torch.Tensor:
    stacked = [batch.images[batch.train_mask] for batch in world_batches]
    if not stacked:
        raise ValueError("No world batches provided.")
    if any(images.numel() == 0 for images in stacked):
        raise ValueError("Each world batch must contribute at least one train image.")
    return torch.cat(stacked, dim=0)


def pixel_statistics(images: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    mean = images.mean(dim=(0, 2, 3), keepdim=True)
    std = images.std(dim=(0, 2, 3), unbiased=False, keepdim=True).clamp_min(eps)
    return mean, std


def normalize_images(images: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (images - mean) / std


def additive_projection(grid: torch.Tensor) -> torch.Tensor:
    row_mean = grid.mean(dim=1, keepdim=True)
    col_mean = grid.mean(dim=0, keepdim=True)
    grand_mean = grid.mean(dim=(0, 1), keepdim=True)
    return row_mean + col_mean - grand_mean


def score_additive_residual(grid: torch.Tensor, eps: float = 1e-8) -> float:
    residual = grid - additive_projection(grid)
    return float(torch.linalg.vector_norm(residual).item() / (eps + torch.linalg.vector_norm(grid).item()))


def score_curvature(grid: torch.Tensor, eps: float = 1e-8) -> float:
    curvature = mixed_difference(grid)
    row_steps = grid[1:, :, :] - grid[:-1, :, :]
    col_steps = grid[:, 1:, :] - grid[:, :-1, :]
    denom = eps + torch.linalg.vector_norm(row_steps).item() + torch.linalg.vector_norm(col_steps).item()
    return float(torch.linalg.vector_norm(curvature).item() / denom)


def score_diagonal_concentration(grid: torch.Tensor, eps: float = 1e-8) -> float:
    curvature = mixed_difference(grid)
    total_energy = float(curvature.pow(2).sum().item())
    if total_energy < eps:
        return 1.0

    diag_energy = 0.0
    rows, cols = curvature.shape[0], curvature.shape[1]
    for diag in range(rows + cols - 1):
        entries = []
        for row in range(rows):
            col = diag - row
            if col < 0 or col >= cols:
                continue
            entries.append(curvature[row, col])
        if not entries:
            continue
        stack = torch.stack(entries, dim=0)
        diag_mean = stack.mean(dim=0)
        diag_energy += float(stack.shape[0]) * float(diag_mean.pow(2).sum().item())
    return float(diag_energy / (total_energy + eps))


def compute_latent_scores(grid: torch.Tensor, eps: float = 1e-8) -> dict[str, float]:
    s_add = score_additive_residual(grid, eps=eps)
    s_curv = score_curvature(grid, eps=eps)
    s_diag = score_diagonal_concentration(grid, eps=eps)
    return {
        "S_add": s_add,
        "S_curv": s_curv,
        "S_diag": s_diag,
        "S_combo": s_curv * (1.0 - s_diag),
        "S_joint": s_diag / (eps + s_curv),
    }


def fit_shared_autoencoder(
    world_batches: list[WorldBatch],
    config: SharedRepresentationConfig,
) -> SharedRepresentationBundle:
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    model = SharedConvAutoencoder(image_size=config.image_size, latent_dim=config.latent_dim).to(device)

    raw_train_images = train_images(world_batches)
    pixel_mean, pixel_std = pixel_statistics(raw_train_images)
    normalized_train = normalize_images(raw_train_images, pixel_mean, pixel_std)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: list[float] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for _ in range(config.epochs):
        permutation = torch.randperm(normalized_train.shape[0])
        epoch_loss = 0.0
        for start in range(0, normalized_train.shape[0], config.batch_size):
            index = permutation[start : start + config.batch_size]
            batch = normalized_train[index].to(device)
            recon, latent = model(batch)
            recon_loss = nn.functional.mse_loss(recon, batch)
            latent_std = latent.std(dim=0, unbiased=False)
            variance_loss = torch.relu(config.sigma_min - latent_std).pow(2).mean()
            loss = recon_loss + config.lambda_var * variance_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * batch.shape[0]
        epoch_loss /= normalized_train.shape[0]
        history.append(epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Autoencoder training did not produce a valid state.")
    model.load_state_dict(best_state)
    model.eval()

    latent_batches = []
    with torch.no_grad():
        for start in range(0, normalized_train.shape[0], config.batch_size):
            batch = normalized_train[start : start + config.batch_size].to(device)
            latent = model.encode(batch)
            latent_batches.append(latent.cpu())
    latent_train = torch.cat(latent_batches, dim=0)
    latent_mean = latent_train.mean(dim=0, keepdim=True)
    latent_std = latent_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)

    return SharedRepresentationBundle(
        model=model,
        config=config,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        latent_mean=latent_mean,
        latent_std=latent_std,
        history=history,
        best_loss=best_loss,
    )


def encode_world_grid(bundle: SharedRepresentationBundle, world_batch: WorldBatch) -> tuple[torch.Tensor, float]:
    device = next(bundle.model.parameters()).device
    grid_size = world_batch.images.shape[0]
    flattened = world_batch.images.reshape(-1, 3, bundle.config.image_size, bundle.config.image_size)
    normalized = normalize_images(flattened, bundle.pixel_mean, bundle.pixel_std).to(device)
    with torch.no_grad():
        recon, latent = bundle.model(normalized)
    recon_mse = float(nn.functional.mse_loss(recon, normalized).item())
    latent = ((latent.cpu() - bundle.latent_mean) / bundle.latent_std).reshape(grid_size, grid_size, -1)
    return latent, recon_mse
