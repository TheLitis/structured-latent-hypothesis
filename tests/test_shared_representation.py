import unittest

import torch

from structured_latent_hypothesis.direct_separable import integrate_curvature_field
from structured_latent_hypothesis.shared_representation import (
    SharedRepresentationConfig,
    WorldBatch,
    compute_latent_scores,
    encode_world_grid,
    fit_shared_autoencoder,
    prepare_world_images,
    score_additive_residual,
    score_curvature,
    score_diagonal_concentration,
)
from structured_latent_hypothesis.synthetic import generate_world


class SharedRepresentationTest(unittest.TestCase):
    def test_prepare_world_images_expands_grayscale_to_rgb(self) -> None:
        world = generate_world("stepcurve_coupled_4.00_0.20", grid_size=4, image_size=10)
        prepared = prepare_world_images(world, image_size=12)
        self.assertEqual(prepared.shape, torch.Size([4, 4, 3, 12, 12]))
        self.assertTrue(torch.isfinite(prepared).all())

    def test_additive_and_curvature_scores_vanish_on_additive_grid(self) -> None:
        row = torch.tensor([[0.0, 1.0], [1.0, 0.5], [2.0, 1.5]])
        col = torch.tensor([[0.5, -0.5], [1.0, 0.0], [1.5, 0.5]])
        grid = row[:, None, :] + col[None, :, :]
        self.assertLess(score_additive_residual(grid), 1e-9)
        self.assertLess(score_curvature(grid), 1e-9)

    def test_diagonal_concentration_is_high_for_diagonal_curvature_field(self) -> None:
        kappa = torch.zeros(5, 5, 2)
        for index in range(5):
            kappa[index, index] = torch.tensor([1.0, -0.5])
        grid = integrate_curvature_field(kappa, grid_size=6)
        concentration = score_diagonal_concentration(grid)
        self.assertGreater(concentration, 0.50)

    def test_shared_autoencoder_smoke_produces_finite_scores(self) -> None:
        worlds = [
            WorldBatch(
                name="stepcurve",
                images=prepare_world_images(generate_world("stepcurve_coupled_4.00_0.20", 4, 12), image_size=12),
                train_mask=torch.ones(4, 4, dtype=torch.bool),
            ),
            WorldBatch(
                name="semireal",
                images=prepare_world_images(generate_world("semireal_coupled_0.20", 4, 12), image_size=12),
                train_mask=torch.ones(4, 4, dtype=torch.bool),
            ),
        ]
        bundle = fit_shared_autoencoder(
            worlds,
            SharedRepresentationConfig(
                image_size=12,
                latent_dim=4,
                batch_size=8,
                epochs=2,
                lr=2e-3,
                lambda_var=1e-3,
                sigma_min=0.1,
                seed=7,
            ),
        )
        latent_grid, recon_mse = encode_world_grid(bundle, worlds[0])
        scores = compute_latent_scores(latent_grid)
        self.assertTrue(torch.isfinite(torch.tensor(recon_mse)))
        self.assertEqual(latent_grid.shape, torch.Size([4, 4, 4]))
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in scores.values()))


if __name__ == "__main__":
    unittest.main()
