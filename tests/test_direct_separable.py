import unittest

import torch

from structured_latent_hypothesis.direct_separable import (
    AdditiveLatentDecoder,
    AdditiveInteractionMLPDecoder,
    AdditiveLowRankDecoder,
    AdditiveOperatorDiagDecoder,
    AdditiveOperatorDecoder,
    CoordLatentDecoder,
    DirectBenchmarkConfig,
    train_direct_with_nested_selection,
    train_one_direct,
)


class DirectSeparableTest(unittest.TestCase):
    def test_additive_decoder_without_residual_has_zero_residual_grid(self) -> None:
        model = AdditiveLatentDecoder(grid_size=6, latent_dim=4, hidden_dim=24, output_dim=16, use_residual=False)
        residual = model.residual_grid()
        self.assertEqual(residual.shape, torch.Size([6, 6, 4]))
        self.assertLess(float(residual.abs().max().item()), 1e-12)

    def test_coord_decoder_outputs_grid_shaped_latents(self) -> None:
        model = CoordLatentDecoder(grid_size=6, latent_dim=4, hidden_dim=24, output_dim=16)
        recon, latent, residual = model()
        self.assertEqual(recon.shape, torch.Size([6, 6, 16]))
        self.assertEqual(latent.shape, torch.Size([6, 6, 4]))
        self.assertLess(float(residual.abs().max().item()), 1e-12)

    def test_low_rank_decoder_centers_interaction_main_effects(self) -> None:
        model = AdditiveLowRankDecoder(grid_size=6, latent_dim=4, hidden_dim=24, output_dim=16, interaction_rank=2)
        residual = model.residual_grid()
        self.assertEqual(residual.shape, torch.Size([6, 6, 4]))
        self.assertLess(float(residual.mean(dim=0).abs().max().item()), 1e-6)
        self.assertLess(float(residual.mean(dim=1).abs().max().item()), 1e-6)

    def test_interaction_mlp_decoder_centers_interaction_main_effects(self) -> None:
        model = AdditiveInteractionMLPDecoder(grid_size=6, latent_dim=4, hidden_dim=24, output_dim=16, interaction_rank=2)
        residual = model.residual_grid()
        self.assertEqual(residual.shape, torch.Size([6, 6, 4]))
        self.assertLess(float(residual.mean(dim=0).abs().max().item()), 1e-6)
        self.assertLess(float(residual.mean(dim=1).abs().max().item()), 1e-6)

    def test_operator_decoder_centers_interaction_main_effects(self) -> None:
        model = AdditiveOperatorDecoder(grid_size=6, latent_dim=4, hidden_dim=24, output_dim=16, interaction_rank=2)
        residual = model.residual_grid()
        self.assertEqual(residual.shape, torch.Size([6, 6, 4]))
        self.assertLess(float(residual.mean(dim=0).abs().max().item()), 1e-6)
        self.assertLess(float(residual.mean(dim=1).abs().max().item()), 1e-6)

    def test_operator_diag_decoder_centers_interaction_main_effects(self) -> None:
        model = AdditiveOperatorDiagDecoder(grid_size=6, latent_dim=4, hidden_dim=24, output_dim=16, interaction_rank=2)
        residual = model.residual_grid()
        self.assertEqual(residual.shape, torch.Size([6, 6, 4]))
        self.assertLess(float(residual.mean(dim=0).abs().max().item()), 1e-6)
        self.assertLess(float(residual.mean(dim=1).abs().max().item()), 1e-6)

    def test_additive_model_generalizes_better_than_cell_latent_on_stepcurve_world(self) -> None:
        shared = {
            "world": "stepcurve_1.00",
            "seed": 3,
            "grid_size": 6,
            "image_size": 12,
            "latent_dim": 4,
            "hidden_dim": 48,
            "train_fraction": 0.78,
            "epochs": 120,
        }
        cell = train_one_direct(DirectBenchmarkConfig(variant="cell", model_type="cell", **shared))
        additive = train_one_direct(DirectBenchmarkConfig(variant="additive", model_type="additive", **shared))

        self.assertLess(additive["test_recon_mse"], cell["test_recon_mse"])

    def test_nested_selection_returns_choice_and_scores(self) -> None:
        config = DirectBenchmarkConfig(
            world="stepcurve_1.00",
            variant="selected",
            seed=3,
            model_type="additive_residual",
            grid_size=8,
            image_size=12,
            latent_dim=4,
            hidden_dim=48,
            train_fraction=0.78,
            epochs=60,
        )
        result = train_direct_with_nested_selection(
            config,
            candidate_recipes={
                "l001": {"model_type": "additive_residual", "lambda_residual": 0.01},
                "l050": {"model_type": "additive_residual", "lambda_residual": 0.05},
            },
        )

        self.assertIn("selection", result)
        self.assertEqual(result["selection"]["mode"], "nested")
        self.assertIn(result["selection"]["chosen_candidate"], {"l001", "l050"})
        self.assertEqual(len(result["selection"]["scores"]), 2)
        self.assertGreater(result["selection"]["realized_inner_train_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
