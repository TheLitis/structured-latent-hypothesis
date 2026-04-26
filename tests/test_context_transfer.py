import unittest

import torch

from structured_latent_hypothesis.context_transfer import (
    CommutingOperatorModel,
    ContextTransferConfig,
    action_vectors,
    configure_adaptation_parameters,
    generate_context_transfer_world,
    ground_truth_adaptation_cost_proxy,
    ground_truth_context_commutator,
    ground_truth_transfer_coupling,
    parse_context_world,
    train_one_context_transfer,
)


class ContextTransferTest(unittest.TestCase):
    def test_alpha_zero_anchor_matches_between_families(self) -> None:
        commuting = generate_context_transfer_world(
            ContextTransferConfig(world="context_commuting_0.00", variant="v", seed=3, model_type="commuting_operator", image_size=12, state_count=6)
        )
        coupled = generate_context_transfer_world(
            ContextTransferConfig(world="context_coupled_0.00", variant="v", seed=3, model_type="commuting_operator", image_size=12, state_count=6)
        )
        self.assertLess(float((commuting.frames - coupled.frames).abs().max().item()), 1e-8)
        self.assertLess(float((commuting.positions - coupled.positions).abs().max().item()), 1e-8)

    def test_semireal_alpha_zero_anchor_matches_between_families(self) -> None:
        commuting = generate_context_transfer_world(
            ContextTransferConfig(
                world="semireal_context_commuting_0.00",
                variant="v",
                seed=3,
                model_type="commuting_operator",
                image_size=12,
                state_count=6,
            )
        )
        coupled = generate_context_transfer_world(
            ContextTransferConfig(
                world="semireal_context_coupled_0.00",
                variant="v",
                seed=3,
                model_type="commuting_operator",
                image_size=12,
                state_count=6,
            )
        )
        self.assertLess(float((commuting.frames - coupled.frames).abs().max().item()), 1e-8)
        self.assertLess(float((commuting.positions - coupled.positions).abs().max().item()), 1e-8)

    def test_metadata_is_monotone_on_coupled_family(self) -> None:
        worlds = [f"context_coupled_{alpha:0.2f}" for alpha in [0.00, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]]
        comm = [ground_truth_context_commutator(world, context_count=5) for world in worlds]
        coupling = [ground_truth_transfer_coupling(world, context_count=5) for world in worlds]
        adaptation = [ground_truth_adaptation_cost_proxy(world, context_count=5) for world in worlds]
        self.assertEqual(comm[0], 0.0)
        self.assertEqual(coupling[0], 0.0)
        self.assertEqual(adaptation[0], 0.0)
        self.assertTrue(all(left <= right + 1e-9 for left, right in zip(comm, comm[1:])))
        self.assertTrue(all(left <= right + 1e-9 for left, right in zip(coupling, coupling[1:])))
        self.assertTrue(all(left <= right + 1e-9 for left, right in zip(adaptation, adaptation[1:])))

    def test_semireal_metadata_is_monotone_on_coupled_family(self) -> None:
        worlds = [f"semireal_context_coupled_{alpha:0.2f}" for alpha in [0.00, 0.20, 0.35, 0.75]]
        comm = [ground_truth_context_commutator(world, context_count=5) for world in worlds]
        coupling = [ground_truth_transfer_coupling(world, context_count=5) for world in worlds]
        adaptation = [ground_truth_adaptation_cost_proxy(world, context_count=5) for world in worlds]
        self.assertEqual(comm[0], 0.0)
        self.assertEqual(coupling[0], 0.0)
        self.assertEqual(adaptation[0], 0.0)
        self.assertTrue(all(left <= right + 1e-9 for left, right in zip(comm, comm[1:])))
        self.assertTrue(all(left <= right + 1e-9 for left, right in zip(coupling, coupling[1:])))
        self.assertTrue(all(left <= right + 1e-9 for left, right in zip(adaptation, adaptation[1:])))

    def test_holdout_context_is_absent_from_transition_train(self) -> None:
        config = ContextTransferConfig(world="context_coupled_0.35", variant="v", seed=3, model_type="commuting_operator")
        world = generate_context_transfer_world(config)
        heldout = world.holdout_context
        self.assertFalse(bool(world.transition_train_mask[heldout].any().item()))
        self.assertTrue(bool(world.transition_eval_mask[heldout].all().item()))
        self.assertTrue(bool(world.support_mask[heldout].any().item()))
        self.assertTrue(bool(world.query_mask[heldout].any().item()))

    def test_commuting_operator_smoke_predicts_finite_images(self) -> None:
        model = CommutingOperatorModel(
            image_size=12,
            latent_dim=8,
            context_dim=4,
            hidden_dim=24,
            context_count=5,
            action_count=4,
        )
        images = torch.rand(6, 3, 12, 12)
        contexts = torch.tensor([0, 1, 2, 3, 4, 0], dtype=torch.long)
        actions = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.long)
        predicted, latent = model.predict_next(images, contexts, actions)
        self.assertEqual(predicted.shape, torch.Size([6, 3, 12, 12]))
        self.assertEqual(latent.shape, torch.Size([6, 8]))
        self.assertTrue(torch.isfinite(predicted).all())

    def test_adaptation_configuration_only_enables_context_parameters(self) -> None:
        model = CommutingOperatorModel(
            image_size=12,
            latent_dim=8,
            context_dim=4,
            hidden_dim=24,
            context_count=5,
            action_count=4,
        )
        trainable = configure_adaptation_parameters(model)
        self.assertTrue(any(name.startswith("context_embeddings") for name in trainable))
        self.assertFalse(any(name.startswith("action_shift") for name in trainable))
        self.assertFalse(model.action_shift.requires_grad)
        self.assertTrue(model.context_embeddings.requires_grad)

    def test_train_one_context_transfer_with_adaptation_returns_metrics(self) -> None:
        result = train_one_context_transfer(
            ContextTransferConfig(
                world="context_coupled_0.20",
                variant="smoke",
                seed=3,
                model_type="operator_plus_residual",
                context_count=4,
                state_count=6,
                action_count=4,
                rollout_length=3,
                image_size=12,
                latent_dim=10,
                context_dim=4,
                hidden_dim=24,
                interaction_rank=2,
                epochs=4,
                adapt_steps=3,
            ),
            evaluate_adaptation=True,
        )
        self.assertIn("adaptation", result)
        self.assertGreaterEqual(result["adaptation"]["steps_to_target"], 0)
        self.assertTrue(torch.isfinite(torch.tensor(result["zero_shot_one_step_mse"])))
        self.assertTrue(torch.isfinite(torch.tensor(result["zero_shot_rollout3_mse"])))


if __name__ == "__main__":
    unittest.main()
