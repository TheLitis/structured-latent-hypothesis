import unittest

import torch

from structured_latent_hypothesis.optimizer_geometry import (
    OptimizerGeometryConfig,
    make_quadratic_objective,
    off_diagonal_curvature_score,
    order_sensitivity,
    random_orthonormal,
    run_low_mixed_curvature_basis,
    run_oja_subspace,
    run_optimizer_geometry_once,
)


class OptimizerGeometryTest(unittest.TestCase):
    def test_quadratic_objective_has_expected_shapes(self) -> None:
        objective = make_quadratic_objective(OptimizerGeometryConfig(alpha=0.20, variant="adam_full", seed=3))
        self.assertEqual(objective.q_matrix.shape, torch.Size([128, 128]))
        self.assertEqual(objective.active_basis.shape, torch.Size([128, 8]))
        self.assertEqual(objective.theta0.shape, torch.Size([128]))

    def test_off_diagonal_curvature_grows_with_alpha_in_active_basis(self) -> None:
        low = make_quadratic_objective(OptimizerGeometryConfig(alpha=0.10, variant="adam_full", seed=3))
        high = make_quadratic_objective(OptimizerGeometryConfig(alpha=0.75, variant="adam_full", seed=3))
        self.assertLessEqual(
            off_diagonal_curvature_score(low.q_matrix, low.active_basis),
            off_diagonal_curvature_score(high.q_matrix, high.active_basis) + 1e-12,
        )

    def test_order_sensitivity_is_finite(self) -> None:
        objective = make_quadratic_objective(OptimizerGeometryConfig(alpha=0.35, variant="adam_full", seed=3))
        basis = random_orthonormal(128, 8)
        value = order_sensitivity(objective.q_matrix, basis, objective.theta0)
        self.assertGreaterEqual(value, 0.0)
        self.assertTrue(torch.isfinite(torch.tensor(value)))

    def test_oja_and_low_mixed_curvature_smoke(self) -> None:
        config = OptimizerGeometryConfig(alpha=0.20, variant="oja_subspace_diag", seed=3, steps=20, basis_refresh=5)
        objective = make_quadratic_objective(config)
        oja = run_oja_subspace(objective, config, full_preconditioner=False)
        proposed = run_low_mixed_curvature_basis(objective, config)
        self.assertEqual(len(oja["losses"]), 20)
        self.assertEqual(len(proposed["losses"]), 20)
        self.assertTrue(torch.isfinite(torch.tensor(oja["off_diagonal_curvature"])))
        self.assertTrue(torch.isfinite(torch.tensor(proposed["off_diagonal_curvature"])))

    def test_optimizer_probe_is_reproducible_for_fixed_seed(self) -> None:
        config = OptimizerGeometryConfig(alpha=0.35, variant="low_mixed_curvature_basis", seed=11, steps=40)
        left = run_optimizer_geometry_once(config)
        right = run_optimizer_geometry_once(config)
        self.assertAlmostEqual(left["final_loss"], right["final_loss"])
        self.assertAlmostEqual(left["loss_auc"], right["loss_auc"])


if __name__ == "__main__":
    unittest.main()
