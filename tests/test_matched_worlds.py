import unittest

import torch

from structured_latent_hypothesis.synthetic import (
    generate_world,
    ground_truth_commutator_magnitude,
    ground_truth_coupling_strength,
    ground_truth_step_drift_magnitude,
)


class MatchedWorldTest(unittest.TestCase):
    def test_ramp_world_zero_strength_matches_commutative_world(self) -> None:
        reference = generate_world("commutative", 8, 20)
        candidate = generate_world("matched_comm_0.00", 8, 20)
        self.assertTrue(torch.allclose(reference, candidate))

    def test_scale_world_zero_strength_matches_commutative_world(self) -> None:
        reference = generate_world("commutative", 8, 20)
        candidate = generate_world("matched_scale_0.00", 8, 20)
        self.assertTrue(torch.allclose(reference, candidate))

    def test_rotation_world_zero_strength_matches_commutative_world(self) -> None:
        reference = generate_world("commutative", 8, 20)
        candidate = generate_world("matched_rotate_0.00", 8, 20)
        self.assertTrue(torch.allclose(reference, candidate))

    def test_step_curve_gamma_one_matches_commutative_world(self) -> None:
        reference = generate_world("commutative", 8, 20)
        candidate = generate_world("stepcurve_1.00", 8, 20)
        self.assertTrue(torch.allclose(reference, candidate))

    def test_step_curve_coupled_zero_matches_stepcurve_world(self) -> None:
        reference = generate_world("stepcurve_4.00", 8, 20)
        candidate = generate_world("stepcurve_coupled_4.00_0.00", 8, 20)
        self.assertTrue(torch.allclose(reference, candidate))

    def test_ramp_commutator_magnitude_increases_with_strength(self) -> None:
        values = [
            ground_truth_commutator_magnitude("matched_comm_0.00", 8, 20),
            ground_truth_commutator_magnitude("matched_comm_0.20", 8, 20),
            ground_truth_commutator_magnitude("matched_comm_0.50", 8, 20),
        ]
        self.assertIsNotNone(values[0])
        self.assertLess(values[0], values[1])
        self.assertLess(values[1], values[2])

    def test_scale_commutator_magnitude_increases_with_strength(self) -> None:
        values = [
            ground_truth_commutator_magnitude("matched_scale_0.00", 8, 20),
            ground_truth_commutator_magnitude("matched_scale_0.20", 8, 20),
            ground_truth_commutator_magnitude("matched_scale_0.50", 8, 20),
        ]
        self.assertIsNotNone(values[0])
        self.assertLess(values[0], values[1])
        self.assertLess(values[1], values[2])

    def test_rotation_commutator_magnitude_increases_with_strength(self) -> None:
        values = [
            ground_truth_commutator_magnitude("matched_rotate_0.00", 8, 20),
            ground_truth_commutator_magnitude("matched_rotate_10.00", 8, 20),
            ground_truth_commutator_magnitude("matched_rotate_30.00", 8, 20),
        ]
        self.assertIsNotNone(values[0])
        self.assertLess(values[0], values[1])
        self.assertLess(values[1], values[2])

    def test_step_curve_has_zero_commutator_but_positive_step_drift(self) -> None:
        self.assertEqual(ground_truth_commutator_magnitude("stepcurve_1.00", 8, 20), 0.0)
        self.assertEqual(ground_truth_commutator_magnitude("stepcurve_2.00", 8, 20), 0.0)
        self.assertLess(float(ground_truth_step_drift_magnitude("stepcurve_1.00", 8)), 1e-12)
        self.assertGreater(float(ground_truth_step_drift_magnitude("stepcurve_2.00", 8)), 0.0)
        self.assertGreater(float(ground_truth_commutator_magnitude("stepcurve_path_1.00", 8, 20)), 0.0)
        self.assertGreater(float(ground_truth_step_drift_magnitude("stepcurve_path_2.00", 8)), 0.0)

    def test_step_curve_coupled_commutator_increases_with_interaction(self) -> None:
        values = [
            ground_truth_commutator_magnitude("stepcurve_coupled_4.00_0.00", 8, 20),
            ground_truth_commutator_magnitude("stepcurve_coupled_4.00_0.20", 8, 20),
            ground_truth_commutator_magnitude("stepcurve_coupled_4.00_0.50", 8, 20),
        ]
        self.assertEqual(values[0], 0.0)
        self.assertLess(values[0], values[1])
        self.assertLess(values[1], values[2])
        self.assertEqual(ground_truth_coupling_strength("stepcurve_coupled_4.00_0.50"), 0.5)

    def test_semireal_world_has_rgb_shape_and_valid_range(self) -> None:
        world = generate_world("semireal_coupled_0.35", 8, 24)
        self.assertEqual(world.shape, torch.Size([8, 8, 3, 24, 24]))
        self.assertGreaterEqual(float(world.min().item()), 0.0)
        self.assertLessEqual(float(world.max().item()), 1.0)

    def test_semireal_commutator_increases_with_coupling(self) -> None:
        values = [
            ground_truth_commutator_magnitude("semireal_coupled_0.00", 6, 24),
            ground_truth_commutator_magnitude("semireal_coupled_0.20", 6, 24),
            ground_truth_commutator_magnitude("semireal_coupled_0.50", 6, 24),
        ]
        self.assertEqual(values[0], 0.0)
        self.assertLess(values[0], values[1])
        self.assertLess(values[1], values[2])
        self.assertEqual(ground_truth_coupling_strength("semireal_coupled_0.35"), 0.35)


if __name__ == "__main__":
    unittest.main()
