import unittest

import torch

from structured_latent_hypothesis.synthetic import generate_world, ground_truth_commutator_magnitude


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


if __name__ == "__main__":
    unittest.main()
