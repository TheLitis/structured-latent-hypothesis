import unittest

from structured_latent_hypothesis.synthetic import ground_truth_commutator_magnitude


class MatchedWorldTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
