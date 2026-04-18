import unittest

from structured_latent_hypothesis.routing import (
    calibrate_threshold,
    mean_metric_for_assignments,
    mean_regret_to_oracle,
    oracle_variant,
    route_variant,
    structured_advantage,
)


class RoutingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.results = {
            "summary": {
                "w0": {
                    "coord_latent": {"test_recon_mse": {"mean": 1.0}},
                    "structured_a": {"test_recon_mse": {"mean": 0.8}},
                    "structured_b": {"test_recon_mse": {"mean": 0.9}},
                },
                "w1": {
                    "coord_latent": {"test_recon_mse": {"mean": 1.1}},
                    "structured_a": {"test_recon_mse": {"mean": 1.3}},
                    "structured_b": {"test_recon_mse": {"mean": 1.0}},
                },
            }
        }

    def test_structured_advantage_uses_best_structured_branch(self) -> None:
        advantage = structured_advantage(self.results, "w0", structured_variants=["structured_a", "structured_b"])
        self.assertAlmostEqual(advantage, 0.2)

    def test_calibrate_threshold_finds_perfect_separator(self) -> None:
        threshold, direction, accuracy = calibrate_threshold([1.0, 2.0, 3.0, 4.0], [0.5, 0.2, -0.1, -0.2])
        self.assertEqual(direction, "low_positive")
        self.assertEqual(accuracy, 1.0)
        self.assertGreater(threshold, 2.0)
        self.assertLess(threshold, 3.0)

    def test_route_variant_respects_direction_and_inversion(self) -> None:
        self.assertEqual(route_variant(5.0, 4.0, "high_positive", "structured_a"), "structured_a")
        self.assertEqual(route_variant(3.0, 4.0, "high_positive", "structured_a"), "coord_latent")
        self.assertEqual(route_variant(5.0, 4.0, "high_positive", "structured_a", invert=True), "coord_latent")

    def test_oracle_variant_picks_best_branch(self) -> None:
        self.assertEqual(oracle_variant(self.results, "w0", structured_variants=["structured_a", "structured_b"]), "structured_a")
        self.assertEqual(oracle_variant(self.results, "w1", structured_variants=["structured_a", "structured_b"]), "structured_b")

    def test_assignment_metrics_helpers(self) -> None:
        assignments = {"w0": "structured_a", "w1": "coord_latent"}
        self.assertAlmostEqual(mean_metric_for_assignments(self.results, assignments), 0.95)
        regret = mean_regret_to_oracle(self.results, assignments, structured_variants=["structured_a", "structured_b"])
        self.assertAlmostEqual(regret, 0.05)


if __name__ == "__main__":
    unittest.main()
