import unittest

from structured_latent_hypothesis.routing import (
    aggregate_metric_records,
    choose_abstain_band,
    choose_best_directional_threshold,
    coverage_rate,
    decisions_to_predictions,
    filtered_sign_accuracy,
    random_router_decisions,
    route_decision_label,
    structured_route_rate,
)


class RoutingV2Test(unittest.TestCase):
    def test_choose_direction_picks_high_positive_region(self) -> None:
        scores = [0.1, 0.2, 0.8, 0.9]
        labels = [False, False, True, True]
        coord_values = [1.0, 1.0, 1.0, 1.0]
        structured_values = [1.2, 1.1, 0.8, 0.7]

        threshold, direction, balanced, regret, margin, accuracy = choose_best_directional_threshold(
            scores,
            labels,
            coord_values,
            structured_values,
        )

        self.assertEqual(direction, "high_positive")
        self.assertGreater(threshold, 0.2)
        self.assertLess(threshold, 0.8)
        self.assertAlmostEqual(balanced, 1.0)
        self.assertAlmostEqual(regret, 0.0)
        self.assertGreater(margin, 0.0)
        self.assertAlmostEqual(accuracy, 1.0)

    def test_choose_direction_picks_low_positive_region(self) -> None:
        scores = [-2.0, -1.0, 1.0, 2.0]
        labels = [True, True, False, False]
        coord_values = [1.0, 1.0, 1.0, 1.0]
        structured_values = [0.7, 0.8, 1.1, 1.2]

        threshold, direction, balanced, regret, margin, accuracy = choose_best_directional_threshold(
            scores,
            labels,
            coord_values,
            structured_values,
        )

        self.assertEqual(direction, "low_positive")
        self.assertGreater(threshold, -1.0)
        self.assertLess(threshold, 1.0)
        self.assertAlmostEqual(balanced, 1.0)
        self.assertAlmostEqual(regret, 0.0)
        self.assertGreater(margin, 0.0)
        self.assertAlmostEqual(accuracy, 1.0)

    def test_choose_abstain_band_prefers_safe_fallback(self) -> None:
        scores = [0.10, 0.20, 0.50, 0.80, 0.90]
        labels = [False, False, False, True, True]
        coord_values = [1.0, 1.0, 1.0, 1.0, 1.0]
        structured_values = [1.2, 1.1, 1.3, 0.8, 0.7]

        band, balanced, regret, coverage, accuracy = choose_abstain_band(
            scores,
            labels,
            coord_values,
            structured_values,
            threshold=0.50,
            direction="high_positive",
        )

        self.assertGreater(band, 0.0)
        self.assertAlmostEqual(balanced, 1.0)
        self.assertLess(coverage, 1.0)
        self.assertAlmostEqual(regret, 0.0)
        self.assertAlmostEqual(accuracy, 1.0)
        self.assertEqual(route_decision_label(0.50, 0.50, "high_positive", abstain_band=band), "abstain")

    def test_coverage_and_filtered_accuracy_respect_abstain(self) -> None:
        decisions = ["coord", "abstain", "structured", "coord"]
        labels = [False, True, True, False]
        predictions = decisions_to_predictions(decisions)
        distances = [0.001, 1e-6, 0.002, 0.003]

        self.assertAlmostEqual(coverage_rate(decisions), 0.75)
        self.assertAlmostEqual(structured_route_rate(decisions), 0.25)
        self.assertAlmostEqual(filtered_sign_accuracy(labels, predictions, distances, minimum_distance=1e-5), 1.0)

    def test_random_router_is_deterministic_for_fixed_seed(self) -> None:
        worlds = ["w0", "w1", "w2", "w3"]
        left = random_router_decisions(worlds, structured_rate=0.35, seed=17)
        right = random_router_decisions(worlds, structured_rate=0.35, seed=17)
        other = random_router_decisions(worlds, structured_rate=0.35, seed=18)

        self.assertEqual(left, right)
        self.assertNotEqual(left, other)

    def test_aggregate_metric_records_returns_mean_std(self) -> None:
        aggregated = aggregate_metric_records(
            [
                {"mse": 1.0, "filtered": None},
                {"mse": 3.0, "filtered": 0.5},
                {"mse": 5.0, "filtered": 0.7},
            ]
        )

        self.assertAlmostEqual(aggregated["mse"]["mean"], 3.0)
        self.assertAlmostEqual(aggregated["mse"]["std"], (8.0 / 3.0) ** 0.5)
        self.assertEqual(aggregated["mse"]["count"], 3.0)
        self.assertAlmostEqual(aggregated["filtered"]["mean"], 0.6)
        self.assertEqual(aggregated["filtered"]["count"], 2.0)


if __name__ == "__main__":
    unittest.main()
