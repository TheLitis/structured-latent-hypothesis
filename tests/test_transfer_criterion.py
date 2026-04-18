from __future__ import annotations

import math
import unittest

from structured_latent_hypothesis.transfer_criterion import (
    evaluate_router,
    leave_one_seed_out_router,
    select_threshold_router,
    spearman_correlation,
)


class TransferCriterionTests(unittest.TestCase):
    def test_spearman_correlation_is_monotone(self) -> None:
        self.assertAlmostEqual(spearman_correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]), 1.0, places=6)
        self.assertAlmostEqual(spearman_correlation([1.0, 2.0, 3.0], [6.0, 4.0, 2.0]), -1.0, places=6)

    def test_select_threshold_router_picks_low_side_when_low_score_is_safe(self) -> None:
        rows = [
            {
                "score_joint_sum": 0.10,
                "structured_zero_shot_rollout5_mse": 0.10,
                "full_zero_shot_rollout5_mse": 0.20,
            },
            {
                "score_joint_sum": 0.12,
                "structured_zero_shot_rollout5_mse": 0.11,
                "full_zero_shot_rollout5_mse": 0.18,
            },
            {
                "score_joint_sum": 0.80,
                "structured_zero_shot_rollout5_mse": 0.25,
                "full_zero_shot_rollout5_mse": 0.12,
            },
            {
                "score_joint_sum": 0.90,
                "structured_zero_shot_rollout5_mse": 0.24,
                "full_zero_shot_rollout5_mse": 0.13,
            },
        ]
        router = select_threshold_router(rows, "score_joint_sum")
        self.assertEqual(router["direction"], "low")
        metrics = evaluate_router(rows, "score_joint_sum", router["threshold"], router["direction"])
        self.assertLess(metrics["routed_rollout5_mse"], metrics["always_full_rollout5_mse"])
        self.assertLess(metrics["routed_rollout5_mse"], metrics["always_structured_rollout5_mse"])

    def test_leave_one_seed_out_router_aggregates_per_seed(self) -> None:
        rows = [
            {
                "seed": 1,
                "score_interaction": 0.05,
                "structured_zero_shot_rollout5_mse": 0.10,
                "full_zero_shot_rollout5_mse": 0.15,
            },
            {
                "seed": 1,
                "score_interaction": 0.80,
                "structured_zero_shot_rollout5_mse": 0.22,
                "full_zero_shot_rollout5_mse": 0.11,
            },
            {
                "seed": 2,
                "score_interaction": 0.04,
                "structured_zero_shot_rollout5_mse": 0.09,
                "full_zero_shot_rollout5_mse": 0.14,
            },
            {
                "seed": 2,
                "score_interaction": 0.82,
                "structured_zero_shot_rollout5_mse": 0.23,
                "full_zero_shot_rollout5_mse": 0.12,
            },
        ]
        metrics = leave_one_seed_out_router(rows, "score_interaction")
        self.assertEqual(len(metrics["per_seed"]), 2)
        self.assertLess(metrics["routed_rollout5_mse_mean"], metrics["always_full_rollout5_mse_mean"])
        self.assertTrue(0.0 <= metrics["accuracy_mean"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
