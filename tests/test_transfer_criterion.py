from __future__ import annotations

import math
import unittest

from structured_latent_hypothesis.transfer_criterion import (
    cross_validate_abstain_by_group,
    cross_validate_classifier_by_group,
    evaluate_cost_sensitive_threshold,
    evaluate_router,
    evaluate_cost_sensitive_abstain,
    leave_one_seed_out_router,
    leave_one_seed_out_classifier,
    select_cost_sensitive_abstain,
    select_cost_sensitive_threshold,
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

    def test_cost_sensitive_threshold_prefers_low_scores_for_safe_positive(self) -> None:
        rows = [
            {"score_diag": 0.10, "task_safe": True},
            {"score_diag": 0.15, "task_safe": True},
            {"score_diag": 0.80, "task_safe": False},
            {"score_diag": 0.85, "task_safe": False},
        ]
        selected = select_cost_sensitive_threshold(
            rows,
            "score_diag",
            "task_safe",
            false_positive_cost=3.0,
            false_negative_cost=1.0,
        )
        self.assertEqual(selected["direction"], "low")
        metrics = evaluate_cost_sensitive_threshold(
            rows,
            "score_diag",
            "task_safe",
            threshold=selected["threshold"],
            direction=selected["direction"],
            false_positive_cost=3.0,
            false_negative_cost=1.0,
        )
        self.assertLess(metrics["average_cost"], metrics["always_positive_cost"])
        self.assertLess(metrics["average_cost"], metrics["always_negative_cost"])

    def test_leave_one_seed_out_classifier_aggregates_per_seed(self) -> None:
        rows = [
            {"seed": 1, "score_joint": 0.10, "task_budget": True},
            {"seed": 1, "score_joint": 0.90, "task_budget": False},
            {"seed": 2, "score_joint": 0.12, "task_budget": True},
            {"seed": 2, "score_joint": 0.88, "task_budget": False},
        ]
        metrics = leave_one_seed_out_classifier(
            rows,
            "score_joint",
            "task_budget",
            false_positive_cost=3.0,
            false_negative_cost=1.0,
        )
        self.assertEqual(len(metrics["per_seed"]), 2)
        self.assertLess(metrics["average_cost_mean"], metrics["always_positive_cost_mean"])
        self.assertLess(metrics["average_cost_mean"], metrics["always_negative_cost_mean"])
        self.assertTrue(0.0 <= metrics["balanced_accuracy_mean"] <= 1.0)

    def test_select_cost_sensitive_abstain_can_use_band(self) -> None:
        rows = [
            {"score_diag": 0.10, "task_safe": True},
            {"score_diag": 0.20, "task_safe": True},
            {"score_diag": 0.49, "task_safe": True},
            {"score_diag": 0.51, "task_safe": False},
            {"score_diag": 0.80, "task_safe": False},
            {"score_diag": 0.90, "task_safe": False},
        ]
        selected = select_cost_sensitive_abstain(
            rows,
            "score_diag",
            "task_safe",
            false_positive_cost=3.0,
            false_negative_cost=1.0,
            abstain_positive_cost=0.75,
            abstain_negative_cost=0.25,
        )
        metrics = evaluate_cost_sensitive_abstain(
            rows,
            "score_diag",
            "task_safe",
            threshold=selected["threshold"],
            direction=selected["direction"],
            band=selected["band"],
            false_positive_cost=3.0,
            false_negative_cost=1.0,
            abstain_positive_cost=0.75,
            abstain_negative_cost=0.25,
        )
        self.assertLessEqual(metrics["average_cost"], metrics["always_positive_cost"])
        self.assertLessEqual(metrics["average_cost"], metrics["always_negative_cost"])
        self.assertLessEqual(metrics["average_cost"], metrics["always_abstain_cost"])
        self.assertTrue(0.0 <= metrics["coverage"] <= 1.0)

    def test_cross_validate_classifier_by_group_uses_requested_group_key(self) -> None:
        rows = [
            {"world": "a", "score_joint": 0.10, "task_budget": True},
            {"world": "a", "score_joint": 0.90, "task_budget": False},
            {"world": "b", "score_joint": 0.12, "task_budget": True},
            {"world": "b", "score_joint": 0.88, "task_budget": False},
        ]
        metrics = cross_validate_classifier_by_group(
            rows,
            "score_joint",
            "task_budget",
            group_key="world",
            false_positive_cost=3.0,
            false_negative_cost=1.0,
        )
        self.assertEqual(len(metrics["per_group"]), 2)
        self.assertLess(metrics["average_cost_mean"], metrics["always_positive_cost_mean"])
        self.assertLess(metrics["average_cost_mean"], metrics["always_negative_cost_mean"])

    def test_cross_validate_abstain_by_group_aggregates_coverage(self) -> None:
        rows = [
            {"family": "commuting", "score_diag": 0.10, "task_safe": True},
            {"family": "commuting", "score_diag": 0.20, "task_safe": True},
            {"family": "coupled", "score_diag": 0.80, "task_safe": False},
            {"family": "coupled", "score_diag": 0.90, "task_safe": False},
        ]
        metrics = cross_validate_abstain_by_group(
            rows,
            "score_diag",
            "task_safe",
            group_key="family",
            false_positive_cost=3.0,
            false_negative_cost=1.0,
            abstain_positive_cost=0.75,
            abstain_negative_cost=0.25,
        )
        self.assertEqual(len(metrics["per_group"]), 2)
        self.assertTrue(0.0 <= metrics["coverage_mean"] <= 1.0)
        self.assertTrue(0.0 <= metrics["abstain_rate_mean"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
