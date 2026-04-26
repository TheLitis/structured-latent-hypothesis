from __future__ import annotations

import math
import unittest

from structured_latent_hypothesis.transfer_criterion import (
    action_cost,
    cross_validate_abstain_by_group,
    cross_validate_classifier_by_group,
    cross_validate_transfer_decision_policy,
    cross_validate_transfer_decision_policy_cost_shift,
    evaluate_cost_sensitive_threshold,
    evaluate_router,
    evaluate_cost_sensitive_abstain,
    evaluate_transfer_decision_policy,
    leave_one_seed_out_router,
    leave_one_seed_out_classifier,
    predict_abstain_label,
    select_cost_sensitive_abstain,
    select_cost_sensitive_threshold,
    select_transfer_decision_policy,
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

    def test_transfer_decision_policy_prefers_structured_then_fallback(self) -> None:
        rows = [
            {
                "world": "a",
                "seed": 1,
                "safe_score": 0.10,
                "budget_score": 0.20,
                "task_safe": True,
                "task_budget": False,
            },
            {
                "world": "b",
                "seed": 1,
                "safe_score": 0.80,
                "budget_score": 0.10,
                "task_safe": False,
                "task_budget": True,
            },
            {
                "world": "c",
                "seed": 1,
                "safe_score": 0.90,
                "budget_score": 0.85,
                "task_safe": False,
                "task_budget": False,
            },
        ]
        metrics = evaluate_transfer_decision_policy(
            rows,
            safe_score_key="safe_score",
            budget_score_key="budget_score",
            safe_threshold=0.20,
            safe_direction="low",
            safe_band=0.0,
            budget_threshold=0.20,
            budget_direction="low",
            budget_band=0.0,
            safe_label_key="task_safe",
            budget_label_key="task_budget",
            structured_violation_cost=5.0,
            fallback_overbudget_cost=2.0,
            escalate_needed_cost=1.0,
            escalate_unneeded_cost=1.5,
        )
        self.assertAlmostEqual(metrics["average_cost"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["always_fallback_cost"], 4.0 / 3.0)
        self.assertAlmostEqual(metrics["always_escalate_cost"], (1.0 + 1.5 + 1.0) / 3.0)
        self.assertEqual(metrics["action_rates"]["structured"], 1 / 3)
        self.assertEqual(metrics["action_rates"]["fallback"], 1 / 3)
        self.assertEqual(metrics["action_rates"]["escalate"], 1 / 3)

    def test_cross_validate_transfer_decision_policy_works_by_group(self) -> None:
        rows = [
            {
                "world": "a",
                "seed": 1,
                "variant": "operator_diag_residual",
                "score_interaction": 0.10,
                "score_residual": 0.10,
                "score_joint_sum": 0.20,
                "score_joint_prod": 0.01,
                "task_safe": True,
                "task_budget": False,
            },
            {
                "world": "a",
                "seed": 2,
                "variant": "operator_diag_residual",
                "score_interaction": 0.12,
                "score_residual": 0.11,
                "score_joint_sum": 0.23,
                "score_joint_prod": 0.0132,
                "task_safe": True,
                "task_budget": False,
            },
            {
                "world": "b",
                "seed": 1,
                "variant": "operator_diag_residual",
                "score_interaction": 0.85,
                "score_residual": 0.20,
                "score_joint_sum": 1.05,
                "score_joint_prod": 0.17,
                "task_safe": False,
                "task_budget": True,
            },
            {
                "world": "b",
                "seed": 2,
                "variant": "operator_diag_residual",
                "score_interaction": 0.88,
                "score_residual": 0.21,
                "score_joint_sum": 1.09,
                "score_joint_prod": 0.1848,
                "task_safe": False,
                "task_budget": True,
            },
            {
                "world": "c",
                "seed": 1,
                "variant": "operator_diag_residual",
                "score_interaction": 0.92,
                "score_residual": 0.40,
                "score_joint_sum": 1.32,
                "score_joint_prod": 0.368,
                "task_safe": False,
                "task_budget": False,
            },
            {
                "world": "c",
                "seed": 2,
                "variant": "operator_diag_residual",
                "score_interaction": 0.95,
                "score_residual": 0.42,
                "score_joint_sum": 1.37,
                "score_joint_prod": 0.399,
                "task_safe": False,
                "task_budget": False,
            },
        ]
        metrics = cross_validate_transfer_decision_policy(
            rows,
            group_key="world",
            safe_score_keys=["score_residual", "score_joint_sum"],
            budget_score_keys=["score_interaction", "score_joint_sum"],
            safe_label_key="task_safe",
            budget_label_key="task_budget",
            structured_violation_cost=5.0,
            fallback_overbudget_cost=2.0,
            escalate_needed_cost=1.0,
            escalate_unneeded_cost=1.5,
        )
        self.assertEqual(len(metrics["per_group"]), 3)
        self.assertLess(metrics["average_cost_mean"], metrics["always_structured_cost_mean"])
        self.assertLess(metrics["average_cost_mean"], metrics["always_fallback_cost_mean"])
        self.assertLessEqual(metrics["average_cost_mean"], metrics["always_escalate_cost_mean"])

    def test_transfer_decision_policy_cost_shift_uses_eval_costs(self) -> None:
        rows = [
            {
                "world": "a",
                "seed": 1,
                "variant": "operator_diag_residual",
                "score_interaction": 0.10,
                "score_residual": 0.10,
                "score_joint_sum": 0.20,
                "score_joint_prod": 0.01,
                "task_safe": True,
                "task_budget": False,
            },
            {
                "world": "a",
                "seed": 2,
                "variant": "operator_diag_residual",
                "score_interaction": 0.12,
                "score_residual": 0.11,
                "score_joint_sum": 0.23,
                "score_joint_prod": 0.0132,
                "task_safe": True,
                "task_budget": False,
            },
            {
                "world": "b",
                "seed": 1,
                "variant": "operator_diag_residual",
                "score_interaction": 0.85,
                "score_residual": 0.20,
                "score_joint_sum": 1.05,
                "score_joint_prod": 0.17,
                "task_safe": False,
                "task_budget": True,
            },
            {
                "world": "b",
                "seed": 2,
                "variant": "operator_diag_residual",
                "score_interaction": 0.88,
                "score_residual": 0.21,
                "score_joint_sum": 1.09,
                "score_joint_prod": 0.1848,
                "task_safe": False,
                "task_budget": True,
            },
        ]
        metrics = cross_validate_transfer_decision_policy_cost_shift(
            rows,
            group_key="world",
            safe_score_keys=["score_residual", "score_joint_sum"],
            budget_score_keys=["score_interaction", "score_joint_sum"],
            safe_label_key="task_safe",
            budget_label_key="task_budget",
            train_structured_violation_cost=5.0,
            train_fallback_overbudget_cost=2.0,
            train_escalate_needed_cost=1.0,
            train_escalate_unneeded_cost=1.5,
            eval_structured_violation_cost=9.0,
            eval_fallback_overbudget_cost=4.0,
            eval_escalate_needed_cost=0.5,
            eval_escalate_unneeded_cost=1.0,
        )
        self.assertEqual(len(metrics["per_group"]), 2)
        self.assertTrue(0.0 <= metrics["structured_rate_mean"] <= 1.0)
        self.assertTrue(0.0 <= metrics["fallback_rate_mean"] <= 1.0)
        self.assertAlmostEqual(metrics["always_escalate_cost_mean"], 0.75)


if __name__ == "__main__":
    unittest.main()
