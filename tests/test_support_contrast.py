import unittest

from structured_latent_hypothesis.support_contrast import (
    add_rank_features,
    build_support_contrast_rows,
    cross_validate_rank_calibrated_transfer,
    curve_value,
    evaluate_rank_calibrated_policy,
    evaluate_rank_external_transfer,
    percentile_rank,
    safe_ratio,
    select_and_evaluate_transfer_policy,
)


def run(world: str, seed: int, variant: str, support_curve: list[float], query_best: float, steps: int) -> dict:
    return {
        "config": {"world": world, "seed": seed, "variant": variant},
        "adaptation": {
            "support_curve": support_curve,
            "support_residual_curve": [0.0 for _ in support_curve],
            "best_query_mse": query_best,
            "steps_to_target": steps,
        },
    }


class SupportContrastTests(unittest.TestCase):
    def test_curve_value_clamps_to_last_point(self) -> None:
        self.assertEqual(curve_value([1.0, 0.8], 8), 0.8)

    def test_safe_ratio_handles_zero_denominator(self) -> None:
        self.assertGreater(safe_ratio(1.0, 0.0), 0.0)

    def test_build_support_contrast_rows_compares_structured_to_fallback(self) -> None:
        results = {
            "worlds": ["context_coupled_0.20"],
            "seeds": [3],
            "runs": [
                run("context_coupled_0.20", 3, "full_transition", [1.0, 0.8, 0.7], 0.20, 6),
                run("context_coupled_0.20", 3, "operator_diag_residual", [1.0, 0.7, 0.6], 0.18, 4),
            ],
        }
        rows = build_support_contrast_rows(results, regret_tolerance=0.0, budget=8)
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0]["score_gain_ratio_1"], 1.0)
        self.assertTrue(rows[0]["task_safe"])
        self.assertTrue(rows[0]["task_budget"])

    def test_rank_features_use_reference_distribution(self) -> None:
        rows = [{"score": 10.0}, {"score": 20.0}]
        reference = [{"score": 0.0}, {"score": 10.0}, {"score": 30.0}]
        ranked = add_rank_features(rows, score_keys=["score"], reference_rows=reference)
        self.assertAlmostEqual(percentile_rank(10.0, [0.0, 10.0, 30.0]), 0.5)
        self.assertAlmostEqual(ranked[0]["rank_score"], 0.5)
        self.assertAlmostEqual(ranked[1]["rank_score"], 2.0 / 3.0)

    def test_rank_external_transfer_returns_trivial_comparison(self) -> None:
        synthetic = [
            {"world": "s1", "seed": 1, "score_a": 0.1, "score_b": 0.1, "task_safe": True, "task_budget": False},
            {"world": "s2", "seed": 1, "score_a": 0.9, "score_b": 0.2, "task_safe": False, "task_budget": True},
            {"world": "s3", "seed": 1, "score_a": 1.0, "score_b": 0.9, "task_safe": False, "task_budget": False},
        ]
        target = [
            {"world": "t1", "seed": 1, "score_a": 10.0, "score_b": 10.0, "task_safe": True, "task_budget": False},
            {"world": "t2", "seed": 1, "score_a": 90.0, "score_b": 20.0, "task_safe": False, "task_budget": True},
            {"world": "t3", "seed": 1, "score_a": 100.0, "score_b": 90.0, "task_safe": False, "task_budget": False},
        ]
        result = evaluate_rank_external_transfer(
            synthetic,
            target,
            raw_safe_score_keys=["score_a"],
            raw_budget_score_keys=["score_b"],
            structured_violation_cost=5.0,
            fallback_overbudget_cost=3.0,
            escalate_needed_cost=0.5,
            escalate_unneeded_cost=1.0,
        )
        self.assertIn("delta_to_best_trivial", result)
        self.assertTrue(0.0 <= result["metrics"]["action_rates"]["structured"] <= 1.0)

    def test_rank_calibrated_cv_works_by_world(self) -> None:
        rows = [
            {"world": "a", "seed": 1, "score_a": 0.1, "score_b": 0.1, "task_safe": True, "task_budget": False},
            {"world": "a", "seed": 2, "score_a": 0.2, "score_b": 0.2, "task_safe": True, "task_budget": False},
            {"world": "b", "seed": 1, "score_a": 0.8, "score_b": 0.3, "task_safe": False, "task_budget": True},
            {"world": "b", "seed": 2, "score_a": 0.9, "score_b": 0.4, "task_safe": False, "task_budget": True},
        ]
        metrics = cross_validate_rank_calibrated_transfer(
            rows,
            group_key="world",
            raw_safe_score_keys=["score_a"],
            raw_budget_score_keys=["score_b"],
            structured_violation_cost=5.0,
            fallback_overbudget_cost=3.0,
            escalate_needed_cost=0.5,
            escalate_unneeded_cost=1.0,
        )
        self.assertEqual(len(metrics["per_group"]), 2)
        self.assertTrue(0.0 <= metrics["structured_rate_mean"] <= 1.0)

    def test_select_and_evaluate_transfer_policy_compares_to_trivial(self) -> None:
        train_rows = [
            {"world": "a", "seed": 1, "score_a": 0.1, "score_b": 0.1, "task_safe": True, "task_budget": False},
            {"world": "b", "seed": 1, "score_a": 0.9, "score_b": 0.9, "task_safe": False, "task_budget": True},
        ]
        result = select_and_evaluate_transfer_policy(
            train_rows,
            train_rows,
            safe_score_keys=["score_a"],
            budget_score_keys=["score_b"],
            structured_violation_cost=5.0,
            fallback_overbudget_cost=3.0,
            escalate_needed_cost=0.5,
            escalate_unneeded_cost=1.0,
        )
        self.assertIn("best_trivial_cost", result)
        self.assertEqual(len(result["per_row"]), 2)

    def test_evaluate_rank_calibrated_policy_supports_source_rows(self) -> None:
        calibration = [
            {"world": "a", "seed": 1, "score_a": 0.1, "score_b": 0.1, "task_safe": True, "task_budget": False},
            {"world": "b", "seed": 1, "score_a": 0.9, "score_b": 0.9, "task_safe": False, "task_budget": True},
        ]
        test = [
            {"world": "c", "seed": 1, "score_a": 0.2, "score_b": 0.2, "task_safe": True, "task_budget": False},
            {"world": "d", "seed": 1, "score_a": 0.8, "score_b": 0.8, "task_safe": False, "task_budget": True},
        ]
        result = evaluate_rank_calibrated_policy(
            calibration,
            test,
            raw_safe_score_keys=["score_a"],
            raw_budget_score_keys=["score_b"],
            structured_violation_cost=5.0,
            fallback_overbudget_cost=3.0,
            escalate_needed_cost=0.5,
            escalate_unneeded_cost=1.0,
            source_rows=calibration,
        )
        self.assertIn("delta_to_best_trivial", result)
        self.assertEqual(len(result["per_row"]), 2)


if __name__ == "__main__":
    unittest.main()
