import unittest

from structured_latent_hypothesis.support_contrast import build_support_contrast_rows, curve_value, safe_ratio


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


if __name__ == "__main__":
    unittest.main()
