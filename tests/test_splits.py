import unittest

import torch

from structured_latent_hypothesis.synthetic import (
    cartesian_block_train_mask,
    cell_mask,
    sample_nested_train_mask,
    sample_random_train_mask,
)


class SplitStrategyTest(unittest.TestCase):
    def test_cartesian_block_split_has_unseen_combinations(self) -> None:
        mask = cartesian_block_train_mask(8)
        self.assertEqual(mask.shape, torch.Size([8, 8]))
        self.assertEqual(int((~mask).sum().item()), 16)
        self.assertGreater(int(mask.sum().item()), int((~mask).sum().item()))
        self.assertGreater(int(cell_mask(mask).sum().item()), 0)
        self.assertTrue(mask[1, 0].item())
        self.assertFalse(mask[1, 1].item())
        self.assertFalse(mask[5, 6].item())

    def test_nested_split_stays_inside_outer_train_mask(self) -> None:
        outer = cartesian_block_train_mask(8)
        inner = sample_nested_train_mask(outer, seed=3)
        inner_val = outer & ~inner

        self.assertTrue(torch.all(inner <= outer))
        self.assertGreater(int(inner.sum().item()), 0)
        self.assertGreater(int(inner_val.sum().item()), 0)
        self.assertGreater(int(cell_mask(inner).sum().item()), 0)
        self.assertTrue(torch.all(inner.sum(dim=1)[outer.sum(dim=1) > 0] >= 2))
        self.assertTrue(torch.all(inner.sum(dim=0)[outer.sum(dim=0) > 0] >= 2))

    def test_nested_split_tracks_requested_keep_fraction(self) -> None:
        outer = cartesian_block_train_mask(8)
        inner = sample_nested_train_mask(outer, seed=3, keep_fraction=0.72)
        realized = float(inner.sum().item()) / float(outer.sum().item())

        self.assertLess(abs(realized - 0.72), 0.08)

    def test_nested_split_supports_random_outer_mask(self) -> None:
        outer = sample_random_train_mask(8, train_fraction=0.78, seed=3)
        inner = sample_nested_train_mask(outer, seed=3, keep_fraction=0.72)
        inner_val = outer & ~inner

        self.assertTrue(torch.all(inner <= outer))
        self.assertGreater(int(inner_val.sum().item()), 0)
        self.assertTrue(torch.all(inner.sum(dim=1)[outer.sum(dim=1) > 0] >= 2))
        self.assertTrue(torch.all(inner.sum(dim=0)[outer.sum(dim=0) > 0] >= 2))


if __name__ == "__main__":
    unittest.main()
