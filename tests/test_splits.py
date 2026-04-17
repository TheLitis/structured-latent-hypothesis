import unittest

import torch

from structured_latent_hypothesis.synthetic import cartesian_block_train_mask, cell_mask


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


if __name__ == "__main__":
    unittest.main()
