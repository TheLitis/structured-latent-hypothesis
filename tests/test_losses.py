import unittest

import torch

from structured_latent_hypothesis.synthetic import mixed_difference_loss


class MixedDifferenceLossTest(unittest.TestCase):
    def test_affine_grid_has_zero_mixed_difference(self) -> None:
        grid_size = 5
        latent_dim = 3
        base = torch.tensor([0.2, -0.1, 0.4])
        u = torch.tensor([1.0, 0.5, -0.25])
        v = torch.tensor([-0.3, 0.7, 0.9])

        rows = []
        for i in range(grid_size):
            cols = []
            for j in range(grid_size):
                cols.append(base + i * u + j * v)
            rows.append(torch.stack(cols))
        z_grid = torch.stack(rows)

        mask = torch.ones((grid_size - 1, grid_size - 1), dtype=torch.bool)
        loss = mixed_difference_loss(z_grid, mask)
        self.assertLess(float(loss.item()), 1e-10)


if __name__ == "__main__":
    unittest.main()
