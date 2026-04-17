from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def ensure_output_dir() -> Path:
    output_dir = Path("docs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def triplet_points(num_triplets: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0 = 0.0
    y0 = 0.0
    dx_triplet = 1.0
    dy_triplet = 0.55
    dx_group = 4.0

    red = []
    green = []
    blue = []
    for n in range(num_triplets):
        red.append((x0 + n * dx_group, y0))
        green.append((x0 + n * dx_group + dx_triplet, y0 + dy_triplet))
        blue.append((x0 + n * dx_group + 2 * dx_triplet, y0 + 2 * dy_triplet))
    return np.asarray(red), np.asarray(green), np.asarray(blue)


def render_triplet_construction(output_dir: Path) -> None:
    red, green, blue = triplet_points(num_triplets=1)
    figure, axis = plt.subplots(figsize=(6.2, 4.6))
    axis.scatter(red[:, 0], red[:, 1], s=220, color="#d62728", label="point 1")
    axis.scatter(green[:, 0], green[:, 1], s=220, color="#2ca02c", label="point 2")
    axis.scatter(blue[:, 0], blue[:, 1], s=220, color="#1f77b4", label="point 3")
    axis.plot([red[0, 0], green[0, 0], blue[0, 0]], [red[0, 1], green[0, 1], blue[0, 1]], color="#555555", linewidth=2.2)
    axis.annotate(
        "equal in-triplet step",
        xy=((red[0, 0] + green[0, 0]) / 2, (red[0, 1] + green[0, 1]) / 2),
        xytext=(0.3, 1.25),
        arrowprops={"arrowstyle": "->", "lw": 1.4},
        fontsize=10,
    )
    axis.annotate(
        "equal in-triplet step",
        xy=((green[0, 0] + blue[0, 0]) / 2, (green[0, 1] + blue[0, 1]) / 2),
        xytext=(1.9, 1.75),
        arrowprops={"arrowstyle": "->", "lw": 1.4},
        fontsize=10,
    )
    axis.set_title("Triplet Construction")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_xlim(-0.8, 3.4)
    axis.set_ylim(-0.4, 2.2)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_dir / "triplet_construction.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_diagonal_subsequence(output_dir: Path) -> None:
    red, green, blue = triplet_points(num_triplets=5)
    figure, axis = plt.subplots(figsize=(9.5, 4.8))

    axis.scatter(red[:, 0], red[:, 1], s=170, color="#d62728", label="point 1")
    axis.scatter(green[:, 0], green[:, 1], s=170, color="#2ca02c", label="point 2")
    axis.scatter(blue[:, 0], blue[:, 1], s=170, color="#1f77b4", label="point 3")

    for n in range(len(red)):
        axis.plot([red[n, 0], green[n, 0], blue[n, 0]], [red[n, 1], green[n, 1], blue[n, 1]], color="#777777", linewidth=1.8)
        if n < len(red) - 1:
            axis.plot([green[n, 0], red[n + 1, 0]], [green[n, 1], red[n + 1, 1]], color="#bbbbbb", linewidth=1.0, linestyle="--")

    for n in range(1, len(green) - 1):
        diagonal_x = [blue[n - 1, 0], green[n, 0], red[n + 1, 0]]
        diagonal_y = [blue[n - 1, 1], green[n, 1], red[n + 1, 1]]
        axis.plot(diagonal_x, diagonal_y, color="#8c3fd1", linewidth=2.3, alpha=0.95)

    axis.fill_between([-1.0, 20.0], -0.2, 1.4, color="#f4f4f4", alpha=0.75, zorder=-5)
    axis.axhline(-0.2, color="#555555", linestyle=":", linewidth=1.2)
    axis.axhline(1.4, color="#555555", linestyle=":", linewidth=1.2)
    axis.text(15.6, 1.47, "y max", fontsize=9)
    axis.text(15.6, -0.34, "y min", fontsize=9)
    axis.annotate(
        "diagonal subsequence:\npoint 3 -> point 2 -> point 1",
        xy=(green[2, 0], green[2, 1]),
        xytext=(9.1, 1.95),
        arrowprops={"arrowstyle": "->", "lw": 1.4},
        fontsize=10,
    )
    axis.set_title("Repeated Triplets and the Diagonal Subsequence")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_xlim(-1.0, 19.0)
    axis.set_ylim(-0.5, 2.5)
    axis.grid(alpha=0.22)
    axis.legend(loc="upper left", ncol=3)
    figure.tight_layout()
    figure.savefig(output_dir / "diagonal_subsequence.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    output_dir = ensure_output_dir()
    render_triplet_construction(output_dir)
    render_diagonal_subsequence(output_dir)


if __name__ == "__main__":
    main()
