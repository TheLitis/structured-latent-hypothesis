from __future__ import annotations

import argparse
import json

from structured_latent_hypothesis.synthetic import run_benchmark_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the initial CFP synthetic benchmark suite.")
    parser.add_argument("--output-json", default="results/initial_synthetic.json")
    parser.add_argument("--output-markdown", default="results/initial_synthetic.md")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=280)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 11, 29])
    parser.add_argument("--worlds", nargs="+", default=["commutative", "noncommutative"])
    parser.add_argument("--variants", nargs="+", default=["baseline", "smooth", "comm", "comm_step"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run_benchmark_suite(
        seeds=args.seeds,
        worlds=args.worlds,
        variants=args.variants,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        grid_size=args.grid_size,
        image_size=args.image_size,
        epochs=args.epochs,
    )
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
