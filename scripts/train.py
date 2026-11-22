"""Train one agent for one seed.

Usage:
    uv run python scripts/train.py configs/fetch_reach_sac_her.yaml --seed 0
"""
import argparse

from niryo_lift.rl.train import load_config, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing run directory")
    args = parser.parse_args()

    run_dir = train(load_config(args.config), seed=args.seed, overwrite=args.overwrite)
    print(f"done: {run_dir}")


if __name__ == "__main__":
    main()
