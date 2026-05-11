from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path


def newest_checkpoint(pattern: str) -> str:
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No checkpoints found matching pattern: {pattern}")
    matches.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume long challenger training from the newest checkpoint."
    )
    parser.add_argument(
        "--checkpoint-pattern",
        type=str,
        default="models/continued/smoke_challenger_cont_iter*.zip",
        help="Glob pattern for selecting the newest challenger checkpoint.",
    )
    parser.add_argument(
        "--target-opponent",
        type=str,
        default="pettingzoo_ppo/models/ppo_vs_smart_final.zip",
        help="Path to PPO opponent model.",
    )
    parser.add_argument("--threshold", type=float, default=0.55, help="Win-rate threshold to stop training.")
    parser.add_argument(
        "--bidirectional-threshold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use aggregated both-direction win rate when checking threshold.",
    )
    parser.add_argument("--timesteps-per-iter", type=int, default=200_000)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--max-iters", type=int, default=30)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=str, default="models/continued")
    parser.add_argument("--model-name", type=str, default="smoke_challenger")
    parser.add_argument("--dry-run", action="store_true", help="Print command and exit without running.")
    args = parser.parse_args()

    challenger = newest_checkpoint(args.checkpoint_pattern)

    cmd = [
        sys.executable,
        "pettingzoo_ppo/continue_training_challenger.py",
        "--challenger",
        challenger,
        "--target-opponent",
        args.target_opponent,
        "--threshold",
        str(args.threshold),
        "--bidirectional-threshold" if args.bidirectional_threshold else "--no-bidirectional-threshold",
        "--timesteps-per-iter",
        str(args.timesteps_per_iter),
        "--eval-episodes",
        str(args.eval_episodes),
        "--max-iters",
        str(args.max_iters),
        "--num-envs",
        str(args.num_envs),
        "--seed",
        str(args.seed),
        "--model-dir",
        args.model_dir,
        "--model-name",
        args.model_name,
    ]

    print(f"Using newest checkpoint: {challenger}")
    print("Running:")
    print(" ".join(cmd))

    if args.dry_run:
        return

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
