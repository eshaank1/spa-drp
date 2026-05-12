#!/usr/bin/env python3
"""
Simple ladder training orchestrator.

Each generation trains against 2 opponents (Previous Gen + Original PPO) for a
fixed number of iterations, evaluates vs all 4 opponents, logs statistics,
and automatically becomes the next generation's champion.

No threshold logic - just train and promote.
"""

import argparse
import glob
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Tuple, Optional


def cleanup_old_champions(champions_dir: Path, out_root: Path, keep_count: int = 5) -> None:
    """Keep only the last `keep_count` champion models, delete older ones and their gen folders."""
    pattern = champions_dir / "champion_gen*.zip"
    models = glob.glob(str(pattern))
    
    if len(models) <= keep_count:
        return
    
    def get_gen_num(path: str) -> int:
        try:
            return int(Path(path).stem.split("gen")[-1])
        except ValueError:
            return -1
    
    models_sorted = sorted(models, key=get_gen_num, reverse=True)
    to_delete = models_sorted[keep_count:]
    
    for model_path in to_delete:
        gen_num = get_gen_num(model_path)
        
        # Delete champion model
        Path(model_path).unlink()
        print(f"Deleted old champion: {Path(model_path).name}")
        
        # Delete corresponding gen folder
        if gen_num >= 0:
            gen_folder = out_root / f"gen_{gen_num}"
            if gen_folder.exists():
                shutil.rmtree(gen_folder)
                print(f"Deleted old generation folder: {gen_folder.name}")


def run_generation(
    generation: int,
    champion_path: str,
    out_root: Path,
    log_file: Path,
    timesteps_total: int,
    eval_episodes: int,
    num_envs: int,
    seed: int,
) -> Tuple[bool, Optional[str]]:
    """Train one generation and return (success, model_path)."""
    gen_dir = out_root / f"gen_{generation}"
    gen_dir.mkdir(parents=True, exist_ok=True)

    model_output = gen_dir / f"champion_gen{generation}.zip"

    cmd = [
        sys.executable,
        "pettingzoo_ppo/train_ladder_challenger.py",
        "--challenger",
        champion_path,
        "--previous-gen",
        champion_path,
        "--original-ppo",
        "pettingzoo_ppo/models/ppo_vs_smart_final.zip",
        "--generation",
        str(generation),
        "--timesteps-total",
        str(timesteps_total),
        "--eval-episodes",
        str(eval_episodes),
        "--num-envs",
        str(num_envs),
        "--seed",
        str(seed + generation),
        "--log-file",
        str(log_file),
        "--model-output",
        str(model_output),
    ]

    print("\n" + "=" * 70)
    print(f"Generation {generation}")
    print("=" * 70)
    print(f"Current champion: {champion_path}")
    print(f"Training for {timesteps_total} timesteps (vs Previous Gen + Original PPO)")
    print(f"Will evaluate vs all 4 opponents")
    print(f"Stats will be logged to: {log_file}")

    subprocess.run(cmd, check=True)

    if model_output.exists():
        return True, str(model_output)

    return False, None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a ladder of bots. Each generation trains against 2 opponents "
            "(Previous Gen + Original PPO) for fixed iterations, evaluates vs 4, "
            "and automatically gets promoted. Statistics logged to CSV."
        )
    )
    parser.add_argument(
        "--initial-champion",
        type=str,
        default="models/ladder/champions/champion_gen1.zip",
        help="Starting champion model path.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=5,
        help="Number of generations to train.",
    )
    parser.add_argument(
        "--timesteps-total",
        type=int,
        default=10000,
        help="Total timesteps per generation (split: 5000 vs each opponent).",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Episodes per opponent during evaluation.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Number of parallel environments.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (will be offset by generation).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="models/ladder",
        help="Output directory for generation artifacts and champions.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="ladder_training_stats.csv",
        help="CSV file to log training statistics (in spa-drp root).",
    )
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    champions_dir = out_root / "champions"
    champions_dir.mkdir(parents=True, exist_ok=True)

    current_champion = str(Path(args.initial_champion))
    if not Path(current_champion).exists():
        raise FileNotFoundError(f"Initial champion not found: {current_champion}")

    # Extract generation number from initial champion
    initial_stem = Path(args.initial_champion).stem
    try:
        initial_gen = int(initial_stem.split("gen")[-1])
    except (ValueError, IndexError):
        initial_gen = 0
    
    log_file = Path(args.log_file)
    
    print(f"\n{'='*70}")
    print("Ladder Training - Simple Mode")
    print(f"{'='*70}")
    print(f"Starting from: {current_champion}")
    print(f"Generations to train: {args.generations}")
    print(f"Timesteps per generation: {args.timesteps_total}")
    print(f"Eval episodes per opponent: {args.eval_episodes}")
    print(f"Statistics log: {log_file}")
    print(f"{'='*70}\n")

    promoted = 0

    for generation in range(initial_gen + 1, initial_gen + args.generations + 1):
        success, trained_model = run_generation(
            generation=generation,
            champion_path=current_champion,
            out_root=out_root,
            log_file=log_file,
            timesteps_total=args.timesteps_total,
            eval_episodes=args.eval_episodes,
            num_envs=args.num_envs,
            seed=args.seed,
        )

        if success and trained_model is not None:
            # Copy to champions directory
            promoted_path = champions_dir / f"champion_gen{generation}.zip"
            shutil.copy2(trained_model, promoted_path)
            current_champion = str(promoted_path)
            promoted += 1
            print(f"\n✓ Promoted: {promoted_path.name}")
            cleanup_old_champions(champions_dir, out_root, keep_count=5)
        else:
            print(f"\n✗ Training failed for generation {generation}")
            break

    print("\n" + "=" * 70)
    print("Ladder training complete")
    print("=" * 70)
    print(f"Generations trained: {promoted}/{args.generations}")
    print(f"Final champion: {current_champion}")
    print(f"Statistics saved to: {log_file}")


if __name__ == "__main__":
    main()
