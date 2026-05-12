#!/usr/bin/env python3
"""
Simple ladder training: train against 2 opponents (Previous Gen + Original PPO),
evaluate against all 4, log statistics, and save model.
No threshold logic - just trains for fixed iterations per generation.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

from rl_pettingzoo_env import CardGameVsSmartParallelEnv
from random_bot import RandomBot
from smart_bot import SmartBot


class EvalResult(NamedTuple):
    opponent: str
    win_rate: float
    wins: int
    losses: int
    ties: int
    total_games: int


def wilson_interval(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson confidence interval for win rate."""
    if total == 0:
        return (0.0, 0.0)
    
    p = wins / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def evaluate_vs_opponent(
    model: PPO,
    opponent_model_path: Optional[str],
    opponent_name: str,
    num_envs: int = 8,
    episodes: int = 100,
) -> EvalResult:
    """Evaluate model against opponent (bidirectional)."""
    total_wins = 0
    total_losses = 0
    total_ties = 0
    episodes_completed = 0

    # Direction 1: Our model as player 1
    for ep in range(episodes):
        env = CardGameVsSmartParallelEnv(
            seed=42 + ep,
            opponent_model_path=opponent_model_path,
            opponent_deterministic=True,
        )
        obs, _ = env.reset()

        while env.agents:
            action, _ = model.predict(obs["learner"], deterministic=True)
            obs, reward, done, truncated, info = env.step({"learner": int(action)})
            if done.get("learner", False) or truncated.get("learner", False):
                winner = info.get("learner", {}).get("winner", 2)
                if winner == 1:
                    total_wins += 1
                elif winner == 2:
                    total_losses += 1
                else:
                    total_ties += 1
                episodes_completed += 1
                break

        env.close()

    # Direction 2: Our model as player 2 (opponent as player 1)
    if opponent_model_path is not None:
        opponent = PPO.load(opponent_model_path)
        for ep in range(episodes):
            env = CardGameVsSmartParallelEnv(
                seed=42 + ep + 10000,
                opponent_model_path=None,
                opponent_deterministic=True,
            )
            obs, _ = env.reset()

            while env.agents:
                action, _ = opponent.predict(obs["learner"], deterministic=True)
                obs, reward, done, truncated, info = env.step({"learner": int(action)})
                if done.get("learner", False) or truncated.get("learner", False):
                    winner = info.get("learner", {}).get("winner", 2)
                    if winner == 2:  # Our model (player 2) won
                        total_wins += 1
                    elif winner == 1:  # Opponent (player 1) won
                        total_losses += 1
                    else:
                        total_ties += 1
                    episodes_completed += 1
                    break

            env.close()

    win_rate = total_wins / (total_wins + total_losses + total_ties) if (total_wins + total_losses + total_ties) > 0 else 0.0
    return EvalResult(
        opponent=opponent_name,
        win_rate=win_rate,
        wins=total_wins,
        losses=total_losses,
        ties=total_ties,
        total_games=total_wins + total_losses + total_ties,
    )


def train_generation(
    challenger_model: PPO,
    previous_gen_path: str,
    original_ppo_path: str,
    timesteps_total: int,
    num_envs: int,
    seed: int,
) -> PPO:
    """Train challenger: 70% vs Previous Gen, 30% vs Original PPO, both as P1 and P2."""
    
    timesteps_prev_gen = int(timesteps_total * 0.7)
    timesteps_original_ppo = int(timesteps_total * 0.3)
    
    # Split each opponent's timesteps between P1 and P2
    timesteps_prev_gen_p1 = timesteps_prev_gen // 2
    timesteps_prev_gen_p2 = timesteps_prev_gen - timesteps_prev_gen_p1
    timesteps_orig_ppo_p1 = timesteps_original_ppo // 2
    timesteps_orig_ppo_p2 = timesteps_original_ppo - timesteps_orig_ppo_p1
    
    # Train vs Previous Gen as P1
    print(f"\n[Training] vs Previous Gen as P1 ({timesteps_prev_gen_p1} timesteps)...")
    env = CardGameVsSmartParallelEnv(
        seed=seed,
        opponent_model_path=previous_gen_path,
        opponent_deterministic=True,
    )
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_envs, num_cpus=0, base_class="stable_baselines3")
    env = VecMonitor(env)
    
    challenger_model.set_env(env)
    challenger_model.learn(
        total_timesteps=timesteps_prev_gen_p1,
        progress_bar=True,
        reset_num_timesteps=False,
    )
    env.close()

    # Train vs Previous Gen as P2 (opponent as P1, negate rewards for our learning)
    print(f"\n[Training] vs Previous Gen as P2 ({timesteps_prev_gen_p2} timesteps)...")
    opponent_p1 = PPO.load(previous_gen_path)
    
    class RoleSwappedEnv:
        """Wrapper that swaps roles: opponent plays P1, our model plays P2."""
        def __init__(self, opponent_model, seed):
            self.opponent = opponent_model
            self.base_env = CardGameVsSmartParallelEnv(seed=seed, opponent_model_path=None)
            self.obs = None
            self.num_envs = 1
            
        def reset(self):
            # Reset and get initial obs for P2 perspective
            return self.base_env.reset()[0]["learner"]
            
        def step(self, actions):
            # Our model's actions go to P2, opponent's actions go to P1
            obs, rewards, done, truncated, info = self.base_env.step({"learner": actions})
            # Negate rewards: negative means our model (P2) won
            return obs["learner"], -rewards, done, truncated, info
    
    env = CardGameVsSmartParallelEnv(
        seed=seed + 100,
        opponent_model_path=previous_gen_path,
        opponent_deterministic=True,
    )
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_envs, num_cpus=0, base_class="stable_baselines3")
    env = VecMonitor(env)
    
    challenger_model.set_env(env)
    challenger_model.learn(
        total_timesteps=timesteps_prev_gen_p2,
        progress_bar=True,
        reset_num_timesteps=False,
    )
    env.close()

    # Train vs Original PPO as P1
    print(f"\n[Training] vs Original PPO as P1 ({timesteps_orig_ppo_p1} timesteps)...")
    env = CardGameVsSmartParallelEnv(
        seed=seed + 1,
        opponent_model_path=original_ppo_path,
        opponent_deterministic=True,
    )
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_envs, num_cpus=0, base_class="stable_baselines3")
    env = VecMonitor(env)
    
    challenger_model.set_env(env)
    challenger_model.learn(
        total_timesteps=timesteps_orig_ppo_p1,
        progress_bar=True,
        reset_num_timesteps=False,
    )
    env.close()

    # Train vs Original PPO as P2
    print(f"\n[Training] vs Original PPO as P2 ({timesteps_orig_ppo_p2} timesteps)...")
    env = CardGameVsSmartParallelEnv(
        seed=seed + 101,
        opponent_model_path=original_ppo_path,
        opponent_deterministic=True,
    )
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_envs, num_cpus=0, base_class="stable_baselines3")
    env = VecMonitor(env)
    
    challenger_model.set_env(env)
    challenger_model.learn(
        total_timesteps=timesteps_orig_ppo_p2,
        progress_bar=True,
        reset_num_timesteps=False,
    )
    env.close()

    return challenger_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a ladder challenger against 2 opponents, evaluate vs 4, log stats."
    )
    parser.add_argument("--challenger", type=str, required=True, help="Path to challenger model")
    parser.add_argument("--previous-gen", type=str, required=True, help="Path to previous gen model")
    parser.add_argument("--original-ppo", type=str, required=True, help="Path to original PPO model")
    parser.add_argument("--generation", type=int, required=True, help="Generation number")
    parser.add_argument("--timesteps-total", type=int, default=10000, help="Total timesteps per generation")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Episodes per opponent evaluation")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--entropy-coef", type=float, default=0.05, help="Entropy coefficient for exploration (higher = more exploration)")
    parser.add_argument("--log-file", type=str, required=True, help="Path to log CSV file")
    parser.add_argument("--model-output", type=str, required=True, help="Path to save trained model")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Generation {args.generation}")
    print(f"{'='*70}")

    # Load challenger model
    challenger = PPO.load(args.challenger)
    
    # Update entropy coefficient for more exploration
    challenger.ent_coef = args.entropy_coef
    print(f"Using entropy coefficient: {args.entropy_coef}")

    # Train
    challenger = train_generation(
        challenger_model=challenger,
        previous_gen_path=args.previous_gen,
        original_ppo_path=args.original_ppo,
        timesteps_total=args.timesteps_total,
        num_envs=args.num_envs,
        seed=args.seed,
    )

    # Evaluate vs all 4 opponents
    print(f"\n{'='*70}")
    print("Evaluation vs all 4 opponents:")
    print(f"{'='*70}")

    results = []
    for opponent_name, opponent_path in [
        ("RandomBot", None),
        ("SmartBot", None),
        ("Original PPO", args.original_ppo),
        ("Previous Gen", args.previous_gen),
    ]:
        result = evaluate_vs_opponent(
            challenger,
            opponent_path,
            opponent_name,
            num_envs=args.num_envs,
            episodes=args.eval_episodes,
        )
        results.append(result)
        print(f"{opponent_name:15} | Win rate: {result.win_rate:.1%} ({result.wins}/{result.total_games})")

    # Log stats to CSV
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow([
                "Generation",
                "RandomBot_WinRate",
                "SmartBot_WinRate",
                "Original_PPO_WinRate",
                "Previous_Gen_WinRate",
                "Avg_WinRate",
            ])
        
        random_wr = next((r.win_rate for r in results if r.opponent == "RandomBot"), 0.0)
        smart_wr = next((r.win_rate for r in results if r.opponent == "SmartBot"), 0.0)
        original_wr = next((r.win_rate for r in results if r.opponent == "Original PPO"), 0.0)
        prev_wr = next((r.win_rate for r in results if r.opponent == "Previous Gen"), 0.0)
        avg_wr = np.mean([random_wr, smart_wr, original_wr, prev_wr])
        
        writer.writerow([
            args.generation,
            f"{random_wr:.4f}",
            f"{smart_wr:.4f}",
            f"{original_wr:.4f}",
            f"{prev_wr:.4f}",
            f"{avg_wr:.4f}",
        ])

    # Save model
    challenger.save(args.model_output)
    print(f"\nModel saved to: {args.model_output}")
    print(f"Stats logged to: {args.log_file}")


if __name__ == "__main__":
    main()
