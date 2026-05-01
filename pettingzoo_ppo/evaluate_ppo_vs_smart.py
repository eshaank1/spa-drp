from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl_pettingzoo_env import CardGameVsSmartParallelEnv


def wilson_interval(successes: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    phat = successes / n
    denom = 1 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) + (z * z) / (4 * n)) / n)) / denom
    return center - margin, center + margin


def run_evaluation(model_path: str, episodes: int, seed: int, deterministic: bool):
    model = PPO.load(model_path)

    wins = 0
    losses = 0
    ties = 0
    rewards = []

    for ep in range(episodes):
        env = CardGameVsSmartParallelEnv(seed=seed + ep)
        obs, _ = env.reset()

        total_reward = 0.0
        final_info = None

        while env.agents:
            action, _ = model.predict(obs["learner"], deterministic=deterministic)
            obs, reward, term, trunc, info = env.step({"learner": int(action)})
            total_reward += reward.get("learner", 0.0)
            final_info = info.get("learner", final_info)

            if term.get("learner", False) or trunc.get("learner", False):
                break

        rewards.append(total_reward)

        if final_info is None:
            losses += 1
            continue

        winner = final_info.get("winner", 2)
        rounds = final_info.get("final_rounds_won", (0, 0))

        if rounds[0] == rounds[1]:
            ties += 1
        elif winner == 1:
            wins += 1
        else:
            losses += 1

    low, high = wilson_interval(wins, episodes)
    print(f"Episodes: {episodes}")
    print(f"Wins: {wins} | Losses: {losses} | Ties: {ties}")
    print(f"Win rate: {wins / episodes * 100:.2f}%")
    print(f"95% CI (Wilson): [{low * 100:.2f}%, {high * 100:.2f}%]")
    print(f"Mean episodic reward: {np.mean(rewards):.3f}")

    if wins / episodes >= 0.7:
        print("Status: beating SmartBot consistently (>= 70% win rate).")
    else:
        print("Status: not yet consistently beating SmartBot. Train longer or tune hyperparameters.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO agent vs SmartBot.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy at evaluation.")
    args = parser.parse_args()

    run_evaluation(
        model_path=args.model_path,
        episodes=args.episodes,
        seed=args.seed,
        deterministic=not args.stochastic,
    )


if __name__ == "__main__":
    main()
