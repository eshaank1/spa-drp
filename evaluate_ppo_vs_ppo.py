from __future__ import annotations

# This script evaluates a challenger PPO bot against a target PPO bot.
# Default mode: challenger is player 1 (learner), opponent is player 2.
# Bidirectional mode: also swaps roles and combines both directions to reduce
# first-player bias in the reported challenger win rate.

import argparse
import math

from stable_baselines3 import PPO

from pettingzoo_ppo.rl_pettingzoo_env import CardGameVsSmartParallelEnv


def wilson_interval(successes: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    phat = successes / n
    denom = 1 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) + (z * z) / (4 * n)) / n)) / denom
    return center - margin, center + margin


def evaluate_direction(model_a_path: str, model_b_path: str, episodes: int, seed: int, deterministic: bool = True):
    """Evaluate model A as learner (player 1) against model B as opponent (player 2)."""
    model_a = PPO.load(model_a_path)

    wins = 0
    losses = 0
    ties = 0

    for ep in range(episodes):
        env = CardGameVsSmartParallelEnv(
            seed=seed + ep,
            opponent_model_path=model_b_path,
            opponent_deterministic=True,
        )
        obs, _ = env.reset()

        final_info = None
        while env.agents:
            action, _ = model_a.predict(obs["learner"], deterministic=deterministic)
            obs, reward, term, trunc, info = env.step({"learner": int(action)})
            final_info = info.get("learner", final_info)

            if term.get("learner", False) or trunc.get("learner", False):
                break

        if final_info is None:
            losses += 1
            env.close()
            continue

        winner = final_info.get("winner", 2)
        rounds = final_info.get("final_rounds_won", (0, 0))

        if rounds[0] == rounds[1]:
            ties += 1
        elif winner == 1:
            wins += 1
        else:
            losses += 1

        env.close()

    return wins, losses, ties


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate challenger PPO vs opponent PPO.")
    parser.add_argument("--challenger-model", type=str, required=True, help="Path to challenger PPO .zip")
    parser.add_argument("--opponent-model", type=str, required=True, help="Path to opponent PPO .zip")
    parser.add_argument("--episodes", type=int, default=1000, help="Episodes per direction")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic challenger policy")
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Evaluate both role directions and aggregate results.",
    )
    args = parser.parse_args()

    deterministic = not args.stochastic

    # Direction 1: challenger as learner (player 1)
    wins_fwd, losses_fwd, ties_fwd = evaluate_direction(
        model_a_path=args.challenger_model,
        model_b_path=args.opponent_model,
        episodes=args.episodes,
        seed=args.seed,
        deterministic=deterministic,
    )

    total_wins = wins_fwd
    total_losses = losses_fwd
    total_ties = ties_fwd
    total_episodes = args.episodes

    if args.bidirectional:
        # Direction 2: opponent as learner vs challenger as opponent
        # Convert back to challenger perspective:
        # challenger_wins_in_reverse = opponent_losses
        wins_rev_for_opp, losses_rev_for_opp, ties_rev = evaluate_direction(
            model_a_path=args.opponent_model,
            model_b_path=args.challenger_model,
            episodes=args.episodes,
            seed=args.seed + 10_000,
            deterministic=deterministic,
        )

        total_wins += losses_rev_for_opp
        total_losses += wins_rev_for_opp
        total_ties += ties_rev
        total_episodes += args.episodes

    low, high = wilson_interval(total_wins, total_episodes)

    print(f"Challenger: {args.challenger_model}")
    print(f"Opponent: {args.opponent_model}")
    print(f"Bidirectional: {args.bidirectional}")
    print(f"Episodes (total): {total_episodes}")
    print(f"Wins: {total_wins} | Losses: {total_losses} | Ties: {total_ties}")
    print(f"Win rate: {total_wins / total_episodes * 100:.2f}%")
    print(f"95% CI (Wilson): [{low * 100:.2f}%, {high * 100:.2f}%]")


if __name__ == "__main__":
    main()
