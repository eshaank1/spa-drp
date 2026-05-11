from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl_pettingzoo_env import CardGameVsSmartParallelEnv


def build_vec_env(num_envs: int, seed: int, opponent_model_path: Optional[str] = None, opponent_deterministic: bool = True):
    env = CardGameVsSmartParallelEnv(
        seed=seed,
        invalid_action_penalty=1.0,
        opponent_model_path=opponent_model_path,
        opponent_deterministic=opponent_deterministic,
    )
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_envs, num_cpus=0, base_class="stable_baselines3")
    env = VecMonitor(env)
    return env


def wilson_interval(successes: int, n: int, z: float = 1.96):
    if n <= 0:
        return 0.0, 0.0
    phat = successes / n
    denom = 1 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) + (z * z) / (4 * n)) / n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def evaluate_direction(learner_model: PPO, opponent_model_path: str, episodes: int, seed: int, deterministic: bool = True):
    """Evaluate learner model as player 1 against opponent model as player 2."""
    wins = 0
    ties = 0
    losses = 0

    for ep in range(episodes):
        env = CardGameVsSmartParallelEnv(seed=seed + ep, opponent_model_path=opponent_model_path, opponent_deterministic=True)
        obs, _ = env.reset()
        final_info = None

        while env.agents:
            action, _ = learner_model.predict(obs["learner"], deterministic=deterministic)
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


def evaluate_model_vs_opponent(
    challenger_model: PPO,
    challenger_model_path: str,
    opponent_model_path: str,
    episodes: int,
    seed: int,
    deterministic: bool = True,
    bidirectional: bool = True,
):
    """Evaluate challenger vs opponent and optionally aggregate both role directions."""
    wins_fwd, losses_fwd, ties_fwd = evaluate_direction(
        learner_model=challenger_model,
        opponent_model_path=opponent_model_path,
        episodes=episodes,
        seed=seed,
        deterministic=deterministic,
    )

    total_wins = wins_fwd
    total_losses = losses_fwd
    total_ties = ties_fwd
    total_episodes = episodes

    if bidirectional:
        # Reverse direction: opponent as learner, challenger as opponent.
        # Convert back to challenger perspective.
        opponent_as_learner = PPO.load(opponent_model_path)
        wins_opp_rev, losses_opp_rev, ties_rev = evaluate_direction(
            learner_model=opponent_as_learner,
            opponent_model_path=challenger_model_path,
            episodes=episodes,
            seed=seed + 10_000,
            deterministic=deterministic,
        )

        total_wins += losses_opp_rev
        total_losses += wins_opp_rev
        total_ties += ties_rev
        total_episodes += episodes

    win_rate = total_wins / total_episodes if total_episodes > 0 else 0.0
    ci = wilson_interval(total_wins, total_episodes)
    return win_rate, ci, (total_wins, total_losses, total_ties, total_episodes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue training a saved challenger against a target PPO until win-rate threshold is met.")
    parser.add_argument("--challenger", type=str, required=True, help="Path to the challenger .zip to continue training.")
    parser.add_argument("--target-opponent", type=str, required=True, help="Path to the target PPO .zip to beat.")
    parser.add_argument("--threshold", type=float, default=0.55, help="Win-rate threshold to stop training.")
    parser.add_argument(
        "--bidirectional-threshold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, threshold uses aggregated win rate from both role directions.",
    )
    parser.add_argument("--timesteps-per-iter", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--max-iters", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--model-name", type=str, default="ppo_challenger")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Build training environment where opponent is the fixed target
    env = build_vec_env(num_envs=args.num_envs, seed=args.seed, opponent_model_path=args.target_opponent, opponent_deterministic=True)

    checkpoint_cb = CheckpointCallback(save_freq=max(10_000 // max(args.num_envs, 1), 1), save_path=str(model_dir / "checkpoints"), name_prefix=args.model_name)

    # Load the challenger and continue training
    model = PPO.load(args.challenger, env=env)

    for it in range(1, args.max_iters + 1):
        print(f"Continue-iter {it}: training {args.timesteps_per_iter} steps against {args.target_opponent}...")
        model.learn(total_timesteps=args.timesteps_per_iter, callback=checkpoint_cb, reset_num_timesteps=False)

        saved = model_dir / f"{args.model_name}_cont_iter{it}"
        model.save(saved)
        print(f"Saved continued model: {saved}.zip")

        print("Evaluating continued model vs target opponent...")
        win_rate, ci, counts = evaluate_model_vs_opponent(
            challenger_model=model,
            challenger_model_path=str(saved) + ".zip",
            opponent_model_path=args.target_opponent,
            episodes=args.eval_episodes,
            seed=args.seed + it,
            deterministic=True,
            bidirectional=args.bidirectional_threshold,
        )
        wins, losses, ties, total_eps = counts
        print(
            f"Bidirectional={args.bidirectional_threshold} | Episodes={total_eps} | "
            f"Wins={wins} Losses={losses} Ties={ties}"
        )
        print(f"Win rate: {win_rate*100:.2f}% | 95% CI: [{ci[0]*100:.2f}%, {ci[1]*100:.2f}%]")

        if win_rate >= args.threshold:
            final_path = model_dir / f"{args.model_name}_final"
            model.save(final_path)
            print(f"Reached threshold. Saved final model to: {final_path}.zip")
            env.close()
            return

    print("Max iterations reached without meeting threshold.")
    env.close()


if __name__ == "__main__":
    main()
