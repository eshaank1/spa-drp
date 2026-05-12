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


def calculate_adaptive_timesteps(eval_results: dict[str, float], total_timesteps: int) -> dict[str, int]:
    """
    Calculate timestep allocation based on evaluation results.
    Opponents with worse performance get more training time.
    
    eval_results: dict with opponent names as keys and win rates as values
    Returns: dict with opponent names as keys and timesteps as values
    """
    # Invert win rates to get "weakness scores" (1 - win_rate = how bad we are)
    weakness_scores = {name: max(0, 1.0 - wr) for name, wr in eval_results.items()}
    
    total_weakness = sum(weakness_scores.values())
    if total_weakness == 0:
        # Equal allocation if no weaknesses detected
        num_opponents = len(eval_results)
        return {name: total_timesteps // num_opponents for name in eval_results.keys()}
    
    # Allocate proportionally to weakness
    allocation = {}
    remaining = total_timesteps
    for i, (name, weakness) in enumerate(weakness_scores.items()):
        if i == len(weakness_scores) - 1:
            # Last opponent gets remainder to avoid rounding errors
            allocation[name] = remaining
        else:
            steps = int((weakness / total_weakness) * total_timesteps)
            allocation[name] = steps
            remaining -= steps
    
    return allocation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a challenger with adaptive timestep allocation based on opponent weaknesses."
    )
    parser.add_argument("--challenger", type=str, required=True, help="Path to the challenger .zip to continue training.")
    parser.add_argument("--previous-gen", type=str, required=True, help="Path to the previous generation champion.")
    parser.add_argument("--original-ppo", type=str, default="pettingzoo_ppo/models/ppo_vs_smart_final.zip", help="Path to original PPO bot.")
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

    # Training order: RandomBot → SmartBot → Original PPO → Previous Gen
    # (None = SmartBot/RandomBot, paths = PPO models)
    opponents_train_order = [
        ("RandomBot", None),
        ("SmartBot", None),
        ("Original PPO", args.original_ppo),
        ("Previous Gen", args.previous_gen),
    ]

    # Build initial training environment
    env = CardGameVsSmartParallelEnv(
        seed=args.seed,
        invalid_action_penalty=1.0,
        opponent_model_path=opponents_train_order[0][1],
        opponent_deterministic=True,
    )
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, args.num_envs, num_cpus=0, base_class="stable_baselines3")
    env = VecMonitor(env)

    checkpoint_cb = CheckpointCallback(save_freq=max(10_000 // max(args.num_envs, 1), 1), save_path=str(model_dir / "checkpoints"), name_prefix=args.model_name)

    # Load the challenger
    model = PPO.load(args.challenger, env=env)

    # Initialize timestep allocation (equal split first iteration)
    timestep_allocation = {name: args.timesteps_per_iter // len(opponents_train_order) for name, _ in opponents_train_order}
    
    for it in range(1, args.max_iters + 1):
        print(f"\nContinue-iter {it}: training {args.timesteps_per_iter} steps with adaptive opponent weighting...")
        print("Timestep allocation:")
        for name, steps in timestep_allocation.items():
            print(f"  {name}: {steps} steps")
        
        # Train against each opponent for allocated timesteps
        for (opp_name, opp_path), steps in zip(opponents_train_order, timestep_allocation.values()):
            print(f"\n  Training {steps} steps vs {opp_name}...")
            
            # Rebuild env for this opponent
            env.close()
            env = CardGameVsSmartParallelEnv(
                seed=args.seed + it,
                invalid_action_penalty=1.0,
                opponent_model_path=opp_path,
                opponent_deterministic=True,
            )
            env = ss.pettingzoo_env_to_vec_env_v1(env)
            env = ss.concat_vec_envs_v1(env, args.num_envs, num_cpus=0, base_class="stable_baselines3")
            env = VecMonitor(env)
            model.set_env(env)
            
            model.learn(total_timesteps=steps, callback=checkpoint_cb, reset_num_timesteps=False)

        saved = model_dir / f"{args.model_name}_cont_iter{it}"
        model.save(saved)
        print(f"\nSaved continued model: {saved}.zip")

        # Evaluate against all four opponents
        print("\n" + "=" * 70)
        print(f"EVALUATION ITERATION {it}")
        print("=" * 70)

        opponents_eval = [
            ("RandomBot", None),
            ("SmartBot", None),
            ("Original PPO", args.original_ppo),
            ("Previous Gen", args.previous_gen),
        ]

        all_passed = True
        results = []
        eval_win_rates = {}

        for opp_name, opp_path in opponents_eval:
            print(f"\nEvaluating vs {opp_name}...")

            if opp_name in ["SmartBot", "RandomBot"]:
                deterministic = opp_name != "RandomBot"
                eval_env = CardGameVsSmartParallelEnv(
                    seed=args.seed + it + (0 if opp_name == "SmartBot" else 100),
                    invalid_action_penalty=1.0,
                    opponent_model_path=None,
                    opponent_deterministic=deterministic,
                )
                wins, losses, ties = 0, 0, 0
                for ep in range(args.eval_episodes):
                    obs, _ = eval_env.reset()
                    final_info = None
                    while eval_env.agents:
                        action, _ = model.predict(obs["learner"], deterministic=True)
                        obs, reward, term, trunc, info = eval_env.step({"learner": int(action)})
                        final_info = info.get("learner", final_info)
                        if term.get("learner", False) or trunc.get("learner", False):
                            break

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

                eval_env.close()
                total_episodes = args.eval_episodes
                win_rate = wins / total_episodes
                ci = wilson_interval(wins, total_episodes)
                counts = (wins, losses, ties, total_episodes)

            else:
                win_rate, ci, counts = evaluate_model_vs_opponent(
                    challenger_model=model,
                    challenger_model_path=str(saved) + ".zip",
                    opponent_model_path=opp_path,
                    episodes=args.eval_episodes,
                    seed=args.seed + it,
                    deterministic=True,
                    bidirectional=args.bidirectional_threshold,
                )

            wins, losses, ties, total_episodes = counts
            passed = win_rate >= args.threshold
            all_passed = all_passed and passed
            status = "✓ PASS" if passed else "✗ FAIL"

            print(f"{status} | {opp_name}: {win_rate*100:.2f}% | 95% CI: [{ci[0]*100:.2f}%, {ci[1]*100:.2f}%]")
            print(f"       Episodes={total_episodes} | Wins={wins} Losses={losses} Ties={ties}")

            results.append((opp_name, win_rate, ci, counts))
            eval_win_rates[opp_name] = win_rate

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for opp_name, wr, ci, counts in results:
            status = "✓" if wr >= args.threshold else "✗"
            print(f"{status} {opp_name}: {wr*100:.2f}%")

        if all_passed:
            final_path = model_dir / f"{args.model_name}_final"
            model.save(final_path)
            print(f"\n✓ ALL THRESHOLDS PASSED. Saved final model to: {final_path}.zip")
            env.close()
            return

        # Calculate adaptive timesteps for next iteration
        timestep_allocation = calculate_adaptive_timesteps(eval_win_rates, args.timesteps_per_iter)
        print(f"\n✗ Not all thresholds passed. Recalculating timestep allocation for next iteration...")

    print("Max iterations reached without meeting all thresholds.")
    env.close()


if __name__ == "__main__":
    main()
