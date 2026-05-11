from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
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


def evaluate_model_vs_opponent(model: PPO, opponent_model_path: str, episodes: int, seed: int, deterministic: bool = True):
    wins = 0
    ties = 0
    losses = 0

    for ep in range(episodes):
        env = CardGameVsSmartParallelEnv(seed=seed + ep, opponent_model_path=opponent_model_path, opponent_deterministic=True)
        obs, _ = env.reset()
        final_info = None

        while env.agents:
            action, _ = model.predict(obs["learner"], deterministic=deterministic)
            obs, reward, term, trunc, info = env.step({"learner": int(action)})
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

        env.close()

    win_rate = wins / episodes if episodes > 0 else 0.0
    # Wilson interval for reporting
    z = 1.96
    denom = 1 + (z * z) / episodes if episodes > 0 else 1
    phat = win_rate
    margin = (z * math.sqrt((phat * (1 - phat) + (z * z) / (4 * episodes)) / episodes)) / denom if episodes > 0 else 0

    return win_rate, (max(0.0, (phat + (z * z) / (2 * episodes)) / denom - margin), min(1.0, (phat + (z * z) / (2 * episodes)) / denom + margin)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Iteratively train PPO until it beats a target PPO opponent.")
    parser.add_argument("--target-opponent", type=str, required=True, help="Path to the target PPO .zip to beat.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Win-rate threshold to stop training.")
    parser.add_argument("--timesteps-per-iter", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--max-iters", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--model-name", type=str, default="ppo_vs_ppo")
    parser.add_argument("--deterministic-eval", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    opponent_path = args.target_opponent

    # Build env for training (opponent is fixed target)
    env = build_vec_env(num_envs=args.num_envs, seed=args.seed, opponent_model_path=opponent_path, opponent_deterministic=True)

    checkpoint_cb = CheckpointCallback(save_freq=max(10_000 // max(args.num_envs, 1), 1), save_path=str(model_dir / "checkpoints"), name_prefix=args.model_name)

    if args.resume_from:
        model = PPO.load(args.resume_from, env=env)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=512,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            clip_range=0.2,
            verbose=1,
            tensorboard_log=str(model_dir / "tb"),
        )

    for it in range(1, args.max_iters + 1):
        print(f"Iteration {it}: training {args.timesteps_per_iter} timesteps...")
        model.learn(total_timesteps=args.timesteps_per_iter, callback=checkpoint_cb, reset_num_timesteps=False)

        iter_path = model_dir / f"{args.model_name}_iter{it}"
        model.save(iter_path)
        print(f"Saved checkpoint: {iter_path}.zip")

        print("Evaluating current model vs target opponent...")
        win_rate, ci = evaluate_model_vs_opponent(model, opponent_path, args.eval_episodes, args.seed + it, deterministic=args.deterministic_eval)
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
