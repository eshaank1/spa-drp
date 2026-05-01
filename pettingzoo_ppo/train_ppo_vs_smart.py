from __future__ import annotations

import argparse
import sys
from pathlib import Path

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl_pettingzoo_env import CardGameVsSmartParallelEnv


def build_vec_env(num_envs: int, seed: int):
    env = CardGameVsSmartParallelEnv(seed=seed, invalid_action_penalty=1.0)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_envs, num_cpus=0, base_class="stable_baselines3")
    env = VecMonitor(env)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO agent vs SmartBot using PettingZoo.")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--model-name", type=str, default="ppo_vs_smart")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to an existing PPO .zip to continue training from.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    env = build_vec_env(num_envs=args.num_envs, seed=args.seed)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(10_000 // max(args.num_envs, 1), 1),
        save_path=str(model_dir / "checkpoints"),
        name_prefix=args.model_name,
    )

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

    model.learn(total_timesteps=args.timesteps, callback=checkpoint_cb, progress_bar=True, reset_num_timesteps=False)

    final_path = model_dir / f"{args.model_name}_final"
    model.save(final_path)
    env.close()

    print(f"Saved model to: {final_path}.zip")
    if args.resume_from:
        print(f"Resumed from: {args.resume_from}")
    print("Next: run evaluate_ppo_vs_smart.py to measure win rate vs SmartBot.")


if __name__ == "__main__":
    main()
