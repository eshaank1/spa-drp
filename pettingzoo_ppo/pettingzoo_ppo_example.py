"""Minimal PettingZoo + PPO example.

This trains a shared PPO policy on a PettingZoo environment by converting it to
an SB3-compatible vectorized environment using SuperSuit.
"""

from __future__ import annotations

from pathlib import Path

from pettingzoo.butterfly import pistonball_v6
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

import supersuit as ss


def make_train_env():
    env = pistonball_v6.parallel_env(
        n_pistons=20,
        time_penalty=-0.1,
        continuous=False,
        random_drop=True,
        random_rotate=True,
        ball_mass=0.75,
        ball_friction=0.3,
        ball_elasticity=1.5,
        max_cycles=125,
    )

    # Convert from multi-agent dict interface to a single-policy vectorized env.
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=2, base_class="stable_baselines3")
    return env


def make_eval_env():
    env = pistonball_v6.parallel_env(n_pistons=20, continuous=False, max_cycles=125)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
    return env


def main() -> None:
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)

    train_env = make_train_env()
    eval_env = make_eval_env()

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best"),
        log_path=str(output_dir / "logs"),
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )

    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=str(output_dir / "tb"),
    )

    model.learn(total_timesteps=100_000, callback=eval_callback)
    model.save(output_dir / "ppo_pistonball")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
