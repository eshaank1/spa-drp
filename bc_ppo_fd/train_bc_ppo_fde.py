"""
End-to-End Training Script: BC + PPO-fD Training Pipeline

Phase 1 (Behavioral Cloning): pre-train the *actual* ActorCriticNetwork used in
PPO directly on real BaselineBot demonstrations, so the learned weights carry
into Phase 2 (no throwaway network, no random-noise demos).

Phase 2 (PPO with auxiliary imitation loss / PPO-fD): refine the policy with PPO
against a *pool* of opponents (BaselineBot, SmartBot, and frozen self-play
snapshots) while a persistent demonstration buffer supplies a non-zero auxiliary
BC loss every update.

Usage:
    python train_bc_ppo_fde.py --num-bc-games 5000 --num-ppo-steps 150
    python train_bc_ppo_fde.py --resume-from agent_final.pt --num-ppo-steps 200
"""

import sys
import copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
import logging
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent

from pettingzoo_ppo.rl_pettingzoo_env import CardGameVsSmartParallelEnv
from bc_ppo_fd.ppo_auxiliary_loss import (
    ActorCriticNetwork,
    PPOWithAuxiliaryLossUpdater,
    RolloutBatch,
)
from bc_ppo_fd.collect_bc_demonstrations import load_or_collect_demonstrations

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(log_file=None):
    """Setup logging to both console and file."""
    if log_file is None:
        log_file = SCRIPT_DIR / "training.log"
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False  # avoid duplicate lines via the root logger
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


logger = setup_logging()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def hand_mask(hand) -> np.ndarray:
    """Action mask: pass + every card rank in hand."""
    mask = np.zeros(14, dtype=np.float32)
    mask[0] = 1.0
    for idx, rank in enumerate(RANKS, start=1):
        if rank in hand:
            mask[idx] = 1.0
    return mask


def demos_to_tensors(demos, device) -> Dict[str, torch.Tensor]:
    """Convert (state, mask, action) demo list into a tensor batch dict."""
    states = np.array([d[0] for d in demos], dtype=np.float32)
    masks = np.array([d[1] for d in demos], dtype=np.float32)
    actions = np.array([d[2] for d in demos], dtype=np.int64)
    return {
        "observations": torch.tensor(states, device=device),
        "action_masks": torch.tensor(masks, device=device),
        "actions": torch.tensor(actions, device=device),
    }


def make_self_play_fn(snapshot_net, device, temperature: float = 1.0):
    """Opponent callable: P2 acts using a frozen snapshot of the agent.

    The env's _get_opponent_observation() already returns a learner-format view
    from Player 2's perspective, so it is fed to the network directly.
    """
    def fn(env):
        obs = env._get_opponent_observation()
        mask = np.zeros(14, dtype=np.float32)
        mask[0] = 1.0
        for idx, rank in enumerate(env.RANKS, start=1):
            if rank in env.player2_hand:
                mask[idx] = 1.0
        with torch.no_grad():
            o = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            m = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = snapshot_net(o, m)
            probs = F.softmax(logits.squeeze(0) / max(temperature, 1e-6), dim=-1)
            action = torch.multinomial(probs, 1).item()
        return int(action)
    return fn


# ---------------------------------------------------------------------------
# Phase 1: Behavioral Cloning on the ActorCriticNetwork
# ---------------------------------------------------------------------------
def behavioral_clone(
    actor_critic: ActorCriticNetwork,
    demos: List,
    device: torch.device,
    num_epochs: int = 50,
    batch_size: int = 128,
    lr: float = 3e-4,
    target_accuracy: float = 0.75,
    val_split: float = 0.1,
) -> float:
    """Pre-train the policy head of the actor-critic to imitate the demos.

    Returns the best validation accuracy reached. Training stops early once the
    target accuracy is hit (to retain exploration capacity for PPO).
    """
    batch = demos_to_tensors(demos, device)
    states, masks, actions = batch["observations"], batch["action_masks"], batch["actions"]

    n = states.shape[0]
    perm = torch.randperm(n, device=device)
    n_val = int(n * val_split)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(num_epochs):
        actor_critic.train()
        epoch_perm = train_idx[torch.randperm(train_idx.shape[0], device=device)]
        for start in range(0, epoch_perm.shape[0], batch_size):
            b = epoch_perm[start:start + batch_size]
            optimizer.zero_grad()
            logits, _ = actor_critic(states[b], masks[b])
            loss = criterion(logits, actions[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 0.5)
            optimizer.step()

        # Validation accuracy
        actor_critic.eval()
        with torch.no_grad():
            logits, _ = actor_critic(states[val_idx], masks[val_idx])
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == actions[val_idx]).float().mean().item()
        best_acc = max(best_acc, acc)

        if (epoch + 1) % 5 == 0:
            logger.info(f"  BC epoch {epoch + 1}/{num_epochs} | val_acc={acc:.3f}")

        if acc >= target_accuracy:
            logger.info(f"  Reached target accuracy {target_accuracy:.0%} at epoch {epoch + 1}; stopping BC.")
            break

    return best_acc


# ---------------------------------------------------------------------------
# Phase 2: PPO-fD trainer
# ---------------------------------------------------------------------------
class PPOTrainer:
    """PPO-fD trainer with an opponent pool and demo-driven auxiliary loss."""

    def __init__(self, env: CardGameVsSmartParallelEnv, device, seed: int = 42):
        self.env = env
        self.device = device
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.self_play_fn = None  # set once a snapshot exists
        self.opponent_names: List[str] = ["baseline", "smart"]

    def set_opponent_pool(self, names: List[str], self_play_fn=None):
        self.opponent_names = names
        self.self_play_fn = self_play_fn

    def _sample_opponent(self):
        name = np.random.choice(self.opponent_names)
        if name == "self" and self.self_play_fn is not None:
            self.env.set_opponent(fn=self.self_play_fn)
        else:
            self.env.set_opponent(kind=name)

    def collect_rollouts(self, actor_critic, num_steps: int = 2048):
        """Collect agent (Player 1) transitions; the env auto-plays Player 2.

        Opponent is resampled from the pool on every episode reset, giving
        within-rollout opponent diversity.
        """
        observations, action_masks, actions = [], [], []
        rewards, values, log_probs, dones = [], [], [], []

        self._sample_opponent()
        obs_dict, _ = self.env.reset()
        obs = obs_dict["learner"]
        total_reward, num_episodes, steps = 0.0, 0, 0

        actor_critic.eval()
        with torch.no_grad():
            while steps < num_steps:
                # The env returns control on Player 1's turn; guard just in case.
                if self.env.current_player != 1:
                    obs_dict, _, done_dict, _, _ = self.env.step({"learner": 0})
                    if done_dict["learner"]:
                        self._sample_opponent()
                        obs_dict, _ = self.env.reset()
                    obs = obs_dict["learner"]
                    continue

                mask = hand_mask(self.env.player1_hand)
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
                logits, value = actor_critic(obs_t, mask_t)
                logits = logits.squeeze(0)
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()
                log_prob = F.log_softmax(logits, dim=-1)[action].item()

                observations.append(obs.copy())
                action_masks.append(mask.copy())
                actions.append(action)
                values.append(value.item())
                log_probs.append(log_prob)

                obs_dict, reward_dict, done_dict, _, info = self.env.step({"learner": action})
                r = reward_dict["learner"]
                done = done_dict["learner"]
                rewards.append(r)
                dones.append(float(done))
                total_reward += r
                steps += 1

                if done:
                    num_episodes += 1
                    self._sample_opponent()
                    obs_dict, _ = self.env.reset()
                    obs = obs_dict["learner"]
                elif obs_dict and "learner" in obs_dict:
                    obs = obs_dict["learner"]

        is_bot_data = torch.zeros(len(actions), dtype=torch.bool, device=self.device)
        rollout = RolloutBatch(
            observations=torch.tensor(np.array(observations), dtype=torch.float32, device=self.device),
            action_masks=torch.tensor(np.array(action_masks), dtype=torch.float32, device=self.device),
            actions=torch.tensor(actions, dtype=torch.long, device=self.device),
            rewards=torch.tensor(rewards, dtype=torch.float32, device=self.device),
            values=torch.tensor(np.array(values), dtype=torch.float32, device=self.device),
            log_probs=torch.tensor(log_probs, dtype=torch.float32, device=self.device),
            dones=torch.tensor(dones, dtype=torch.float32, device=self.device),
            is_bot_data=is_bot_data,
        )
        metrics = {
            "avg_episode_return": total_reward / max(num_episodes, 1),
            "num_episodes": num_episodes,
        }
        return rollout, metrics

    def evaluate(self, actor_critic, opponent: str, num_games: int = 50,
                 deterministic: bool = True) -> float:
        """Win rate of the agent (P1) vs a fixed opponent. argmax by default."""
        self.env.set_opponent(kind=opponent)
        wins = 0
        actor_critic.eval()
        with torch.no_grad():
            for _ in range(num_games):
                obs_dict, _ = self.env.reset()
                obs = obs_dict["learner"]
                done = False
                while not done:
                    if self.env.current_player != 1:
                        obs_dict, _, done_dict, _, info = self.env.step({"learner": 0})
                        done = done_dict["learner"]
                        if not done:
                            obs = obs_dict["learner"]
                        continue
                    mask = hand_mask(self.env.player1_hand)
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
                    logits, _ = actor_critic(obs_t, mask_t)
                    logits = logits.squeeze(0)
                    if deterministic:
                        action = torch.argmax(logits).item()
                    else:
                        action = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
                    obs_dict, _, done_dict, _, info = self.env.step({"learner": action})
                    done = done_dict["learner"]
                    if done:
                        rw = info["learner"]["final_rounds_won"]
                        if rw[0] > rw[1]:
                            wins += 1
                    elif obs_dict and "learner" in obs_dict:
                        obs = obs_dict["learner"]
        return wins / num_games


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    logger.info("=" * 70)
    logger.info("BC + PPO-fD Training Pipeline")
    logger.info("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    actor_critic = ActorCriticNetwork(
        obs_dim=50, action_dim=14, hidden_dim=256, num_hidden_layers=2,
    ).to(device)

    # ===== Demonstration buffer (real BaselineBot vs BaselineBot) =====
    demo_batch = None
    if args.num_bc_games > 0:
        demo_path = SCRIPT_DIR / "bc_demos.pkl"
        logger.info("\n" + "=" * 70)
        logger.info("Loading / collecting BaselineBot demonstrations")
        logger.info("=" * 70)
        demos = load_or_collect_demonstrations(
            str(demo_path), num_games=args.num_bc_games, seed=42
        )
        logger.info(f"✓ {len(demos)} demonstrations available")
        demo_batch = demos_to_tensors(demos, device)

        if not args.resume_from:
            logger.info("\nPHASE 1: Behavioral Cloning (warm-start the PPO network)")
            best_acc = behavioral_clone(
                actor_critic, demos, device,
                num_epochs=args.bc_epochs, target_accuracy=args.bc_target_acc,
            )
            logger.info(f"✓ BC complete | best val accuracy: {best_acc:.2%}")

    # ===== Resume from checkpoint (skips BC weights init) =====
    if args.resume_from:
        ckpt = Path(args.resume_from)
        logger.info(f"Loading checkpoint from {ckpt}...")
        actor_critic.load_state_dict(torch.load(ckpt, map_location=device))
        logger.info("✓ Checkpoint loaded")

    # ===== PHASE 2: PPO-fD =====
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: PPO-FD ONLINE TRAINING")
    logger.info("=" * 70)

    env = CardGameVsSmartParallelEnv(seed=42)
    trainer = PPOTrainer(env, device=device)
    updater = PPOWithAuxiliaryLossUpdater(
        actor_critic, device=device, bc_lambda=args.bc_lambda_start, entropy_coeff=args.entropy_coeff,
    )

    # Frozen self-play snapshot (initialized to the current policy).
    snapshot = copy.deepcopy(actor_critic).to(device)
    snapshot.eval()
    self_play_fn = make_self_play_fn(snapshot, device)

    logger.info(f"Starting PPO training for {args.num_ppo_steps} steps...")
    for step in range(args.num_ppo_steps):
        # Introduce self-play once the agent has had a little PPO refinement.
        if step < args.self_play_after:
            trainer.set_opponent_pool(["baseline", "smart"])
        else:
            trainer.set_opponent_pool(["baseline", "smart", "self"], self_play_fn)

        # Decay the auxiliary BC loss weight over training so PPO is
        # increasingly free to move away from the BaselineBot-clone anchor
        # instead of being pinned near it for the whole run.
        progress = step / max(args.num_ppo_steps - 1, 1)
        updater.bc_lambda = args.bc_lambda_start + (args.bc_lambda_end - args.bc_lambda_start) * progress

        rollout, env_metrics = trainer.collect_rollouts(actor_critic, num_steps=args.rollout_steps)
        ppo_metrics = updater.update(
            rollout, num_epochs=4, gamma=0.99, gae_lambda=0.95, demo_batch=demo_batch,
        )

        # Refresh the self-play snapshot periodically.
        if (step + 1) % args.snapshot_every == 0:
            snapshot.load_state_dict(actor_critic.state_dict())

        if (step + 1) % 5 == 0 or step == 0:
            wr_base = trainer.evaluate(actor_critic, "baseline", num_games=args.eval_games)
            wr_smart = trainer.evaluate(actor_critic, "smart", num_games=args.eval_games)
            logger.info(
                f"Step {step + 1:3d}: Loss={ppo_metrics['total_loss']:7.4f} | "
                f"PPO={ppo_metrics['ppo_loss']:7.4f} | BC={ppo_metrics['bc_loss']:7.4f} | "
                f"bc_lambda={updater.bc_lambda:.3f} | "
                f"WR(baseline)={wr_base:.1%} | WR(smart)={wr_smart:.1%} | "
                f"Eps={env_metrics['num_episodes']}"
            )

    logger.info("\n✓ Training complete!")

    # ===== Final evaluation =====
    logger.info("\nFinal Evaluation (200 games each)...")
    final_base = trainer.evaluate(actor_critic, "baseline", num_games=200)
    final_smart = trainer.evaluate(actor_critic, "smart", num_games=200)
    logger.info(f"Final Win Rate vs BaselineBot: {final_base:.1%}")
    logger.info(f"Final Win Rate vs SmartBot:    {final_smart:.1%}")

    save_path = SCRIPT_DIR / "agent_final.pt"
    torch.save(actor_critic.state_dict(), save_path)
    logger.info(f"\n✓ Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BC + PPO-fD Training")
    parser.add_argument("--num-bc-games", type=int, default=5000,
                        help="BaselineBot games for the demo buffer (0 to skip BC + aux loss)")
    parser.add_argument("--bc-epochs", type=int, default=50)
    parser.add_argument("--bc-target-acc", type=float, default=0.75)
    parser.add_argument("--num-ppo-steps", type=int, default=150)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--bc-lambda-start", type=float, default=0.2,
                        help="Initial weight of the auxiliary BC loss in PPO-fD")
    parser.add_argument("--bc-lambda-end", type=float, default=0.02,
                        help="Final weight of the auxiliary BC loss, linearly decayed "
                             "over --num-ppo-steps so PPO can climb above the BC anchor")
    parser.add_argument("--entropy-coeff", type=float, default=0.02)
    parser.add_argument("--self-play-after", type=int, default=20,
                        help="PPO step at which self-play opponents join the pool")
    parser.add_argument("--snapshot-every", type=int, default=10,
                        help="Refresh the frozen self-play snapshot every N steps")
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Checkpoint to resume from (skips BC warm-start)")
    args = parser.parse_args()

    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.is_absolute():
            resume_path = SCRIPT_DIR / args.resume_from
        args.resume_from = str(resume_path)
        logger.info(f"Resuming from checkpoint: {resume_path}")

    main(args)
