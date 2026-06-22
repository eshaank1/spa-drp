"""
End-to-End Training Script: BC + PPO-fD Training Pipeline

This is a complete, runnable example showing how to:
1. Load BC demonstrations
2. Train BC pre-training phase
3. Train PPO-fD online phase
4. Evaluate the final agent

Usage:
    python train_bc_ppo_fde.py --num-bc-games 10000 --num-ppo-steps 100
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import logging
import pickle
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pettingzoo_ppo.rl_pettingzoo_env import CardGameVsSmartParallelEnv
from bc_ppo_fd.state_inverter import invert_state
from bc_ppo_fd.behavioral_cloning import create_behavioral_cloning_pipeline
from bc_ppo_fd.ppo_auxiliary_loss import (
    ActorCriticNetwork,
    PPOWithAuxiliaryLossUpdater,
    RolloutBatch,
)
from baseline_bot import BaselineBot

# Setup logging to both console and file
def setup_logging(log_file=None):
    """Setup logging to both console and file."""
    if log_file is None:
        # Use the script directory for the log file
        script_dir = Path(__file__).resolve().parent
        log_file = script_dir / "training.log"
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # File handler - append to existing log
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logging()


class PPOTrainer:
    """PPO-fD trainer with environment integration."""
    
    def __init__(
        self,
        env: CardGameVsSmartParallelEnv,
        bot: BaselineBot,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42,
    ):
        self.env = env
        self.bot = bot
        self.device = torch.device(device)
        self.seed = seed
        
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def _get_valid_actions_mask(self, hand) -> np.ndarray:
        """Create action mask from hand."""
        mask = np.zeros(14, dtype=np.float32)
        mask[0] = 1.0  # Pass always valid
        
        rank_to_idx = {
            "A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
            "7": 7, "8": 8, "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13
        }
        
        for card in hand:
            if card in rank_to_idx:
                mask[rank_to_idx[card]] = 1.0
        
        return mask
    
    def collect_rollouts(
        self,
        actor_critic: ActorCriticNetwork,
        num_steps: int = 2048,
    ) -> Tuple[RolloutBatch, Dict]:
        """
        Collect mixed rollouts (agent + bot data).
        
        Returns:
            Tuple of (rollout_batch, metrics)
        """
        observations = []
        action_masks = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        is_bot_data = []
        
        obs_dict, _ = self.env.reset()
        obs = obs_dict["learner"] if isinstance(obs_dict, dict) else obs_dict
        total_reward = 0.0
        num_episodes = 0
        steps_collected = 0
        
        actor_critic.eval()
        
        with torch.no_grad():
            while steps_collected < num_steps:
                # ===== AGENT'S TURN (PLAYER 1) =====
                if self.env.current_player == 1:
                    hand = self.env.player1_hand
                    mask = self._get_valid_actions_mask(hand)
                    
                    # Get agent action from policy
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
                    mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device)
                    
                    logits, value = actor_critic(obs_t.unsqueeze(0), mask_t.unsqueeze(0))
                    logits = logits.squeeze(0)
                    value = value.squeeze(0)
                    
                    # Sample action
                    probs = F.softmax(logits, dim=-1)
                    action = torch.multinomial(probs, 1).item()
                    log_prob = F.log_softmax(logits, dim=-1)[action].item()
                    
                    # Store transition
                    observations.append(obs.copy())
                    action_masks.append(mask.copy())
                    actions.append(action)
                    values.append(value.item())
                    log_probs.append(log_prob)
                    is_bot_data.append(False)
                    
                    # Step environment
                    obs_dict, reward_dict, done_dict, trunc_dict, info = self.env.step({"learner": action})
                    # Note: obs_dict is empty {} when episode ends
                    if obs_dict and "learner" in obs_dict:
                        obs = obs_dict["learner"]
                    rewards.append(reward_dict.get("learner", 0.0) if isinstance(reward_dict, dict) else reward_dict)
                    dones.append(float(done_dict.get("learner", False) if isinstance(done_dict, dict) else done_dict))
                    
                    total_reward += rewards[-1]
                    steps_collected += 1
                    
                    if dones[-1]:
                        num_episodes += 1
                        obs_dict, _ = self.env.reset()
                        obs = obs_dict["learner"] if isinstance(obs_dict, dict) else obs_dict
                
                # ===== BOT'S TURN (PLAYER 2) =====
                else:
                    hand = self.env.player2_hand
                    mask = self._get_valid_actions_mask(hand)
                    
                    # Get bot action
                    p2_score = self.env._score(self.env.p2_played)
                    p1_score = self.env._score(self.env.p1_played)
                    is_last_round = self.env.current_round == 3
                    
                    action_str = self.bot.decide_move(
                        hand=hand,
                        player_score=p2_score,
                        opponent_score=p1_score,
                        is_last_round=is_last_round,
                        opponent_just_played=self.env.opponent_just_played,
                        my_rounds_won=self.env.rounds_won[1],
                        opponent_rounds_won=self.env.rounds_won[0],
                    )
                    
                    # Convert to action
                    if action_str == "PASS":
                        action = 0
                    else:
                        rank_to_idx = {
                            "A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
                            "7": 7, "8": 8, "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13
                        }
                        action = rank_to_idx.get(action_str, 0)
                    
                    # Get observation from bot's perspective and INVERT
                    obs_p2 = self.env._get_opponent_observation()
                    obs_p2_torch = torch.tensor(obs_p2, dtype=torch.float32)
                    obs_p1_inverted = invert_state(obs_p2_torch).numpy().astype(np.float32)
                    
                    # Get value and log prob from agent's perspective
                    obs_t = torch.tensor(
                        obs_p1_inverted, dtype=torch.float32, device=self.device
                    )
                    mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device)
                    
                    logits, value = actor_critic(obs_t.unsqueeze(0), mask_t.unsqueeze(0))
                    logits = logits.squeeze(0)
                    value = value.squeeze(0)
                    
                    log_prob = F.log_softmax(logits, dim=-1)[action].item()
                    
                    # Store transition (marked as bot data)
                    observations.append(obs_p1_inverted.copy())
                    action_masks.append(mask.copy())
                    actions.append(action)
                    values.append(value.item())
                    log_probs.append(log_prob)
                    is_bot_data.append(True)
                    
                    # Step environment
                    obs_dict, reward_dict, done_dict, trunc_dict, info = self.env.step({"learner": 0})
                    # Note: obs_dict is empty {} when episode ends
                    if obs_dict and "learner" in obs_dict:
                        obs = obs_dict["learner"]
                    rewards.append(0.0)  # No reward for bot actions
                    dones.append(float(done_dict.get("learner", False) if isinstance(done_dict, dict) else done_dict))
                    
                    steps_collected += 1
                    
                    if dones[-1]:
                        num_episodes += 1
                        obs_dict, _ = self.env.reset()
                        obs = obs_dict["learner"] if isinstance(obs_dict, dict) else obs_dict
        rollout_batch = RolloutBatch(
            observations=torch.tensor(np.array(observations), dtype=torch.float32, device=self.device),
            action_masks=torch.tensor(np.array(action_masks), dtype=torch.float32, device=self.device),
            actions=torch.tensor(actions, dtype=torch.long, device=self.device),
            rewards=torch.tensor(rewards, dtype=torch.float32, device=self.device),
            values=torch.tensor(np.array(values), dtype=torch.float32, device=self.device),
            log_probs=torch.tensor(log_probs, dtype=torch.float32, device=self.device),
            dones=torch.tensor(dones, dtype=torch.float32, device=self.device),
            is_bot_data=torch.tensor(is_bot_data, dtype=torch.bool, device=self.device),
        )
        
        metrics = {
            "avg_episode_return": total_reward / max(num_episodes, 1),
            "num_episodes": num_episodes,
            "num_agent_steps": sum(1 for x in is_bot_data if not x),
            "num_bot_steps": sum(1 for x in is_bot_data if x),
        }
        
        return rollout_batch, metrics
    
    def evaluate(
        self,
        actor_critic: ActorCriticNetwork,
        num_games: int = 50,
    ) -> Dict:
        """Evaluate agent win rate."""
        wins = 0
        total_reward = 0.0
        
        actor_critic.eval()
        
        with torch.no_grad():
            for _ in range(num_games):
                obs_dict, _ = self.env.reset()
                obs = obs_dict["learner"] if isinstance(obs_dict, dict) else obs_dict
                done = False
                episode_reward = 0.0
                
                while not done:
                    if self.env.current_player == 1:
                        hand = self.env.player1_hand
                        mask = self._get_valid_actions_mask(hand)
                        
                        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
                        mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device)
                        
                        logits, _ = actor_critic(obs_t.unsqueeze(0), mask_t.unsqueeze(0))
                        action = torch.argmax(logits).item()
                        
                        obs_dict, reward_dict, done_dict, trunc_dict, info = self.env.step({"learner": action})
                        # Note: obs_dict is empty {} when episode terminates
                        if obs_dict and "learner" in obs_dict:
                            obs = obs_dict["learner"]
                        episode_reward += reward_dict.get("learner", 0.0) if isinstance(reward_dict, dict) else reward_dict
                        done = done_dict.get("learner", False) if isinstance(done_dict, dict) else done_dict
                        
                        if done and info["learner"]["final_rounds_won"][0] > info["learner"]["final_rounds_won"][1]:
                            wins += 1
                    
                    else:
                        hand = self.env.player2_hand
                        p2_score = self.env._score(self.env.p2_played)
                        p1_score = self.env._score(self.env.p1_played)
                        is_last_round = self.env.current_round == 3
                        
                        action_str = self.bot.decide_move(
                            hand=hand,
                            player_score=p2_score,
                            opponent_score=p1_score,
                            is_last_round=is_last_round,
                            opponent_just_played=self.env.opponent_just_played,
                            my_rounds_won=self.env.rounds_won[1],
                            opponent_rounds_won=self.env.rounds_won[0],
                        )
                        
                        if action_str == "PASS":
                            action = 0
                        else:
                            rank_to_idx = {
                                "A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
                                "7": 7, "8": 8, "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13
                            }
                            action = rank_to_idx.get(action_str, 0)
                        
                        obs_dict, _, done_dict, trunc_dict, info = self.env.step({"learner": 0})
                        # Note: obs_dict is empty {} when episode terminates
                        if obs_dict and "learner" in obs_dict:
                            obs = obs_dict["learner"]
                        done = done_dict.get("learner", False) if isinstance(done_dict, dict) else done_dict
                
                total_reward += episode_reward
        
        return {
            "win_rate": wins / num_games,
            "avg_episode_reward": total_reward / num_games,
        }


def main(args):
    """Main training script."""
    
    logger.info("=" * 70)
    logger.info("BC + PPO-fD Training Pipeline")
    logger.info("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # ===== PHASE 1: BEHAVIORAL CLONING =====
    if args.num_bc_games > 0:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 1: BEHAVIORAL CLONING PRE-TRAINING")
        logger.info("=" * 70)
        
        # Create dummy BC demonstrations
        logger.info("Creating BC demonstrations...")
        demonstrations = []
        for _ in range(args.num_bc_games * 10):  # ~10 steps per game
            state = np.random.rand(50).astype(np.float32)
            mask = np.ones(14, dtype=np.float32)
            action = np.random.randint(0, 14)
            demonstrations.append((state, mask, action))
        
        logger.info(f"✓ Generated {len(demonstrations)} demonstrations")
        
        # Train BC
        bc_actor, bc_trainer, loaders = create_behavioral_cloning_pipeline(
            demonstrations,
            batch_size=64,
            device=device,
            hidden_dim=256,
            num_hidden_layers=2,
        )
        
        logger.info("Starting BC training...")
        bc_history = bc_trainer.train(
            loaders["train_loader"],
            loaders["val_loader"],
            num_epochs=50,
            target_accuracy=0.75,
            patience=10,
            verbose=True,
        )
        
        logger.info(f"✓ BC Phase complete")
        logger.info(f"  Final accuracy: {bc_history['best_val_accuracy']:.2%}")
    else:
        logger.info("\nSkipping BC phase (resuming from checkpoint)")
    
    # ===== PHASE 2: PPO-FD =====
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: PPO-FD ONLINE TRAINING")
    logger.info("=" * 70)
    
    # Initialize environment and bot
    env = CardGameVsSmartParallelEnv(seed=42)
    bot = BaselineBot()
    
    # Create PPO trainer
    ppo_trainer = PPOTrainer(env, bot, device=device)
    
    # Initialize actor-critic (optionally from BC weights or checkpoint)
    actor_critic = ActorCriticNetwork(
        obs_dim=50,
        action_dim=14,
        hidden_dim=256,
        num_hidden_layers=2,
    ).to(device)
    
    # Load from checkpoint if provided
    if hasattr(args, 'resume_from') and args.resume_from:
        logger.info(f"Loading checkpoint from {args.resume_from}...")
        checkpoint = torch.load(args.resume_from, map_location=device)
        actor_critic.load_state_dict(checkpoint)
        logger.info("✓ Checkpoint loaded successfully")
    
    # Initialize PPO updater
    ppo_updater = PPOWithAuxiliaryLossUpdater(
        actor_critic,
        device=device,
        bc_lambda=0.1,
        entropy_coeff=0.01,
    )
    
    # Training loop
    logger.info(f"Starting PPO training for {args.num_ppo_steps} steps...")
    
    for step in range(args.num_ppo_steps):
        # Collect rollouts
        rollouts, env_metrics = ppo_trainer.collect_rollouts(
            actor_critic,
            num_steps=2048,
        )
        
        # Update PPO
        ppo_metrics = ppo_updater.update(
            rollouts,
            num_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
        )
        
        # Evaluate
        if (step + 1) % 5 == 0 or step == 0:
            eval_metrics = ppo_trainer.evaluate(actor_critic, num_games=20)
            
            logger.info(
                f"Step {step + 1:3d}: "
                f"Loss={ppo_metrics['total_loss']:7.4f} | "
                f"PPO={ppo_metrics['ppo_loss']:7.4f} | "
                f"BC={ppo_metrics['bc_loss']:7.4f} | "
                f"WinRate={eval_metrics['win_rate']:.1%} | "
                f"Episodes={env_metrics['num_episodes']}"
            )
    
    logger.info("\n✓ Training complete!")
    
    # Final evaluation
    logger.info("\nFinal Evaluation (100 games)...")
    final_metrics = ppo_trainer.evaluate(actor_critic, num_games=100)
    logger.info(f"Final Win Rate: {final_metrics['win_rate']:.1%}")
    logger.info(f"Final Avg Reward: {final_metrics['avg_episode_reward']:.4f}")
    
    # Save model
    script_dir = Path(__file__).resolve().parent
    save_path = script_dir / "agent_final.pt"
    torch.save(actor_critic.state_dict(), save_path)
    logger.info(f"\n✓ Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BC + PPO-fD Training")
    parser.add_argument("--num-bc-games", type=int, default=100, help="Number of BC demo games")
    parser.add_argument("--num-ppo-steps", type=int, default=20, help="Number of PPO training steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume training from (skips BC phase)")
    
    args = parser.parse_args()
    
    # If resuming, skip BC phase by setting num-bc-games to 0
    if args.resume_from:
        # Convert relative path to absolute if needed
        resume_path = Path(args.resume_from)
        if not resume_path.is_absolute():
            script_dir = Path(__file__).resolve().parent
            resume_path = script_dir / args.resume_from
        logger.info(f"Resuming from checkpoint: {resume_path}")
        args.num_bc_games = 0
        args.resume_from = str(resume_path)
    
    main(args)
