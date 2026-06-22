"""
Integration Guide: Complete BC + PPO-fD Training Pipeline

This module demonstrates how to integrate all three components:
1. State Inverter (for perspective swapping)
2. Behavioral Cloning (pre-training phase)
3. PPO with Auxiliary Loss (online learning phase)

In a real setup, this would be integrated with your environment and bot.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

# Import the three components
from state_inverter import invert_state
from behavioral_cloning import (
    ActorNetwork as BCActorNetwork,
    BCDataset,
    BehavioralCloningTrainer,
    create_behavioral_cloning_pipeline,
)
from ppo_auxiliary_loss import (
    ActorCriticNetwork,
    PPOWithAuxiliaryLossUpdater,
    RolloutBatch,
)

logger = logging.getLogger(__name__)


@dataclass
class BCPPOConfig:
    """Configuration for BC + PPO-fD training."""
    
    # General
    device: str = "cuda"
    seed: int = 42
    
    # BC Phase
    bc_num_epochs: int = 100
    bc_batch_size: int = 64
    bc_target_accuracy: float = 0.75
    bc_learning_rate: float = 3e-4
    bc_patience: int = 10
    
    # PPO Phase
    ppo_num_epochs: int = 10
    ppo_batch_size: int = 64
    ppo_learning_rate: float = 3e-4
    ppo_clip_ratio: float = 0.2
    ppo_entropy_coeff: float = 0.01
    ppo_value_coeff: float = 0.5
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_bc_lambda: float = 0.1  # Weight for auxiliary BC loss (0.05-0.2)
    
    # Network architecture
    obs_dim: int = 50
    action_dim: int = 14
    hidden_dim: int = 256
    num_hidden_layers: int = 2


class RolloutBuffer:
    """Simple rollout buffer for storing trajectories."""
    
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.reset()
    
    def reset(self):
        """Clear the buffer."""
        self.observations = []
        self.action_masks = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.is_bot_data = []
    
    def add_transition(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        is_bot: bool,
    ):
        """Add a single transition to the buffer."""
        self.observations.append(obs)
        self.action_masks.append(action_mask)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.is_bot_data.append(is_bot)
    
    def get_rollout_batch(self) -> RolloutBatch:
        """Convert buffer to a RolloutBatch for PPO update."""
        return RolloutBatch(
            observations=torch.tensor(
                np.array(self.observations),
                dtype=torch.float32,
                device=self.device
            ),
            action_masks=torch.tensor(
                np.array(self.action_masks),
                dtype=torch.float32,
                device=self.device
            ),
            actions=torch.tensor(
                self.actions,
                dtype=torch.long,
                device=self.device
            ),
            rewards=torch.tensor(
                self.rewards,
                dtype=torch.float32,
                device=self.device
            ),
            values=torch.tensor(
                self.values,
                dtype=torch.float32,
                device=self.device
            ),
            log_probs=torch.tensor(
                self.log_probs,
                dtype=torch.float32,
                device=self.device
            ),
            dones=torch.tensor(
                self.dones,
                dtype=torch.float32,
                device=self.device
            ),
            is_bot_data=torch.tensor(
                self.is_bot_data,
                dtype=torch.bool,
                device=self.device
            ),
        )


class BCPPOTrainer:
    """
    High-level trainer coordinating BC pre-training and PPO-fD online learning.
    """
    
    def __init__(self, config: BCPPOConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Set seed for reproducibility
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        # Will be initialized later
        self.bc_actor: Optional[BCActorNetwork] = None
        self.actor_critic: Optional[ActorCriticNetwork] = None
        self.ppo_updater: Optional[PPOWithAuxiliaryLossUpdater] = None
    
    def phase1_behavioral_cloning(
        self,
        demonstrations: List[Tuple[np.ndarray, np.ndarray, int]],
    ) -> Dict:
        """
        Phase 1: Pre-train actor on heuristic bot demonstrations.
        
        Args:
            demonstrations: List of (state, action_mask, action) tuples from bot games
        
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting Behavioral Cloning phase with {len(demonstrations)} demos")
        
        # Create BC pipeline
        self.bc_actor, trainer, loaders = create_behavioral_cloning_pipeline(
            demonstrations,
            val_split=0.2,
            batch_size=self.config.bc_batch_size,
            device=self.device,
            obs_dim=self.config.obs_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
            num_hidden_layers=self.config.num_hidden_layers,
        )
        
        # Train
        history = trainer.train(
            loaders["train_loader"],
            loaders["val_loader"],
            num_epochs=self.config.bc_num_epochs,
            target_accuracy=self.config.bc_target_accuracy,
            patience=self.config.bc_patience,
            verbose=True,
        )
        
        logger.info(f"BC phase complete. Final accuracy: {history['best_val_accuracy']:.2%}")
        
        return history
    
    def phase2_ppo_with_auxiliary_loss(
        self,
        rollout_fn,
        num_training_steps: int = 10,
        num_env_steps_per_update: int = 2048,
    ) -> Dict:
        """
        Phase 2: Online training with PPO + auxiliary BC loss.
        
        Args:
            rollout_fn: Callable that takes actor_critic and returns (RolloutBatch, metrics)
            num_training_steps: Number of PPO update steps
            num_env_steps_per_update: Rollout length between updates
        
        Returns:
            Training metrics dictionary
        """
        logger.info("Starting PPO-fD phase")
        
        # Initialize actor-critic from BC pre-trained actor if available
        if self.bc_actor is not None:
            # Transfer BC weights to actor-critic actor head
            logger.info("Initializing actor-critic from BC pre-trained weights")
            self.actor_critic = ActorCriticNetwork(
                obs_dim=self.config.obs_dim,
                action_dim=self.config.action_dim,
                hidden_dim=self.config.hidden_dim,
                num_hidden_layers=self.config.num_hidden_layers,
            )
            
            # Copy BC backbone to actor-critic backbone
            with torch.no_grad():
                ac_backbone_params = list(self.actor_critic.backbone.parameters())
                bc_net_params = list(self.bc_actor.net.parameters())
                
                # Copy up to num_hidden_layers * 2 (each layer is Linear + ReLU)
                num_params_to_copy = min(len(ac_backbone_params), len(bc_net_params))
                for i in range(num_params_to_copy):
                    ac_backbone_params[i].copy_(bc_net_params[i])
        else:
            # Initialize from scratch
            logger.info("Initializing actor-critic from scratch")
            self.actor_critic = ActorCriticNetwork(
                obs_dim=self.config.obs_dim,
                action_dim=self.config.action_dim,
                hidden_dim=self.config.hidden_dim,
                num_hidden_layers=self.config.num_hidden_layers,
            )
        
        # Create PPO updater
        self.ppo_updater = PPOWithAuxiliaryLossUpdater(
            self.actor_critic,
            device=self.device,
            learning_rate=self.config.ppo_learning_rate,
            clip_ratio=self.config.ppo_clip_ratio,
            entropy_coeff=self.config.ppo_entropy_coeff,
            value_coeff=self.config.ppo_value_coeff,
            bc_lambda=self.config.ppo_bc_lambda,
        )
        
        # Training loop
        metrics = {
            "total_loss": [],
            "ppo_loss": [],
            "bc_loss": [],
            "value_loss": [],
            "entropy_bonus": [],
            "episode_returns": [],
        }
        
        for step in range(num_training_steps):
            logger.info(f"PPO step {step + 1}/{num_training_steps}")
            
            # Collect rollouts
            rollouts, env_metrics = rollout_fn(self.actor_critic)
            
            # Update PPO
            update_metrics = self.ppo_updater.update(
                rollouts,
                num_epochs=self.config.ppo_num_epochs,
                gamma=self.config.ppo_gamma,
                gae_lambda=self.config.ppo_gae_lambda,
            )
            
            # Record metrics
            for k, v in update_metrics.items():
                if k in metrics:
                    metrics[k].append(v)
            
            for k, v in env_metrics.items():
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(v)
            
            # Log progress
            if (step + 1) % 5 == 0:
                logger.info(
                    f"  PPO Loss: {update_metrics['ppo_loss']:.4f} | "
                    f"BC Loss: {update_metrics['bc_loss']:.4f} | "
                    f"Value Loss: {update_metrics['value_loss']:.4f}"
                )
        
        logger.info("PPO-fD phase complete")
        
        return metrics


# Example usage structure
def example_integration_flow():
    """
    Demonstrates the complete integration flow.
    
    In practice, you would replace the dummy components with your actual
    game environment and bot implementations.
    """
    
    # Configuration
    config = BCPPOConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        bc_target_accuracy=0.75,
        ppo_bc_lambda=0.1,
    )
    
    # ===== PHASE 1: BEHAVIORAL CLONING =====
    
    # Create dummy demonstrations (in practice, generate from bot vs. bot games)
    logger.info("Generating dummy BC demonstrations...")
    demonstrations = []
    for _ in range(10000):
        state = np.random.rand(50).astype(np.float32)
        action_mask = np.ones(14, dtype=np.float32)
        # Make 2-3 actions invalid
        invalid_indices = np.random.choice(14, size=np.random.randint(1, 4), replace=False)
        action_mask[invalid_indices] = 0
        action = np.random.choice(np.where(action_mask == 1)[0])
        demonstrations.append((state, action_mask, action))
    
    # Initialize trainer
    trainer = BCPPOTrainer(config)
    
    # Run BC phase
    bc_history = trainer.phase1_behavioral_cloning(demonstrations)
    
    # ===== PHASE 2: PPO-FD ONLINE LEARNING =====
    
    def dummy_rollout_fn(actor_critic):
        """
        In practice, this function would:
        1. Interact with the environment for N steps
        2. Collect states where agent acted
        3. Invert states where bot acted (using state_inverter)
        4. Return RolloutBatch with is_bot_data flags
        """
        # Dummy implementation
        rollout_buffer = RolloutBuffer(device=config.device)
        
        for step in range(100):
            obs = np.random.rand(50).astype(np.float32)
            mask = np.ones(14, dtype=np.float32)
            action = np.random.randint(0, 14)
            reward = np.random.randn()
            value = np.random.randn()
            log_prob = np.random.randn()
            done = step == 99
            is_bot = step % 2 == 0
            
            rollout_buffer.add_transition(
                obs, mask, action, reward, value, log_prob, done, is_bot
            )
        
        rollout_batch = rollout_buffer.get_rollout_batch()
        env_metrics = {"episode_return": 1.0}
        
        return rollout_batch, env_metrics
    
    # Run PPO-fD phase
    ppo_metrics = trainer.phase2_ppo_with_auxiliary_loss(
        rollout_fn=dummy_rollout_fn,
        num_training_steps=5,
    )
    
    logger.info(f"Training complete!")
    logger.info(f"BC final accuracy: {bc_history['best_val_accuracy']:.2%}")
    logger.info(f"PPO avg loss: {np.mean(ppo_metrics['total_loss']):.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("BC + PPO-fD Integration Example")
    print("=" * 50)
    print()
    print("To use this in your training pipeline:")
    print()
    print("1. Generate 10k bot vs. bot games and save as demonstrations")
    print("2. Call trainer.phase1_behavioral_cloning(demonstrations)")
    print("3. Implement rollout_fn that:")
    print("   - Interacts with env using actor_critic policy")
    print("   - Inverts bot states using invert_state()")
    print("   - Marks which steps were bot actions with is_bot_data flag")
    print("4. Call trainer.phase2_ppo_with_auxiliary_loss(rollout_fn)")
    print()
    print("Key integration points:")
    print("  - state_inverter.invert_state(obs) for converting bot observations")
    print("  - RolloutBuffer for collecting mixed agent/bot trajectories")
    print("  - PPOWithAuxiliaryLossUpdater.update() for computing combined loss")
    print()
