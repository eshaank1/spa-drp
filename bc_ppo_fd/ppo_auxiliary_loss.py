"""
Phase 2: Custom PPO Update with Auxiliary Imitation Loss (PPO-fD).

This module implements a custom PPO algorithm that combines:
1. Standard PPO clipped surrogate objective for agent-generated rollouts
2. Cross-Entropy BC loss for heuristic bot actions (learned from inverted states)
3. Value loss for value function regression
4. Entropy bonus for exploration

The rollout buffer contains states where:
- Agent acted (to be trained with PPO surrogate loss)
- Heuristic bot acted (to be trained with BC loss)

We use the `is_bot_data` flag to distinguish between the two.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RolloutBatch:
    """Batch of rollouts from environment interaction."""
    
    # Core trajectory data
    observations: torch.Tensor  # (traj_len, 50)
    action_masks: torch.Tensor  # (traj_len, 14)
    actions: torch.Tensor  # (traj_len,)
    rewards: torch.Tensor  # (traj_len,)
    values: torch.Tensor  # (traj_len,)
    log_probs: torch.Tensor  # (traj_len,)
    dones: torch.Tensor  # (traj_len,)
    
    # Flags to distinguish agent vs. bot actions
    is_bot_data: torch.Tensor  # (traj_len,) - True where bot acted, False where agent acted
    
    @property
    def agent_mask(self) -> torch.Tensor:
        """Mask for agent-generated data."""
        return ~self.is_bot_data
    
    @property
    def bot_mask(self) -> torch.Tensor:
        """Mask for bot-generated data."""
        return self.is_bot_data


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network: shared backbone + separate heads for policy and value.
    """
    
    def __init__(
        self,
        obs_dim: int = 50,
        action_dim: int = 14,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
    ):
        """
        Args:
            obs_dim: Observation dimension
            action_dim: Action space dimension
            hidden_dim: Hidden layer size
            num_hidden_layers: Number of hidden layers in backbone
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Shared backbone
        layers = []
        prev_dim = obs_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # Policy head (actor)
        self.policy_head = nn.Linear(prev_dim, action_dim)
        
        # Value head (critic)
        self.value_head = nn.Linear(prev_dim, 1)
    
    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            obs: Observations (batch_size, obs_dim)
            action_mask: Action mask (batch_size, action_dim) [1=valid, 0=invalid]
        
        Returns:
            Tuple of (policy_logits, values)
              - policy_logits: (batch_size, action_dim) with masking applied
              - values: (batch_size, 1)
        """
        features = self.backbone(obs)
        
        # Policy logits with action masking
        policy_logits = self.policy_head(features)
        if action_mask is not None:
            policy_logits = policy_logits.masked_fill(action_mask == 0, -1e9)
        
        # Value predictions
        values = self.value_head(features)
        
        return policy_logits, values
    
    def get_action_log_prob(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get log probability of action and entropy.
        
        Args:
            obs: (batch_size, obs_dim)
            action: (batch_size,) with values in [0, action_dim-1]
            action_mask: (batch_size, action_dim)
        
        Returns:
            Tuple of (log_probs, entropy)
              - log_probs: (batch_size,)
              - entropy: (batch_size,) or scalar
        """
        policy_logits, _ = self.forward(obs, action_mask=action_mask)
        
        # Softmax to get probabilities
        probs = F.softmax(policy_logits, dim=-1)
        
        # Log probabilities
        log_probs = F.log_softmax(policy_logits, dim=-1)
        
        # Gather log prob for sampled actions
        # action shape: (batch_size,), so we need to gather along dim 1
        action_log_probs = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)  # (batch_size,)
        
        # Entropy: -sum(p * log_p)
        entropy = -(probs * log_probs).sum(dim=-1)  # (batch_size,)
        
        return action_log_probs, entropy


class PPOWithAuxiliaryLossUpdater:
    """
    PPO update with auxiliary BC loss for bot demonstrations.
    """
    
    def __init__(
        self,
        actor_critic: ActorCriticNetwork,
        device: torch.device = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,
        # PPO hyperparameters
        clip_ratio: float = 0.2,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        # Auxiliary BC loss
        bc_lambda: float = 0.1,
    ):
        """
        Args:
            actor_critic: ActorCriticNetwork instance
            device: torch device
            learning_rate: Adam learning rate
            weight_decay: L2 regularization
            clip_ratio: PPO clipping parameter (epsilon)
            entropy_coeff: Weight for entropy bonus
            value_coeff: Weight for value loss
            max_grad_norm: Gradient clipping threshold
            bc_lambda: Weight for auxiliary BC loss (0.05-0.2 recommended)
        """
        self.device = device or torch.device("cpu")
        self.actor_critic = actor_critic.to(self.device)
        
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # PPO hyperparameters
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        
        # Auxiliary BC loss
        self.bc_lambda = bc_lambda
        self.bc_criterion = nn.CrossEntropyLoss(reduction="mean")
    
    def compute_advantages(
        self,
        rollouts: RolloutBatch,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            rollouts: RolloutBatch
            gamma: Discount factor
            gae_lambda: GAE smoothing parameter
        
        Returns:
            Tuple of (advantages, returns)
              - advantages: (traj_len,)
              - returns: (traj_len,)
        """
        # Compute TD residuals
        values = rollouts.values.squeeze(-1)  # (traj_len,)
        rewards = rollouts.rewards  # (traj_len,)
        dones = rollouts.dones  # (traj_len,)
        
        # Value targets: r_t + gamma * V(s_{t+1}) * (1 - done)
        # Shift values for bootstrapping
        next_values = torch.cat([values[1:], torch.zeros(1, device=self.device)])
        td_targets = rewards + gamma * next_values * (1.0 - dones)
        td_residuals = td_targets - values
        
        # GAE computation: advantages = sum of discounted TD residuals
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            gae = td_residuals[t] + gamma * gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        
        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def compute_ppo_loss(
        self,
        agent_obs: torch.Tensor,
        agent_actions: torch.Tensor,
        agent_masks: torch.Tensor,
        agent_old_log_probs: torch.Tensor,
        agent_advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute PPO surrogate loss for agent-generated data.
        
        Args:
            agent_obs: (num_agent_steps, obs_dim)
            agent_actions: (num_agent_steps,)
            agent_masks: (num_agent_steps, action_dim)
            agent_old_log_probs: (num_agent_steps,) - old log probs from rollout
            agent_advantages: (num_agent_steps,) - advantages (normalized)
        
        Returns:
            Tuple of (ppo_loss, entropy_loss)
        """
        # Get new policy and log probs
        new_log_probs, entropy = self.actor_critic.get_action_log_prob(
            agent_obs,
            agent_actions,
            action_mask=agent_masks
        )
        
        # Importance sampling ratio: r_t = pi_new(a|s) / pi_old(a|s)
        ratio = torch.exp(new_log_probs - agent_old_log_probs)
        
        # Clipped surrogate loss
        surr1 = ratio * agent_advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * agent_advantages
        ppo_loss = -torch.min(surr1, surr2).mean()
        
        # Entropy bonus (maximize entropy for exploration)
        entropy_bonus = entropy.mean()
        
        return ppo_loss, entropy_bonus
    
    def compute_bc_loss(
        self,
        bot_obs: torch.Tensor,
        bot_actions: torch.Tensor,
        bot_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute behavioral cloning loss for bot-generated data.
        
        For bot-generated trajectories, we don't use advantage estimates.
        Instead, we simply train the actor to predict the bot's actions
        using cross-entropy loss (with action masking applied).
        
        Args:
            bot_obs: (num_bot_steps, obs_dim)
            bot_actions: (num_bot_steps,)
            bot_masks: (num_bot_steps, action_dim)
        
        Returns:
            bc_loss: scalar
        """
        # Get policy logits (with masking)
        policy_logits, _ = self.actor_critic.forward(bot_obs, action_mask=bot_masks)
        
        # Cross-entropy loss on the bot's actions
        bc_loss = self.bc_criterion(policy_logits, bot_actions)
        
        return bc_loss
    
    def compute_value_loss(
        self,
        obs: torch.Tensor,
        returns: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute value function loss (MSE on returns).
        
        Args:
            obs: (traj_len, obs_dim)
            returns: (traj_len,) - computed returns
        
        Returns:
            value_loss: scalar
        """
        _, predicted_values = self.actor_critic.forward(obs)
        predicted_values = predicted_values.squeeze(-1)  # (traj_len,)
        
        value_loss = F.mse_loss(predicted_values, returns)
        
        return value_loss
    
    def update(
        self,
        rollouts: RolloutBatch,
        num_epochs: int = 3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Dict[str, float]:
        """
        Perform PPO update with auxiliary BC loss.
        
        Args:
            rollouts: RolloutBatch with combined agent and bot data
            num_epochs: Number of passes over the data
            gamma: Discount factor
            gae_lambda: GAE parameter
        
        Returns:
            Dictionary with loss components:
              - total_loss
              - ppo_loss
              - bc_loss
              - value_loss
              - entropy_bonus
        """
        self.actor_critic.train()
        
        # Compute advantages and returns for ALL data
        # (we only use advantages for agent data, but compute for all for value loss)
        advantages, returns = self.compute_advantages(rollouts, gamma=gamma, gae_lambda=gae_lambda)
        
        # Masks for separating agent and bot data
        agent_mask = rollouts.agent_mask  # (traj_len,)
        bot_mask = rollouts.bot_mask  # (traj_len,)
        
        # Aggregate losses over epochs
        loss_history = {
            "total_loss": [],
            "ppo_loss": [],
            "bc_loss": [],
            "value_loss": [],
            "entropy_bonus": [],
        }
        
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            
            # ===== PPO Loss (Agent Data) =====
            # Shape annotations for clarity:
            #   observations: (traj_len, 50)
            #   actions: (traj_len,)
            #   action_masks: (traj_len, 14)
            #   agent_mask: (traj_len,) boolean
            #   agent_advantages: (traj_len,)
            
            if agent_mask.any():
                agent_obs = rollouts.observations[agent_mask]  # (num_agent, 50)
                agent_actions = rollouts.actions[agent_mask]  # (num_agent,)
                agent_masks = rollouts.action_masks[agent_mask]  # (num_agent, 14)
                agent_old_log_probs = rollouts.log_probs[agent_mask]  # (num_agent,)
                agent_advantages = advantages[agent_mask]  # (num_agent,)
                
                ppo_loss, entropy_bonus = self.compute_ppo_loss(
                    agent_obs,
                    agent_actions,
                    agent_masks,
                    agent_old_log_probs,
                    agent_advantages,
                )
            else:
                ppo_loss = torch.tensor(0.0, device=self.device)
                entropy_bonus = torch.tensor(0.0, device=self.device)
            
            # ===== BC Loss (Bot Data) =====
            # For bot data, we ignore advantage estimates and just do supervised learning.
            # Shape annotations:
            #   bot_obs: (num_bot, 50)
            #   bot_actions: (num_bot,)
            #   bot_masks: (num_bot, 14)
            
            if bot_mask.any():
                bot_obs = rollouts.observations[bot_mask]  # (num_bot, 50)
                bot_actions = rollouts.actions[bot_mask]  # (num_bot,)
                bot_masks = rollouts.action_masks[bot_mask]  # (num_bot, 14)
                
                bc_loss = self.compute_bc_loss(bot_obs, bot_actions, bot_masks)
            else:
                bc_loss = torch.tensor(0.0, device=self.device)
            
            # ===== Value Loss (All Data) =====
            # We train the value function on both agent and bot data.
            # Shape annotations:
            #   observations: (traj_len, 50)
            #   returns: (traj_len,)
            
            value_loss = self.compute_value_loss(rollouts.observations, returns)
            
            # ===== Combined Loss =====
            # Total Loss = PPO_Loss + lambda * BC_Loss + value_coeff * Value_Loss - entropy_coeff * Entropy_Bonus
            
            total_loss = (
                ppo_loss
                + self.bc_lambda * bc_loss
                + self.value_coeff * value_loss
                - self.entropy_coeff * entropy_bonus
            )
            
            # Backward and optimize
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Record losses
            loss_history["total_loss"].append(total_loss.item())
            loss_history["ppo_loss"].append(ppo_loss.item() if isinstance(ppo_loss, torch.Tensor) else ppo_loss)
            loss_history["bc_loss"].append(bc_loss.item() if isinstance(bc_loss, torch.Tensor) else bc_loss)
            loss_history["value_loss"].append(value_loss.item() if isinstance(value_loss, torch.Tensor) else value_loss)
            loss_history["entropy_bonus"].append(entropy_bonus.item() if isinstance(entropy_bonus, torch.Tensor) else entropy_bonus)
        
        # Return average losses over epochs
        return {
            k: np.mean(v) for k, v in loss_history.items()
        }


if __name__ == "__main__":
    print("Testing PPO-fD update mechanism...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a dummy actor-critic network
    actor_critic = ActorCriticNetwork(obs_dim=50, action_dim=14)
    
    # Create updater
    updater = PPOWithAuxiliaryLossUpdater(
        actor_critic,
        device=device,
        bc_lambda=0.1
    )
    
    # Create dummy rollouts: 50% agent data, 50% bot data
    traj_len = 100
    agent_portion = 50
    
    obs = torch.randn(traj_len, 50)
    masks = torch.ones(traj_len, 14)
    actions = torch.randint(0, 14, (traj_len,))
    rewards = torch.randn(traj_len)
    values = torch.randn(traj_len, 1)
    old_log_probs = torch.randn(traj_len)
    dones = torch.zeros(traj_len)
    dones[traj_len - 1] = 1.0
    
    is_bot_data = torch.zeros(traj_len, dtype=torch.bool)
    is_bot_data[agent_portion:] = True
    
    rollouts = RolloutBatch(
        observations=obs,
        action_masks=masks,
        actions=actions,
        rewards=rewards,
        values=values,
        log_probs=old_log_probs,
        dones=dones,
        is_bot_data=is_bot_data,
    )
    
    # Update
    losses = updater.update(rollouts, num_epochs=2)
    
    print("Losses after 1 update:")
    for k, v in losses.items():
        print(f"  {k}: {v:.4f}")
