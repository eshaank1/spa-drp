"""
Wrapper for trained BC+PPO-fD agent to play in game_with_bots.py
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bc_ppo_fd.ppo_auxiliary_loss import ActorCriticNetwork


class AgentBot:
    """Trained agent bot that can play in game_with_bots framework."""
    
    RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    RANK_VALUES = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                   '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13}

    def __init__(self, model_path: str = "bc_ppo_fd/agent_final.pt", device: str = "cpu"):
        """Initialize agent with trained model."""
        self.device = torch.device(device)
        
        # Load trained model
        self.actor_critic = ActorCriticNetwork(
            obs_dim=50,
            action_dim=14,
            hidden_dim=256,
            num_hidden_layers=2,
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint)
        self.actor_critic.eval()

    def decide_move(self, hand, my_score, opp_score, round_num,
                    my_wins, opp_wins, opponent_has_passed):
        """
        Decide move using trained neural network.
        
        Returns:
            0 for PASS, or a card value (1-13)
        """
        if not hand:
            return 0  # PASS
        
        # Build observation from agent's perspective
        obs = self._get_obs(hand, my_score, opp_score, round_num, my_wins, opp_wins, opponent_has_passed)
        mask = self._get_valid_actions_mask(hand)
        
        # Get action from network
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            logits, _ = self.actor_critic(obs_t.unsqueeze(0), mask_t.unsqueeze(0))
            logits = logits.squeeze(0)
            action = torch.argmax(logits).item()
        
        if action == 0:
            return 0  # PASS
        else:
            # Convert action to card value
            rank = self.RANKS[action - 1]
            if rank in hand:
                return self.RANK_VALUES[rank]
            else:
                # Invalid action, pass instead
                return 0

    def _get_obs(self, my_hand, my_score, opp_score, round_num, my_wins, opp_wins, opponent_has_passed):
        """Build 50-dim observation vector."""
        obs = np.zeros(50, dtype=np.float32)
        
        # Hand (13)
        for rank in my_hand:
            obs[self.RANKS.index(rank)] = 1.0
        
        # My played cards (13) - We need to track cards played this round
        # For this we approximate by tracking score and round state
        # Note: This is computed in game_with_bots context where we have full state
        
        # Opponent's played cards (13)
        # Note: This is computed in game_with_bots context where we have full state
        
        # For now, we use score information as proxy
        # Better approach: pass actual played cards from game context
        
        # Metadata (11)
        metadata_start = 39
        obs[metadata_start + 0] = (round_num - 1) / 3.0  # Round (0-indexed)
        obs[metadata_start + 1] = my_wins / 2.0  # My rounds won
        obs[metadata_start + 2] = opp_wins / 2.0  # Opponent rounds won
        obs[metadata_start + 3] = 0.0  # Current player (agent is player 2)
        obs[metadata_start + 4] = 1.0  # Current player (agent is player 2)
        obs[metadata_start + 5] = 0.0  # First player indicator (will be set properly)
        obs[metadata_start + 6] = 1.0  # First player indicator (will be set properly)
        obs[metadata_start + 7] = 1.0 if opponent_has_passed else 0.0  # Opponent passed
        obs[metadata_start + 8] = 0.0  # Agent passed (not relevant)
        obs[metadata_start + 9] = len(my_hand) / 13.0  # Hand size
        obs[metadata_start + 10] = my_score / 70.0  # Approximate score ratio
        
        return obs

    def _get_valid_actions_mask(self, hand):
        """Create action mask (PASS always valid, then cards in hand)."""
        mask = np.zeros(14, dtype=np.float32)
        mask[0] = 1.0  # Pass always valid
        
        for card in hand:
            if card in self.RANKS:
                mask[self.RANKS.index(card) + 1] = 1.0
        
        return mask
