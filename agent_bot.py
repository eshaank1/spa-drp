"""
Wrapper for the trained BC+PPO-fD agent so it can play in game_with_bots.py.

The network is trained entirely from the "learner" (Player 1) perspective, i.e.
observations are laid out as [my_hand | my_played | opp_played | metadata]. This
wrapper therefore builds the agent's *own* perspective regardless of which seat
it occupies, matching the exact 50-dim format used during training
(see rl_pettingzoo_env._get_observation). The previous version left all 26
played-card features as zeros and put the score in the opponent-hand slot, so the
agent was effectively blind at play time — that is fixed here.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bc_ppo_fd.ppo_auxiliary_loss import ActorCriticNetwork


class AgentBot:
    """Trained agent that plays in the game_with_bots framework."""

    RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    RANK_VALUES = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                   '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13}

    def __init__(self, model_path: str = None, device: str = "cpu",
                 deterministic: bool = False, temperature: float = 1.0):
        """
        Args:
            model_path: path to the trained state_dict (default: bc_ppo_fd/agent_final.pt)
            deterministic: if True, take argmax; otherwise sample from the policy
                (a little stochasticity makes the agent far less exploitable by a
                human who would otherwise learn its fixed responses).
            temperature: softmax temperature used when sampling.
        """
        if model_path is None:
            model_path = str(PROJECT_ROOT / "bc_ppo_fd" / "agent_final.pt")
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.temperature = temperature

        self.actor_critic = ActorCriticNetwork(
            obs_dim=50, action_dim=14, hidden_dim=256, num_hidden_layers=2,
        ).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint)
        self.actor_critic.eval()

    def decide_move(self, hand, my_score, opp_score, round_num,
                    my_wins, opp_wins, opponent_has_passed,
                    my_played=None, opp_played=None, opp_hand_size=None,
                    first_player_is_me=None, i_have_passed=False):
        """
        Decide a move from the agent's own perspective.

        Returns:
            0 for PASS, or a card value (1-13).

        The optional keyword args carry the full round state from the game loop.
        When they are omitted the agent still plays, but the played-card features
        are treated as empty (degraded — callers should pass them).
        """
        if not hand:
            return 0  # PASS

        obs = self._build_observation(
            hand, my_score, opp_score, round_num, my_wins, opp_wins,
            opponent_has_passed, my_played or [], opp_played or [],
            opp_hand_size, first_player_is_me, i_have_passed,
        )
        mask = self._valid_actions_mask(hand)
        if self._is_critical(round_num, opp_wins):
            mask[0] = 0.0  # final round: cards have no future value, never pass while holding one

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits, _ = self.actor_critic(obs_t, mask_t)
            logits = logits.squeeze(0)
            if self.deterministic:
                action = torch.argmax(logits).item()
            else:
                probs = F.softmax(logits / max(self.temperature, 1e-6), dim=-1)
                action = torch.multinomial(probs, 1).item()

        if action == 0:
            return 0  # PASS
        rank = self.RANKS[action - 1]
        if rank in hand:
            return self.RANK_VALUES[rank]
        return 0  # masked out should prevent this, but pass defensively

    def _build_observation(self, hand, my_score, opp_score, round_num,
                           my_wins, opp_wins, opponent_has_passed,
                           my_played, opp_played, opp_hand_size,
                           first_player_is_me, i_have_passed) -> np.ndarray:
        """Build the 50-dim observation in the learner (own-perspective) format,
        matching rl_pettingzoo_env._get_observation exactly."""
        obs = np.zeros(50, dtype=np.float32)

        # [0:13] my hand
        for rank in hand:
            obs[self.RANKS.index(rank)] = 1.0
        # [13:26] my played cards this round
        for rank in my_played:
            obs[13 + self.RANKS.index(rank)] = 1.0
        # [26:39] opponent's played cards this round
        for rank in opp_played:
            obs[26 + self.RANKS.index(rank)] = 1.0

        # [39:50] metadata (from "me as the learner" perspective)
        obs[39] = round_num / 3.0
        obs[40] = my_wins / 2.0
        obs[41] = opp_wins / 2.0
        obs[42] = 1.0   # it is my turn (the learner is always to-move in training)
        obs[43] = 0.0
        if first_player_is_me is None:
            # Unknown: leave both first-player flags off rather than guess wrong.
            obs[44] = 0.0
            obs[45] = 0.0
        else:
            obs[44] = 1.0 if first_player_is_me else 0.0
            obs[45] = 0.0 if first_player_is_me else 1.0
        obs[46] = 1.0 if i_have_passed else 0.0
        obs[47] = 1.0 if opponent_has_passed else 0.0
        obs[48] = len(hand) / 13.0
        if opp_hand_size is not None:
            obs[49] = opp_hand_size / 13.0
        return obs

    @staticmethod
    def _is_critical(round_num, opp_wins) -> bool:
        """A must-win round: either the literal last round, or the opponent
        already has 1 round win so losing this one ends the game."""
        return round_num >= 3 or opp_wins == 1

    def _valid_actions_mask(self, hand) -> np.ndarray:
        mask = np.zeros(14, dtype=np.float32)
        mask[0] = 1.0  # Pass always valid
        for card in hand:
            if card in self.RANKS:
                mask[self.RANKS.index(card) + 1] = 1.0
        return mask
