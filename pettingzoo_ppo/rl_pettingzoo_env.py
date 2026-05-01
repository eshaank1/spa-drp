from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_bot import SmartBot


class CardGameVsSmartParallelEnv(ParallelEnv):
    """PettingZoo ParallelEnv: one learning agent vs the existing SmartBot.

    The learning agent is always Player 1. Player 2 follows SmartBot policy.
    Action space:
    - 0: pass
    - 1..13: play rank index mapped to A..K
    """

    metadata = {"name": "card_game_vs_smart_v0", "render_modes": [None], "is_parallelizable": True}

    RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    RANK_VALUES = {
        "A": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "10": 10,
        "J": 11,
        "Q": 12,
        "K": 13,
    }

    def __init__(
        self,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        invalid_action_penalty: float = 0.2,
        round_win_reward: float = 1.0,
        game_win_reward: float = 5.0,
        score_delta_scale: float = 0.02,
    ):
        self.possible_agents = ["learner"]
        self.agents: List[str] = []

        self._action_spaces = {"learner": spaces.Discrete(14)}
        # 13 (hand) + 13 (my played) + 13 (opp played) + 11 (metadata)
        self._observation_spaces = {
            "learner": spaces.Box(low=0.0, high=1.0, shape=(50,), dtype=np.float32)
        }

        self._rank_to_action = {rank: idx + 1 for idx, rank in enumerate(self.RANKS)}
        self._action_to_rank = {idx + 1: rank for idx, rank in enumerate(self.RANKS)}

        self.smart_bot = SmartBot()
        self.rng = random.Random(seed)
        self.render_mode = render_mode

        self.invalid_action_penalty = invalid_action_penalty
        self.round_win_reward = round_win_reward
        self.game_win_reward = game_win_reward
        self.score_delta_scale = score_delta_scale

        self.player1_deck: List[str] = []
        self.player2_deck: List[str] = []
        self.player1_hand: List[str] = []
        self.player2_hand: List[str] = []
        self.p1_played: List[str] = []
        self.p2_played: List[str] = []

        self.rounds_won = [0, 0]
        self.current_round = 1
        self.first_player = 1
        self.current_player = 1
        self.passed_players = set()
        self.opponent_just_played = False

    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    def action_space(self, agent: str):
        return self._action_spaces[agent]

    def reset(self, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.rng.seed(seed)

        self.player1_deck = self.RANKS.copy()
        self.player2_deck = self.RANKS.copy()
        self.rng.shuffle(self.player1_deck)
        self.rng.shuffle(self.player2_deck)

        self.player1_hand = self.player1_deck[:7]
        self.player1_deck = self.player1_deck[7:]

        self.player2_hand = self.player2_deck[:7]
        self.player2_deck = self.player2_deck[7:]

        self.rounds_won = [0, 0]
        self.current_round = 1
        self.first_player = self.rng.choice([1, 2])

        self._reset_round_state(self.first_player)
        self.agents = self.possible_agents[:]

        auto_reward = self._advance_to_next_available_player()
        obs = self._get_observation()

        return {"learner": obs}, {"learner": {"auto_reward": auto_reward}}

    def step(self, actions: Dict[str, int]):
        if not self.agents:
            return {}, {}, {}, {}, {}

        reward = 0.0

        self._advance_to_next_available_player()

        action = int(actions.get("learner", 0))

        if not self._is_game_over() and self.current_player == 1:
            reward += self._apply_learner_action(action)
            self._advance_to_next_available_player()

        terminated = self._is_game_over()
        if terminated:
            self.agents = []

        rewards = {"learner": float(reward)}
        terminations = {"learner": terminated}
        truncations = {"learner": False}

        info = {
            "final_rounds_won": tuple(self.rounds_won),
            "winner": 1 if self.rounds_won[0] > self.rounds_won[1] else 2,
        }

        observations = {} if terminated else {"learner": self._get_observation()}
        infos = {"learner": info}

        return observations, rewards, terminations, truncations, infos

    def _apply_learner_action(self, action: int) -> float:
        reward = 0.0
        prev_delta = self._round_score_delta()

        played_card = None
        if not self.player1_hand:
            pass
        elif action == 0:
            pass
        elif action in self._action_to_rank and self._action_to_rank[action] in self.player1_hand:
            played_card = self._action_to_rank[action]
            self.player1_hand.remove(played_card)
            self.p1_played.append(played_card)
        else:
            reward -= self.invalid_action_penalty

        if played_card is None:
            self.passed_players.add(1)
            self.opponent_just_played = False
        else:
            self.opponent_just_played = True

        self.current_player = 2

        new_delta = self._round_score_delta()
        reward += self.score_delta_scale * (new_delta - prev_delta)
        reward += self._resolve_round_if_needed()
        return reward

    def _apply_opponent_action(self) -> float:
        reward = 0.0
        prev_delta = self._round_score_delta()

        played_card = None
        if self.player2_hand:
            p1_score = self._score(self.p1_played)
            p2_score = self._score(self.p2_played)
            is_last_round = self.current_round == 3

            choice = self.smart_bot.decide_move(
                hand=self.player2_hand,
                player_score=p2_score,
                opponent_score=p1_score,
                is_last_round=is_last_round,
                opponent_just_played=self.opponent_just_played,
                my_rounds_won=self.rounds_won[1],
                opponent_rounds_won=self.rounds_won[0],
            )

            if choice != "PASS" and choice in self.player2_hand:
                played_card = choice
                self.player2_hand.remove(played_card)
                self.p2_played.append(played_card)

        if played_card is None:
            self.passed_players.add(2)
            self.opponent_just_played = False
        else:
            self.opponent_just_played = True

        self.current_player = 1

        new_delta = self._round_score_delta()
        reward += self.score_delta_scale * (new_delta - prev_delta)
        reward += self._resolve_round_if_needed()
        return reward

    def _advance_to_next_available_player(self) -> float:
        reward = 0.0

        while not self._is_game_over():
            if self.current_player not in self.passed_players:
                break

            other_player = 2 if self.current_player == 1 else 1
            if other_player in self.passed_players:
                break

            self.current_player = other_player

        while not self._is_game_over() and self.current_player == 2:
            reward += self._apply_opponent_action()
        return reward

    def _resolve_round_if_needed(self) -> float:
        if len(self.passed_players) < 2:
            return 0.0

        reward = 0.0
        p1_score = self._score(self.p1_played)
        p2_score = self._score(self.p2_played)

        if p1_score > p2_score:
            self.rounds_won[0] += 1
            self.first_player = 1
            reward += self.round_win_reward
        elif p2_score > p1_score:
            self.rounds_won[1] += 1
            self.first_player = 2
            reward -= self.round_win_reward
        else:
            self.rounds_won[0] += 1
            self.rounds_won[1] += 1

        if self._is_game_over():
            if self.rounds_won[0] > self.rounds_won[1]:
                reward += self.game_win_reward
            else:
                reward -= self.game_win_reward
            return reward

        if self.current_round < 3:
            self._draw_cards()

        self.current_round += 1
        self._reset_round_state(self.first_player)
        return reward

    def _draw_cards(self):
        if self.player1_deck:
            self.player1_hand.append(self.player1_deck.pop(0))
        if self.player2_deck:
            self.player2_hand.append(self.player2_deck.pop(0))

    def _reset_round_state(self, starting_player: int):
        self.p1_played = []
        self.p2_played = []
        self.passed_players = set()
        self.opponent_just_played = False
        self.current_player = starting_player

    def _is_game_over(self) -> bool:
        return self.current_round > 3 and len(self.passed_players) >= 2

    def _get_action_mask(self) -> np.ndarray:
        """Return a mask of valid actions (1 = valid, 0 = invalid)."""
        mask = np.zeros(14, dtype=np.uint8)
        mask[0] = 1  # Action 0 (pass) is always valid
        
        # Mark card actions as valid if card is in hand
        for action, rank in self._action_to_rank.items():
            if rank in self.player1_hand:
                mask[action] = 1
        
        return mask

    def _get_observation(self):
        obs = np.zeros(50, dtype=np.float32)

        # 13 cards in hand
        for index, rank in enumerate(self.RANKS):
            if rank in self.player1_hand:
                obs[index] = 1.0

        # 13 cards already played by the learner
        for index, rank in enumerate(self.RANKS, start=13):
            if rank in self.p1_played:
                obs[index] = 1.0

        # 13 cards already played by the opponent
        for index, rank in enumerate(self.RANKS, start=26):
            if rank in self.p2_played:
                obs[index] = 1.0

        # 11 metadata features
        metadata_start = 39
        obs[metadata_start + 0] = self.current_round / 3.0
        obs[metadata_start + 1] = self.rounds_won[0] / 2.0
        obs[metadata_start + 2] = self.rounds_won[1] / 2.0
        obs[metadata_start + 3] = 1.0 if self.current_player == 1 else 0.0
        obs[metadata_start + 4] = 1.0 if self.current_player == 2 else 0.0
        obs[metadata_start + 5] = 1.0 if self.first_player == 1 else 0.0
        obs[metadata_start + 6] = 1.0 if self.first_player == 2 else 0.0
        obs[metadata_start + 7] = 1.0 if 1 in self.passed_players else 0.0
        obs[metadata_start + 8] = 1.0 if 2 in self.passed_players else 0.0
        obs[metadata_start + 9] = len(self.player1_hand) / 13.0
        obs[metadata_start + 10] = len(self.player2_hand) / 13.0

        return obs

    def _score(self, cards: List[str]) -> int:
        return sum(self.RANK_VALUES[card] for card in cards)

    def _round_score_delta(self) -> int:
        return self._score(self.p1_played) - self._score(self.p2_played)
