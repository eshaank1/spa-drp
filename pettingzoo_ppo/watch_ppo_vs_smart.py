from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_bot import SmartBot


class WatchPPOVsSmart:
    RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    RANK_VALUES = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                   '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13}

    def __init__(self, model_path: str, seed: int = 123):
        self.model = PPO.load(model_path)
        self.smart_bot = SmartBot()
        self.rng = random.Random(seed)

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
        self.current_player = self.first_player
        self.passed_players = set()
        self.p1_played: List[str] = []
        self.p2_played: List[str] = []
        self.opponent_just_played = False

    def display_game_state(self, player_num: int):
        if player_num == 1:
            hand = self.player1_hand
            deck_size = len(self.player1_deck)
            label = 'PPO Bot'
        else:
            hand = self.player2_hand
            deck_size = len(self.player2_deck)
            label = 'Smart Bot'

        print(f"\n--- Player {player_num}'s Turn ({label}) ---")
        rank_order = self.RANKS
        sorted_hand = sorted(hand, key=lambda card: rank_order.index(card))
        print(f"Hand: {', '.join(sorted_hand) if sorted_hand else '[empty]'}")
        print(f"Cards remaining in deck: {deck_size}")
        print(f"Round {self.current_round} | Wins: {self.rounds_won[player_num - 1]}")

    def display_round_state(self):
        p1_score = sum(self.RANK_VALUES[card] for card in self.p1_played)
        p2_score = sum(self.RANK_VALUES[card] for card in self.p2_played)
        print(f"Player 1 played: {self.p1_played if self.p1_played else 'None'} (Score: {p1_score})")
        print(f"Player 2 played: {self.p2_played if self.p2_played else 'None'} (Score: {p2_score})")

    def _build_observation(self):
        obs = np.zeros(50, dtype=np.float32)

        for index, rank in enumerate(self.RANKS):
            if rank in self.player1_hand:
                obs[index] = 1.0

        for index, rank in enumerate(self.RANKS, start=13):
            if rank in self.p1_played:
                obs[index] = 1.0

        for index, rank in enumerate(self.RANKS, start=26):
            if rank in self.p2_played:
                obs[index] = 1.0

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

    def _score_round(self):
        p1_score = sum(self.RANK_VALUES[card] for card in self.p1_played)
        p2_score = sum(self.RANK_VALUES[card] for card in self.p2_played)
        print(f"\n--- Round {self.current_round} Results ---")
        print(f"Player 1 total: {p1_score}")
        print(f"Player 2 total: {p2_score}")

        if p1_score > p2_score:
            print("PPO Bot WINS this round!")
            self.rounds_won[0] += 1
            self.first_player = 1
        elif p2_score > p1_score:
            print("Smart Bot WINS this round!")
            self.rounds_won[1] += 1
            self.first_player = 2
        else:
            print("TIED this round! Both players get a point.")
            self.rounds_won[0] += 1
            self.rounds_won[1] += 1

        print(f"Score: PPO Bot: {self.rounds_won[0]} | Smart Bot: {self.rounds_won[1]}")

    def _draw_cards(self):
        for _ in range(3):
            if self.player1_deck:
                self.player1_hand.append(self.player1_deck.pop(0))
            if self.player2_deck:
                self.player2_hand.append(self.player2_deck.pop(0))

    def _bot_action(self):
        if not self.player1_hand:
            print("PPO Bot has no cards - forced pass")
            self.passed_players.add(1)
            self.opponent_just_played = False
            return

        obs = self._build_observation()
        action, _ = self.model.predict(obs, deterministic=True)
        action = int(action)

        if action == 0:
            print("PPO Bot passes")
            self.passed_players.add(1)
            self.opponent_just_played = False
            return

        if action not in range(1, 14):
            print(f"PPO Bot produced invalid action {action} - treating as pass")
            self.passed_players.add(1)
            self.opponent_just_played = False
            return

        card = self.RANKS[action - 1]
        if card not in self.player1_hand:
            print(f"PPO Bot tried to play {card} but does not have it - treating as pass")
            self.passed_players.add(1)
            self.opponent_just_played = False
            return

        self.p1_played.append(card)
        self.player1_hand.remove(card)
        print(f"PPO Bot played: {card}")
        self.opponent_just_played = True

    def _smart_action(self):
        if not self.player2_hand:
            print("Smart Bot has no cards - forced pass")
            self.passed_players.add(2)
            self.opponent_just_played = False
            return

        p1_score = sum(self.RANK_VALUES[card] for card in self.p1_played)
        p2_score = sum(self.RANK_VALUES[card] for card in self.p2_played)
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

        if choice == 'PASS':
            print("Smart Bot passes")
            self.passed_players.add(2)
            self.opponent_just_played = False
            return

        if choice not in self.player2_hand:
            print(f"Smart Bot tried to play {choice} but does not have it - treating as pass")
            self.passed_players.add(2)
            self.opponent_just_played = False
            return

        self.p2_played.append(choice)
        self.player2_hand.remove(choice)
        print(f"Smart Bot played: {choice}")
        self.opponent_just_played = True

    def play_round(self):
        self.p1_played = []
        self.p2_played = []
        self.passed_players = set()
        self.current_player = self.first_player
        self.opponent_just_played = False

        print(f"\n{'=' * 50}")
        print(f"ROUND {self.current_round}")
        print(f"{'=' * 50}")
        print(f"Player {self.first_player} plays first this round.")

        while True:
            if self.current_player in self.passed_players:
                other_player = 2 if self.current_player == 1 else 1
                if other_player in self.passed_players:
                    break
                self.current_player = other_player
                continue

            if self.current_player == 1:
                self.display_game_state(1)
                self.display_round_state()
                self._bot_action()
                self.current_player = 2
            else:
                self.display_game_state(2)
                self.display_round_state()
                self._smart_action()
                self.current_player = 1

        self._score_round()

        if self.rounds_won[0] >= 2 or self.rounds_won[1] >= 2:
            return

        if self.current_round < 3:
            self._draw_cards()

        self.current_round += 1

    def play_game(self):
        print("\n" + "=" * 50)
        print("CARD GAME: PPO BOT vs SMART BOT")
        print("=" * 50)
        print("PPO Bot is Player 1. Smart Bot is Player 2.")
        print("First to win 2 rounds wins!")
        print("=" * 50)

        while self.rounds_won[0] < 2 and self.rounds_won[1] < 2:
            self.play_round()

        print(f"\n{'=' * 50}")
        print("GAME OVER!")
        print(f"{'=' * 50}")
        if self.rounds_won[0] > self.rounds_won[1]:
            print("PPO BOT WINS THE GAME!")
        else:
            print("SMART BOT WINS THE GAME!")
        print(f"Final Score: PPO Bot: {self.rounds_won[0]} | Smart Bot: {self.rounds_won[1]}")


def main():
    parser = argparse.ArgumentParser(description="Watch PPO bot play SmartBot.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    game = WatchPPOVsSmart(model_path=args.model_path, seed=args.seed)
    game.play_game()


if __name__ == "__main__":
    main()
