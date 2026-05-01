# 2 human players against each other

import random
from typing import List


class CardGame:
    RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    RANK_VALUES = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                   '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13}

    def __init__(self):
        self.player1_deck = self.RANKS.copy()
        self.player2_deck = self.RANKS.copy()

        # Draw initial 7 cards
        random.shuffle(self.player1_deck)
        random.shuffle(self.player2_deck)

        self.player1_hand = self.player1_deck[:7]
        self.player1_deck = self.player1_deck[7:]

        self.player2_hand = self.player2_deck[:7]
        self.player2_deck = self.player2_deck[7:]

        self.rounds_won = [0, 0]  # [Player 1 wins, Player 2 wins]
        self.current_round = 1
        self.first_player = random.choice([1, 2])

    def display_game_state(self, player_num: int):
        """Display current state for a player"""
        if player_num == 1:
            hand = self.player1_hand
            deck_size = len(self.player1_deck)
        else:
            hand = self.player2_hand
            deck_size = len(self.player2_deck)

        # Display hand in sorted order
        rank_order = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        sorted_hand = sorted(hand, key=lambda card: rank_order.index(card))

        print(f"\n--- Player {player_num}'s Turn ---")
        print(f"Your hand: {', '.join(sorted_hand)}")
        print(f"Cards remaining in deck: {deck_size}")
        print(f"Round {self.current_round}/3 | Your wins: {self.rounds_won[player_num-1]}")

    def display_round_state(self, p1_played: List[str], p2_played: List[str]):
        """Display cards played this round"""
        p1_score = sum(self.RANK_VALUES[card] for card in p1_played)
        p2_score = sum(self.RANK_VALUES[card] for card in p2_played)

        print(f"\nPlayer 1 played: {p1_played if p1_played else 'None'} (Score: {p1_score})")
        print(f"Player 2 played: {p2_played if p2_played else 'None'} (Score: {p2_score})")

    def play_round(self):
        """Play a single round"""
        p1_played = []
        p2_played = []
        current_player = self.first_player
        passed_players = set()

        print(f"\n{'='*50}")
        print(f"ROUND {self.current_round}")
        print(f"{'='*50}")
        print(f"Player {self.first_player} plays first this round.")

        while True:
            if current_player in passed_players:
                other_player = 2 if current_player == 1 else 1
                if other_player in passed_players:
                    break
                current_player = other_player
                continue

            if current_player == 1:
                self.display_game_state(1)
                self.display_round_state(p1_played, p2_played)

                # Auto-pass if no cards in hand
                if not self.player1_hand:
                    print("Player 1 has no cards - forced pass")
                    passed_players.add(1)
                    if len(passed_players) == 2:
                        break
                    current_player = 2
                    continue
                else:
                    # Get player input
                    while True:
                        choice = input("Play a card or pass? (card name or 'pass'): ").strip().upper()
                        if choice == 'PASS':
                            print("Player 1 passes")
                            passed_players.add(1)
                            break
                        elif choice in self.player1_hand:
                            p1_played.append(choice)
                            self.player1_hand.remove(choice)
                            print(f"Player 1 played: {choice}")
                            break
                        else:
                            print("Invalid choice. Please play a card from your hand or pass.")

                current_player = 2

            else:  # Player 2
                self.display_game_state(2)
                self.display_round_state(p1_played, p2_played)

                # Auto-pass if no cards in hand
                if not self.player2_hand:
                    print("Player 2 has no cards - forced pass")
                    passed_players.add(2)
                    if len(passed_players) == 2:
                        break
                    current_player = 1
                    continue
                else:
                    # Get player input
                    while True:
                        choice = input("Play a card or pass? (card name or 'pass'): ").strip().upper()
                        if choice == 'PASS':
                            print("Player 2 passes")
                            passed_players.add(2)
                            break
                        elif choice in self.player2_hand:
                            p2_played.append(choice)
                            self.player2_hand.remove(choice)
                            print(f"Player 2 played: {choice}")
                            break
                        else:
                            print("Invalid choice. Please play a card from your hand or pass.")

                current_player = 1

        # Calculate scores and determine winner
        p1_score = sum(self.RANK_VALUES[card] for card in p1_played)
        p2_score = sum(self.RANK_VALUES[card] for card in p2_played)

        print(f"\n--- Round {self.current_round} Results ---")
        print(f"Player 1 total: {p1_score}")
        print(f"Player 2 total: {p2_score}")

        if p1_score > p2_score:
            print("Player 1 WINS this round!")
            self.rounds_won[0] += 1
            self.first_player = 1
        elif p2_score > p1_score:
            print("Player 2 WINS this round!")
            self.rounds_won[1] += 1
            self.first_player = 2
        else:
            print("TIED this round! Both players get a point.")
            self.rounds_won[0] += 1
            self.rounds_won[1] += 1

        print(f"Score: Player 1: {self.rounds_won[0]} | Player 2: {self.rounds_won[1]}")

        # Check if game is over
        if self.rounds_won[0] >= 2 or self.rounds_won[1] >= 2:
            return

        # Draw 3 new cards if game continues
        if self.current_round < 3:
            self._draw_cards()

        self.current_round += 1

    def _draw_cards(self):
        """Draw 3 cards for each player from their deck"""
        for _ in range(3):
            if self.player1_deck:
                self.player1_hand.append(self.player1_deck.pop(0))
            if self.player2_deck:
                self.player2_hand.append(self.player2_deck.pop(0))

    def play_game(self):
        """Play the complete game"""
        print("\n" + "="*50)
        print("CARD STRATEGY GAME")
        print("="*50)
        print("First to win 2 out of 3 rounds wins!")
        print(f"Player {self.first_player} goes first!")
        print("="*50)

        while sum(w >= 2 for w in self.rounds_won) == 0:
            self.play_round()

        print(f"\n{'='*50}")
        print("GAME OVER!")
        print(f"{'='*50}")
        if self.rounds_won[0] > self.rounds_won[1]:
            print("PLAYER 1 WINS THE GAME!")
        else:
            print("PLAYER 2 WINS THE GAME!")
        print(f"Final Score: Player 1: {self.rounds_won[0]} | Player 2: {self.rounds_won[1]}")


if __name__ == "__main__":
    game = CardGame()
    game.play_game()
