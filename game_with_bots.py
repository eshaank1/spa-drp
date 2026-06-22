# Human vs either bot, or bot vs bot

import random
from typing import List, Optional
from smart_bot import SmartBot
from random_bot import RandomBot
from baseline_bot import BaselineBot
from agent_bot import AgentBot


class CardGameWithBots:
    RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    RANK_VALUES = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                   '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13}

    def __init__(self, player1_type='human', player2_type='smart'):
        """
        Initialize game with player types.

        Args:
            player1_type: 'human', 'smart', 'random', 'baseline', or 'agent'
            player2_type: 'human', 'smart', 'random', 'baseline', or 'agent'
        """
        self.player1_type = player1_type
        self.player2_type = player2_type

        self.player1_bot = self._create_bot(player1_type)
        self.player2_bot = self._create_bot(player2_type)

        self.player1_deck = self.RANKS.copy()
        self.player2_deck = self.RANKS.copy()

        random.shuffle(self.player1_deck)
        random.shuffle(self.player2_deck)

        self.player1_hand = self.player1_deck[:7]
        self.player1_deck = self.player1_deck[7:]

        self.player2_hand = self.player2_deck[:7]
        self.player2_deck = self.player2_deck[7:]

        self.rounds_won = [0, 0]
        self.current_round = 1
        self.first_player = random.choice([1, 2])

    def _create_bot(self, player_type):
        """Create a bot instance if needed"""
        if player_type == 'smart':
            return SmartBot()
        elif player_type == 'random':
            return RandomBot()
        elif player_type == 'baseline':
            return BaselineBot()
        elif player_type == 'agent':
            return AgentBot()
        return None

    def _get_bot_move(self, player_num: int, p1_played: List[str], p2_played: List[str], 
                      opponent_just_played: bool, opponent_has_passed: bool = False) -> str:
        """Get move from a bot, handling different bot interfaces"""
        if player_num == 1:
            bot = self.player1_bot
            bot_type = self.player1_type
            hand = self.player1_hand
            player_score = sum(self.RANK_VALUES[card] for card in p1_played)
            opponent_score = sum(self.RANK_VALUES[card] for card in p2_played)
            my_wins = self.rounds_won[0]
            opp_wins = self.rounds_won[1]
        else:
            bot = self.player2_bot
            bot_type = self.player2_type
            hand = self.player2_hand
            player_score = sum(self.RANK_VALUES[card] for card in p2_played)
            opponent_score = sum(self.RANK_VALUES[card] for card in p1_played)
            my_wins = self.rounds_won[1]
            opp_wins = self.rounds_won[0]

        is_last_round = (self.current_round == 3)

        # BaselineBot and AgentBot use the updated interface
        if bot_type in ['baseline', 'agent']:
            result = bot.decide_move(
                hand=hand,
                my_score=player_score,
                opp_score=opponent_score,
                round_num=self.current_round,
                my_wins=my_wins,
                opp_wins=opp_wins,
                opponent_has_passed=opponent_has_passed,
            )
            # Returns 0 for PASS or a card value (1-13)
            if result == 0:
                return 'PASS'
            else:
                # Convert card value to card name
                for rank, value in self.RANK_VALUES.items():
                    if value == result:
                        return rank
                return 'PASS'  # Fallback if invalid
        else:
            # SmartBot and RandomBot use the legacy interface
            return bot.decide_move(
                hand=hand,
                player_score=player_score,
                opponent_score=opponent_score,
                is_last_round=is_last_round,
                opponent_just_played=opponent_just_played,
                my_rounds_won=my_wins,
                opponent_rounds_won=opp_wins,
            )

    def display_game_state(self, player_num: int):
        """Display current state for a player"""
        if player_num == 1:
            hand = self.player1_hand
            deck_size = len(self.player1_deck)
            player_type = self.player1_type
        else:
            hand = self.player2_hand
            deck_size = len(self.player2_deck)
            player_type = self.player2_type

        print(f"\n--- Player {player_num}'s Turn ({player_type.upper()}) ---")

        # Only show hand if it's a human player
        if (player_num == 1 and self.player1_type == 'human') or (player_num == 2 and self.player2_type == 'human'):
            # Display hand in sorted order
            rank_order = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
            sorted_hand = sorted(hand, key=lambda card: rank_order.index(card))
            print(f"Hand: {', '.join(sorted_hand)}")
        else:
            print(f"Hand: [hidden]")

        print(f"Cards remaining in deck: {deck_size}")
        print(f"Round {self.current_round} | Wins: {self.rounds_won[player_num-1]}")

    def display_round_state(self, p1_played: List[str], p2_played: List[str]):
        """Display cards played this round"""
        p1_score = sum(self.RANK_VALUES[card] for card in p1_played)
        p2_score = sum(self.RANK_VALUES[card] for card in p2_played)

        print(f"Player 1 played: {p1_played if p1_played else 'None'} (Score: {p1_score})")
        print(f"Player 2 played: {p2_played if p2_played else 'None'} (Score: {p2_score})")

    def play_round(self):
        """Play a single round"""
        p1_played = []
        p2_played = []
        current_player = self.first_player
        passed_players = set()
        opponent_just_played = False

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

                # Get player 1's move
                if self.player1_type == 'human':
                    # Auto-pass if no cards in hand
                    if not self.player1_hand:
                        print("Player 1 has no cards - forced pass")
                        passed_players.add(1)
                        opponent_just_played = False
                        if len(passed_players) == 2:
                            break
                        current_player = 2
                        continue
                    else:
                        while True:
                            choice = input("Play a card or pass? (card name or 'pass'): ").strip().upper()
                            if choice == 'PASS':
                                print("Player 1 passes")
                                passed_players.add(1)
                                opponent_just_played = False
                                if len(passed_players) == 2:
                                    break
                                break
                            elif choice in self.player1_hand:
                                p1_played.append(choice)
                                self.player1_hand.remove(choice)
                                print(f"Player 1 played: {choice}")
                                opponent_just_played = True
                                break
                            else:
                                print("Invalid choice. Please play a card from your hand or pass.")
                else:
                    # Bot decides
                    p1_score = sum(self.RANK_VALUES[card] for card in p1_played)
                    p2_score = sum(self.RANK_VALUES[card] for card in p2_played)

                    # Pass in '2 in passed_players' so BaselineBot knows if Player 2 passed
                    choice = self._get_bot_move(1, p1_played, p2_played, opponent_just_played, opponent_has_passed=(2 in passed_players))

                    if choice == 'PASS':
                        print(f"Player 1 ({self.player1_type}) passes")
                        passed_players.add(1)
                        opponent_just_played = False
                        if len(passed_players) == 2:
                            break
                    else:
                        p1_played.append(choice)
                        self.player1_hand.remove(choice)
                        print(f"Player 1 ({self.player1_type}) played: {choice}")
                        opponent_just_played = True

                current_player = 2

            else:  # Player 2
                self.display_game_state(2)
                self.display_round_state(p1_played, p2_played)

                if self.player2_type == 'human':
                    # Auto-pass if no cards in hand
                    if not self.player2_hand:
                        print("Player 2 has no cards - forced pass")
                        passed_players.add(2)
                        opponent_just_played = False
                        if len(passed_players) == 2:
                            break
                        current_player = 1
                        continue
                    else:
                        while True:
                            choice = input("Play a card or pass? (card name or 'pass'): ").strip().upper()
                            if choice == 'PASS':
                                print("Player 2 passes")
                                passed_players.add(2)
                                opponent_just_played = False
                                if len(passed_players) == 2:
                                    break
                                break
                            elif choice in self.player2_hand:
                                p2_played.append(choice)
                                self.player2_hand.remove(choice)
                                print(f"Player 2 played: {choice}")
                                opponent_just_played = True
                                break
                            else:
                                print("Invalid choice. Please play a card from your hand or pass.")
                else:
                    # Bot decides
                    p1_score = sum(self.RANK_VALUES[card] for card in p1_played)
                    p2_score = sum(self.RANK_VALUES[card] for card in p2_played)

                    # Pass in '1 in passed_players' so BaselineBot knows if Player 1 passed
                    choice = self._get_bot_move(2, p1_played, p2_played, opponent_just_played, opponent_has_passed=(1 in passed_players))

                    if choice == 'PASS':
                        print(f"Player 2 ({self.player2_type}) passes")
                        passed_players.add(2)
                        opponent_just_played = False
                        if len(passed_players) == 2:
                            break
                    else:
                        p2_played.append(choice)
                        self.player2_hand.remove(choice)
                        print(f"Player 2 ({self.player2_type}) played: {choice}")
                        opponent_just_played = True

                current_player = 1

        # Calculate scores
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

        # Draw 3 new cards
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
        print(f"CARD STRATEGY GAME: {self.player1_type.upper()} vs {self.player2_type.upper()}")
        print("="*50)
        print("First to win 2 rounds wins!")
        print("="*50)

        while sum(w >= 2 for w in self.rounds_won) == 0:
            self.play_round()

        print(f"\n{'='*50}")
        print("GAME OVER!")
        print(f"{'='*50}")
        if self.rounds_won[0] > self.rounds_won[1]:
            print(f"PLAYER 1 ({self.player1_type.upper()}) WINS THE GAME!")
        else:
            print(f"PLAYER 2 ({self.player2_type.upper()}) WINS THE GAME!")
        print(f"Final Score: Player 1: {self.rounds_won[0]} | Player 2: {self.rounds_won[1]}")

        return self.rounds_won


if __name__ == "__main__":
    print("\n" + "="*50)
    print("CHOOSE YOUR MATCHUP")
    print("="*50)
    print("1. Human vs Smart Bot")
    print("2. Human vs Random Bot")
    print("3. Human vs Baseline Bot")
    print("4. Human vs Trained Agent")
    print("5. Smart Bot vs Random Bot")
    print("6. Smart Bot vs Baseline Bot")
    print("7. Agent vs Baseline Bot")
    print("="*50)

    while True:
        choice = input("Enter 1-7: ").strip()
        if choice == '1':
            game = CardGameWithBots('human', 'smart')
            break
        elif choice == '2':
            game = CardGameWithBots('human', 'random')
            break
        elif choice == '3':
            game = CardGameWithBots('human', 'baseline')
            break
        elif choice == '4':
            game = CardGameWithBots('human', 'agent')
            break
        elif choice == '5':
            game = CardGameWithBots('smart', 'random')
            break
        elif choice == '6':
            game = CardGameWithBots('smart', 'baseline')
            break
        elif choice == '7':
            game = CardGameWithBots('agent', 'baseline')
            break
        else:
            print("Invalid choice. Please enter 1-7.")

    game.play_game()