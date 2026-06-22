"""
Play against the trained BC+PPO-fD agent.

Usage:
    python play_vs_trained_agent.py --model agent_final.pt
"""

import argparse
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bc_ppo_fd.ppo_auxiliary_loss import ActorCriticNetwork


class PlayVsTrainedAgent:
    """Interactive game where human plays against trained agent."""
    
    RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    RANK_VALUES = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                   '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13}

    def __init__(self, model_path: str, device: str = "cpu"):
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
        
        print(f"✓ Loaded model from {model_path}")
        
        # Initialize game state
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

    def display_game_state(self, player_num: int):
        """Display current state for a player"""
        if player_num == 1:
            hand = self.player1_hand
            deck_size = len(self.player1_deck)
        else:
            hand = self.player2_hand
            deck_size = len(self.player2_deck)

        print(f"\n--- Player {player_num}'s Turn ---")

        # Only show hand if it's human player (player 1)
        if player_num == 1:
            # Display hand in sorted order by card value
            sorted_hand = sorted(hand, key=lambda card: self.RANK_VALUES[card])
            print(f"Hand: {', '.join(sorted_hand)}")
        else:
            print(f"Hand: [Agent]")

        print(f"Cards remaining in deck: {deck_size}")
        print(f"Round {self.current_round} | Wins: {self.rounds_won[player_num-1]}")

    def display_round_state(self, p1_played: List[str], p2_played: List[str]):
        """Display cards played this round"""
        p1_score = sum(self.RANK_VALUES[card] for card in p1_played)
        p2_score = sum(self.RANK_VALUES[card] for card in p2_played)

        print(f"You played: {p1_played if p1_played else 'None'} (Score: {p1_score})")
        print(f"Agent played: {p2_played if p2_played else 'None'} (Score: {p2_score})")

    def _get_agent_action(self, hand: List[str], p1_played: List[str], p2_played: List[str], opponent_just_played: bool, opponent_has_passed: bool) -> str:
        """Get action from trained agent."""
        if not hand:
            return 'PASS'
        
        # Get observation from agent's perspective (player 2)
        obs = self._get_obs_from_perspective(hand, p1_played, p2_played, opponent_just_played, opponent_has_passed)
        mask = self._get_valid_actions_mask(hand)
        
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            logits, _ = self.actor_critic(obs_t.unsqueeze(0), mask_t.unsqueeze(0))
            logits = logits.squeeze(0)
            action = torch.argmax(logits).item()
        
        if action == 0:
            return 'PASS'
        else:
            rank = self.RANKS[action - 1]
            if rank in hand:
                return rank
            else:
                # Invalid action, pass instead
                return 'PASS'

    def _get_obs_from_perspective(self, my_hand: List[str], p1_played: List[str], p2_played: List[str], opponent_just_played: bool, opponent_has_passed: bool = False) -> np.ndarray:
        """Get 50-dim observation vector from agent's perspective (player 2)."""
        obs = np.zeros(50, dtype=np.float32)
        
        # Hand (13)
        for rank in my_hand:
            obs[self.RANKS.index(rank)] = 1.0
        
        # My played cards (13) - agent is player 2, so p2_played
        for rank in p2_played:
            obs[13 + self.RANKS.index(rank)] = 1.0
        
        # Opponent's played cards (13) - opponent is player 1, so p1_played
        for rank in p1_played:
            obs[26 + self.RANKS.index(rank)] = 1.0
        
        # Metadata (11)
        metadata_start = 39
        obs[metadata_start + 0] = (self.current_round - 1) / 3.0  # Round (0-indexed)
        obs[metadata_start + 1] = self.rounds_won[1] / 2.0  # My rounds won (agent is player 2)
        obs[metadata_start + 2] = self.rounds_won[0] / 2.0  # Opponent rounds won (player 1)
        obs[metadata_start + 3] = 0.0  # Current player is 1 (always 0 for agent perspective)
        obs[metadata_start + 4] = 1.0  # Current player is 2 (always 1 for agent perspective)
        obs[metadata_start + 5] = 1.0 if self.first_player == 1 else 0.0
        obs[metadata_start + 6] = 1.0 if self.first_player == 2 else 0.0
        obs[metadata_start + 7] = 1.0 if opponent_has_passed else 0.0  # Player 1 passed (opponent)
        obs[metadata_start + 8] = 0.0  # Player 2 passed (agent) - not relevant in this context
        obs[metadata_start + 9] = len(my_hand) / 13.0  # Agent hand size
        obs[metadata_start + 10] = len(p1_played) / 13.0  # Opponent played count
        
        return obs

    def _get_valid_actions_mask(self, hand: List[str]) -> np.ndarray:
        """Create action mask from hand."""
        mask = np.zeros(14, dtype=np.float32)
        mask[0] = 1.0  # Pass always valid
        
        for card in hand:
            if card in self.RANKS:
                mask[self.RANKS.index(card) + 1] = 1.0
        
        return mask

    def _draw_cards(self):
        """Draw 3 new cards for each player from their deck"""
        for _ in range(3):
            if self.player1_deck:
                self.player1_hand.append(self.player1_deck.pop(0))
            if self.player2_deck:
                self.player2_hand.append(self.player2_deck.pop(0))

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

                # Human player's move
                if not self.player1_hand:
                    print("You have no cards - forced pass")
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
                            print("You pass")
                            passed_players.add(1)
                            opponent_just_played = False
                            if len(passed_players) == 2:
                                break
                            break
                        elif choice in self.player1_hand:
                            p1_played.append(choice)
                            self.player1_hand.remove(choice)
                            print(f"You played: {choice}")
                            opponent_just_played = True
                            break
                        else:
                            print("Invalid choice. Please play a card from your hand or pass.")

                current_player = 2

            else:  # Agent (Player 2)
                self.display_game_state(2)
                self.display_round_state(p1_played, p2_played)

                # Agent's move
                if not self.player2_hand:
                    print("Agent has no cards - forced pass")
                    passed_players.add(2)
                    opponent_just_played = False
                    if len(passed_players) == 2:
                        break
                    current_player = 1
                    continue
                else:
                    choice = self._get_agent_action(self.player2_hand, p1_played, p2_played, opponent_just_played, 1 in passed_players)

                    if choice == 'PASS':
                        print("Agent passes")
                        passed_players.add(2)
                        opponent_just_played = False
                        if len(passed_players) == 2:
                            break
                    else:
                        p2_played.append(choice)
                        self.player2_hand.remove(choice)
                        print(f"Agent played: {choice}")
                        opponent_just_played = True

                current_player = 1

        # Calculate scores
        p1_score = sum(self.RANK_VALUES[card] for card in p1_played)
        p2_score = sum(self.RANK_VALUES[card] for card in p2_played)

        print(f"\n--- Round {self.current_round} Results ---")
        print(f"You total: {p1_score}")
        print(f"Agent total: {p2_score}")

        if p1_score > p2_score:
            print("YOU WIN THIS ROUND!")
            self.rounds_won[0] += 1
            self.first_player = 1
        elif p2_score > p1_score:
            print("AGENT WINS THIS ROUND!")
            self.rounds_won[1] += 1
            self.first_player = 2
        else:
            print("TIED THIS ROUND! Both get a point.")
            self.rounds_won[0] += 1
            self.rounds_won[1] += 1

        print(f"Score: You: {self.rounds_won[0]} | Agent: {self.rounds_won[1]}")

        # Check if game is over
        if self.rounds_won[0] >= 2 or self.rounds_won[1] >= 2:
            return

        # Draw 3 new cards for next round
        if self.current_round < 3:
            self._draw_cards()

        self.current_round += 1

    def play_game(self):
        """Play the complete game"""
        print("\n" + "="*50)
        print("CARD STRATEGY GAME: YOU vs TRAINED AGENT")
        print("="*50)
        print("First to win 2 rounds wins!")
        print("="*50)

        while sum(w >= 2 for w in self.rounds_won) == 0:
            self.play_round()

        print(f"\n{'='*50}")
        print("GAME OVER!")
        print(f"{'='*50}")
        if self.rounds_won[0] > self.rounds_won[1]:
            print(f"🎉 YOU WIN THE GAME!")
        else:
            print(f"💔 AGENT WINS THE GAME!")
        print(f"Final Score: You: {self.rounds_won[0]} | Agent: {self.rounds_won[1]}")

        return self.rounds_won


def main():
    parser = argparse.ArgumentParser(description="Play against trained BC+PPO-fD agent")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model (default: agent_final.pt in same folder)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    
    args = parser.parse_args()
    
    # If no model specified, use agent_final.pt in the same folder
    if args.model is None:
        script_dir = Path(__file__).resolve().parent
        args.model = str(script_dir / "agent_final.pt")
    
    game = PlayVsTrainedAgent(args.model, device=args.device)
    
    while True:
        game.play_game()
        if input("\nPlay another game? (y/n): ").lower() != 'y':
            print("Thanks for playing!")
            break
        # Reset for new game
        game = PlayVsTrainedAgent(args.model, device=args.device)


if __name__ == "__main__":
    main()
