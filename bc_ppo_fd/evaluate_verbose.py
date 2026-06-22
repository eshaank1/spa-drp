"""
Evaluate trained agent against BaselineBot with move-by-move output.

Usage:
    python evaluate_verbose.py --model agent_final.pt --num-games 5
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pettingzoo_ppo.rl_pettingzoo_env import CardGameVsSmartParallelEnv
from bc_ppo_fd.ppo_auxiliary_loss import ActorCriticNetwork
from baseline_bot import BaselineBot


class VerboseEvaluator:
    """Evaluates trained agent with detailed move-by-move output."""
    
    RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    RANK_VALUES = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                   '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13}
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42,
    ):
        self.device = torch.device(device)
        self.seed = seed
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
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
        
        print(f"✓ Loaded model from {model_path}\n")
        
        # Initialize environment and bot
        self.env = CardGameVsSmartParallelEnv()
        self.bot = BaselineBot()
    
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
    
    def _action_to_str(self, action: int) -> str:
        """Convert action index to card name."""
        if action == 0:
            return "PASS"
        return self.RANKS[action - 1]
    
    def _score_cards(self, cards):
        """Calculate score from list of cards."""
        return sum(self.RANK_VALUES.get(card, 0) for card in cards)
    
    def play_single_game(self, game_num: int) -> bool:
        """Play a single game with verbose output. Returns True if agent won."""
        obs_dict, _ = self.env.reset()
        obs = obs_dict["learner"] if isinstance(obs_dict, dict) else obs_dict
        done = False
        agent_won = False
        last_round = 0

        print(f"\n{'='*60}")
        print(f"GAME {game_num}")
        print(f"{'='*60}")

        # If bot went first, its opening move happened inside reset().
        # Display it now before the main loop (step() will clear the log).
        if self.env._action_log:
            print(f"\n--- ROUND 1 ---")
            print(f"Score: Agent 0 | Bot 0")
            last_round = 1
            for entry in self.env._action_log:
                card = entry['card']
                card_str = card if card is not None else "PASS"
                hand_str = ", ".join(sorted(entry['hand_before'], key=lambda c: self.RANK_VALUES[c]))
                print(f"  [Bot]   plays {card_str:4} | hand: {hand_str:40} | scores: A={entry['p1_score']:2} B={entry['p2_score']:2}")

        while not done:
            round_num = self.env.current_round

            # Show round header when round changes
            if round_num != last_round:
                print(f"\n--- ROUND {round_num} ---")
                print(f"Score: Agent {self.env.rounds_won[0]} | Bot {self.env.rounds_won[1]}")
                last_round = round_num
            
            if self.env.current_player == 1:
                # Agent's turn
                hand = self.env.player1_hand
                mask = self._get_valid_actions_mask(hand)
                
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
                mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device)
                
                with torch.no_grad():
                    logits, _ = self.actor_critic(obs_t.unsqueeze(0), mask_t.unsqueeze(0))
                    action = torch.argmax(logits).item()
                
                action_str = self._action_to_str(action)
                agent_score = self._score_cards(self.env.p1_played)
                bot_score = self._score_cards(self.env.p2_played)
                
                hand_str = ", ".join(sorted(hand, key=lambda c: self.RANK_VALUES[c]))
                print(f"  [Agent] plays {action_str:4} | hand: {hand_str:40} | scores: A={agent_score:2} B={bot_score:2}")
                
                obs_dict, reward_dict, done_dict, trunc_dict, info = self.env.step({"learner": action})
                if obs_dict and "learner" in obs_dict:
                    obs = obs_dict["learner"]
                done = done_dict.get("learner", False) if isinstance(done_dict, dict) else done_dict

                if done and info["learner"]["final_rounds_won"][0] > info["learner"]["final_rounds_won"][1]:
                    agent_won = True

                # Display each bot action from the log, inserting round headers when the round changes
                for entry in self.env._action_log:
                    entry_round = entry['round']
                    if entry_round != last_round:
                        rw = entry['rounds_won']
                        print(f"\n--- ROUND {entry_round} ---")
                        print(f"Score: Agent {rw[0]} | Bot {rw[1]}")
                        last_round = entry_round
                    card = entry['card']
                    card_str = card if card is not None else "PASS"
                    hand_str = ", ".join(sorted(entry['hand_before'], key=lambda c: self.RANK_VALUES[c]))
                    print(f"  [Bot]   plays {card_str:4} | hand: {hand_str:40} | scores: A={entry['p1_score']:2} B={entry['p2_score']:2}")
        
        # Show final result
        print(f"\n{'─'*60}")
        agent_final_score = self.env.rounds_won[0]
        bot_final_score = self.env.rounds_won[1]
        winner = "Agent" if agent_won else "Bot"
        print(f"Final Score - Agent: {agent_final_score} | Bot: {bot_final_score}")
        print(f"WINNER: {winner}")
        print(f"{'='*60}")
        
        return agent_won
    
    def evaluate(self, num_games: int = 5) -> Dict:
        """Evaluate agent with verbose output."""
        wins = 0
        
        self.actor_critic.eval()
        
        with torch.no_grad():
            for game_num in range(1, num_games + 1):
                if self.play_single_game(game_num):
                    wins += 1
        
        return {
            "win_rate": wins / num_games,
            "wins": wins,
            "total_games": num_games,
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agent with verbose move-by-move output")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model")
    parser.add_argument("--num-games", type=int, default=5, help="Number of games to play (default: 5)")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or cuda)")
    
    args = parser.parse_args()
    
    # If no model specified, use agent_final.pt in the same folder
    if args.model is None:
        script_dir = Path(__file__).resolve().parent
        args.model = str(script_dir / "agent_final.pt")
    
    # Auto-detect device if not specified
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Device: {args.device}")
    print(f"Playing {args.num_games} games with verbose output...\n")
    
    evaluator = VerboseEvaluator(args.model, device=args.device)
    results = evaluator.evaluate(num_games=args.num_games)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Win Rate: {results['win_rate']:.1%} ({results['wins']}/{results['total_games']} games)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
