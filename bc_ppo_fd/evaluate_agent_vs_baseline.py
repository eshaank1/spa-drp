"""
Evaluate trained agent against BaselineBot.

Usage:
    python evaluate_agent_vs_baseline.py --model agent_final.pt --num-games 50
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


class EvaluatorBot:
    """Evaluates trained agent against BaselineBot."""
    
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
        
        print(f"✓ Loaded model from {model_path}")
        
        # Player 2 is played by BaselineBot inside the env (pluggable opponent),
        # so the agent is genuinely evaluated against BaselineBot.
        self.env = CardGameVsSmartParallelEnv(opponent="baseline")
    
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
    
    def evaluate(self, num_games: int = 50) -> Dict:
        """Evaluate agent win rate against BaselineBot."""
        wins = 0
        total_reward = 0.0
        
        self.actor_critic.eval()
        
        with torch.no_grad():
            for game_num in range(num_games):
                obs_dict, _ = self.env.reset()
                obs = obs_dict["learner"] if isinstance(obs_dict, dict) else obs_dict
                done = False
                episode_reward = 0.0
                
                while not done:
                    if self.env.current_player == 1:
                        # Agent's turn (Player 1)
                        hand = self.env.player1_hand
                        mask = self._get_valid_actions_mask(hand)
                        
                        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
                        mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device)
                        
                        logits, _ = self.actor_critic(obs_t.unsqueeze(0), mask_t.unsqueeze(0))
                        action = torch.argmax(logits).item()
                        
                        obs_dict, reward_dict, done_dict, trunc_dict, info = self.env.step({"learner": action})
                        if obs_dict and "learner" in obs_dict:
                            obs = obs_dict["learner"]
                        episode_reward += reward_dict.get("learner", 0.0) if isinstance(reward_dict, dict) else reward_dict
                        done = done_dict.get("learner", False) if isinstance(done_dict, dict) else done_dict
                        
                        if done and info["learner"]["final_rounds_won"][0] > info["learner"]["final_rounds_won"][1]:
                            wins += 1
                    
                    else:
                        # Player 2 (BaselineBot) is played automatically by the
                        # env; just advance with a no-op learner action.
                        obs_dict, _, done_dict, trunc_dict, info = self.env.step({"learner": 0})
                        if obs_dict and "learner" in obs_dict:
                            obs = obs_dict["learner"]
                        done = done_dict.get("learner", False) if isinstance(done_dict, dict) else done_dict
                
                total_reward += episode_reward
                
                # Progress indicator
                if (game_num + 1) % max(1, num_games // 10) == 0:
                    print(f"  Completed {game_num + 1}/{num_games} games...")
        
        return {
            "win_rate": wins / num_games,
            "avg_episode_reward": total_reward / num_games,
            "wins": wins,
            "total_games": num_games,
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agent against BaselineBot")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model (default: agent_final.pt in same folder)")
    parser.add_argument("--num-games", type=int, default=50, help="Number of evaluation games")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or cuda, default: auto)")
    
    args = parser.parse_args()
    
    # If no model specified, use agent_final.pt in the same folder
    if args.model is None:
        script_dir = Path(__file__).resolve().parent
        args.model = str(script_dir / "agent_final.pt")
    
    # Auto-detect device if not specified
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Device: {args.device}")
    print(f"Evaluating for {args.num_games} games...\n")
    
    evaluator = EvaluatorBot(args.model, device=args.device)
    results = evaluator.evaluate(num_games=args.num_games)
    
    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Win Rate: {results['win_rate']:.1%} ({results['wins']}/{results['total_games']} games)")
    print(f"Avg Episode Reward: {results['avg_episode_reward']:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
