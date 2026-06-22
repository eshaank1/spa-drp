"""
Data Collection Script: Generate BC Demonstrations

This script shows how to collect 10,000 games of BaselineBot vs BaselineBot
and extract (state, action_mask, action) tuples for behavioral cloning.

In your specific setup, you would integrate this with your environment and bot.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from typing import List, Tuple, Optional
import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pettingzoo_ppo.rl_pettingzoo_env import CardGameVsSmartParallelEnv
from baseline_bot import BaselineBot
from state_inverter import invert_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BCDemoCollector:
    """Collect behavioral cloning demonstrations from bot vs bot games."""
    
    def __init__(self, num_games: int = 10000, seed: int = 42):
        """
        Args:
            num_games: Number of games to simulate
            seed: Random seed for reproducibility
        """
        self.num_games = num_games
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Initialize environment and bot
        # Note: This uses the existing environment from your codebase
        self.env = CardGameVsSmartParallelEnv(seed=seed)
        self.bot = BaselineBot()
        
        self.demonstrations: List[Tuple[np.ndarray, np.ndarray, int]] = []
    
    def _get_valid_actions_mask(self, hand: List[str]) -> np.ndarray:
        """Create action mask from hand."""
        mask = np.zeros(14, dtype=np.float32)
        mask[0] = 1.0  # Pass is always valid
        
        # Map card ranks to action indices (1-13)
        rank_to_idx = {
            "A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
            "7": 7, "8": 8, "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13
        }
        
        for card in hand:
            if card in rank_to_idx:
                mask[rank_to_idx[card]] = 1.0
        
        return mask
    
    def _action_to_card(self, action: int) -> Optional[str]:
        """Convert action index to card rank."""
        if action == 0:
            return None  # Pass
        
        rank_map = {
            1: "A", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6",
            7: "7", 8: "8", 9: "9", 10: "10", 11: "J", 12: "Q", 13: "K"
        }
        
        return rank_map.get(action)
    
    def collect_from_single_game(self) -> Tuple[int, int]:
        """
        Simulate one game and collect demonstrations.
        
        Returns:
            Tuple of (num_demos_collected, winner)
        """
        obs, _ = self.env.reset()
        done = False
        game_winner = None
        demos_this_game = 0
        
        while not done:
            # Determine whose turn it is
            if self.env.current_player == 1:
                # Agent's turn (player 1) - we track this
                # (In full training, this would be replaced by agent decisions,
                # but for demos we use the bot's policy)
                hand = self.env.player1_hand
                mask = self._get_valid_actions_mask(hand)
                
                # Get bot decision (same strategy as player 2)
                p1_score = self.env._score(self.env.p1_played)
                p2_score = self.env._score(self.env.p2_played)
                is_last_round = self.env.current_round == 3
                
                action_str = self.bot.decide_move(
                    hand=hand,
                    player_score=p1_score,
                    opponent_score=p2_score,
                    is_last_round=is_last_round,
                    opponent_just_played=self.env.opponent_just_played,
                    my_rounds_won=self.env.rounds_won[0],
                    opponent_rounds_won=self.env.rounds_won[1],
                )
                
                # Convert to action index
                if action_str == "PASS":
                    action = 0
                else:
                    rank_to_idx = {
                        "A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
                        "7": 7, "8": 8, "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13
                    }
                    action = rank_to_idx.get(action_str, 0)
                
                # Store demonstration (player 1's perspective, no inversion needed)
                self.demonstrations.append((obs.copy(), mask.copy(), action))
                demos_this_game += 1
                
                # Step environment
                obs, _, done, info = self.env.step({"learner": action})
                if done:
                    game_winner = info["learner"]["final_rounds_won"]
            
            else:
                # Player 2's turn (bot's perspective)
                hand = self.env.player2_hand
                mask = self._get_valid_actions_mask(hand)
                
                # Get bot decision
                p2_score = self.env._score(self.env.p2_played)
                p1_score = self.env._score(self.env.p1_played)
                is_last_round = self.env.current_round == 3
                
                action_str = self.bot.decide_move(
                    hand=hand,
                    player_score=p2_score,
                    opponent_score=p1_score,
                    is_last_round=is_last_round,
                    opponent_just_played=self.env.opponent_just_played,
                    my_rounds_won=self.env.rounds_won[1],
                    opponent_rounds_won=self.env.rounds_won[0],
                )
                
                # Convert to action index
                if action_str == "PASS":
                    action = 0
                else:
                    rank_to_idx = {
                        "A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
                        "7": 7, "8": 8, "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13
                    }
                    action = rank_to_idx.get(action_str, 0)
                
                # Get observation from player 2's perspective
                obs_p2 = self.env._get_opponent_observation()
                
                # CRITICAL: Invert to player 1's perspective for training
                obs_p2_torch = torch.tensor(obs_p2, dtype=torch.float32)
                obs_p1_inverted = invert_state(obs_p2_torch).numpy().astype(np.float32)
                
                # Store demonstration (inverted perspective)
                self.demonstrations.append((obs_p1_inverted.copy(), mask.copy(), action))
                demos_this_game += 1
                
                # Step environment
                obs, _, done, info = self.env.step({"learner": 0})  # Dummy action
                if done:
                    game_winner = info["learner"]["final_rounds_won"]
        
        return demos_this_game, game_winner
    
    def collect_demonstrations(self, verbose: bool = True) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        """
        Collect demonstrations from num_games games.
        
        Args:
            verbose: Print progress
        
        Returns:
            List of (observation, action_mask, action) tuples
        """
        self.demonstrations = []
        
        for game_idx in range(self.num_games):
            demos_added, winner = self.collect_from_single_game()
            
            if verbose and (game_idx + 1) % 1000 == 0:
                logger.info(
                    f"Completed {game_idx + 1}/{self.num_games} games | "
                    f"Total demos: {len(self.demonstrations)}"
                )
            
            if verbose and (game_idx + 1) % 50 == 0:
                logger.debug(f"  Game {game_idx + 1}: {demos_added} demos collected")
        
        logger.info(
            f"✓ Collection complete: {len(self.demonstrations)} total demonstrations"
        )
        
        return self.demonstrations
    
    def save_demonstrations(self, filepath: str):
        """Save demonstrations to disk."""
        data = {
            "demonstrations": self.demonstrations,
            "num_games": self.num_games,
            "num_demos": len(self.demonstrations),
        }
        
        import pickle
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved {len(self.demonstrations)} demos to {filepath}")
    
    def load_demonstrations(self, filepath: str):
        """Load demonstrations from disk."""
        import pickle
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        self.demonstrations = data["demonstrations"]
        logger.info(f"Loaded {len(self.demonstrations)} demos from {filepath}")
        
        return self.demonstrations


def main():
    """Main collection script."""
    
    logger.info("=" * 70)
    logger.info("Behavioral Cloning Demo Collection")
    logger.info("=" * 70)
    
    # Configuration
    NUM_GAMES = 10000
    SAVE_PATH = "bc_demonstrations_10k.pkl"
    
    # Create collector
    collector = BCDemoCollector(num_games=NUM_GAMES, seed=42)
    
    # Collect demonstrations
    logger.info(f"Starting collection of {NUM_GAMES} games...")
    demonstrations = collector.collect_demonstrations(verbose=True)
    
    # Save to disk
    collector.save_demonstrations(SAVE_PATH)
    
    # Print statistics
    logger.info("\n" + "=" * 70)
    logger.info("Collection Statistics")
    logger.info("=" * 70)
    logger.info(f"Total games: {NUM_GAMES}")
    logger.info(f"Total demonstrations: {len(demonstrations)}")
    logger.info(f"Avg demos per game: {len(demonstrations) / NUM_GAMES:.1f}")
    
    # Analyze action distribution
    action_counts = np.zeros(14)
    for _, _, action in demonstrations:
        action_counts[action] += 1
    
    logger.info("\nAction distribution:")
    action_names = ["PASS"] + [str(i) for i in range(1, 14)]
    for i, (name, count) in enumerate(zip(action_names, action_counts)):
        pct = 100.0 * count / len(demonstrations)
        logger.info(f"  Action {i:2d} ({name:4s}): {count:7d} ({pct:5.2f}%)")
    
    logger.info("\n✓ Ready for BC pre-training!")
    logger.info(f"  Load demonstrations with:")
    logger.info(f"    from behavioral_cloning import create_behavioral_cloning_pipeline")
    logger.info(f"    with open('{SAVE_PATH}', 'rb') as f:")
    logger.info(f"        import pickle")
    logger.info(f"        data = pickle.load(f)")
    logger.info(f"    demonstrations = data['demonstrations']")
    logger.info(f"    actor, trainer, loaders = create_behavioral_cloning_pipeline(")
    logger.info(f"        demonstrations, batch_size=64")
    logger.info(f"    )")


if __name__ == "__main__":
    main()
