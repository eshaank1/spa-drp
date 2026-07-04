"""
Data Collection Script: Generate BC Demonstrations

Collects games of BaselineBot vs BaselineBot and extracts
(state, action_mask, action) tuples for behavioral cloning.

The learning agent is always Player 1 in the environment, and the environment
now plays Player 2 with BaselineBot (opponent="baseline"). We drive Player 1
with the *same* BaselineBot and record only Player 1's decisions, so every
demonstration is a genuine BaselineBot move captured in the exact Player-1
observation format the agent trains and is evaluated in. No state inversion is
required (and the previously-broken Player-2 branch is gone).
"""

import sys
from pathlib import Path
import numpy as np
from typing import List, Tuple, Optional
import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pettingzoo_ppo.rl_pettingzoo_env import CardGameVsSmartParallelEnv
from baseline_bot import BaselineBot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BCDemoCollector:
    """Collect behavioral cloning demonstrations from BaselineBot vs BaselineBot."""

    def __init__(self, num_games: int = 10000, seed: int = 42):
        """
        Args:
            num_games: Number of games to simulate
            seed: Random seed for reproducibility
        """
        self.num_games = num_games
        self.seed = seed

        # Player 2 is BaselineBot (played internally by the env); we drive
        # Player 1 with BaselineBot too -> true BaselineBot-vs-BaselineBot games.
        self.env = CardGameVsSmartParallelEnv(seed=seed, opponent="baseline")
        self.bot = BaselineBot()

        self.demonstrations: List[Tuple[np.ndarray, np.ndarray, int]] = []

    def _get_valid_actions_mask(self, hand: List[str]) -> np.ndarray:
        """Create action mask from hand (Pass + cards in hand)."""
        mask = np.zeros(14, dtype=np.float32)
        mask[0] = 1.0  # Pass is always valid
        for card_idx, rank in enumerate(self.env.RANKS, start=1):
            if rank in hand:
                mask[card_idx] = 1.0
        return mask

    def _baseline_p1_action(self) -> int:
        """BaselineBot's action for Player 1 in the current env state (0..13)."""
        # BaselineBot returns 0 (pass) or the card value 1..13, which equals our
        # action index directly.
        return int(
            self.bot.decide_move(
                hand=self.env.player1_hand,
                my_score=self.env._score(self.env.p1_played),
                opp_score=self.env._score(self.env.p2_played),
                round_num=self.env.current_round,
                my_wins=self.env.rounds_won[0],
                opp_wins=self.env.rounds_won[1],
                opponent_has_passed=(2 in self.env.passed_players),
            )
        )

    def collect_from_single_game(self) -> Tuple[int, Optional[tuple]]:
        """
        Simulate one game and collect Player-1 BaselineBot demonstrations.

        Returns:
            Tuple of (num_demos_collected, final_rounds_won)
        """
        obs_dict, _ = self.env.reset()
        obs = obs_dict["learner"]
        done = False
        game_winner = None
        demos_this_game = 0

        while not done:
            # The env auto-plays Player 2; control returns here on Player 1's turn.
            mask = self._get_valid_actions_mask(self.env.player1_hand)
            action = self._baseline_p1_action()

            # Record the Player-1 decision in the agent's native observation format.
            self.demonstrations.append((obs.copy(), mask.copy(), action))
            demos_this_game += 1

            obs_dict, _, done_dict, _, info = self.env.step({"learner": action})
            done = done_dict["learner"]
            if obs_dict and "learner" in obs_dict:
                obs = obs_dict["learner"]
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


def load_or_collect_demonstrations(
    filepath: str,
    num_games: int = 5000,
    seed: int = 42,
    verbose: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """Load demonstrations from ``filepath`` if it exists, else collect & save.

    Returns the list of (state, action_mask, action) tuples.
    """
    path = Path(filepath)
    collector = BCDemoCollector(num_games=num_games, seed=seed)
    if path.exists():
        demos = collector.load_demonstrations(str(path))
        if demos:
            return demos
    demos = collector.collect_demonstrations(verbose=verbose)
    collector.save_demonstrations(str(path))
    return demos


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
