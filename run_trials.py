# Run trials, print raw win rates

import sys
import time
from io import StringIO
from game_with_bots import CardGameWithBots

def run_trials(num_trials=1000000):
    """Run multiple games between SmartBot and RandomBot (silent mode)"""
    smart_wins = 0
    random_wins = 0
    
    print(f"Running {num_trials:,} trials of SmartBot vs RandomBot...\n")
    start_time = time.time()
    
    for trial in range(1, num_trials + 1):
        # Suppress game output
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        game = CardGameWithBots('smart', 'random')
        result = game.play_game()
        
        # Restore stdout
        sys.stdout = old_stdout
        
        if result[0] > result[1]:
            smart_wins += 1
        else:
            random_wins += 1
        
        # Progress update every 10,000 trials
        if trial % 10000 == 0:
            elapsed = time.time() - start_time
            print(f"Progress: {trial:,} / {num_trials:,} trials ({trial/num_trials*100:.1f}%) - {elapsed:.1f}s")
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS - {num_trials:,} TRIALS")
    print(f"{'='*60}")
    print(f"SmartBot wins:  {smart_wins:,}")
    print(f"RandomBot wins: {random_wins:,}")
    print(f"SmartBot win rate:  {smart_wins/num_trials*100:.2f}%")
    print(f"RandomBot win rate: {random_wins/num_trials*100:.2f}%")
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_trials(1000000)
