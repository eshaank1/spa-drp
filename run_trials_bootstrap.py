# Run trials, win rates + statistical estimations and + more detailed analysis tracking

import sys
import time
import numpy as np
from io import StringIO
from game_with_bots import CardGameWithBots

def run_trials_bootstrap(num_trials=1000000, group_size=10000, ci=95):
    """Run trials and bootstrap resample to get confidence intervals"""
    results = []  # Store 1 for SmartBot win, 0 for RandomBot win
    first_player_results = []  # Store 1 if whoever started first won, else 0
    smart_win_first_player_flags = []  # For SmartBot wins: 1 if SmartBot started first
    random_win_first_player_flags = []  # For RandomBot wins: 1 if RandomBot started first
    
    print(f"Running {num_trials:,} trials of SmartBot vs RandomBot...\n")
    start_time = time.time()
    
    for trial in range(1, num_trials + 1):
        # Suppress game output
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        game = CardGameWithBots('smart', 'random')
        starting_player = game.first_player
        result = game.play_game()
        
        # Restore stdout
        sys.stdout = old_stdout
        
        # Store 1 if SmartBot wins, 0 if RandomBot wins
        results.append(1 if result[0] > result[1] else 0)

        # Store 1 if starting player wins the game, else 0
        winner = 1 if result[0] > result[1] else 2
        first_player_results.append(1 if winner == starting_player else 0)

        # Conditional first-player flags within each bot's wins
        if winner == 1:
            smart_win_first_player_flags.append(1 if starting_player == 1 else 0)
        else:
            random_win_first_player_flags.append(1 if starting_player == 2 else 0)
        
        # Progress update every 10,000 trials
        if trial % 10000 == 0:
            elapsed = time.time() - start_time
            print(f"Progress: {trial:,} / {num_trials:,} trials ({trial/num_trials*100:.1f}%) - {elapsed:.1f}s")
    
    elapsed = time.time() - start_time
    results = np.array(results)
    first_player_results = np.array(first_player_results)
    smart_win_first_player_flags = np.array(smart_win_first_player_flags)
    random_win_first_player_flags = np.array(random_win_first_player_flags)
    
    # Bootstrap resampling
    num_groups = num_trials // group_size
    bootstrap_samples = np.random.choice(results, size=(1000, num_trials), replace=True)
    first_player_bootstrap_samples = np.random.choice(first_player_results, size=(1000, num_trials), replace=True)

    smart_conditional_bootstrap_samples = None
    random_conditional_bootstrap_samples = None

    if smart_win_first_player_flags.size > 0:
        smart_conditional_bootstrap_samples = np.random.choice(
            smart_win_first_player_flags,
            size=(1000, smart_win_first_player_flags.size),
            replace=True
        )

    if random_win_first_player_flags.size > 0:
        random_conditional_bootstrap_samples = np.random.choice(
            random_win_first_player_flags,
            size=(1000, random_win_first_player_flags.size),
            replace=True
        )
    
    # Calculate win rates for each bootstrap sample (grouped by group_size)
    smart_win_rates = []
    random_win_rates = []
    first_player_win_rates = []
    smart_first_player_given_win_rates = []
    random_first_player_given_win_rates = []
    
    for sample in bootstrap_samples:
        # Split into groups and calculate mean for each group
        groups = sample[:num_groups * group_size].reshape(num_groups, group_size)
        group_means = groups.mean(axis=1)
        smart_win_rates.append(group_means.mean())
        random_win_rates.append(1 - group_means.mean())

    for sample in first_player_bootstrap_samples:
        groups = sample[:num_groups * group_size].reshape(num_groups, group_size)
        group_means = groups.mean(axis=1)
        first_player_win_rates.append(group_means.mean())

    if smart_conditional_bootstrap_samples is not None:
        for sample in smart_conditional_bootstrap_samples:
            smart_first_player_given_win_rates.append(sample.mean())

    if random_conditional_bootstrap_samples is not None:
        for sample in random_conditional_bootstrap_samples:
            random_first_player_given_win_rates.append(sample.mean())
    
    smart_win_rates = np.array(smart_win_rates) * 100
    random_win_rates = np.array(random_win_rates) * 100
    first_player_win_rates = np.array(first_player_win_rates) * 100
    smart_first_player_given_win_rates = np.array(smart_first_player_given_win_rates) * 100
    random_first_player_given_win_rates = np.array(random_first_player_given_win_rates) * 100
    
    # Calculate confidence intervals
    alpha = (100 - ci) / 2
    smart_lower = np.percentile(smart_win_rates, alpha)
    smart_upper = np.percentile(smart_win_rates, 100 - alpha)
    random_lower = np.percentile(random_win_rates, alpha)
    random_upper = np.percentile(random_win_rates, 100 - alpha)
    first_player_lower = np.percentile(first_player_win_rates, alpha)
    first_player_upper = np.percentile(first_player_win_rates, 100 - alpha)

    smart_conditional_lower = None
    smart_conditional_upper = None
    random_conditional_lower = None
    random_conditional_upper = None

    if smart_first_player_given_win_rates.size > 0:
        smart_conditional_lower = np.percentile(smart_first_player_given_win_rates, alpha)
        smart_conditional_upper = np.percentile(smart_first_player_given_win_rates, 100 - alpha)

    if random_first_player_given_win_rates.size > 0:
        random_conditional_lower = np.percentile(random_first_player_given_win_rates, alpha)
        random_conditional_upper = np.percentile(random_first_player_given_win_rates, 100 - alpha)
    
    print(f"\n{'='*70}")
    print(f"BOOTSTRAP RESULTS - {num_trials:,} TRIALS ({ci}% Confidence Interval)")
    print(f"{'='*70}")
    print(f"\nSmartBot Win Rate:")
    print(f"  Point estimate:  {results.mean()*100:.2f}%")
    print(f"  {ci}% CI: [{smart_lower:.2f}%, {smart_upper:.2f}%]")
    print(f"\nRandomBot Win Rate:")
    print(f"  Point estimate:  {(1-results.mean())*100:.2f}%")
    print(f"  {ci}% CI: [{random_lower:.2f}%, {random_upper:.2f}%]")
    print(f"\nFirst-Player Win Rate:")
    print(f"  Point estimate:  {first_player_results.mean()*100:.2f}%")
    print(f"  {ci}% CI: [{first_player_lower:.2f}%, {first_player_upper:.2f}%]")

    print(f"\nOf SmartBot's Wins, % Where SmartBot Was First Player:")
    if smart_win_first_player_flags.size > 0:
        print(f"  Point estimate:  {smart_win_first_player_flags.mean()*100:.2f}%")
        print(f"  {ci}% CI: [{smart_conditional_lower:.2f}%, {smart_conditional_upper:.2f}%]")
    else:
        print("  Point estimate:  N/A (no SmartBot wins)")
        print(f"  {ci}% CI: N/A")

    print(f"\nOf RandomBot's Wins, % Where RandomBot Was First Player:")
    if random_win_first_player_flags.size > 0:
        print(f"  Point estimate:  {random_win_first_player_flags.mean()*100:.2f}%")
        print(f"  {ci}% CI: [{random_conditional_lower:.2f}%, {random_conditional_upper:.2f}%]")
    else:
        print("  Point estimate:  N/A (no RandomBot wins)")
        print(f"  {ci}% CI: N/A")

    print(f"\nTime elapsed: {elapsed:.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_trials_bootstrap(1000000, group_size=10000, ci=95)
