# spa-drp

*Reinforcement Learning for Optimal Game-Theoretic Strategies*

**Directed Reading & Research Project** | University of Washington - Statistics Department

**Author:** Eshaan Kumar  
**Mentor:** Ph.D. Candidate Alex Kokot  
**Term:** Spring 2026

---

## Overview

This project develops reinforcement learning agents that learn to play a two-player sequential card game through iterative ladder-style training. Using Proximal Policy Optimization (PPO) from Stable-Baselines3 and PettingZoo's multi-agent framework, we train agents that:

- **Generalize effectively** against diverse opponents by training in both player roles (P1 and P2)
- **Avoid overfitting** through entropy regularization and action masking
- **Improve iteratively** via ladder training where each generation competes against previous generations
- **Exhibit strategic diversity** by sampling stochastically from learned policies rather than deterministic play
- **Handle invalid actions gracefully** by automatically correcting impossible moves to valid alternatives

The system includes comprehensive evaluation metrics (win rates, confidence intervals), automatic model promotion, and statistical tracking across generations.

---

## Project Structure

```
.
├── card_game.py                    # Core card game logic and rules
├── smart_bot.py                    # Heuristic-based baseline strategy
├── random_bot.py                   # Random play baseline
├── play_vs_ppo.py                  # Interactive human vs bot gameplay
├── evaluate_ppo_vs_ppo.py          # Bot vs bot evaluation framework
├── run_ladder_training.py          # Main ladder training orchestrator
├── run_continue_training.py        # Continued training (legacy)
│
├── pettingzoo_ppo/
│   ├── rl_pettingzoo_env.py       # PettingZoo ParallelEnv with action masking
│   ├── train_ladder_challenger.py  # Single generation trainer
│   ├── train_ppo_vs_smart.py      # Direct SmartBot training
│   └── models/                     # Trained PPO models
│
├── models/
│   ├── ladder/
│   │   ├── champions/              # Promoted generation champions
│   │   ├── gen_N/                  # Generation-specific artifacts
│   │   └── ppo_vs_smart_final.zip  # Original baseline PPO model
│   └── continued/                  # Continued training checkpoints
│
├── NOTES.md                        # Quick reference for all commands
├── ladder_notes.md                 # Detailed ladder training documentation
└── ladder_training_stats.csv       # Training statistics log
```

---

## Key Features

### Ladder Training System
- **Automatic progression:** Each generation competes against previous generations and the original PPO baseline
- **Bidirectional evaluation:** Tests both player 1 and player 2 perspectives to avoid player 1 advantage bias
- **Automatic promotion:** All generations advance to maintain iterative improvement without threshold bottlenecks
- **Disk cleanup:** Automatically maintains only the 5 most recent models to manage storage

### Generalization Improvements
- **Role diversity:** 50/50 split between P1 and P2 training ensures agents learn both offensive and defensive strategies
- **Entropy regularization:** Configurable entropy coefficient (default 0.05) encourages exploration of diverse strategies
- **Stochastic play:** Interactive mode samples from learned policies rather than deterministic max-probability moves
- **Action masking:** Both training and gameplay prevent invalid actions (e.g., playing cards not in hand) by sampling from valid alternatives

### Evaluation & Monitoring
- **Multi-opponent evaluation:** Assesses performance against 4 different strategies (RandomBot, SmartBot, Original PPO, Previous Gen)
- **Wilson confidence intervals:** Statistical rigor for win rate estimation
- **CSV logging:** Automatic tracking of metrics across generations
- **Bidirectional scoring:** Aggregates results from both player perspectives for comprehensive assessment

---

## Technical Stack

- **RL Framework:** Stable-Baselines3 (PPO algorithm)
- **Multi-Agent Framework:** PettingZoo
- **Vectorization:** SuperSuit (environment parallelization)
- **Evaluation:** Gymnasium (metric tracking)
- **Language:** Python 3.9+

---

## Results

Training statistics are logged to `ladder_training_stats.csv` with per-generation metrics:
- Win rates vs each opponent (RandomBot, SmartBot, Original PPO, Previous Gen)
- Bidirectional aggregation (both player perspectives)
- Confidence intervals for statistical validation

View results:
```bash
cat ladder_training_stats.csv
```

---

## References & Resources

- See [NOTES.md](NOTES.md) for complete command reference
- See [GAME_RULES.md](GAME_RULES.md) for card game mechanics