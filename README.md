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

## Play against the trained agent

Try beating the agent yourself, right in your browser — no setup required:

**[Play now](https://eshaank1.github.io/spa-drp/)**

Pick your opponent from the dropdown: the trained BC+PPO-fD agent (`bc_ppo_fd/agent_final.pt`), or any of the three heuristic bots — BaselineBot, SmartBot, RandomBot — then play a full best-of-3 game. Everything runs entirely client-side in JavaScript, no backend:

- `docs/agent.js` reimplements the trained network's forward pass (weights exported once via `scripts/export_agent_weights.py`)
- `docs/bots.js` ports `baseline_bot.py` / `smart_bot.py` / `random_bot.py`'s decision logic
- `docs/game.js` is the game-rules engine; `docs/app.js` wires it all to the page

`tests/web/` has the JS test suite (`node --test tests/web/*.test.js`), including a numerical-fidelity check of the ported network against the real PyTorch model.

---

## BC + PPO-fD Training Pipeline

The agent behind the web demo (`bc_ppo_fd/agent_final.pt`) is trained by a separate, newer pipeline from the ladder system described above — Behavioral Cloning warm-start followed by PPO from Demonstrations (PPO-fD):

1. **Phase 1 — Behavioral Cloning warm-start.** The actual PPO actor-critic network (not a throwaway model) is pre-trained by supervised learning on real BaselineBot-vs-BaselineBot demonstration games (`bc_ppo_fd/collect_bc_demonstrations.py`), so PPO starts from a competent policy instead of random weights.
2. **Phase 2 — PPO-fD online training.** The warm-started network is then refined with PPO against a rotating opponent pool (BaselineBot, SmartBot, and frozen self-play snapshots of earlier checkpoints), while an auxiliary imitation (cross-entropy) loss against the offline demonstration buffer keeps every update anchored to sound fundamentals — this is the "fD" (from Demonstrations) part of PPO-fD.
3. **Decaying the imitation anchor.** Early runs kept the auxiliary loss weight (`bc_lambda`) fixed for the whole run, which permanently pinned the policy to a BaselineBot clone and prevented PPO from ever improving past it (win rate plateaued around 50-54%, a coin flip, no matter how many more steps were added). The fix: `bc_lambda` now decays linearly (0.2 → 0.02) over training so PPO is increasingly free to diverge from the BC anchor once it has a decent starting point.

**Current result:** a consistent, reproducible edge over both heuristic bots — **56-63% win rate vs BaselineBot and vs SmartBot**, measured over 500 games per opponent across 3 random seeds (deterministic policy). The deployed web demo samples stochastically from the same policy, matching how it's evaluated interactively via `game_with_bots.py`.

```bash
# Train from scratch (BC warm-start + PPO-fD)
python3 bc_ppo_fd/train_bc_ppo_fde.py --num-bc-games 5000 --num-ppo-steps 4000 \
--bc-lambda-start 0.2 --bc-lambda-end 0.02 --entropy-coeff 0.02

# Evaluate vs BaselineBot / SmartBot
python3 bc_ppo_fd/evaluate_agent_vs_baseline.py --num-games 200

# Play it yourself in the terminal, or in the browser (see above)
python3 bc_ppo_fd/play_vs_trained_agent.py
```

See `bc_ppo_fd/COMMANDS.md` and `bc_ppo_fd/BC_PPO_FD_GUIDE.md` for the full command reference.

---

## Project Structure

```
.
├── card_game.py                    # Core card game logic and rules
├── baseline_bot.py                 # Deficit-tracking heuristic strategy
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
├── bc_ppo_fd/                      # BC + PPO-fD training pipeline
│   ├── train_bc_ppo_fde.py        # BC warm-start + PPO-fD vs an opponent pool
│   ├── ppo_auxiliary_loss.py      # ActorCriticNetwork + PPO-fD update
│   ├── agent_final.pt             # Trained checkpoint (source of the web demo)
│   ├── play_vs_trained_agent.py   # Terminal human-vs-agent play
│   └── evaluate_agent_vs_baseline.py
│
├── docs/                           # Play-vs-agent web demo (GitHub Pages)
│   ├── index.html, style.css, app.js
│   ├── game.js                    # Game-rules engine (pure JS)
│   ├── agent.js                   # Trained network forward pass (pure JS)
│   └── bots.js                    # BaselineBot/SmartBot/RandomBot ports
│
├── scripts/
│   └── export_agent_weights.py    # Exports agent_final.pt -> docs/agent_weights.json
│
├── tests/web/                      # node --test suite for docs/*.js
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