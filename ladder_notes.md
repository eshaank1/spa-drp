# Ladder commands

## Keep training the ladder

Simple training: each generation trains for 10000 timesteps (split between 2 opponents), evaluates vs all 4, and automatically gets promoted. Statistics logged to CSV.

```bash
python3 run_ladder_training.py --generations 10 --timesteps-total 10000 --eval-episodes 100 --num-envs 8
```

This logs training statistics to `ladder_training_stats.csv` with columns:
- Generation
- RandomBot_WinRate
- SmartBot_WinRate  
- Original_PPO_WinRate
- Previous_Gen_WinRate
- Avg_WinRate

If you want to start from a specific champion:

```bash
python3 run_ladder_training.py --initial-champion models/ladder/champions/champion_gen5.zip --generations 10 --timesteps-total 100000 --eval-episodes 100 --num-envs 8
```

### What happens during ladder training

Each generation trains for a fixed number of timesteps against **2 opponents only**:

**Training phase (10000 timesteps default):**
- **70%** (7000 timesteps) vs **Previous Gen** (current champion)
- **30%** (3000 timesteps) vs **Original PPO** (baseline model)

No threshold checking during training - just pure training iterations.

**Evaluation phase (100 episodes per opponent):**
After training completes, the model is evaluated against all 4 opponents:
- **RandomBot**: Random play strategy
- **SmartBot**: Heuristic-based smart bot
- **Original PPO**: The original ppo_vs_smart_final.zip model
- **Previous Gen**: Current champion

Win rates are calculated bidirectionally (both player roles) and averaged.

**Promotion:**
Every generation **automatically gets promoted** to become the next champion. No threshold requirement - this allows the models to iteratively improve without getting stuck.

**Statistics logging:**
After each generation, a row is added to `ladder_training_stats.csv` with win rates vs all 4 opponents, allowing you to track learning progress over generations.

## Play against the current champion

Play as a human against the champion model:

```bash
python3 play_vs_ppo.py --model-path models/ladder/champions/champion_gen0.zip
```

If you have a newer promoted champion, swap in that file instead:

```bash
python3 play_vs_ppo.py --model-path models/ladder/champions/champion_gen5.zip
```

To automatically play against the newest promoted champion:

```bash
python3 play_vs_ppo.py --model-path "$(ls -1t models/ladder/champions/champion_gen*.zip | head -n 1)"
```

## Test champion win rate vs other bots

Evaluate a champion against another bot with bidirectional scoring:

```bash
python3 evaluate_ppo_vs_ppo.py --challenger-model models/ladder/champions/champion_gen0.zip --opponent-model pettingzoo_ppo/models/ppo_vs_smart_final.zip --episodes 100 --bidirectional
```

You can also compare a champion against another ladder champion:

```bash
python3 evaluate_ppo_vs_ppo.py --challenger-model models/ladder/champions/champion_gen5.zip --opponent-model models/ladder/champions/champion_gen4.zip --episodes 100 --bidirectional
```

To automatically test the newest champion against the PPO bot:

```bash
python3 evaluate_ppo_vs_ppo.py --challenger-model "$(ls -1t models/ladder/champions/champion_gen*.zip | head -n 1)" --opponent-model pettingzoo_ppo/models/ppo_vs_smart_final.zip --episodes 100 --bidirectional
```

## Useful notes

- Each generation trains vs only **2 opponents** (Previous Gen + Original PPO) to keep training focused and efficient.
  - **70% of timesteps** go to Previous Gen (most important)
  - **30% of timesteps** go to Original PPO (baseline)
- Each generation evaluates vs all **4 opponents** (RandomBot, SmartBot, Original PPO, Previous Gen) for comprehensive tracking.
- **No threshold logic** - every generation automatically gets promoted to keep the training loop flowing.
- Statistics are logged to `ladder_training_stats.csv` in the spa-drp root directory, tracking win rates over generations.
- Only the **5 most recent champion models** are kept; older ones and their gen folders are automatically deleted to save disk space.
- `models/ladder/champions/champion_gen1.zip` is the seeded starting champion.
- Each promoted generation is copied into `models/ladder/champions/`.
- The trainer uses `pettingzoo_ppo/train_ladder_challenger.py` internally for simple fixed-iteration training.
