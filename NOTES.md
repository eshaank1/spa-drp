# SPA-DRP Training & Evaluation Commands

## Current Workflow: Ladder Training

The primary workflow now uses **ladder training** for iterative bot improvement.

### Run Ladder Training

```bash
python3 run_ladder_training.py --generations 10 --timesteps-total 100000 --eval-episodes 100 --num-envs 8 --entropy-coef 0.05
```

**Parameters:**
- `--generations N`: Number of generations to train
- `--timesteps-total`: Training timesteps per generation (default: 100000)
- `--eval-episodes`: Episodes to evaluate each opponent (default: 100)
- `--num-envs`: Parallel environments (default: 8)
- `--entropy-coef`: Exploration coefficient (default: 0.05, higher = more diverse)
- `--initial-champion`: Starting model (default: champion_gen1.zip)

**Example: Continue from gen 5 with more exploration**

```bash
python3 run_ladder_training.py --initial-champion models/ladder/champions/champion_gen5.zip --generations 10 --timesteps-total 100000 --eval-episodes 100 --num-envs 8 --entropy-coef 0.1
```

### What Happens During Ladder Training

Each generation trains for a fixed number of timesteps against **2 opponents only**, **playing both Player 1 and Player 2 roles equally**:

**Training Phase (timesteps split across roles):**
- **35%** vs Previous Gen as Player 1
- **35%** vs Previous Gen as Player 2
- **15%** vs Original PPO as Player 1
- **15%** vs Original PPO as Player 2

Playing both roles forces the bot to learn both offensive and defensive strategies, making it more robust against human players.

The training uses **entropy regularization** (default 0.05) which encourages exploration of diverse strategies instead of always picking the same "safe" move. This makes the bot less predictable and more adaptable to novel opponents.

**Evaluation Phase (after training):**
After training completes, the model is evaluated against all 4 opponents (100 episodes each):
- **RandomBot**: Random play strategy
- **SmartBot**: Heuristic-based smart bot
- **Original PPO**: The original ppo_vs_smart_final.zip model
- **Previous Gen**: Current champion

Win rates are calculated bidirectionally (both player roles) and aggregated into `ladder_training_stats.csv`.

**Promotion:**
Every generation **automatically gets promoted** to become the next champion—no threshold requirement. This allows models to iteratively improve without getting stuck, and statistics are logged to track progress.

---

## Ladder Training Details

For ongoing training against a fixed opponent without ladder progression:

```bash
python3 run_continue_training.py --timesteps-per-iter 300000 --eval-episodes 400 --max-iters 60 --num-envs 8 --threshold 0.55
```

**Parameters:**
- `--timesteps-per-iter`: Timesteps per training iteration (default: 100000)
- `--eval-episodes`: Episodes per evaluation (default: 100)
- `--max-iters`: Maximum iterations to train (default: 10)
- `--num-envs`: Parallel environments (default: 8)
- `--threshold`: Win rate threshold to stop (default: 0.55 = 55%)
- `--no-bidirectional-threshold`: Use one-way evaluation only (old behavior)
- `--dry-run`: Preview commands without starting training

---

## Evaluation: Bot vs Bot

### Compare against original PPO bot

```bash
python3 evaluate_ppo_vs_ppo.py --challenger-model models/ladder/champions/champion_gen5.zip --opponent-model pettingzoo_ppo/models/ppo_vs_smart_final.zip --episodes 100 --bidirectional
```

### Compare two ladder champions

```bash
python3 evaluate_ppo_vs_ppo.py --challenger-model models/ladder/champions/champion_gen5.zip --opponent-model models/ladder/champions/champion_gen4.zip --episodes 100 --bidirectional
```

### Auto-test latest champion against original PPO bot

```bash
python3 evaluate_ppo_vs_ppo.py --challenger-model "$(ls -1t models/ladder/champions/champion_gen*.zip | head -n 1)" --opponent-model pettingzoo_ppo/models/ppo_vs_smart_final.zip --episodes 100 --bidirectional
```

---

## Human Play: Test Your Bot

### Play against specific generation

```bash
python3 play_vs_ppo.py --model-path models/ladder/gen_105/champion_gen105.zip
```

### Play against latest champion

```bash
python3 play_vs_ppo.py --model-path "$(ls -1t models/ladder/champions/champion_gen*.zip | head -n 1)"
```

### Play against original PPO

```bash
python3 play_vs_ppo.py --model-path pettingzoo_ppo/models/ppo_vs_smart_final.zip
```

---

## Legacy: Direct SmartBot Training

**Note:** This was used before ladder training. Kept for reference.

### Train new bot against SmartBot

```bash
cd pettingzoo_ppo
python3 train_ppo_vs_smart.py --resume-from models/ppo_vs_smart_v7_final.zip --timesteps 1000000 --model-name ppo_vs_smart_v8
```

### Evaluate new bot vs SmartBot

```bash
cd pettingzoo_ppo
python3 evaluate_ppo_vs_smart.py --model-path models/ppo_vs_smart_v7_final.zip --episodes 1000
```

### Human play vs new bot

```bash
python3 play_vs_ppo.py --model-path pettingzoo_ppo/models/ppo_vs_smart_v7_final.zip
```

**Note:** Increment version numbers (v7 → v8 → v9) for each iteration.

---

## Key Concepts

**Ladder Training:** Generational training where each bot fights previous generation + original PPO. Automatically promotes winners and evaluates against 4 opponents.

**Bidirectional Evaluation:** Tests both player 1 and player 2 perspectives, averaging results for fairness.

**Action Masking:** Both training and play prevent invalid moves (e.g., playing cards not in hand) by auto-correcting to valid actions.

**Entropy Coefficient:** Controls exploration. Higher = more diverse strategies. Start at 0.05, increase to 0.1 for more robustness.

---

## Monitoring Training

After ladder training, check results in:

```bash
cat ladder_training_stats.csv
```

Columns: `Generation, RandomBot_WinRate, SmartBot_WinRate, Original_PPO_WinRate, Previous_Gen_WinRate, Avg_WinRate`

---

## Ladder Training Implementation Details

**Training Structure:**
- Each generation trains vs only **2 opponents** (Previous Gen + Original PPO) but plays **both Player 1 and Player 2 roles** (50/50 split)
- Within each opponent, the split is: **70% vs Previous Gen, 30% vs Original PPO** (Previous Gen is stronger)
- Each generation evaluates vs all **4 opponents** (RandomBot, SmartBot, Original PPO, Previous Gen) for comprehensive tracking

**Model Management:**
- `models/ladder/champions/champion_gen1.zip` is the seeded starting champion
- Each promoted generation is copied into `models/ladder/champions/`
- Only the **5 most recent champion models** are kept; older ones and their gen folders are automatically deleted to save disk space
- Training artifacts stored in `models/ladder/gen_N/` directories

**Advanced Features:**
- **Action masking:** Both training and human play prevent invalid moves (trying to play cards you don't have) by sampling from valid actions. No invalid plays allowed!
- **Stochastic play during interactive mode:** The bot samples from its policy rather than always picking the "best" move, making it less predictable and more human-like
- **Entropy regularization:** Encourages exploration during training; use `--entropy-coef 0.1` for more robustness against human opponents
- **No threshold logic:** Every generation automatically gets promoted to keep the training loop flowing

---

## Troubleshooting

**Bot tries to play cards it doesn't have:**
- Action masking is active. Bot should auto-correct invalid actions. If errors persist, check rl_pettingzoo_env.py

**Training exits early:**
- Check `--max-iters` setting for continued training
- Ladder training always trains full generations

**Model files not found:**
- Ensure paths are correct (relative from spa-drp root)
- Check `models/ladder/champions/` for latest champions
- Check `models/ladder/gen_N/` for generation-specific models
