**How to keep training whenever you want:**

**Note:** Training uses bidirectional threshold by default (evaluates win rate in both player directions, requires ≥55% aggregated win rate to stop).

for regular progress

python3 run_continue_training.py

or longer version with flags for a serious training push

python3 run_continue_training.py --timesteps-per-iter 300000 --eval-episodes 400 --max-iters 60 --num-envs 8 --threshold 0.55

to train with one-way threshold only (old behavior)

python3 run_continue_training.py --no-bidirectional-threshold

**If you want to preview what it will run without starting training:**

python3 run_continue_training.py --dry-run

**To run test vs other bot:**

python3 evaluate_ppo_vs_ppo.py --challenger-model models/continued/smoke_challenger_final.zip --opponent-model pettingzoo_ppo/models/ppo_vs_smart_final.zip --episodes 100 --bidirectional