# BC + PPO-fD commands

## Collect BaselineBot demonstrations (optional; train auto-collects if missing)
python3 bc_ppo_fd/collect_bc_demonstrations.py        # writes bc_demonstrations_10k.pkl
# (training uses bc_ppo_fd/bc_demos.pkl, created automatically on first run)

## Train a new agent (BC warm-start + PPO-fD vs opponent pool)
python3 bc_ppo_fd/train_bc_ppo_fde.py --num-bc-games 5000 --num-ppo-steps 300

## Continue training from the saved checkpoint (skips BC warm-start)
python3 bc_ppo_fd/train_bc_ppo_fde.py --resume-from agent_final.pt --num-ppo-steps 200

# Useful knobs:
#   --bc-lambda 0.2          weight of the auxiliary imitation loss (lower => more PPO freedom)
#   --entropy-coeff 0.01     exploration bonus
#   --self-play-after 20     PPO step at which frozen self-play snapshots join the opponent pool
#   --eval-games 50          games per opponent at each eval

## Evaluate vs BaselineBot (true opponent now — env plays P2 with BaselineBot)
python3 bc_ppo_fd/evaluate_agent_vs_baseline.py --num-games 200

## Evaluate with move-by-move output
python3 bc_ppo_fd/evaluate_verbose.py --num-games 5

## Play against the agent
python3 game_with_bots.py                 # menu option 4 (you vs agent) or 7 (agent vs baseline)
python3 bc_ppo_fd/play_vs_trained_agent.py

## View training log
cat bc_ppo_fd/training.log