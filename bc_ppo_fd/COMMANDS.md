# Train (from root folder)
python3 bc_ppo_fd/train_bc_ppo_fde.py --resume-from agent_final.pt --num-ppo-steps 100

# Play against agent (from root folder)
python3 bc_ppo_fd/play_vs_trained_agent.py

# Or specify model explicitly
python3 bc_ppo_fd/play_vs_trained_agent.py --model bc_ppo_fd/agent_final.pt

# Watch one game step-by-step (from root folder)
python3 bc_ppo_fd/watch_agent_vs_baseline.py

# Evaluate agent vs BaselineBot (from root folder)
python3 bc_ppo_fd/evaluate_agent_vs_baseline.py --num-games 50

# Specify model explicitly for evaluation
python3 bc_ppo_fd/evaluate_agent_vs_baseline.py --model bc_ppo_fd/agent_final.pt --num-games 100

# View logs
tail -50 bc_ppo_fd/training.log
cat bc_ppo_fd/training.log