# Train new agent
python3 bc_ppo_fd/train_bc_ppo_fde.py --num-ppo-steps 100

# Continue training agent
python3 bc_ppo_fd/train_bc_ppo_fde.py --resume-from agent_final.pt --num-ppo-steps 500

# Play against agent
python3 game_with_bots.py

# Evaluate agent vs BaselineBot (step by step)
python3 bc_ppo_fd/evaluate_verbose.py

# View logs
cat bc_ppo_fd/training.log