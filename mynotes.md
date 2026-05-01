**To run consistently winning bot simulations vs smart bot:**

in pettingzoo_ppo dir, run

/usr/bin/python3 evaluate_ppo_vs_smart.py --model-path models/ppo_vs_smart_v7_final.zip --episodes 1000

**to train a new bot against smart bot:**

in pettingzoo_ppo dir, run

usr/bin/python3 train_ppo_vs_smart.py --resume-from models/ppo_vs_smart_v7_final.zip --timesteps 1000000 --model-name ppo_vs_smart_v8


**to play against bot:**

in root dir, run

/usr/bin/python3 play_vs_ppo.py --model-path pettingzoo_ppo/models/ppo_vs_smart_v7_final.zip


**Change the numbers in every v<> for every iteration**