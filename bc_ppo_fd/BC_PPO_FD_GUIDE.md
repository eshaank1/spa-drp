"""
COMPREHENSIVE GUIDE: BC + PPO-fD for Card Game RL Training

This document provides complete implementation details and integration instructions
for the two-phase training approach: Behavioral Cloning (BC) followed by PPO with
auxiliary Imitation Loss (PPO-fD).

======== TABLE OF CONTENTS ========

1. ARCHITECTURAL OVERVIEW
2. COMPONENT DETAILS
3. INTEGRATION WORKFLOW
4. ACTION MASKING MECHANICS
5. REWARD SHAPING GUIDELINES
6. PRACTICAL USAGE EXAMPLES
7. HYPERPARAMETER TUNING
8. DEBUGGING TIPS

======== 1. ARCHITECTURAL OVERVIEW ========

The training pipeline consists of three phases:

┌─────────────────────────────────────────────────────────────┐
│                    PHASE 1: BEHAVIORAL CLONING               │
│                 (Pre-train on bot demonstrations)            │
├─────────────────────────────────────────────────────────────┤
│ Input:  10,000 games of BaselineBot vs BaselineBot         │
│         → Extract (state, action_mask, action) tuples       │
│ Output: Pre-trained Actor network (MLP)                     │
│ Loss:   Cross-Entropy with action masking                   │
│ Target: ~75% accuracy (prevents overfitting)                │
│ Time:   ~10-20 minutes (single GPU)                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 PHASE 2A: PPO WITH AUXILIARY LOSS             │
│        (Online learning against BaselineBot via self-play)   │
├─────────────────────────────────────────────────────────────┤
│ Input:  Pre-trained Actor from Phase 1                      │
│         Live rollouts with mixed data:                       │
│           - Agent actions (PPO surrogate loss)              │
│           - Bot actions (auxiliary BC loss)                  │
│ Process:                                                     │
│   For each step t:                                           │
│     1. If agent acted:                                      │
│        - Compute PPO clipped surrogate loss                 │
│     2. If bot acted:                                        │
│        - Invert observation using state_inverter()         │
│        - Compute cross-entropy BC loss                      │
│     3. Compute value loss on ALL steps                      │
│     4. Total = PPO_loss + λ*BC_loss + α*Value_loss         │
│        - Add entropy bonus for exploration                  │
│ Output: Trained Actor-Critic for competitive play          │
│ Time:   Varies by target performance                        │
└─────────────────────────────────────────────────────────────┘


======== 2. COMPONENT DETAILS ========

### 2.1 STATE INVERTER (state_inverter.py)

The state inverter is critical because the game is symmetric but observations
are not. When the bot (player 2) plays, its observation is from its perspective:
  Bot's observation:   [Bot_hand | Bot_played | Agent_played | metadata]

But we want to train our agent using the bot's data:
  Agent's observation: [Agent_hand | Agent_played | Bot_played | metadata]

The inverter swaps these perspectives:

  Transformation:
    [0-12]   (my hand) ↔ N/A (we keep bot's hand)
    [13-25]  (my cards played) ↔ [26-38] (opp cards played)
    [26-38]  (opp cards played) ↔ [13-25]
    [40]     (my_wins) ↔ [41] (opp_wins)
    [42-43]  (is_current_player flags) ← swap
    [44-45]  (is_first_player flags) ← swap
    [46-47]  (passed flags) ← swap
    [48-49]  (hand_sizes) ← swap

Usage:
    from state_inverter import invert_state
    
    # Single observation
    bot_obs = torch.tensor([...])  # From bot's perspective
    agent_obs = invert_state(bot_obs)  # From agent's perspective
    
    # Batch processing
    bot_obs_batch = torch.tensor([[...], [...], ...])  # Shape: (32, 50)
    agent_obs_batch = invert_state(bot_obs_batch)  # Shape: (32, 50)


### 2.2 BEHAVIORAL CLONING (behavioral_cloning.py)

Pre-training teaches the agent to imitate the heuristic bot.

Key classes:
  - BCDataset: PyTorch dataset for demonstrations
  - ActorNetwork: MLP policy network with action masking
  - BehavioralCloningTrainer: Training loop with early stopping

Training process:

  1. Load demonstrations: List of (state, action_mask, action) tuples
  2. Split: 80% train, 20% validation
  3. For each epoch:
     a. Forward pass: logits = actor(state, action_mask)
     b. Apply masking: logits[invalid_actions] = -1e9
     c. Loss = CrossEntropyLoss(logits, action)
     d. Backward + optimizer step
     e. Track accuracy (correct_actions / total_actions)
  4. Early stopping when:
     - Accuracy reaches target (75%)
     - No improvement for 'patience' epochs (default 10)

Why 75% accuracy?
  - Too high (>85%): Overfits to bot's fixed strategy → rigid agent
  - Too low (<60%): Insufficient learning → poor performance
  - 75%: Good balance → learns general strategy without memorizing

Usage:
    from behavioral_cloning import create_behavioral_cloning_pipeline
    
    demonstrations = [
        (state1, mask1, action1),
        (state2, mask2, action2),
        ...
    ]
    
    actor, trainer, loaders = create_behavioral_cloning_pipeline(
        demonstrations,
        batch_size=64,
        hidden_dim=256,
        num_hidden_layers=2
    )
    
    history = trainer.train(
        loaders["train_loader"],
        loaders["val_loader"],
        num_epochs=100,
        target_accuracy=0.75
    )


### 2.3 PPO WITH AUXILIARY LOSS (ppo_auxiliary_loss.py)

The core of Phase 2. Combines PPO surrogate loss with auxiliary BC loss.

Key classes:
  - ActorCriticNetwork: Shared backbone + separate policy/value heads
  - PPOWithAuxiliaryLossUpdater: Implements combined loss computation

The combined loss:

  Total_Loss = PPO_Loss + λ*BC_Loss + α*Value_Loss - β*Entropy

  Where:
    PPO_Loss = mean(-min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t))
      - r_t = exp(log_π_new - log_π_old)  [importance ratio]
      - A_t = advantage estimate
      - ε = clip_ratio (0.2)
    
    BC_Loss = CrossEntropyLoss(logits[bot_actions], bot_actions)
      - Computed only on bot action states
      - Encourages agent to learn from bot's decisions
    
    Value_Loss = MSE(V_predicted - Returns)
      - Computed on ALL states (agent + bot)
      - Returns = A_t + V_t
    
    Entropy = -sum(π * log(π))
      - Bonus for exploration (negative loss coefficient)
    
    λ = 0.05-0.2  [adjust based on bot data quantity]
    α = 0.5       [value loss weight]
    β = 0.01      [entropy bonus weight]

Algorithm steps (per update):

  For each PPO epoch:
    1. Compute advantages using GAE:
       gae_λ = 0.95 (smoothing parameter)
       γ = 0.99 (discount factor)
    
    2. Separate data into agent_mask and bot_mask:
       agent_mask: Where agent_policy acted
       bot_mask: Where bot_policy acted
    
    3. For agent data:
       new_logits, new_values = AC_network(obs, mask)
       new_logits_probs = softmax(new_logits)
       new_log_probs = log_softmax(new_logits)
       ratio = exp(new_log_probs - old_log_probs)
       surrogate = min(ratio*advantages, clipped_ratio*advantages)
       ppo_loss = -mean(surrogate)
    
    4. For bot data:
       bot_logits, _ = AC_network(bot_obs, bot_mask)
       bc_loss = CrossEntropy(bot_logits, bot_actions)
    
    5. Value loss on all data:
       value_loss = MSE(values - returns)
    
    6. Entropy bonus on all policy distributions
    
    7. Total_loss = ppo_loss + λ*bc_loss + α*value_loss - β*entropy
    
    8. Backward + clip gradients + optimizer.step()

Usage:
    from ppo_auxiliary_loss import ActorCriticNetwork, PPOWithAuxiliaryLossUpdater
    
    actor_critic = ActorCriticNetwork(
        obs_dim=50,
        action_dim=14,
        hidden_dim=256
    )
    
    updater = PPOWithAuxiliaryLossUpdater(
        actor_critic,
        bc_lambda=0.1,  # Auxiliary loss weight
        entropy_coeff=0.01
    )
    
    losses = updater.update(
        rollouts=rollout_batch,
        num_epochs=3,
        gamma=0.99,
        gae_lambda=0.95
    )


======== 3. INTEGRATION WORKFLOW ========

Step 1: Collect BC demonstrations
───────────────────────────────────
# Generate 10,000 games of BaselineBot vs BaselineBot
# For each game, collect (state, action_mask, action) at each step
# When bot is player 2, INVERT the state using state_inverter

from state_inverter import invert_state

demonstrations = []
for game_idx in range(10000):
    game_states = play_bot_vs_bot()  # Your game simulation
    
    for step in game_states:
        obs, mask, action = step
        
        # If bot acted (player 2), invert perspective
        if step.actor == 2:
            obs = invert_state(obs)
        
        demonstrations.append((obs, mask, action))


Step 2: Phase 1 - Behavioral Cloning
─────────────────────────────────────
from bc_ppo_integration import BCPPOTrainer, BCPPOConfig

config = BCPPOConfig(
    device="cuda",
    bc_target_accuracy=0.75,
    hidden_dim=256,
)

trainer = BCPPOTrainer(config)
bc_history = trainer.phase1_behavioral_cloning(demonstrations)

# bc_history contains:
# - final accuracy
# - training curves
# - best model


Step 3: Collect PPO rollouts with mixed data
──────────────────────────────────────────────
# During environment interaction:
# 1. Agent acts → store with is_bot_data=False
# 2. Bot acts → invert state, store with is_bot_data=True

def rollout_fn(actor_critic):
    buffer = RolloutBuffer()
    
    while not done and steps < max_steps:
        if current_player == 1:  # Agent's turn
            obs = get_observation()
            mask = get_valid_actions()
            
            # Get agent's action
            with torch.no_grad():
                logits, value = actor_critic(obs, mask)
                action = sample_from_logits(logits)
                log_prob = get_log_prob(logits, action)
            
            # Step environment
            next_obs, reward, done = env.step(action)
            
            # Store transition
            buffer.add_transition(
                obs, mask, action, reward, value,
                log_prob, done, is_bot=False
            )
        
        else:  # Bot's turn
            obs = get_observation()  # From bot's perspective
            mask = get_valid_actions()
            
            # Get bot's action
            action = baseline_bot.decide_move(obs, mask)
            
            # CRITICAL: Invert state perspective
            obs_inverted = invert_state(obs)
            
            # Get log prob from agent's network
            with torch.no_grad():
                logits, value = actor_critic(obs_inverted, mask)
                log_prob = get_log_prob(logits, action)
            
            # Step environment
            next_obs, reward, done = env.step(action)
            
            # Store transition with bot flag
            buffer.add_transition(
                obs_inverted, mask, action, reward, value,
                log_prob, done, is_bot=True
            )
    
    return buffer.get_rollout_batch(), metrics


Step 4: Phase 2 - PPO-fD Online Learning
─────────────────────────────────────────
ppo_metrics = trainer.phase2_ppo_with_auxiliary_loss(
    rollout_fn=rollout_fn,
    num_training_steps=100,
)


======== 4. ACTION MASKING MECHANICS ========

Why action masking?
  - Invalid actions would waste model capacity learning -inf gradients
  - With masking, policy only allocates probability mass to valid actions
  - Prevents probabilistic leakage to invalid actions

Implementation in forward pass:

  def forward(self, obs, action_mask=None):
      logits = self.net(obs)
      
      if action_mask is not None:
          # action_mask: 1=valid, 0=invalid
          # Set invalid logits to -1e9 (approximately -infinity)
          logits = logits.masked_fill(action_mask == 0, -1e9)
      
      return logits
  
  # After softmax:
  probs = softmax(logits)
  # Invalid actions ≈ 0
  # Valid actions = normalized probabilities

Practical integration:
  - action_mask[0] = 1 always (can always pass)
  - action_mask[i] = 1 if card i is in hand
  - action_mask[i] = 0 otherwise

From environment:
  mask = np.zeros(14)
  mask[0] = 1  # Can always pass
  for rank in hand:
      mask[rank_to_action[rank]] = 1


======== 5. REWARD SHAPING GUIDELINES ========

CRITICAL: Reward must reflect GAME outcome, not round outcome.

❌ DON'T:
    reward = 1.0 if round_won else -1.0
    # This incentivizes winning round 1 even at the cost of cards
    # → Agent burns all high cards early → Loses game

✓ DO:
    reward = 0.0  # Most steps
    
    if round_complete:
        if round_won:
            reward += 0.1  # Small bonus
        elif round_lost:
            reward -= 0.1
        else:
            reward += 0.05  # Tie
    
    if game_complete:
        if game_won:
            reward += 5.0  # Large bonus at end
        else:
            reward -= 5.0

Why?
  - Main reward signal = game outcome (±5.0)
  - Round rewards (±0.1) provide guidance without distorting priorities
  - Score delta (±0.02 * Δscore) provides fine-grained shaping

The value function learns the long-horizon strategy from game rewards,
while round rewards provide local guidance.


======== 6. PRACTICAL USAGE EXAMPLES ========

Example 1: Minimal integration
────────────────────────────────

from state_inverter import invert_state
from behavioral_cloning import create_behavioral_cloning_pipeline
from ppo_auxiliary_loss import ActorCriticNetwork, PPOWithAuxiliaryLossUpdater

# Phase 1
bc_demos = load_bot_demonstrations()  # 10k games
actor, trainer, loaders = create_behavioral_cloning_pipeline(bc_demos)
trainer.train(loaders["train_loader"], loaders["val_loader"])

# Phase 2
ac = ActorCriticNetwork()
updater = PPOWithAuxiliaryLossUpdater(ac, bc_lambda=0.1)

for epoch in range(100):
    rollouts = collect_rollouts(ac)
    losses = updater.update(rollouts)
    print(f"Epoch {epoch}: Loss={losses['total_loss']:.4f}")


Example 2: Full integration with environment
──────────────────────────────────────────────

class CardGameTrainer:
    def __init__(self, env, bot, device="cuda"):
        self.env = env
        self.bot = bot
        self.device = device
    
    def collect_bot_demonstrations(self, num_games=10000):
        demos = []
        for _ in range(num_games):
            obs, _ = self.env.reset()
            done = False
            
            while not done:
                if self.env.current_player == 1:
                    action = self.bot.decide_move(obs)
                    obs, reward, done, _ = self.env.step(action)
                    
                    # Invert because bot's observation
                    obs_inv = invert_state(obs)
                    mask = self.env.get_action_mask()
                    
                    demos.append((obs_inv, mask, action))
        
        return demos
    
    def train_bc_phase(self, demonstrations):
        actor, trainer, loaders = create_behavioral_cloning_pipeline(
            demonstrations,
            batch_size=64
        )
        return trainer.train(
            loaders["train_loader"],
            loaders["val_loader"],
            num_epochs=100,
            target_accuracy=0.75
        )
    
    def collect_ppo_rollouts(self, actor_critic):
        buffer = RolloutBuffer(self.device)
        obs, _ = self.env.reset()
        done = False
        
        while not done and buffer.__len__() < 2048:
            if self.env.current_player == 1:
                mask = self.env.get_action_mask()
                
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32)
                    mask_t = torch.tensor(mask, dtype=torch.float32)
                    logits, value = actor_critic(obs_t, mask_t)
                    action = torch.multinomial(
                        torch.softmax(logits, -1), 1
                    ).item()
                    log_prob = torch.log_softmax(logits, -1)[action].item()
                
                obs_next, reward, done, _ = self.env.step(action)
                
                buffer.add_transition(
                    obs, mask, action, reward, value.item(),
                    log_prob, done, is_bot=False
                )
                
                obs = obs_next
            
            else:
                mask = self.env.get_action_mask()
                action = self.bot.decide_move(obs)
                
                obs_inv = invert_state(torch.tensor(obs, dtype=torch.float32))
                
                with torch.no_grad():
                    mask_t = torch.tensor(mask, dtype=torch.float32)
                    logits, value = actor_critic(obs_inv, mask_t)
                    log_prob = torch.log_softmax(logits, -1)[action].item()
                
                obs_next, reward, done, _ = self.env.step(action)
                
                buffer.add_transition(
                    obs_inv.numpy(), mask, action, reward, value.item(),
                    log_prob, done, is_bot=True
                )
                
                obs = obs_next
        
        return buffer.get_rollout_batch()
    
    def train_ppo_phase(self, num_updates=100):
        ac = ActorCriticNetwork()
        updater = PPOWithAuxiliaryLossUpdater(ac, bc_lambda=0.1)
        
        for update in range(num_updates):
            rollouts = self.collect_ppo_rollouts(ac)
            losses = updater.update(rollouts, num_epochs=4)
            
            if (update + 1) % 10 == 0:
                win_rate = self.evaluate(ac)
                print(f"Update {update+1}: Loss={losses['total_loss']:.4f}, WinRate={win_rate:.1%}")
    
    def evaluate(self, actor_critic, num_games=100):
        wins = 0
        for _ in range(num_games):
            obs, _ = self.env.reset()
            done = False
            
            while not done:
                if self.env.current_player == 1:
                    mask = self.env.get_action_mask()
                    with torch.no_grad():
                        logits, _ = actor_critic(
                            torch.tensor(obs, dtype=torch.float32),
                            torch.tensor(mask, dtype=torch.float32)
                        )
                        action = torch.argmax(logits).item()
                    obs, _, done, info = self.env.step(action)
                    if done and info["winner"] == 1:
                        wins += 1
                else:
                    action = self.bot.decide_move(obs)
                    obs, _, done, _ = self.env.step(action)
        
        return wins / num_games


======== 7. HYPERPARAMETER TUNING ========

BC Phase:
  bc_target_accuracy: 0.70-0.80
    - Lower: faster training, more exploration room
    - Higher: more faithful to bot, but may overfit
  
  bc_batch_size: 32-128
    - Batch size; 64 is typical
  
  bc_learning_rate: 1e-4 to 1e-3
    - 3e-4 is good default
  
  bc_patience: 5-15
    - Early stopping patience (epochs without improvement)

PPO Phase:
  ppo_clip_ratio: 0.1-0.3
    - Default 0.2 is standard; controls trust region
  
  ppo_gamma: 0.95-0.99
    - Discount factor; 0.99 for long horizons
  
  ppo_gae_lambda: 0.9-0.99
    - GAE smoothing; higher = more variance but less bias
  
  ppo_bc_lambda: 0.05-0.2
    - Auxiliary loss weight
    - If bot_action % << 1%: use lower (0.05-0.1)
    - If bot_action % >> 1%: use higher (0.1-0.2)
  
  ppo_entropy_coeff: 0.001-0.1
    - Too high: random exploration
    - Too low: premature convergence
    - 0.01 is standard


======== 8. DEBUGGING TIPS ========

Issue: BC phase doesn't converge
  - Check: Is action_mask being applied correctly?
  - Check: Are invalid actions in demonstrations?
  - Check: Is learning rate too high? Try 1e-4
  - Check: Model capacity too low? Try hidden_dim=512

Issue: PPO loss oscillates wildly
  - Check: Is reward clipping enabled?
  - Check: Gradient norm clipping (max_grad_norm=0.5)
  - Check: Batch size too small? Try 128+
  - Check: Learning rate? Try 1e-4

Issue: Agent plays invalid actions
  - Check: Is action_mask being passed to forward()?
  - Check: After softmax, verify invalid_probs ≈ 0
  - Check: In environment step, handle invalid actions

Issue: BC_loss is NaN
  - Check: Are bot observations correctly inverted?
  - Check: Are logits being computed correctly?
  - Check: Verify no -inf in logits (action masking overflow)

Issue: Agent learns quickly but plateaus
  - Increase bc_lambda (more auxiliary loss guidance)
  - Decrease entropy_coeff (less exploration noise)
  - Increase GAE smoothing (gae_lambda to 0.99)
"""

# Quick test script
if __name__ == "__main__":
    print("=" * 70)
    print("BC + PPO-fD DOCUMENTATION")
    print("=" * 70)
    print(__doc__)
