"""
QUICK START GUIDE: BC + PPO-fD Implementation

This file provides quick reference and TL;DR for the complete implementation.
"""

# =============================================================================
# FILE STRUCTURE
# =============================================================================

CREATED_FILES = {
    "state_inverter.py": {
        "purpose": "Fast vectorized state perspective inversion",
        "key_function": "invert_state(obs) -> inverted_obs",
        "shapes": {
            "input": "(50,) or (batch, 50)",
            "output": "same as input",
        },
        "usage": """
            from state_inverter import invert_state
            import torch
            
            obs_bot = torch.tensor([...])  # Player 2 perspective
            obs_agent = invert_state(obs_bot)  # Player 1 perspective
        """
    },
    
    "behavioral_cloning.py": {
        "purpose": "BC pre-training on bot demonstrations",
        "key_classes": [
            "BCDataset - PyTorch dataset wrapper",
            "ActorNetwork - MLP policy network",
            "BehavioralCloningTrainer - Training loop with early stopping",
        ],
        "key_function": "create_behavioral_cloning_pipeline(...)",
        "usage": """
            from behavioral_cloning import create_behavioral_cloning_pipeline
            
            demonstrations = [(state, action_mask, action), ...]
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
        """
    },
    
    "ppo_auxiliary_loss.py": {
        "purpose": "PPO with auxiliary BC loss for online learning",
        "key_classes": [
            "ActorCriticNetwork - Shared backbone + separate heads",
            "PPOWithAuxiliaryLossUpdater - Implements combined loss",
            "RolloutBatch - Dataclass for mixed rollouts",
        ],
        "key_loss": """
            Total_Loss = PPO_Loss + λ*BC_Loss + α*Value_Loss - β*Entropy
            
            PPO_Loss: Computed on agent actions only
            BC_Loss: Computed on bot actions only
            Value_Loss: Computed on all actions
            Entropy: Bonus term for exploration
        """,
        "usage": """
            from ppo_auxiliary_loss import ActorCriticNetwork, PPOWithAuxiliaryLossUpdater
            
            ac = ActorCriticNetwork()
            updater = PPOWithAuxiliaryLossUpdater(ac, bc_lambda=0.1)
            
            losses = updater.update(rollout_batch, num_epochs=4)
        """
    },
    
    "bc_ppo_integration.py": {
        "purpose": "High-level trainer coordinating both phases",
        "key_classes": [
            "BCPPOConfig - Configuration dataclass",
            "RolloutBuffer - Simple trajectory buffer",
            "BCPPOTrainer - High-level coordinator",
        ],
        "usage": """
            from bc_ppo_integration import BCPPOTrainer, BCPPOConfig
            
            config = BCPPOConfig(device="cuda")
            trainer = BCPPOTrainer(config)
            
            # Phase 1
            bc_history = trainer.phase1_behavioral_cloning(demonstrations)
            
            # Phase 2
            ppo_metrics = trainer.phase2_ppo_with_auxiliary_loss(
                rollout_fn=collect_rollouts_fn,
                num_training_steps=100
            )
        """
    },
    
    "collect_bc_demonstrations.py": {
        "purpose": "Collect BC demos from bot vs bot games",
        "key_class": "BCDemoCollector",
        "main_function": "main()",
        "usage": """
            python collect_bc_demonstrations.py
            # Generates 10,000 bot vs bot games
            # Saves demonstrations to bc_demonstrations_10k.pkl
        """
    },
    
    "train_bc_ppo_fde.py": {
        "purpose": "End-to-end training script (complete pipeline)",
        "key_class": "PPOTrainer",
        "main_function": "main(args)",
        "usage": """
            python train_bc_ppo_fde.py --num-ppo-steps 100
        """
    },
    
    "BC_PPO_FD_GUIDE.md": {
        "purpose": "Comprehensive documentation with all details",
        "sections": [
            "Architectural overview",
            "Component details",
            "Integration workflow",
            "Action masking mechanics",
            "Reward shaping",
            "Practical examples",
            "Hyperparameter tuning",
            "Debugging tips",
        ]
    }
}

# =============================================================================
# QUICK START (5 MINUTES)
# =============================================================================

QUICK_START = """
Step 1: Import components
    from state_inverter import invert_state
    from behavioral_cloning import create_behavioral_cloning_pipeline
    from ppo_auxiliary_loss import ActorCriticNetwork, PPOWithAuxiliaryLossUpdater

Step 2: Prepare BC data
    # Generate 10k games of BaselineBot vs BaselineBot
    demonstrations = [
        (state_1, action_mask_1, action_1),
        (state_2, action_mask_2, action_2),
        ...
    ]
    # Important: If bot acted, invert state using invert_state()

Step 3: Train BC
    actor, trainer, loaders = create_behavioral_cloning_pipeline(
        demonstrations,
        batch_size=64
    )
    history = trainer.train(
        loaders["train_loader"],
        loaders["val_loader"],
        num_epochs=100,
        target_accuracy=0.75
    )

Step 4: Initialize PPO
    ac = ActorCriticNetwork()
    updater = PPOWithAuxiliaryLossUpdater(ac, bc_lambda=0.1)

Step 5: Collect rollouts (agent + bot)
    rollouts = RolloutBatch(
        observations=...,  # (traj_len, 50)
        action_masks=...,  # (traj_len, 14)
        actions=...,  # (traj_len,)
        rewards=...,  # (traj_len,)
        values=...,  # (traj_len, 1)
        log_probs=...,  # (traj_len,)
        dones=...,  # (traj_len,)
        is_bot_data=...,  # (traj_len,) boolean
    )

Step 6: Update PPO
    losses = updater.update(rollouts, num_epochs=4)
    print(f"Loss: {losses['total_loss']:.4f}")

Done! Your agent is training!
"""

# =============================================================================
# KEY CONCEPTS
# =============================================================================

KEY_CONCEPTS = {
    "State Inverter": {
        "why": "Game is symmetric but observations are perspective-dependent",
        "what": "Swaps player 1/2 roles in observation vector",
        "when": "When bot acts (player 2), to train agent (player 1)",
        "how": "invert_state(obs) -> swaps card positions and metadata flags"
    },
    
    "Action Masking": {
        "why": "Prevent agent from learning invalid actions",
        "what": "Set invalid logits to -1e9 before softmax",
        "implementation": """
            logits = network(obs)
            logits[action_mask == 0] = -1e9
            probs = softmax(logits)  # Invalid actions ≈ 0
        """
    },
    
    "BC Pre-training": {
        "why": "Warm-start agent with bot's strategy",
        "target": "~75% accuracy (balance between learning and exploration)",
        "loss": "Cross-entropy with action masking",
        "duration": "~10-20 min with 10k demos on single GPU"
    },
    
    "PPO-fD": {
        "why": "Combine PPO with auxiliary BC loss to leverage bot expertise",
        "agent_data": "PPO surrogate loss (learn from own experience)",
        "bot_data": "BC loss (learn from bot decisions)",
        "λ parameter": "Trade-off between PPO and BC learning (0.05-0.2)"
    },
    
    "Advantage Estimation": {
        "formula": "GAE: gae_t = δ_t + (γ*λ)*gae_{t+1}",
        "γ": "Discount factor (0.99)",
        "λ": "GAE smoothing (0.95)",
        "used_for": "PPO surrogate loss only (not for BC loss)"
    }
}

# =============================================================================
# COMMON ERRORS & FIXES
# =============================================================================

ERROR_FIXES = {
    "BC accuracy stuck at ~1/14": {
        "problem": "Network not learning",
        "checklist": [
            "Check action_mask is being applied in forward()",
            "Verify demonstrations have correct shape (50, 14, int)",
            "Try higher learning rate (1e-3) or larger batch size (128)",
        ]
    },
    
    "Agent plays invalid actions": {
        "problem": "Action masking not working",
        "checklist": [
            "Verify action_mask is passed to forward()",
            "Check -1e9 is large enough (test: exp(-1e9) ≈ 0)",
            "Ensure mask[invalid] = 0, mask[valid] = 1",
        ]
    },
    
    "PPO loss is NaN": {
        "problem": "Numerical instability",
        "checklist": [
            "Gradient clipping (max_grad_norm=0.5) enabled?",
            "Advantage normalization present?",
            "Check for -inf in state/reward",
        ]
    },
    
    "Agent learns fast then plateaus": {
        "problem": "Not using bot knowledge effectively",
        "solutions": [
            "Increase bc_lambda (0.1 → 0.2)",
            "Decrease entropy_coeff (0.01 → 0.001)",
            "Increase GAE smoothing (gae_lambda=0.99)",
        ]
    },
    
    "State inversion issues": {
        "problem": "Agent gets confused about whose perspective",
        "checklist": [
            "Only invert when bot acts (player 2)",
            "Verify inversion is element-wise (same shape in/out)",
            "Check: inverted[40] ↔ inverted[41] (wins swapped)",
        ]
    }
}

# =============================================================================
# TENSOR SHAPES REFERENCE
# =============================================================================

TENSOR_SHAPES = {
    "Observation (obs)": "(50,) or (batch, 50)",
    "Action mask": "(14,) or (batch, 14)",
    "Actions": "() or (batch,) - values in [0, 13]",
    "Rewards": "(batch,)",
    "Values": "(batch, 1) or (batch,)",
    "Log probs": "(batch,)",
    "Dones": "(batch,)",
    "is_bot_data": "(batch,) - boolean",
    
    "RolloutBatch": {
        "observations": "(traj_len, 50)",
        "action_masks": "(traj_len, 14)",
        "actions": "(traj_len,)",
        "rewards": "(traj_len,)",
        "values": "(traj_len, 1)",
        "log_probs": "(traj_len,)",
        "dones": "(traj_len,)",
        "is_bot_data": "(traj_len,) bool",
    }
}

# =============================================================================
# HYPERPARAMETER CHEAT SHEET
# =============================================================================

HYPERPARAMETERS = {
    "BC Phase": {
        "batch_size": 64,
        "learning_rate": 3e-4,
        "target_accuracy": 0.75,
        "patience": 10,
        "num_hidden_layers": 2,
        "hidden_dim": 256,
    },
    
    "PPO Phase": {
        "batch_size": 64,
        "learning_rate": 3e-4,
        "clip_ratio": 0.2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "entropy_coeff": 0.01,
        "value_coeff": 0.5,
        "bc_lambda": 0.1,
        "max_grad_norm": 0.5,
    }
}

# =============================================================================
# EXPECTED PERFORMANCE METRICS
# =============================================================================

METRICS = {
    "BC Phase": {
        "target_accuracy": 0.75,
        "training_time": "10-20 minutes (10k demos, 1 GPU)",
        "typical_curve": "Accuracy increases quickly first 20 epochs, then plateaus",
    },
    
    "PPO Phase": {
        "step_1_win_rate": 0.2,  # Right after BC
        "step_50_win_rate": 0.4,  # Mid-training
        "step_100_win_rate": 0.5,  # Full training
        "typical_curve": "Win rate increases gradually, may plateau",
    }
}

# =============================================================================
# INTEGRATION CHECKLIST
# =============================================================================

INTEGRATION_CHECKLIST = """
□ Environment Setup
  □ Have 50-dimensional observation space (Box(0, 1))
  □ Have 14-dimensional action space (Discrete(14))
  □ Observation includes action_mask (14-dim binary)
  □ Observation format matches: [hand | played | opp_played | metadata]

□ BC Phase
  □ Generate 10,000+ bot vs bot games
  □ Invert states when bot acts using invert_state()
  □ Create demonstrations list: [(obs, mask, action), ...]
  □ Run behavioral cloning until ~75% accuracy
  □ Save actor weights

□ PPO Phase  
  □ Collect mixed rollouts:
    □ Agent actions: is_bot_data=False
    □ Bot actions: invert state, is_bot_data=True
  □ Create RolloutBatch with all required fields
  □ Call updater.update(rollouts, num_epochs=4)
  □ Track PPO loss, BC loss, and win rate
  □ Evaluate every N steps

□ Action Masking
  □ Generate action_mask at each step
  □ Pass to forward() in both BC and PPO phases
  □ Verify invalid actions have probability ≈ 0

□ Reward Shaping
  □ Large reward for game win (±5.0)
  □ Small reward for round win (±0.1)
  □ Fine-grained reward for score delta (±0.02)
  □ Avoid round-level rewards dominating game-level rewards

□ Testing
  □ Verify state inversion doesn't corrupt data
  □ Check agent doesn't play invalid actions
  □ Monitor that BC loss decreases during training
  □ Confirm PPO loss converges
"""

# =============================================================================
# REFERENCE FORMULAS
# =============================================================================

FORMULAS = """
===== STATE INVERSION =====
For player 2 -> player 1:
  my_hand: keep (player 2's hand stays)
  my_played: swap with opponent_played
  opponent_played: swap with my_played
  my_wins: swap with opponent_wins
  current_player flags: swap
  first_player flags: swap
  passed flags: swap
  hand_sizes: swap

===== ACTION MASKING =====
logits_masked = logits.masked_fill(action_mask == 0, -1e9)
probs = softmax(logits_masked)  # Only valid actions have prob > 0

===== PPO OBJECTIVE =====
L^CLIP(θ) = -E_t[min(r_t * Â_t, clip(r_t, 1-ε, 1+ε) * Â_t)]

where:
  r_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) = exp(log_π_new - log_π_old)
  Â_t = advantage estimate from GAE
  ε = clip_ratio (0.2)

===== COMBINED LOSS =====
L_total = L_PPO + λ*L_BC + α*L_value - β*H[π]

where:
  λ = 0.05-0.2 (auxiliary loss weight)
  α = 0.5 (value coefficient)
  β = 0.01 (entropy coefficient)

===== GENERALIZED ADVANTAGE ESTIMATION =====
GAE_t = δ_t + (γλ)*δ_{t+1} + (γλ)²*δ_{t+2} + ...

where:
  δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
  γ = discount factor
  λ = GAE smoothing parameter
"""

# =============================================================================
# EXAMPLE INTEGRATION CODE
# =============================================================================

EXAMPLE_CODE = """
# ===== PHASE 1: BEHAVIORAL CLONING =====

from behavioral_cloning import create_behavioral_cloning_pipeline

# Load or generate 10k demonstrations
demonstrations = load_bot_demonstrations()  # List of (obs, mask, action)

# Create and train
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
    target_accuracy=0.75,
    patience=10
)

print(f"BC training done. Final accuracy: {history['best_val_accuracy']:.2%}")

# ===== PHASE 2: PPO-FD =====

from ppo_auxiliary_loss import ActorCriticNetwork, PPOWithAuxiliaryLossUpdater, RolloutBatch

# Initialize networks
ac = ActorCriticNetwork(obs_dim=50, action_dim=14, hidden_dim=256)
updater = PPOWithAuxiliaryLossUpdater(ac, bc_lambda=0.1)

# Training loop
for step in range(100):
    # Collect rollouts with mixed data
    rollouts = collect_mixed_rollouts(ac)  # Returns RolloutBatch
    
    # Update
    losses = updater.update(rollouts, num_epochs=4)
    
    # Evaluate
    if (step + 1) % 10 == 0:
        win_rate = evaluate(ac, num_games=50)
        print(f"Step {step+1}: Loss={losses['total_loss']:.4f}, WinRate={win_rate:.1%}")

print("Training complete!")
"""

if __name__ == "__main__":
    print("=" * 80)
    print("BC + PPO-fD QUICK START REFERENCE")
    print("=" * 80)
    print()
    print("FILES CREATED:")
    for name, info in CREATED_FILES.items():
        print(f"  • {name}")
    print()
    print("QUICK START:")
    print(QUICK_START)
    print()
    print("For detailed documentation, see BC_PPO_FD_GUIDE.md")
