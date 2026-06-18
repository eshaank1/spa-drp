import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset

# ============================================================================
# 1. THE BUG-FIXED HEURISTIC BOT
# ============================================================================
class BaselineBot:
    """
    Fixed Heuristic Bot - Safe from the 'Negative Deficit' hand-dumping bug.
    """
    RANK_VALUES = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                   '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13}

    # Map indices to card strings for the environment
    IDX_TO_CARD = {1:'A', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 
                   8:'8', 9:'9', 10:'10', 11:'J', 12:'Q', 13:'K'}

    def decide_move(self, hand, my_score, opp_score, round_num,
                    my_wins, opp_wins, opponent_has_passed):

        if not hand:
            return 0  # PASS

        sorted_hand = sorted(hand, key=lambda c: self.RANK_VALUES[c])
        deficit = opp_score - my_score  # positive = we're behind, negative = we're ahead

        must_win = (opp_wins == 1)          # losing this round loses the match
        is_last_round = (round_num >= 3)
        critical = must_win or is_last_round
        can_sacrifice = (my_wins == 1 and opp_wins == 0)  # 1-0 up, can afford to drop a round

        # Cards that flip the lead in one play (value must exceed current deficit)
        winning_cards = [c for c in sorted_hand if self.RANK_VALUES[c] > deficit]

        # ── Opponent has passed: their score is locked, we act freely ───────
        if opponent_has_passed:
            if deficit < 0:           # already winning — stop spending cards
                return 0
            if winning_cards:         # can take the lead with smallest efficient card
                return self.RANK_VALUES[winning_cards[0]]
            if critical:              # can't win but must try — best effort
                return self.RANK_VALUES[sorted_hand[-1]]
            return 0                  # can't win, save cards for next round

        # ── Opponent still active ────────────────────────────────────────────

        # Currently ahead
        if deficit < 0:
            if critical:
                # Never pass here — opponent can still play and overtake us
                return self.RANK_VALUES[sorted_hand[0]]
            lead = -deficit
            if can_sacrifice or lead >= 8:
                return 0              # large lead or round is expendable
            return self.RANK_VALUES[sorted_hand[0]]  # protect a modest lead

        # Tied
        if deficit == 0:
            mid_cards = [c for c in sorted_hand if 3 <= self.RANK_VALUES[c] <= 7]
            if mid_cards:
                return self.RANK_VALUES[random.choice(mid_cards)]
            return self.RANK_VALUES[sorted_hand[0]]

        # Behind
        if winning_cards:
            best = winning_cards[0]
            # Round 1 only: don't burn a face card to gain a tiny edge
            if not critical and self.RANK_VALUES[best] >= 11 and deficit <= 3:
                return 0
            return self.RANK_VALUES[best]

        # Behind with no winning card
        if critical:
            return self.RANK_VALUES[sorted_hand[-1]]  # best effort
        if can_sacrifice or deficit >= 8:
            return 0                                   # concede, save cards
        return self.RANK_VALUES[sorted_hand[-1]]       # close the gap


# ============================================================================
# 2. NEURAL NETWORK WITH NATIVE ACTION MASKING
# ============================================================================
class MaskedActorCritic(nn.Module):
    def __init__(self, obs_dim=50, action_dim=14):
        super().__init__()
        # Shared Feature Extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        # Actor Head (Outputs raw logits for 14 actions)
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        # Critic Head (Outputs state value)
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs, action_mask=None):
        features = self.shared(obs)
        logits = self.actor(features)
        
        if action_mask is not None:
            # Replace invalid action logits with a large negative number
            # so their softmax probability becomes 0.
            logits = logits.masked_fill(action_mask == 0, -1e9)
            
        value = self.critic(features)
        return logits, value

    def get_action(self, obs, action_mask):
        logits, value = self.forward(obs, action_mask)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value


# ============================================================================
# 3. STATE INVERTER (Component A)
# ============================================================================
def invert_state(obs_tensor):
    """
    Swaps Player 1 and Player 2 specific features so the PPO agent can
    evaluate the state from the opponent's perspective.
    Expects obs_tensor of shape [Batch, 50].
    """
    inverted = obs_tensor.clone()
    
    # 0-12: My Hand (Cannot invert this perfectly if hand is hidden, 
    # but for BC training, you MUST save the bot's actual hand during data collection.
    # If this tensor represents the exact state the bot saw, you don't need to invert it.
    # We only invert if the environment logged the state from Agent's perspective.)
    
    # Assuming standard indices:
    # 13-25: P1 played cards
    # 26-38: P2 played cards
    inverted[:, 13:26] = obs_tensor[:, 26:39] # Put P2's played cards into "My Played"
    inverted[:, 26:39] = obs_tensor[:, 13:26] # Put P1's played cards into "Opp Played"
    
    # Index 39: P1 Wins, Index 40: P2 Wins
    inverted[:, 39] = obs_tensor[:, 40]
    inverted[:, 40] = obs_tensor[:, 39]
    
    # Index 41: P1 Deck Size, Index 42: P2 Deck Size
    inverted[:, 41] = obs_tensor[:, 42]
    inverted[:, 42] = obs_tensor[:, 41]
    
    # Index 43: P1 Passed, Index 44: P2 Passed
    inverted[:, 43] = obs_tensor[:, 44]
    inverted[:, 44] = obs_tensor[:, 43]
    
    return inverted


# ============================================================================
# 4. PHASE 1: BEHAVIORAL CLONING (Component B)
# ============================================================================
def train_behavioral_cloning(agent, expert_dataset, epochs=15, batch_size=256, lr=1e-3):
    """
    expert_dataset: A TensorDataset containing (observations, action_masks, actions)
    gathered from 10,000 games of BaselineBot vs BaselineBot.
    """
    print("Starting Phase 1: Behavioral Cloning...")
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    dataloader = DataLoader(expert_dataset, batch_size=batch_size, shuffle=True)
    
    agent.train()
    for epoch in range(epochs):
        total_loss = 0
        correct_preds = 0
        total_samples = 0
        
        for obs, mask, action in dataloader:
            optimizer.zero_grad()
            
            # Forward pass with action masking applied
            logits, _ = agent(obs, action_mask=mask)
            
            # Cross Entropy Loss
            loss = F.cross_entropy(logits, action)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Track Accuracy
            preds = torch.argmax(logits, dim=-1)
            correct_preds += (preds == action).sum().item()
            total_samples += action.size(0)
            
        accuracy = correct_preds / total_samples
        print(f"BC Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f} | Accuracy: {accuracy*100:.1f}%")
        
        # Early stopping to prevent overfitting to the script
        if accuracy >= 0.75:
            print("Reached target 75% accuracy. Stopping BC early to preserve exploration capability.")
            break
            
    print("Behavioral Cloning Complete. Model ready for PPO.\n")
    return agent


# ============================================================================
# 5. PHASE 2: PPO WITH AUXILIARY IMITATION LOSS (Component C)
# ============================================================================
def compute_ppo_fd_loss(agent, ppo_batch, bot_batch, lambda_weight=0.1, clip_ratio=0.2):
    """
    Computes the joint loss for a single PPO update step.
    
    ppo_batch: Dict containing PPO agent's rollout tensors (obs, masks, actions, log_probs, returns, advantages)
    bot_batch: Dict containing opponent's turn tensors (inverted_obs, masks, actions)
    """
    
    # --- 1. Standard PPO Loss Calculation ---
    agent_obs = ppo_batch['obs']
    agent_masks = ppo_batch['masks']
    agent_actions = ppo_batch['actions']
    old_log_probs = ppo_batch['log_probs']
    advantages = ppo_batch['advantages']
    returns = ppo_batch['returns']
    
    logits, values = agent(agent_obs, action_mask=agent_masks)
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    new_log_probs = dist.log_prob(agent_actions)
    entropy = dist.entropy().mean()
    
    # Policy Clipping
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()
    
    # Value Loss
    critic_loss = F.mse_loss(values.squeeze(-1), returns)
    
    # Combined Standard PPO Loss
    ppo_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
    
    
    # --- 2. Auxiliary Imitation Loss (Learning from the smart opponent) ---
    if bot_batch is not None and bot_batch['obs'].size(0) > 0:
        bot_obs = bot_batch['obs'] # These must already be inverted using invert_state()!
        bot_masks = bot_batch['masks']
        bot_actions = bot_batch['actions']
        
        # Get agent's predictions on the board states the opponent faced
        bot_logits, _ = agent(bot_obs, action_mask=bot_masks)
        
        # Supervised penalty: cross entropy against the opponent's chosen action
        imitation_loss = F.cross_entropy(bot_logits, bot_actions)
    else:
        imitation_loss = torch.tensor(0.0, device=ppo_loss.device)
        
        
    # --- 3. Joint Update ---
    total_loss = ppo_loss + (lambda_weight * imitation_loss)
    
    return total_loss, ppo_loss, imitation_loss


# ============================================================================
# USAGE EXAMPLE (How to integrate this into your training loop)
# ============================================================================
if __name__ == "__main__":
    # 1. Initialize Network
    agent = MaskedActorCritic(obs_dim=50, action_dim=14)
    
    # 2. Run Phase 1 (Assuming you have pre-saved tensor data)
    # mock data for compilation check
    mock_obs = torch.rand(1000, 50)
    mock_masks = torch.ones(1000, 14) 
    mock_actions = torch.randint(0, 14, (1000,))
    dataset = TensorDataset(mock_obs, mock_masks, mock_actions)
    
    agent = train_behavioral_cloning(agent, dataset, epochs=5)
    
    # 3. Phase 2 Setup (Inside your actual Rollout generation loop)
    # When generating rollouts against BaselineBot:
    # - If P1 (Agent) acts: Save to `ppo_batch` buffer
    # - If P2 (Baseline) acts: Invert the state with `invert_state(obs)`, 
    #   and save to `bot_batch` buffer.
    
    # Mocking a PPO update step
    ppo_batch = {
        'obs': torch.rand(64, 50),
        'masks': torch.ones(64, 14),
        'actions': torch.randint(0, 14, (64,)),
        'log_probs': torch.zeros(64),
        'advantages': torch.rand(64),
        'returns': torch.rand(64)
    }
    
    bot_batch = {
        'obs': torch.rand(16, 50), # 16 opponent moves recorded in this rollout
        'masks': torch.ones(16, 14),
        'actions': torch.randint(0, 14, (16,))
    }
    
    optimizer = optim.Adam(agent.parameters(), lr=3e-4)
    
    total_loss, ppo_loss, bc_loss = compute_ppo_fd_loss(agent, ppo_batch, bot_batch, lambda_weight=0.1)
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print(f"PPO Step Complete. Total Loss: {total_loss.item():.4f} (PPO: {ppo_loss.item():.4f}, BC: {bc_loss.item():.4f})")