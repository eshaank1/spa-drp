"""
State Inverter Utility: Fast, vectorized PyTorch function to invert game observations.

When the heuristic bot plays a turn, its observation is from its own perspective 
(player 2). To use the bot's moves as training data for our agent (player 1), 
we must invert the board state by swapping perspectives.

Observation format (50-dim):
  [0-12]:   My hand (binary, 13 dims)
  [13-25]:  My cards played this round (binary, 13 dims)
  [26-38]:  Opponent cards played this round (binary, 13 dims)
  [39-49]:  Metadata (11 dims):
            [39]: current_round / 3.0
            [40]: my_wins / 2.0
            [41]: opp_wins / 2.0
            [42]: is_current_player==1
            [43]: is_current_player==2
            [44]: is_first_player==1
            [45]: is_first_player==2
            [46]: has_player_1_passed
            [47]: has_player_2_passed
            [48]: my_hand_size / 13.0
            [49]: opp_hand_size / 13.0
"""

import torch
from typing import Union


def invert_state(obs: Union[torch.Tensor, list]) -> torch.Tensor:
    """
    Invert a game observation from one player's perspective to the other's.
    
    When the opponent (player 2) makes a decision with obs from their perspective,
    we need to invert it to player 1's perspective to use as training data.
    
    Args:
        obs: Observation tensor of shape (50,) or (batch_size, 50).
            Can be a PyTorch tensor or a list.
    
    Returns:
        inverted_obs: Inverted observation tensor, same shape as input.
    
    Example:
        # Single observation inversion
        obs_p2 = torch.tensor([...])  # Player 2's observation
        obs_p1 = invert_state(obs_p2)  # Player 1's perspective
        
        # Batch inversion
        obs_batch = torch.tensor([[...], [...], ...])  # Shape: (batch_size, 50)
        inv_batch = invert_state(obs_batch)
    """
    # Convert to tensor if needed
    if isinstance(obs, list):
        obs = torch.tensor(obs, dtype=torch.float32)
    elif not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, dtype=torch.float32)
    else:
        obs = obs.float()
    
    # Handle both single and batch observations
    is_batched = obs.dim() == 2
    if not is_batched:
        obs = obs.unsqueeze(0)  # Add batch dimension: (50,) -> (1, 50)
    
    batch_size = obs.shape[0]
    inverted = obs.clone()
    
    # ===== CARD HAND SWAP =====
    # [0-12] (my hand) <-> [26-38] (opponent's played)
    # From player 2's perspective, "my hand" is what player 1 sees as "opp hand"
    # But player 1 doesn't see the opponent's hand in observations!
    # So: player 2's [0-12] (its hand) -> player 1 doesn't track this directly
    # However, inverted [0-12] should be what player 1 has in their own hand.
    #
    # The swap is:
    # - Player 2's observation [0-12] (what player 2 has) -> Player 1 observation [26-38]? NO
    # - Player 2's observation [26-38] (what player 1 played) -> Player 1 observation [13-25]
    # - Player 2's observation [13-25] (what player 2 played) -> Player 1 observation [26-38]
    #
    # Actually, let's think more carefully:
    # Player 2 obs format: [P2_hand | P2_played | P1_played | metadata]
    # Player 1 obs format: [P1_hand | P1_played | P2_played | metadata]
    #
    # So to go from P2's perspective to P1's:
    # - P2's [0-12] (P2's hand) should map to P1's [26-38]? NO - P1 never sees P2's hand
    # - P2's [13-25] (P2 played) should map to P1's [26-38] (P2 played from P1's view)
    # - P2's [26-38] (P1 played) should map to P1's [13-25] (P1 played from P1's view)
    #
    # But P1's hand [0-12] is not in P2's observation at all (by design).
    # So we must preserve P2's hand as the inverted hand.
    
    temp_hand = inverted[:, 0:13].clone()  # Player 2's hand
    temp_p2_played = inverted[:, 13:26].clone()  # Player 2's played cards
    temp_p1_played = inverted[:, 26:39].clone()  # Player 1's played cards (from P2's view)
    
    # Rearrange for Player 1's perspective:
    inverted[:, 0:13] = temp_hand  # P2's hand becomes P1's "hand" (after invert)
    inverted[:, 13:26] = temp_p1_played  # P1's played becomes P1's played in inverted
    inverted[:, 26:39] = temp_p2_played  # P2's played becomes P2's played in inverted
    
    # ===== METADATA SWAP =====
    # [39]: current_round - no swap needed (symmetric)
    # [40]: my_wins (P2's wins) <-> [41]: opp_wins (P1's wins)
    temp_my_wins = inverted[:, 40].clone()
    inverted[:, 40] = inverted[:, 41]  # P1's wins -> my_wins
    inverted[:, 41] = temp_my_wins  # P2's wins -> opp_wins
    
    # [42-43]: is_current_player (1 or 2) - swap the flags
    temp_cp1 = inverted[:, 42].clone()
    inverted[:, 42] = inverted[:, 43]  # is_current_player==2 -> is_current_player==1
    inverted[:, 43] = temp_cp1  # is_current_player==1 -> is_current_player==2
    
    # [44-45]: is_first_player (1 or 2) - swap the flags
    temp_fp1 = inverted[:, 44].clone()
    inverted[:, 44] = inverted[:, 45]  # is_first_player==2 -> is_first_player==1
    inverted[:, 45] = temp_fp1  # is_first_player==1 -> is_first_player==2
    
    # [46-47]: has_player_X_passed - swap the flags
    temp_p1_passed = inverted[:, 46].clone()
    inverted[:, 46] = inverted[:, 47]  # has_player_2_passed -> has_player_1_passed
    inverted[:, 47] = temp_p1_passed  # has_player_1_passed -> has_player_2_passed
    
    # [48-49]: hand_size - swap
    temp_p1_size = inverted[:, 48].clone()
    inverted[:, 48] = inverted[:, 49]  # opp_hand_size -> my_hand_size
    inverted[:, 49] = temp_p1_size  # my_hand_size -> opp_hand_size
    
    # Remove batch dimension if input was unbatched
    if not is_batched:
        inverted = inverted.squeeze(0)
    
    return inverted


if __name__ == "__main__":
    # Quick test
    print("Testing state_inverter...")
    
    # Create a dummy observation
    test_obs = torch.zeros(50)
    test_obs[0:5] = 1.0  # My hand has first 5 cards
    test_obs[13:18] = 1.0  # I played 5 cards
    test_obs[26:30] = 1.0  # Opponent played 4 cards
    test_obs[40] = 0.5  # My wins (1/2)
    test_obs[41] = 0.0  # Opponent wins (0/2)
    
    print(f"Original obs: {test_obs}")
    inverted_obs = invert_state(test_obs)
    print(f"Inverted obs: {inverted_obs}")
    
    # Test batch inversion
    batch_obs = torch.zeros(8, 50)
    batch_obs[:, 40] = 0.5
    batch_obs[:, 41] = 0.0
    batch_inverted = invert_state(batch_obs)
    print(f"Batch shape OK: {batch_inverted.shape == torch.Size([8, 50])}")
    print(f"Wins swapped in batch: {torch.allclose(batch_inverted[:, 40], torch.tensor(0.0)) and torch.allclose(batch_inverted[:, 41], torch.tensor(0.5))}")
