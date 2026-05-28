import random

class BaselineBot:
    """
    Upgraded Heuristic Bot:
    - Prioritizes card advantage (willing to lose Round 1 to save high cards).
    - Uses "Survival Mode" when the opponent is one win away from taking the game.
    - Probes with mid-low cards on ties to force the opponent to commit.
    - Refuses to burn Face cards (J, Q, K) early just to win a minor bidding war.
    """

    RANK_VALUES = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                   '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13}

    def decide_move(self, hand, my_score, opp_score, round_num,
                    my_wins, opp_wins, opp_just_played):
        """
        Decide what move to make based on game state and card advantage.
        """
        if not hand:
            return 'PASS'

        # Sort hand from lowest to highest value
        sorted_hand = sorted(hand, key=lambda c: self.RANK_VALUES[c])
        
        # 1. State Assessment
        must_win = (opp_wins == 1) # If they win this round, we lose the game
        is_last_round = (my_wins == 1 and opp_wins == 1) or (round_num >= 3)
        deficit = opp_score - my_score

        # 2. Survival Mode / Endgame Execution
        if must_win or is_last_round:
            # If we are already winning and they just passed, take the win
            if my_score > opp_score and not opp_just_played:
                return 'PASS'
            
            # Find all cards that put us strictly ahead
            winning_cards = [c for c in sorted_hand if self.RANK_VALUES[c] > deficit]
            
            if winning_cards:
                # Play the cheapest card that keeps us alive
                return winning_cards[0]
            else:
                # Hail Mary: we can't get ahead with one card. 
                # Play our highest card to close the gap and hope we have more cards left.
                return sorted_hand[-1]

        # 3. Strategic Round Abandonment (Conserving Cards)
        # If we are trailing by 10+ points early in the game, let them have the round.
        # They overcommitted, so we take the card advantage into the next rounds.
        if deficit >= 10 and not must_win:
            return 'PASS'

        # 4. Opening / Tied Game
        if my_score == opp_score:
            # Don't open with an Ace (too weak) or a King (too valuable).
            # Probe with a mid-low card (3 through 7) to test the waters.
            mid_cards = [c for c in sorted_hand if 3 <= self.RANK_VALUES[c] <= 7]
            if mid_cards:
                return random.choice(mid_cards) # Add slight randomness to avoid predictability
            return sorted_hand[0]

        # 5. Over-topping Efficiently
        if my_score < opp_score:
            winning_cards = [c for c in sorted_hand if self.RANK_VALUES[c] > deficit]
            
            if winning_cards:
                card_to_play = winning_cards[0]
                
                # RESOURCE CHECK: If the only way to win Round 1 is to burn a Face card (11+),
                # and we aren't in survival mode, just pass and save it for later.
                if self.RANK_VALUES[card_to_play] >= 11 and round_num == 1:
                    return 'PASS'
                
                return card_to_play
            else:
                # We can't beat their score with a single card.
                # Instead of throwing away a card for nothing, fold the round.
                return 'PASS'

        # 6. We are currently ahead
        if my_score > opp_score:
            # If they passed, we pass to claim the round win.
            if not opp_just_played:
                return 'PASS'
            # If they played but are STILL behind us (e.g., they played a tiny card),
            # pass and force them to play another card to catch up.
            return 'PASS'