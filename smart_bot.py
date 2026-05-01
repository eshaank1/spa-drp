# Smart Bot

class SmartBot:
    """
    Smart Bot Strategy:
    - Plays lowest cards first
    - Always plays to get ahead with the smallest possible card
    - If already ahead and opponent just passed, pass too
    - If can't get ahead, pass ONLY if we won't lose the game
    - Last round: play all remaining cards
    """

    RANK_VALUES = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                   '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13}

    def decide_move(self, hand, player_score, opponent_score, is_last_round, opponent_just_played,
                    my_rounds_won=0, opponent_rounds_won=0):
        """
        Decide what move to make.

        Args:
            hand: list of cards in bot's hand
            player_score: bot's current score this round
            opponent_score: opponent's current score this round
            is_last_round: boolean, is this the final round of the game
            opponent_just_played: boolean, did opponent just play (vs pass)
            my_rounds_won: number of rounds I've won so far
            opponent_rounds_won: number of rounds opponent has won so far

        Returns:
            card to play (str) or 'PASS'
        """

        if not hand:
            return 'PASS'

        # Last round: play all cards (play lowest card to deplete hand)
        if is_last_round:
            return self._get_lowest_card(hand)

        # If already ahead and opponent just passed, pass too (no need to keep playing)
        if player_score > opponent_score and not opponent_just_played:
            return 'PASS'

        # Find cards that get us ahead (beat opponent's score)
        cards_to_win = [card for card in hand if self.RANK_VALUES[card] + player_score > opponent_score]

        if cards_to_win:
            # Play the smallest card that gets us ahead
            return min(cards_to_win, key=lambda c: self.RANK_VALUES[c])

        # If no cards get us ahead, only pass if we won't lose the game
        # If opponent already has 2 wins, we MUST try to win this round
        if opponent_rounds_won >= 2:
            # We're about to lose - play lowest card to try to get ahead
            return self._get_lowest_card(hand)
        
        # Otherwise safe to pass and save cards
        return 'PASS'

    def _get_lowest_card(self, hand):
        """Return the lowest card in hand"""
        return min(hand, key=lambda c: self.RANK_VALUES[c])
