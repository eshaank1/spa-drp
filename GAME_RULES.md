# Card Game Rules

## Objective

Win 2 out of 3 rounds by accumulating higher card values than your opponent in strategic bidding rounds.

---

## Game Setup

### Deck Composition
- **Standard 13-card deck** per player: A, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K
- Each player has their own independent deck (no shared deck)
- **Card values:** A=1, 2-10=face value, J=11, Q=12, K=13

### Starting Hand
- Each player draws **7 cards** from their 13-card deck
- The remaining **6 cards** stay in their personal deck
- **First player is randomly selected** for Round 1

---

## Round Mechanics

### Structure
A game consists of **up to 3 rounds**. The first player to win 2 rounds wins the game.

### Playing a Round

#### Turn Order
1. Players alternate taking turns, starting with the designated first player
2. On each turn, a player must either:
   - **Play a card** from their hand (adds to their round total)
   - **Pass** (no card is played, and they cannot play again that round)

#### Passing & Round End
- Once both players have passed, the round ends
- A player who passes cannot play again in that round
- If a player has no cards left in their hand, they are forced to pass

#### Round Duration
- Rounds continue with alternating plays until both players have passed
- This means:
  - Both playing → Player 1 passes → Player 2 plays again → Player 1 has passed → Player 2 passes → Both have now passed → **Round ends**
  - Or: Both players pass consecutively → **Round ends**

### Card Drawing

After each round completes (before the next round begins):
- Each player draws **3 cards** from their personal deck (if available)
- Players who have fewer than 3 remaining cards only draw what's available
- This ensures players have more cards for subsequent rounds

**Exception:** If a player has already won 2 rounds, the game ends immediately—no drawing occurs.

---

## Scoring

### Per-Round Scoring
1. Sum all cards played by each player during the round
2. Player with the **higher total score wins the round**
3. In case of a **tie**, **both players receive 1 round win**

### Example Round
```
Round 1:
  Player 1 plays: K, 5, 2 → Total: 13 + 5 + 2 = 20
  Player 2 plays: Q, 9, 3 → Total: 12 + 9 + 3 = 24
  Result: Player 2 wins (24 > 20)
```

### Game Outcome
First player to reach **2 round wins** wins the game.

**Possible final scores:**
- 2-0 (win 2 rounds straight)
- 2-1 (win 2 out of 3 rounds)
- 2-2 (all 3 rounds played, but only first to 2 wins counts)

---

## Strategic Elements

### Decision Points
Each turn requires a strategic decision:
- **Play a strong card now** to build lead early in the round
- **Play a weak card** to test opponent's response
- **Pass strategically** to:
  - Preserve strong cards for future plays
  - Force opponent to commit more cards
  - Control round pacing

### Incomplete Information
- Players **cannot see** cards in opponent's hands
- Players **cannot predict** opponent's future plays
- Decisions must account for uncertainty about remaining opponent cards

### Card Management
- Limited deck (13 cards per player total)
- Cards drawn between rounds are limited (3 per round)
- Strategic hand management becomes critical as the game progresses

### Psychological Elements
- Opponent behavior reveals information about hand composition
- Passing patterns can signal hand strength or weakness
- Card play sequencing affects future round dynamics

---

## Game Flow Example

```
Setup:
  Player 1 hand: K, Q, 7, 4, 2, A, 5 (7 cards)
  Player 2 hand: J, 10, 9, 6, 3, 2, 8 (7 cards)
  First player: Player 1

Round 1:
  Player 1 plays: Q (12) | Player 2 plays: J (11) | Running: P1=12, P2=11
  Player 1 plays: K (13) | Player 2 plays: 10 (10) | Running: P1=25, P2=21
  Player 1 passes | Player 2 plays: 9 (9) | Running: P1=25, P2=30
  Player 1 has passed | Player 2 passes | ROUND ENDS
  Final: Player 2 wins 30-25
  Scores: Player 1: 0 wins, Player 2: 1 win

Drawing Phase:
  Player 1 draws 3 cards from remaining deck
  Player 2 draws 3 cards from remaining deck

Round 2 & 3:
  Continue similarly until one player reaches 2 round wins

Game End:
  First player to 2 wins: WINNER!
```

---

## Special Cases

### All Cards Played
If both players play all their cards before both passing:
- Round ends when cards are exhausted
- Scoring proceeds normally

### Tied Rounds
Both players get a round win if their total scores are equal.
- Example: If both total 25, both get +1 round win
- This can lead to a 2-2 scenario before the 2-win threshold is reached

### No Cards in Hand
- Player automatically passes (cannot choose otherwise)
- Opponent can continue playing if they have cards

---

## Reinforcement Learning Context

### Action Space
The game is represented as a discrete action space:
- **Action 0:** Pass
- **Actions 1-13:** Play specific card rank (A, 2, 3, ..., K)

### Invalid Actions
- Attempting to play a card not in hand: **Masked to valid alternatives**
- The RL agent learns to avoid predicting invalid actions through:
  - Training with action masking
  - Penalty for invalid attempts
  - Automatic correction to valid alternatives

### Observation Space
The agent receives a 50-dimensional observation vector:
- **13 dimensions:** Cards in agent's hand
- **13 dimensions:** Cards agent has played this round
- **13 dimensions:** Cards opponent has played this round
- **11 dimensions:** Game metadata (round number, round wins, first player, passes, deck sizes)

---

## Win Conditions

### For Human Players
Win 2 rounds and achieve the highest final score.

### For RL Agents
Trained to maximize cumulative reward, which includes:
- Reward for winning rounds
- Penalty for losing rounds
- Scaling based on score differential
- Penalty for invalid actions (training only)

