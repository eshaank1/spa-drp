// docs/bots.js
// Client-side ports of the three heuristic bots (baseline_bot.py, smart_bot.py,
// random_bot.py) so the web demo can offer them as alternative opponents to
// the trained Agent. Pure decision logic, no DOM access — same dual-load
// pattern as game.js/agent.js (browser <script> global `Bots`, or Node
// `require()` for tests).
//
// Each *Move function returns 'PASS' or a card rank string, for a uniform
// interface regardless of what the Python original returned (BaselineBot's
// Python version returns a numeric rank value instead of a card string, but
// since RANK_VALUES is a bijection this is the same decision, just
// represented consistently with the other two bots here).

const Bots = (function () {
  const RANK_VALUES = {
    A: 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
    '8': 8, '9': 9, '10': 10, J: 11, Q: 12, K: 13,
  };

  function lowestCard(hand) {
    return hand.reduce((min, c) => (RANK_VALUES[c] < RANK_VALUES[min] ? c : min));
  }

  function randomChoice(arr) {
    return arr[Math.floor(Math.random() * arr.length)];
  }

  // Mirrors smart_bot.py's SmartBot.decide_move.
  function smartBotMove(
    hand, playerScore, opponentScore, isLastRound, opponentJustPlayed,
    myRoundsWon = 0, opponentRoundsWon = 0
  ) {
    if (hand.length === 0) return 'PASS';
    if (isLastRound) return lowestCard(hand);
    if (playerScore > opponentScore && !opponentJustPlayed) return 'PASS';

    const cardsToWin = hand.filter((c) => RANK_VALUES[c] + playerScore > opponentScore);
    if (cardsToWin.length) return lowestCard(cardsToWin);

    if (opponentRoundsWon >= 2) return lowestCard(hand);
    return 'PASS';
  }

  // Mirrors random_bot.py's RandomBot.decide_move.
  function randomBotMove(
    hand, playerScore, opponentScore, isLastRound, opponentJustPlayed,
    myRoundsWon = 0, opponentRoundsWon = 0
  ) {
    if (hand.length === 0) return 'PASS';
    if (isLastRound) return randomChoice(hand);
    if (playerScore > opponentScore && !opponentJustPlayed) return 'PASS';

    const cardsToWin = hand.filter((c) => RANK_VALUES[c] + playerScore > opponentScore);
    if (cardsToWin.length) return randomChoice(cardsToWin);

    if (opponentRoundsWon >= 2) return randomChoice(hand);
    return 'PASS';
  }

  // Mirrors baseline_bot.py's BaselineBot.decide_move.
  function baselineBotMove(hand, myScore, oppScore, roundNum, myWins, oppWins, opponentHasPassed) {
    if (hand.length === 0) return 'PASS';

    const sortedHand = hand.slice().sort((a, b) => RANK_VALUES[a] - RANK_VALUES[b]);
    const deficit = oppScore - myScore;
    const mustWin = oppWins === 1;
    const isLastRound = roundNum >= 3;
    const critical = mustWin || isLastRound;
    const canSacrifice = myWins === 1 && oppWins === 0;
    const winningCards = sortedHand.filter((c) => RANK_VALUES[c] > deficit);

    if (opponentHasPassed) {
      if (deficit < 0) {
        if (critical) return sortedHand[0]; // final round: cards have no future value, keep padding score
        return 'PASS';
      }
      if (winningCards.length) return winningCards[0];
      if (critical) return sortedHand[sortedHand.length - 1];
      return 'PASS';
    }

    if (deficit < 0) {
      if (critical) return sortedHand[0];
      const lead = -deficit;
      if (canSacrifice || lead >= 8) return 'PASS';
      return sortedHand[0];
    }

    if (deficit === 0) {
      const midCards = sortedHand.filter((c) => RANK_VALUES[c] >= 3 && RANK_VALUES[c] <= 7);
      if (midCards.length) return randomChoice(midCards);
      return sortedHand[0];
    }

    if (winningCards.length) {
      const best = winningCards[0];
      if (!critical && RANK_VALUES[best] >= 11 && deficit <= 3) return 'PASS';
      return best;
    }

    if (critical) return sortedHand[sortedHand.length - 1];
    if (canSacrifice || deficit >= 8) return 'PASS';
    return sortedHand[sortedHand.length - 1];
  }

  return { RANK_VALUES, smartBotMove, randomBotMove, baselineBotMove };
})();

if (typeof module !== 'undefined' && module.exports) {
  module.exports = Bots;
}
