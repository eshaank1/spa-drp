const test = require('node:test');
const assert = require('node:assert/strict');
const Bots = require('../../docs/bots.js');

// Regression test for the "opponent has passed while already ahead" branch,
// which used to return PASS even in a critical (must-win/final) round,
// leaving cards unplayed for no benefit.
test('baselineBotMove keeps playing when already ahead, opponent passed, and round is critical', () => {
  const move = Bots.baselineBotMove(
    ['2', '3'], // hand
    10, // myScore
    5, // oppScore (deficit = -5, we're ahead)
    3, // roundNum -> isLastRound
    0, // myWins
    0, // oppWins
    true // opponentHasPassed
  );
  assert.notEqual(move, 'PASS');
});

test('baselineBotMove still passes when already ahead, opponent passed, and round is NOT critical', () => {
  const move = Bots.baselineBotMove(
    ['2', '3'],
    10,
    5,
    1, // roundNum -> not last round
    0,
    0, // oppWins = 0 -> not must-win
    true
  );
  assert.equal(move, 'PASS');
});

test('baselineBotMove treats round 2 after a round-1 tie as critical', () => {
  const move = Bots.baselineBotMove(
    ['2', '3'],
    10,
    5,
    2, // roundNum
    1, // myWins (from the round-1 tie)
    1, // oppWins = 1 -> must-win
    true
  );
  assert.notEqual(move, 'PASS');
});

test('baselineBotMove conserves in round 2 when the deficit exceeds the round-1 margin', () => {
  const move = Bots.baselineBotMove(
    ['K', 'Q'], // even a card that could flip the current lead
    2, // myScore
    10, // oppScore -> deficit = 8
    2, // roundNum
    1, // myWins (won round 1)
    0, // oppWins
    false,
    5 // round1Margin: won round 1 by only 5, deficit (8) exceeds it
  );
  assert.equal(move, 'PASS');
});

test('baselineBotMove keeps fighting in round 2 when the deficit is still within the round-1 margin', () => {
  const move = Bots.baselineBotMove(
    ['K', 'Q'],
    2,
    10, // deficit = 8
    2,
    1,
    0,
    false,
    10 // round1Margin: won round 1 by 10, still ahead of the round-2 deficit
  );
  assert.notEqual(move, 'PASS');
});
