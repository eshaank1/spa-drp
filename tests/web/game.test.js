const test = require('node:test');
const assert = require('node:assert/strict');
const CardGame = require('../../docs/game.js');

function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function freshState(overrides) {
  const base = {
    player1Hand: [],
    player1Deck: [],
    player2Hand: [],
    player2Deck: [],
    roundsWon: [0, 0],
    currentRound: 1,
    firstPlayer: 1,
    p1Played: [],
    p2Played: [],
    passedPlayers: new Set(),
    currentPlayer: 1,
    gameOver: false,
    winner: null,
    lastRoundResult: null,
  };
  return Object.assign(base, overrides);
}

test('createGame deals 7+6 to each independent 13-card deck', () => {
  const state = CardGame.createGame(mulberry32(42));
  assert.equal(state.player1Hand.length, 7);
  assert.equal(state.player1Deck.length, 6);
  assert.equal(state.player2Hand.length, 7);
  assert.equal(state.player2Deck.length, 6);
  assert.deepEqual(
    [...state.player1Hand, ...state.player1Deck].sort(),
    [...CardGame.RANKS].sort()
  );
  assert.deepEqual(
    [...state.player2Hand, ...state.player2Deck].sort(),
    [...CardGame.RANKS].sort()
  );
  assert.ok(state.firstPlayer === 1 || state.firstPlayer === 2);
});

test('playing a card moves it from hand to played and switches the turn', () => {
  const state = freshState({ player1Hand: ['K', '2'], player2Hand: ['5'] });
  CardGame.applyMove(state, 1, { type: 'card', rank: 'K' });
  assert.deepEqual(state.player1Hand, ['2']);
  assert.deepEqual(state.p1Played, ['K']);
  assert.equal(CardGame.actingPlayer(state), 2);
});

test('both players passing ends the round and scores it', () => {
  const state = freshState({
    player1Hand: ['K'],
    player2Hand: ['2'],
    player1Deck: [],
    player2Deck: [],
  });
  CardGame.applyMove(state, 1, { type: 'card', rank: 'K' }); // P1: 13
  CardGame.applyMove(state, 2, { type: 'card', rank: '2' }); // P2: 2
  CardGame.applyMove(state, 1, { type: 'pass' });
  CardGame.applyMove(state, 2, { type: 'pass' });
  assert.equal(state.roundsWon[0], 1);
  assert.equal(state.roundsWon[1], 0);
  assert.equal(state.lastRoundResult.winner, 1);
  assert.equal(state.currentRound, 2);
});

test('a tied round awards both players a round win', () => {
  const state = freshState({ player1Hand: ['5'], player2Hand: ['5'] });
  CardGame.applyMove(state, 1, { type: 'card', rank: '5' });
  CardGame.applyMove(state, 2, { type: 'card', rank: '5' });
  CardGame.applyMove(state, 1, { type: 'pass' });
  CardGame.applyMove(state, 2, { type: 'pass' });
  assert.equal(state.roundsWon[0], 1);
  assert.equal(state.roundsWon[1], 1);
  assert.equal(state.lastRoundResult.winner, 'tie');
});

test('game ends as soon as a player reaches 2 round wins, skipping the draw', () => {
  const state = freshState({
    roundsWon: [1, 0],
    currentRound: 2,
    player1Hand: ['K'],
    player2Hand: ['2'],
    player1Deck: ['3'],
    player2Deck: ['4'],
  });
  CardGame.applyMove(state, 1, { type: 'card', rank: 'K' });
  CardGame.applyMove(state, 2, { type: 'card', rank: '2' });
  CardGame.applyMove(state, 1, { type: 'pass' });
  CardGame.applyMove(state, 2, { type: 'pass' });
  assert.equal(state.gameOver, true);
  assert.equal(state.winner, 1);
  // draw phase must NOT have run since the game already ended
  assert.deepEqual(state.player1Deck, ['3']);
  assert.deepEqual(state.player2Deck, ['4']);
});

test('an empty hand is auto-treated as a forced pass', () => {
  const state = freshState({ player1Hand: [], player2Hand: ['5'] });
  CardGame.applyMove(state, 1, { type: 'card', rank: 'K' }); // no K in hand -> forced pass
  assert.ok(state.passedPlayers.has(1));
  assert.deepEqual(state.p1Played, []);
});

test('playing a card not in hand throws', () => {
  const state = freshState({ player1Hand: ['5'], player2Hand: ['2'] });
  assert.throws(() => CardGame.applyMove(state, 1, { type: 'card', rank: 'K' }));
});

test('acting out of turn throws', () => {
  const state = freshState({ player1Hand: ['5'], player2Hand: ['2'], currentPlayer: 1 });
  assert.throws(() => CardGame.applyMove(state, 2, { type: 'card', rank: '2' }));
});

test('if a player has already passed, the other player keeps acting alone', () => {
  const state = freshState({ player1Hand: ['K'], player2Hand: ['2', '3'] });
  CardGame.applyMove(state, 1, { type: 'pass' });
  assert.equal(CardGame.actingPlayer(state), 2);
  CardGame.applyMove(state, 2, { type: 'card', rank: '2' });
  // P1 already passed, so P2 must act again immediately
  assert.equal(CardGame.actingPlayer(state), 2);
});
