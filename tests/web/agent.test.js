const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Agent = require('../../docs/agent.js');
const CardGame = require('../../docs/game.js');

const weights = JSON.parse(
  fs.readFileSync(path.join(__dirname, '../../docs/agent_weights.json'), 'utf8')
);
const fixtures = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures.json'), 'utf8'));

function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

test('forward pass matches the real PyTorch model on fixed fixtures', () => {
  for (const [i, c] of fixtures.entries()) {
    const logits = Agent.forward(weights, c.obs, c.mask);
    for (let j = 0; j < 14; j++) {
      assert.ok(
        Math.abs(logits[j] - c.logits[j]) < 1e-3,
        `fixture ${i}, action ${j}: JS=${logits[j]} py=${c.logits[j]}`
      );
    }
  }
});

test('buildMask marks pass and only the ranks present in hand', () => {
  const mask = Agent.buildMask(['K', '2']);
  assert.equal(mask[0], 1); // pass
  assert.equal(mask[Agent.RANKS.indexOf('K') + 1], 1);
  assert.equal(mask[Agent.RANKS.indexOf('2') + 1], 1);
  assert.equal(mask.reduce((a, b) => a + b, 0), 3); // pass + 2 cards, nothing else
});

test('buildObservation sets the agent (Player 2) hand/played bits correctly', () => {
  const state = CardGame.createGame(mulberry32(1));
  state.player2Hand = ['K', 'A'];
  state.p2Played = ['5'];
  state.p1Played = ['3'];
  const obs = Agent.buildObservation(state);
  assert.equal(obs[Agent.RANKS.indexOf('K')], 1);
  assert.equal(obs[Agent.RANKS.indexOf('A')], 1);
  assert.equal(obs[13 + Agent.RANKS.indexOf('5')], 1);
  assert.equal(obs[26 + Agent.RANKS.indexOf('3')], 1);
  assert.equal(obs[39 + 3], 1.0); // "it is my turn" is always 1 at decision time
});

test('chooseMove never proposes a card the agent does not hold', () => {
  const rng = mulberry32(99);
  for (let trial = 0; trial < 200; trial++) {
    const state = CardGame.createGame(rng);
    const move = Agent.chooseMove(weights, state, rng);
    if (move.type === 'card') {
      assert.ok(state.player2Hand.includes(move.rank));
    } else {
      assert.equal(move.type, 'pass');
    }
  }
});

test('chooseMove always passes when the agent has an empty hand', () => {
  const state = CardGame.createGame(mulberry32(5));
  state.player2Hand = [];
  const move = Agent.chooseMove(weights, state);
  assert.deepEqual(move, { type: 'pass' });
});
