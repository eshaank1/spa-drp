// docs/agent.js
// Client-side re-implementation of bc_ppo_fd's ActorCriticNetwork forward
// pass, so the trained agent (bc_ppo_fd/agent_final.pt) can run entirely in
// the browser. Weights are exported by scripts/export_agent_weights.py into
// agent_weights.json. No DOM access here; see app.js for wiring.

const Agent = (function () {
  // Must match CardGame.RANKS in game.js / GAME_RULES.md ordering.
  const RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'];

  // A must-win round: either the literal last round, or the opponent (P1)
  // already has 1 round win so losing this one ends the game. Cards held
  // past this point have zero future value, so pass is never optimal while
  // any remain.
  function isCriticalRound(state) {
    return state.currentRound >= 3 || state.roundsWon[0] === 1;
  }

  function cardValue(rank) {
    return RANKS.indexOf(rank) + 1; // A=1, 2=2, ..., K=13
  }

  function scoreOf(playedRanks) {
    return playedRanks.reduce((total, r) => total + cardValue(r), 0);
  }

  // Round 2, already up 1-0 (won round 1 outright): if the current deficit
  // already exceeds how much round 1 was won by, passing now guarantees a
  // round 3 (and game) win — see baseline_bot.py's decide_move for the full
  // argument. state.lastRoundResult still holds round 1's result at this
  // point since startNextRound (game.js) doesn't clear it.
  function shouldConserveForRound3(state) {
    if (state.currentRound !== 2 || state.roundsWon[1] !== 1 || state.roundsWon[0] !== 0) return false;
    const r1 = state.lastRoundResult;
    if (!r1 || r1.round !== 1) return false;
    const round1Margin = r1.p2Score - r1.p1Score; // agent is P2
    const deficit = scoreOf(state.p1Played) - scoreOf(state.p2Played); // opponent - me
    return deficit > round1Margin;
  }

  function buildMask(hand, state) {
    const mask = new Array(14).fill(0);
    mask[0] = state && isCriticalRound(state) ? 0 : 1;
    for (const rank of hand) {
      const idx = RANKS.indexOf(rank);
      if (idx !== -1) mask[idx + 1] = 1;
    }
    if (state && shouldConserveForRound3(state)) {
      mask.fill(0);
      mask[0] = 1; // force pass
    }
    return mask;
  }

  // Builds the 50-dim observation from Player 2's (the agent's) perspective,
  // matching rl_pettingzoo_env.py's _get_opponent_observation() /
  // play_vs_trained_agent.py's _get_obs_from_perspective(). The agent is
  // always Player 2 in this web demo; the human is always Player 1.
  function buildObservation(state) {
    const obs = new Array(50).fill(0);
    for (const rank of state.player2Hand) obs[RANKS.indexOf(rank)] = 1;
    for (const rank of state.p2Played) obs[13 + RANKS.indexOf(rank)] = 1;
    for (const rank of state.p1Played) obs[26 + RANKS.indexOf(rank)] = 1;

    const m = 39;
    obs[m + 0] = state.currentRound / 3.0;
    obs[m + 1] = state.roundsWon[1] / 2.0; // my (agent's) rounds won
    obs[m + 2] = state.roundsWon[0] / 2.0; // opponent's rounds won
    obs[m + 3] = 1.0; // it is my turn (always true at decision time)
    obs[m + 4] = 0.0; // opponent's-turn flag; always 0 since the agent is to-move at decision time
    obs[m + 5] = state.firstPlayer === 2 ? 1.0 : 0.0;
    obs[m + 6] = state.firstPlayer === 1 ? 1.0 : 0.0;
    obs[m + 7] = 0.0; // I have passed — false at decision time
    obs[m + 8] = state.passedPlayers.has(1) ? 1.0 : 0.0; // opponent has passed
    obs[m + 9] = state.player2Hand.length / 13.0;
    obs[m + 10] = state.player1Hand.length / 13.0;
    return obs;
  }

  function dot(a, b) {
    let total = 0;
    for (let i = 0; i < a.length; i++) total += a[i] * b[i];
    return total;
  }

  function linear(x, W, b) {
    return W.map((row, i) => dot(row, x) + b[i]);
  }

  function relu(v) {
    return v.map((x) => Math.max(0, x));
  }

  // Returns masked logits (invalid actions set to -1e9), matching
  // ActorCriticNetwork.forward()'s masked_fill behavior exactly.
  function forward(weights, obs, mask) {
    const h1 = relu(linear(obs, weights.w1, weights.b1));
    const h2 = relu(linear(h1, weights.w2, weights.b2));
    const logits = linear(h2, weights.w3, weights.b3);
    return logits.map((v, i) => (mask[i] ? v : -1e9));
  }

  function softmax(logits) {
    const max = Math.max(...logits);
    const exps = logits.map((v) => Math.exp(v - max));
    const total = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / total);
  }

  // Samples an action index from `probs` using `rng() -> [0,1)`.
  function sampleAction(probs, rng) {
    rng = rng || Math.random;
    const r = rng();
    let cumulative = 0;
    for (let i = 0; i < probs.length; i++) {
      cumulative += probs[i];
      if (r < cumulative) return i;
    }
    return probs.length - 1; // floating-point rounding fallback
  }

  // Picks the agent's (Player 2) move for the current state by sampling
  // from its policy, matching AgentBot(deterministic=False) — the same
  // stochastic behavior game_with_bots.py uses for human-vs-agent play.
  function chooseMove(weights, state, rng) {
    const hand = state.player2Hand;
    if (hand.length === 0) return { type: 'pass' };
    const obs = buildObservation(state);
    const mask = buildMask(hand, state);
    const logits = forward(weights, obs, mask);
    const probs = softmax(logits);
    const action = sampleAction(probs, rng);
    if (action === 0) return { type: 'pass' };
    return { type: 'card', rank: RANKS[action - 1] };
  }

  return {
    RANKS,
    isCriticalRound,
    shouldConserveForRound3,
    buildMask,
    buildObservation,
    forward,
    softmax,
    sampleAction,
    chooseMove,
  };
})();

if (typeof module !== 'undefined' && module.exports) {
  module.exports = Agent;
}
