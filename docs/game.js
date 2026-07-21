// docs/game.js
// Pure game-state logic for the card strategy game (see ../GAME_RULES.md).
// No DOM access here so this can be loaded as a <script> in the browser
// (defines the global `CardGame`) or required from Node for tests.

const CardGame = (function () {
  const RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'];
  const RANK_VALUES = {
    A: 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
    '8': 8, '9': 9, '10': 10, J: 11, Q: 12, K: 13,
  };

  function shuffle(array, rng) {
    const a = array.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = a[i];
      a[i] = a[j];
      a[j] = tmp;
    }
    return a;
  }

  function createGame(rng) {
    rng = rng || Math.random;
    const deck1 = shuffle(RANKS, rng);
    const deck2 = shuffle(RANKS, rng);
    const firstPlayer = rng() < 0.5 ? 1 : 2;
    return {
      player1Hand: deck1.slice(0, 7),
      player1Deck: deck1.slice(7),
      player2Hand: deck2.slice(0, 7),
      player2Deck: deck2.slice(7),
      roundsWon: [0, 0],
      currentRound: 1,
      firstPlayer,
      p1Played: [],
      p2Played: [],
      passedPlayers: new Set(),
      currentPlayer: firstPlayer,
      gameOver: false,
      winner: null,
      lastRoundResult: null,
    };
  }

  function opponent(player) {
    return player === 1 ? 2 : 1;
  }

  function handOf(state, player) {
    return player === 1 ? state.player1Hand : state.player2Hand;
  }

  function playedOf(state, player) {
    return player === 1 ? state.p1Played : state.p2Played;
  }

  // Resolves state.currentPlayer forward past any player who has already
  // passed this round. Returns the player who must act next, or null if
  // both players have passed (round is over).
  function actingPlayer(state) {
    let guard = 0;
    while (state.passedPlayers.has(state.currentPlayer)) {
      const other = opponent(state.currentPlayer);
      if (state.passedPlayers.has(other)) {
        return null;
      }
      state.currentPlayer = other;
      guard += 1;
      if (guard > 4) {
        throw new Error('actingPlayer failed to resolve (both players stuck)');
      }
    }
    return state.currentPlayer;
  }

  function sum(cards) {
    return cards.reduce((total, c) => total + RANK_VALUES[c], 0);
  }

  function drawPhase(state) {
    for (let i = 0; i < 3; i++) {
      if (state.player1Deck.length) state.player1Hand.push(state.player1Deck.shift());
      if (state.player2Deck.length) state.player2Hand.push(state.player2Deck.shift());
    }
  }

  function finishRound(state) {
    const p1Score = sum(state.p1Played);
    const p2Score = sum(state.p2Played);
    let winner;
    if (p1Score > p2Score) {
      state.roundsWon[0] += 1;
      state.firstPlayer = 1;
      winner = 1;
    } else if (p2Score > p1Score) {
      state.roundsWon[1] += 1;
      state.firstPlayer = 2;
      winner = 2;
    } else {
      state.roundsWon[0] += 1;
      state.roundsWon[1] += 1;
      winner = 'tie';
    }
    state.lastRoundResult = {
      round: state.currentRound,
      p1Score,
      p2Score,
      winner,
      roundsWon: state.roundsWon.slice(),
    };

    // A 2-2 split (reachable only via a tie round) is resolved the same way
    // the rest of this codebase resolves it (evaluate_agent_vs_baseline.py,
    // play_vs_trained_agent.py): a strict ">" comparison, so a tie at 2-2
    // does not count as a Player 1 win. Kept for consistency, not "fixed"
    // here since it's an existing project-wide convention.
    if (state.roundsWon[0] >= 2 || state.roundsWon[1] >= 2) {
      state.gameOver = true;
      state.winner = state.roundsWon[0] > state.roundsWon[1] ? 1 : 2;
      return;
    }

    state.p1Played = [];
    state.p2Played = [];
    state.passedPlayers = new Set();
    state.currentPlayer = state.firstPlayer;
    if (state.currentRound < 3) {
      drawPhase(state);
      state.currentRound += 1;
    }
  }

  // Applies a move for `player`. `move` is either {type:'pass'} or
  // {type:'card', rank}. An empty hand is automatically treated as a
  // forced pass regardless of the requested move. Mutates and returns
  // `state`.
  function applyMove(state, player, move) {
    if (state.gameOver) {
      throw new Error('game is already over');
    }
    const active = actingPlayer(state);
    if (active !== player) {
      throw new Error(`it is player ${active}'s turn, not ${player}`);
    }

    const hand = handOf(state, player);
    if (hand.length === 0 || move.type === 'pass') {
      state.passedPlayers.add(player);
    } else {
      const idx = hand.indexOf(move.rank);
      if (idx === -1) {
        throw new Error(`player ${player} does not have ${move.rank} in hand`);
      }
      hand.splice(idx, 1);
      playedOf(state, player).push(move.rank);
    }

    state.currentPlayer = opponent(player);

    if (actingPlayer(state) === null) {
      finishRound(state);
    }
    return state;
  }

  return {
    RANKS,
    RANK_VALUES,
    createGame,
    applyMove,
    actingPlayer,
    opponent,
    handOf,
    playedOf,
  };
})();

if (typeof module !== 'undefined' && module.exports) {
  module.exports = CardGame;
}
