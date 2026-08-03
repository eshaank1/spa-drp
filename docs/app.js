// docs/app.js
// Wires game.js (pure state machine) + agent.js (policy inference) to the
// DOM. No game rules or network math live here.

(function () {
  const TALLY_KEY = 'spa-drp-tally';
  const OPPONENT_NAMES = {
    agent: 'Agent',
    baseline: 'BaselineBot',
    smart: 'SmartBot',
    random: 'RandomBot',
  };
  let weights = null;
  let state = null;
  let lastLoggedRound = 0;
  let opponentKind = 'agent';

  const el = {
    opponentSelect: document.getElementById('opponent-select'),
    opponentNames: document.querySelectorAll('.opp-name'),
    tallyYou: document.getElementById('tally-you'),
    tallyAgent: document.getElementById('tally-agent'),
    roundNum: document.getElementById('round-num'),
    roundsYou: document.getElementById('rounds-you'),
    roundsAgent: document.getElementById('rounds-agent'),
    agentHandCount: document.getElementById('agent-hand-count'),
    agentRoundScore: document.getElementById('agent-round-score'),
    youRoundScore: document.getElementById('you-round-score'),
    agentPlayed: document.getElementById('agent-played'),
    agentPassIndicator: document.getElementById('agent-pass-indicator'),
    youPlayed: document.getElementById('you-played'),
    yourHand: document.getElementById('your-hand'),
    passBtn: document.getElementById('pass-btn'),
    roundOver: document.getElementById('round-over'),
    roundOverMessage: document.getElementById('round-over-message'),
    nextRoundBtn: document.getElementById('next-round-btn'),
    log: document.getElementById('log'),
    gameOver: document.getElementById('game-over'),
    gameOverMessage: document.getElementById('game-over-message'),
    playAgainBtn: document.getElementById('play-again-btn'),
  };

  function loadTally() {
    try {
      const raw = localStorage.getItem(TALLY_KEY);
      return raw ? JSON.parse(raw) : { you: 0, agent: 0 };
    } catch (e) {
      return { you: 0, agent: 0 };
    }
  }

  function saveTally(tally) {
    try {
      localStorage.setItem(TALLY_KEY, JSON.stringify(tally));
    } catch (e) {
      // localStorage unavailable (private browsing, etc.) — tally just
      // won't persist across page loads.
    }
  }

  function logLine(text) {
    const p = document.createElement('div');
    p.textContent = text;
    el.log.prepend(p);
  }

  function cardEl(rank, tag, onClick) {
    const node = document.createElement(tag);
    node.className = 'card';
    node.textContent = rank;
    if (onClick) node.addEventListener('click', () => onClick(rank));
    return node;
  }

  function scoreOf(playedCards) {
    return playedCards.reduce((total, rank) => total + CardGame.RANK_VALUES[rank], 0);
  }

  // Resolves Player 2's move using whichever opponent is currently selected.
  // The trained Agent uses the fetched network weights; the three heuristic
  // bots (docs/bots.js) are derived straight from the current game state.
  function chooseOpponentMove(kind, gameState) {
    const hand = gameState.player2Hand;
    if (kind === 'agent') return Agent.chooseMove(weights, gameState);

    const myScore = scoreOf(gameState.p2Played);
    const oppScore = scoreOf(gameState.p1Played);
    const isLastRound = gameState.currentRound >= 3;
    const opponentJustPlayed = !gameState.passedPlayers.has(1);
    const myRoundsWon = gameState.roundsWon[1];
    const oppRoundsWon = gameState.roundsWon[0];

    // Round 1's result, still held in lastRoundResult while round 2 is in
    // progress (startNextRound doesn't clear it) — needed for the round-2
    // conservation rule.
    const r1 = gameState.lastRoundResult;
    const round1Margin = r1 && r1.round === 1 ? r1.p2Score - r1.p1Score : null;

    let result;
    if (kind === 'baseline') {
      result = Bots.baselineBotMove(
        hand, myScore, oppScore, gameState.currentRound, myRoundsWon, oppRoundsWon,
        gameState.passedPlayers.has(1), round1Margin
      );
    } else if (kind === 'random') {
      result = Bots.randomBotMove(
        hand, myScore, oppScore, isLastRound, opponentJustPlayed, myRoundsWon, oppRoundsWon
      );
    } else {
      result = Bots.smartBotMove(
        hand, myScore, oppScore, isLastRound, opponentJustPlayed, myRoundsWon, oppRoundsWon
      );
    }
    return result === 'PASS' ? { type: 'pass' } : { type: 'card', rank: result };
  }

  function render() {
    el.opponentNames.forEach((node) => {
      node.textContent = OPPONENT_NAMES[opponentKind];
    });
    el.roundNum.textContent = Math.min(state.currentRound, 3);
    el.roundsYou.textContent = state.roundsWon[0];
    el.roundsAgent.textContent = state.roundsWon[1];
    el.agentHandCount.textContent = state.player2Hand.length;
    el.agentRoundScore.textContent = scoreOf(state.p2Played);
    el.youRoundScore.textContent = scoreOf(state.p1Played);

    el.agentPassIndicator.hidden = !state.passedPlayers.has(2);
    el.agentPlayed.replaceChildren(...state.p2Played.map((r) => cardEl(r, 'span')));
    el.youPlayed.replaceChildren(...state.p1Played.map((r) => cardEl(r, 'span')));

    const isHumanTurn =
      !state.gameOver && CardGame.actingPlayer(state) === 1 && state.player1Hand.length > 0;

    el.yourHand.replaceChildren(
      ...state.player1Hand
        .slice()
        .sort((a, b) => CardGame.RANK_VALUES[a] - CardGame.RANK_VALUES[b])
        .map((r) => {
          const btn = cardEl(r, 'button', isHumanTurn ? onPlayCard : null);
          btn.disabled = !isHumanTurn;
          return btn;
        })
    );
    el.passBtn.disabled = !isHumanTurn;

    const tally = loadTally();
    el.tallyYou.textContent = tally.you;
    el.tallyAgent.textContent = tally.agent;

    if (state.roundOver && !state.gameOver) {
      const r = state.lastRoundResult;
      const oppName = OPPONENT_NAMES[opponentKind];
      const outcome =
        r.winner === 'tie' ? 'Tied the round.' : r.winner === 1 ? 'You won the round.' : `${oppName} won the round.`;
      el.roundOverMessage.textContent = `Round ${r.round} complete — You ${r.p1Score}, ${oppName} ${r.p2Score}. ${outcome}`;
      el.roundOver.hidden = false;
    } else {
      el.roundOver.hidden = true;
    }

    if (state.gameOver) {
      el.gameOver.hidden = false;
      el.gameOverMessage.textContent =
        state.winner === 1 ? 'Game Over — You won!' : `Game Over — ${OPPONENT_NAMES[opponentKind]} won!`;
    } else {
      el.gameOver.hidden = true;
    }
  }

  function maybeLogRoundResult() {
    const r = state.lastRoundResult;
    if (r && r.round !== lastLoggedRound) {
      lastLoggedRound = r.round;
      const outcome = r.winner === 'tie' ? 'Tied' : r.winner === 1 ? 'You won' : 'Agent won';
      logLine(`Round ${r.round}: You ${r.p1Score} — Agent ${r.p2Score}. ${outcome} the round.`);
    }
  }

  function runAutoMovesIfAny() {
    // Resolve any number of consecutive automatic turns: the agent's own
    // turns, plus a forced pass for the human whenever it's their turn but
    // their hand is empty (mirrors GAME_RULES.md's "no cards left -> forced
    // pass", which game.js already enforces for whichever side calls
    // applyMove — the human side just has no UI control to click when their
    // hand is empty, so we must submit that forced pass here instead).
    while (!state.gameOver) {
      const active = CardGame.actingPlayer(state);
      if (active === 2) {
        const move =
          state.player2Hand.length === 0 ? { type: 'pass' } : chooseOpponentMove(opponentKind, state);
        CardGame.applyMove(state, 2, move);
        maybeLogRoundResult();
      } else if (active === 1 && state.player1Hand.length === 0) {
        CardGame.applyMove(state, 1, { type: 'pass' });
        maybeLogRoundResult();
      } else {
        break;
      }
    }
  }

  function onPlayCard(rank) {
    CardGame.applyMove(state, 1, { type: 'card', rank });
    maybeLogRoundResult();
    afterHumanMove();
  }

  function onPass() {
    if (state.player1Hand.length === 0) return; // button should be disabled already
    CardGame.applyMove(state, 1, { type: 'pass' });
    maybeLogRoundResult();
    afterHumanMove();
  }

  function afterHumanMove() {
    runAutoMovesIfAny();
    if (state.gameOver) {
      const tally = loadTally();
      if (state.winner === 1) tally.you += 1;
      else tally.agent += 1;
      saveTally(tally);
    }
    render();
  }

  function onNextRound() {
    CardGame.startNextRound(state);
    runAutoMovesIfAny();
    render();
  }

  function startGame() {
    state = CardGame.createGame(Math.random);
    lastLoggedRound = 0;
    el.log.replaceChildren();
    logLine(`New game. ${state.firstPlayer === 1 ? 'You go' : `${OPPONENT_NAMES[opponentKind]} goes`} first.`);
    runAutoMovesIfAny(); // in case the agent goes first, or the human somehow starts hand-empty
    render();
  }

  el.passBtn.addEventListener('click', onPass);
  el.nextRoundBtn.addEventListener('click', onNextRound);
  el.playAgainBtn.addEventListener('click', startGame);
  el.opponentSelect.addEventListener('change', () => {
    opponentKind = el.opponentSelect.value;
    if (weights !== null) startGame(); // only once the initial load has completed
  });

  fetch('agent_weights.json')
    .then((res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    })
    .then((w) => {
      weights = w;
      startGame();
    })
    .catch((err) => {
      el.log.textContent = "Couldn't load the agent — please refresh.";
      console.error(err);
    });
})();
