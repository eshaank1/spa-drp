# Play-vs-Agent Web Demo — Design

**Date:** 2026-07-21
**Status:** Approved

## Goal

Let anyone viewing the GitHub repo README play a full game of the card game against the trained BC+PPO-fD agent (`bc_ppo_fd/agent_final.pt`), directly in their browser, with no setup.

## Architecture

A static site in `docs/` served by GitHub Pages at `https://eshaank1.github.io/spa-drp/`, linked from the README. No backend, no build step — plain HTML/CSS/JS.

The trained network is exported once to a JSON file of its weights via a Python script. The browser loads that JSON and runs the forward pass itself — the network is small enough (`[50] → Linear+ReLU → [256] → Linear+ReLU → [256] → policy Linear → [14]`) that a hand-written matmul in JS is sufficient; no ML framework (TensorFlow.js, ONNX runtime) is needed. This mirrors what `agent_bot.py`'s `ActorCriticNetwork` does at inference time, reimplemented in JS.

Game logic (deck, hands, rounds, scoring, draw phases) is reimplemented in JS directly from `GAME_RULES.md` / `card_game.py`, since it must run entirely client-side with no server round-trip.

## Components

- **`docs/index.html`** — page shell: hand display, play/pass buttons, round & score display, win/loss tally, "play again" button.
- **`docs/style.css`** — clean/minimal styling: card-like buttons, color coding for round state (ahead/behind/tied), responsive layout for mobile.
- **`docs/game.js`** — pure game state machine, no DOM code: deck setup, turn order/alternation, playing/passing, round scoring, draw phase (3 cards after each round unless someone already has 2 wins), game-over detection (first to 2 round wins). Independently testable (e.g. in a browser console) since it has no rendering side effects.
- **`docs/agent.js`** — loads `agent_weights.json` once at startup; given the current game state, builds the 50-dim observation vector from the agent's perspective (mirrors `rl_pettingzoo_env.py`'s `_get_opponent_observation()` / `play_vs_trained_agent.py`'s `_get_obs_from_perspective`: 13 hand dims, 13 own-played dims, 13 opp-played dims, 11 metadata dims), builds the 14-dim action mask from the agent's hand, runs the masked forward pass, and **samples** from the resulting softmax distribution (matching `AgentBot(deterministic=False)`, the same behavior `game_with_bots.py` uses for human-vs-agent play).
- **`docs/app.js`** — wires `game.js` + `agent.js` to the DOM: renders state on every change, handles card/pass clicks, drives the agent's turn after the human's, and persists the session win/loss tally to `localStorage` (key e.g. `spa-drp-tally`, reset only if the user clears storage — no explicit reset button in v1).
- **`scripts/export_agent_weights.py`** — loads `ActorCriticNetwork` from `bc_ppo_fd/agent_final.pt` via the existing `bc_ppo_fd/ppo_auxiliary_loss.py` class, and dumps `backbone.{0,2}.{weight,bias}` and `policy_head.{weight,bias}` (value head is unused for gameplay and omitted) as plain nested JSON arrays to `docs/agent_weights.json`. Rerunnable any time the agent is retrained, to refresh the web demo.

## Data flow

1. Page load → `fetch('agent_weights.json')` → parse into JS arrays → init a fresh game: shuffle both independent 13-card decks, deal 7 cards each, randomly choose the first player for round 1.
2. Human's turn → click a card (or "Pass") → `game.js` applies the move, updates state.
3. Agent's turn → `agent.js` builds the obs vector + mask from current state → forward pass → masked softmax → sample → `game.js` applies the resulting move (card or pass).
4. Alternate turns (with each side's "already passed" rule) until both have passed for the round → score the round (higher sum wins; tie = both get a round win) → check game-over (first to 2 round wins) → if not over, draw 3 cards each (capped by deck remainder) → next round.
5. Game over → determine winner (higher round-win count; a 2-2 split is possible via tie rounds, first to *reach* 2 still decides it per existing rules) → update `localStorage` tally → show result + "Play again."

## Error handling

Everything is client-side with fixed-shape tensors (50-dim obs, 14-dim action space), so there isn't much that can go wrong at runtime. The one real failure mode is `agent_weights.json` failing to load (network hiccup on GitHub Pages) — show a plain "Couldn't load the agent — please refresh" message instead of a broken/frozen UI. Action masking guarantees the agent never proposes an illegal move, so no separate legality-validation layer is needed on its output.

## Testing

- **Numerical fidelity:** spot-check the JS forward pass against the real PyTorch model — same fixed observation vectors in, same logits out (within float tolerance) — to confirm the ported inference is faithful to `agent_final.pt`, not just plausible-looking.
- **Manual playtest:** play multiple full games through in a real browser (via the agent-browser skill) covering: winning 2-0, winning 2-1, a tied round, running out of cards, and the draw-phase card counts — checked against `GAME_RULES.md`.

## Out of scope (v1)

- Backend-hosted "real" PyTorch inference (rejected in favor of client-side JS — no server cost/maintenance).
- Move log / turn-by-turn commentary panel (deferred; only final round summaries shown).
- Explicit tally-reset control, difficulty levels, deterministic-mode toggle.
