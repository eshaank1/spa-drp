"""Export bc_ppo_fd/agent_final.pt weights + numerical test fixtures for the
web demo in docs/.

Rerun this any time the agent is retrained, to refresh the browser demo:
    python3 scripts/export_agent_weights.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bc_ppo_fd.ppo_auxiliary_loss import ActorCriticNetwork

MODEL_PATH = PROJECT_ROOT / "bc_ppo_fd" / "agent_final.pt"
WEIGHTS_OUT = PROJECT_ROOT / "docs" / "agent_weights.json"
FIXTURES_OUT = PROJECT_ROOT / "tests" / "web" / "fixtures.json"


def export_weights(model: ActorCriticNetwork) -> dict:
    sd = model.state_dict()
    return {
        "w1": sd["backbone.0.weight"].tolist(),
        "b1": sd["backbone.0.bias"].tolist(),
        "w2": sd["backbone.2.weight"].tolist(),
        "b2": sd["backbone.2.bias"].tolist(),
        "w3": sd["policy_head.weight"].tolist(),
        "b3": sd["policy_head.bias"].tolist(),
    }


def make_fixtures(model: ActorCriticNetwork, num_cases: int = 8, seed: int = 7) -> list:
    """Random (obs, mask) -> logits cases computed by the real model, used to
    verify the ported JS forward pass produces identical logits."""
    rng = np.random.default_rng(seed)
    cases = []
    model.eval()
    with torch.no_grad():
        for _ in range(num_cases):
            obs = rng.random(50).astype(np.float32)
            mask = np.zeros(14, dtype=np.float32)
            mask[0] = 1.0  # pass always valid
            num_cards = int(rng.integers(1, 8))
            card_idx = rng.choice(np.arange(1, 14), size=num_cards, replace=False)
            mask[card_idx] = 1.0

            obs_t = torch.tensor(obs).unsqueeze(0)
            mask_t = torch.tensor(mask).unsqueeze(0)
            logits, _ = model(obs_t, mask_t)
            cases.append({
                "obs": obs.tolist(),
                "mask": mask.tolist(),
                "logits": logits.squeeze(0).tolist(),
            })
    return cases


def main():
    model = ActorCriticNetwork(obs_dim=50, action_dim=14, hidden_dim=256, num_hidden_layers=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(WEIGHTS_OUT, "w") as f:
        json.dump(export_weights(model), f)
    print(f"Wrote {WEIGHTS_OUT}")

    FIXTURES_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(FIXTURES_OUT, "w") as f:
        json.dump(make_fixtures(model), f)
    print(f"Wrote {FIXTURES_OUT}")


if __name__ == "__main__":
    main()
