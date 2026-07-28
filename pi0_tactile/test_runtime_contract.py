"""Lightweight runtime contract checks that do not instantiate pi0 weights."""
from __future__ import annotations

import numpy as np
import torch

from pi0_tactile.config import Pi0TactileConfig
from pi0_tactile.guidance import Pi0ActionAdapter
from pi0_tactile.prompt import PromptTokenizer


def fit_stat_dim(arr: np.ndarray, dim: int, pad_value: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.shape[0] >= dim:
        return arr[:dim]
    return np.concatenate([
        arr,
        np.full(dim - arr.shape[0], float(pad_value), dtype=np.float32),
    ])


def test_pi05_defaults_and_padding():
    cfg = Pi0TactileConfig(pi05=True, action_dim=32, robot_action_dim=7)
    assert cfg.max_token_len == 200
    adapter = Pi0ActionAdapter(model_action_dim=cfg.action_dim, robot_action_dim=cfg.robot_action_dim)
    action7 = torch.ones(2, 4, 7)
    action32 = adapter.pad_robot_action(action7)
    assert tuple(action32.shape) == (2, 4, 32)
    assert torch.allclose(action32[..., :7], action7)
    assert torch.count_nonzero(action32[..., 7:]) == 0


def test_ascii_pi05_prompt_includes_state():
    tokenizer = PromptTokenizer(max_token_len=200, pi05=True, backend="ascii")
    tokens, mask = tokenizer.tokenize_np("insert board", state=np.zeros(32, dtype=np.float32))
    assert tokens.shape == (200,)
    assert mask.shape == (200,)
    text = "".join(chr(int(t)) for t in tokens[mask])
    assert "State:" in text
    assert "Action:" in text
    assert len(text.split("State:", 1)[1].split(";", 1)[0].split()) == 32


def test_norm_helpers():
    stat = fit_stat_dim(np.array([1.0, 2.0], dtype=np.float32), 4, 0.0)
    assert stat.tolist() == [1.0, 2.0, 0.0, 0.0]
    x = (np.array([0.5], dtype=np.float32) - np.array([0.0])) / (np.array([1.0]) - np.array([0.0])) * 2 - 1
    assert np.allclose(x, np.array([0.0], dtype=np.float32))


def main():
    test_pi05_defaults_and_padding()
    test_ascii_pi05_prompt_includes_state()
    test_norm_helpers()
    print("pi0_tactile runtime contract tests passed")


if __name__ == "__main__":
    main()
