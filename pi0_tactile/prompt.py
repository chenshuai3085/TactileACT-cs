"""Prompt tokenization helpers for pi0/pi0.5 tactile runs."""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

for _OPENPI in (
    os.environ.get("OPENPI_SRC", ""),
    os.path.join(os.path.dirname(_ROOT), "openpi", "src"),
    "/home/chenshuai/Project/openpi/src",
):
    if _OPENPI and os.path.exists(_OPENPI) and _OPENPI not in sys.path:
        sys.path.insert(0, _OPENPI)


@dataclass
class PromptTokenizer:
    """Small wrapper around the official PaliGemma tokenizer with an ASCII fallback."""

    max_token_len: int
    pi05: bool = False
    backend: str = "ascii"

    def __post_init__(self):
        self.backend = str(self.backend or "ascii").lower()
        self._tokenizer = None
        if self.backend in {"paligemma", "official", "openpi"}:
            from openpi.models.tokenizer import PaligemmaTokenizer

            self._tokenizer = PaligemmaTokenizer(max_len=int(self.max_token_len))
        elif self.backend != "ascii":
            raise ValueError(f"Unsupported tokenizer_backend: {self.backend}")

    def tokenize_np(
        self,
        prompt: str,
        state: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return int32 tokens and bool mask as NumPy arrays."""
        if self._tokenizer is not None:
            state_arg = state if self.pi05 else None
            tokens, mask = self._tokenizer.tokenize(prompt, state_arg)
            return tokens.astype(np.int32), mask.astype(bool)

        text = str(prompt).strip().replace("_", " ").replace("\n", " ")
        if self.pi05 and state is not None:
            state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
            discretized = np.digitize(
                state_arr,
                bins=np.linspace(-1, 1, 256 + 1, dtype=np.float32)[:-1],
            ) - 1
            state_str = " ".join(map(str, discretized.tolist()))
            text = f"Task: {text}, State: {state_str};\nAction: "
        elif not self.pi05:
            text = text + "\n"

        if self.backend == "ascii":
            logging.getLogger(__name__).debug(
                "Using ASCII prompt tokenizer placeholder. Use --tokenizer_backend paligemma "
                "for checkpoint-compatible pi0/pi0.5 runs."
            )

        ids = [ord(c) for c in text[: int(self.max_token_len)]]
        mask = [True] * len(ids)
        if len(ids) < self.max_token_len:
            pad_len = int(self.max_token_len) - len(ids)
            ids.extend([0] * pad_len)
            mask.extend([False] * pad_len)
        return np.asarray(ids, dtype=np.int32), np.asarray(mask, dtype=bool)

    def tokenize_torch(
        self,
        prompt: str,
        state: Optional[np.ndarray] = None,
        *,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, mask = self.tokenize_np(prompt, state)
        return (
            torch.as_tensor(tokens, dtype=torch.int32, device=device),
            torch.as_tensor(mask, dtype=torch.bool, device=device),
        )
