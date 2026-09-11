from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Sequence

import torch


class NonFiniteScoreError(RuntimeError):
    """Invalid numerical model output, never an incorrect answer or fallback."""


@dataclass(frozen=True)
class OptionScores:
    options: list[str]
    logprobs: list[float]

    def __post_init__(self) -> None:
        if not self.options or len(self.options) != len(self.logprobs):
            raise ValueError("options and logprobs must be non-empty and aligned")
        if not all(math.isfinite(value) for value in self.logprobs):
            raise NonFiniteScoreError("non-finite canonical option scores")

    @property
    def predicted_index(self) -> int:
        return max(range(len(self.logprobs)), key=self.logprobs.__getitem__)

    @property
    def predicted_option(self) -> str:
        return self.options[self.predicted_index]

    def gold_margin(self, gold_index: int) -> float:
        if not 0 <= gold_index < len(self.logprobs):
            raise IndexError(gold_index)
        wrong = max(v for i, v in enumerate(self.logprobs) if i != gold_index)
        return float(self.logprobs[gold_index] - wrong)


@dataclass
class RoutingContext:
    query_repr: torch.Tensor
    audio_repr: torch.Tensor
    audio_mask: torch.Tensor
    relevance_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.audio_mask.dtype is not torch.bool:
            raise ValueError("audio_mask must be a boolean tensor")
        if self.audio_repr.shape[:-1] != self.audio_mask.shape:
            raise ValueError("audio_mask must match audio_repr batch/sequence dimensions")


class AudioLMWrapper(ABC):
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def score_options(self, audio_path: str, query: str, options: Sequence[str]) -> OptionScores: ...

    @abstractmethod
    def get_routing_context(self, audio_path: str, query: str) -> RoutingContext: ...
