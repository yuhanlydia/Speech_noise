from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from sar.models.base import RoutingContext


@dataclass
class RoutingDecision:
    gate: torch.Tensor
    metadata: dict[str, float] | None = None


class RoutingMethod(Protocol):
    def route(self, context: RoutingContext) -> RoutingDecision: ...
    def trainable_parameter_count(self) -> int: ...
