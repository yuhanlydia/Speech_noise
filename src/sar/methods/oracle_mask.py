from __future__ import annotations

import torch

from .base import RoutingDecision
from sar.models.base import RoutingContext


class OracleMaskRouter:
    def route(self, context: RoutingContext) -> RoutingDecision:
        if context.relevance_mask is None:
            raise ValueError("OracleMaskRouter requires context.relevance_mask")
        if context.relevance_mask.shape != context.audio_mask.shape:
            raise ValueError("relevance_mask must match audio_mask")
        gate = context.relevance_mask.to(dtype=context.audio_repr.dtype)
        gate = torch.where(context.audio_mask, gate, torch.ones_like(gate))
        return RoutingDecision(gate=gate)

    def trainable_parameter_count(self) -> int:
        return 0
