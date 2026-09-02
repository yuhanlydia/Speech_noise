from __future__ import annotations

import math
import torch
from torch import nn

from .base import RoutingDecision
from sar.models.base import RoutingContext


class StaticAudioGate(nn.Module):
    def __init__(self, initial_gate: float = 1.0):
        super().__init__()
        if not 0 < initial_gate < 1:
            initial_gate = min(max(initial_gate, 1e-6), 1 - 1e-6)
        logit = math.log(initial_gate / (1 - initial_gate))
        self.logit = nn.Parameter(torch.tensor(float(logit)))

    def route(self, context: RoutingContext) -> RoutingDecision:
        g = torch.sigmoid(self.logit).to(context.audio_repr.dtype)
        gate = torch.where(context.audio_mask, torch.ones_like(context.audio_mask, dtype=context.audio_repr.dtype) * g, torch.ones_like(context.audio_mask, dtype=context.audio_repr.dtype))
        return RoutingDecision(gate=gate)

    def trainable_parameter_count(self) -> int:
        return 1
