from __future__ import annotations

import torch
from torch import nn


class LayerRouter(nn.Module):
    def __init__(self, query_dim: int, num_layers: int):
        super().__init__()
        self.proj = nn.Linear(query_dim, num_layers)

    def layer_weights(self, query_repr: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(query_repr))

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
