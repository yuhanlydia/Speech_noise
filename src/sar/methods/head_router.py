from __future__ import annotations

import torch
from torch import nn


class HeadRouter(nn.Module):
    def __init__(self, query_dim: int, num_heads: int):
        super().__init__()
        self.proj = nn.Linear(query_dim, num_heads)

    def head_weights(self, query_repr: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(query_repr))

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
