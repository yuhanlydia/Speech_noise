from __future__ import annotations

import torch


class FixedKVSubspace:
    """Query-independent projection/suppression control from the prior KV line."""

    def __init__(self, basis: torch.Tensor, strength: float = 1.0):
        if basis.ndim != 2:
            raise ValueError("basis must have shape [hidden, rank]")
        q, _ = torch.linalg.qr(basis, mode="reduced")
        self.basis = q
        self.strength = float(strength)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        basis = self.basis.to(device=x.device, dtype=x.dtype)
        return x - self.strength * ((x @ basis) @ basis.transpose(-1, -2))

    def trainable_parameter_count(self) -> int:
        return 0
