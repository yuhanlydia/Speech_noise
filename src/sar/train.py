from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass
class RouterTrainingBatch:
    query_repr: torch.Tensor
    audio_repr: torch.Tensor
    audio_mask: torch.Tensor
    relevance_mask: torch.Tensor | None = None


def freeze_backbone(backbone: nn.Module) -> None:
    for p in backbone.parameters():
        p.requires_grad_(False)


def routing_supervision_loss(gate: torch.Tensor, relevance_mask: torch.Tensor | None, identity_weight: float = 0.0) -> torch.Tensor:
    loss = gate.new_zeros(())
    if relevance_mask is not None:
        target = relevance_mask.to(dtype=gate.dtype)
        loss = loss + nn.functional.binary_cross_entropy(gate, target)
    if identity_weight:
        loss = loss + float(identity_weight) * torch.mean((gate - 1.0) ** 2)
    return loss


def train_router_epoch(router: nn.Module, batches: Iterable[RouterTrainingBatch], optimizer: torch.optim.Optimizer, identity_weight: float = 0.0) -> float:
    router.train(); total = 0.0; n = 0
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        gate = router(batch.query_repr, batch.audio_repr, batch.audio_mask)
        loss = routing_supervision_loss(gate, batch.relevance_mask, identity_weight)
        loss.backward(); optimizer.step()
        total += float(loss.detach().cpu()); n += 1
    if n == 0:
        raise ValueError('no training batches')
    return total / n


def qacr_record_loss(
    scores: torch.Tensor,
    gold_index: int,
    gate: torch.Tensor,
    audio_mask: torch.Tensor,
    event_mask: torch.Tensor | None,
    role: str,
    lambda_switch: float,
    lambda_identity: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervised MVP objective for one same-waveform relevance record.

    Event-span supervision is only valid when the event/source is temporally localizable;
    it is not treated as source separation for overlapping speech.
    """
    if scores.ndim != 1 or not 0 <= gold_index < scores.numel():
        raise ValueError("scores must be [options] with a valid gold_index")
    if role not in {"use", "ignore"}:
        raise ValueError("role must be use or ignore")
    qa = nn.functional.cross_entropy(scores[None, :], torch.tensor([gold_index], device=scores.device))
    switch = gate.new_zeros(())
    identity = gate.new_zeros(())
    if event_mask is not None:
        if event_mask.shape != gate.shape or audio_mask.shape != gate.shape:
            raise ValueError("event/audio masks must match gate shape")
        event = event_mask.to(gate.device) & audio_mask.to(gate.device)
        if event.any():
            target_value = 1.0 if role == "use" else 0.0
            target = torch.full_like(gate[event], target_value)
            switch = nn.functional.binary_cross_entropy(gate[event], target)
        protected = audio_mask.to(gate.device) & ~event_mask.to(gate.device)
        if protected.any():
            identity = torch.mean((gate[protected] - 1.0) ** 2)
    loss = qa + float(lambda_switch) * switch + float(lambda_identity) * identity
    parts = {
        "qa": float(qa.detach().cpu()),
        "switch": float(switch.detach().cpu()),
        "identity": float(identity.detach().cpu()),
    }
    return loss, parts
