"""Stop after the first complete declaration recognized by the existing parser."""

import torch

from sar.data.blocks import _BLOCK_TAG_RE
from sar.methods.daa import _FOCUS_RE
from sar.models.base import NonFiniteScoreError


class FiniteGenerationScores:
    """Reject numerical faults while allowing deliberately suppressed tokens."""

    def __call__(self, input_ids, scores):
        if (torch.isnan(scores).any() or torch.isposinf(scores).any()
                or not torch.isfinite(scores).any(dim=-1).all()):
            raise NonFiniteScoreError("non-finite generation scores")
        return scores


class DeclarationStoppingCriteria:
    def __init__(self, tokenizer, prompt_length: int, kind: str):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.pattern = {"focus": _FOCUS_RE, "blocks": _BLOCK_TAG_RE}[kind]

    def __call__(self, input_ids, scores, **kwargs):
        complete = [
            self.pattern.search(self.tokenizer.decode(
                row[self.prompt_length:], skip_special_tokens=True,
            )) is not None
            for row in input_ids
        ]
        return torch.tensor(complete, device=input_ids.device, dtype=torch.bool)
