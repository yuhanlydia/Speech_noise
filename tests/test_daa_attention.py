import math

import torch

from sar.methods.daa import daa_eager_attention_forward


class Dummy:
    num_key_value_groups = 1
    training = False


def test_focus_mask_is_applied_before_softmax_and_renormalizes():
    module = Dummy()
    q = torch.zeros(1, 1, 1, 1)
    k = torch.zeros(1, 1, 3, 1)
    v = torch.tensor([[[[1.0], [10.0], [100.0]]]])
    audio_mask = torch.tensor([[True, True, False]])
    allowed = torch.tensor([[True, False, False]])
    consumer = torch.tensor([[True]])
    out, weights = daa_eager_attention_forward(
        module,
        q,
        k,
        v,
        None,
        1.0,
        audio_mask=audio_mask,
        allowed_audio_mask=allowed,
        consumer_mask=consumer,
    )
    assert torch.allclose(
        weights[0, 0, 0],
        torch.tensor([0.5, 0.0, 0.5]),
        atol=1e-6,
    )
    assert torch.allclose(out[0, 0, 0], torch.tensor([50.5]), atol=1e-6)


def test_non_consumer_queries_retain_full_attention():
    module = Dummy()
    q = torch.zeros(1, 1, 1, 1)
    k = torch.zeros(1, 1, 3, 1)
    v = torch.tensor([[[[1.0], [10.0], [100.0]]]])
    audio_mask = torch.tensor([[True, True, False]])
    allowed = torch.tensor([[True, False, False]])
    consumer = torch.tensor([[False]])
    out, weights = daa_eager_attention_forward(
        module,
        q,
        k,
        v,
        None,
        1.0,
        audio_mask=audio_mask,
        allowed_audio_mask=allowed,
        consumer_mask=consumer,
    )
    assert torch.allclose(
        weights[0, 0, 0],
        torch.tensor([1 / 3, 1 / 3, 1 / 3]),
        atol=1e-6,
    )
    assert torch.allclose(out[0, 0, 0], torch.tensor([37.0]), atol=1e-6)


def test_allowing_all_audio_reduces_to_base_attention():
    module = Dummy()
    q = torch.randn(1, 1, 2, 3)
    k = torch.randn(1, 1, 4, 3)
    v = torch.randn(1, 1, 4, 3)
    audio_mask = torch.tensor([[True, True, False, False]])
    allowed = audio_mask.clone()
    consumer = torch.tensor([[True, True]])
    focused, _ = daa_eager_attention_forward(
        module,
        q,
        k,
        v,
        None,
        1 / math.sqrt(3),
        audio_mask=audio_mask,
        allowed_audio_mask=allowed,
        consumer_mask=consumer,
    )
    base, _ = daa_eager_attention_forward(
        module,
        q,
        k,
        v,
        None,
        1 / math.sqrt(3),
    )
    assert torch.allclose(focused, base, atol=1e-6)
