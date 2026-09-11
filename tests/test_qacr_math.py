import torch

from sar.methods.qacr import (
    QACRRouter,
    decompose_attention_contributions,
    routed_attention_output,
)


def fixture():
    attn = torch.tensor([[[[0.1, 0.2, 0.3, 0.4]]]], dtype=torch.float32)
    values = torch.tensor([[[[1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 4.0]]]], dtype=torch.float32)
    audio_mask = torch.tensor([[False, True, True, False]])
    return attn, values, audio_mask


def test_gate_one_recovers_base_attention_output():
    attn, values, audio_mask = fixture()
    base = torch.matmul(attn, values)
    gate = torch.ones(1, 4)
    routed = routed_attention_output(attn, values, audio_mask, gate)
    assert torch.allclose(routed, base, atol=1e-6)


def test_changing_audio_gate_leaves_text_contribution_unchanged():
    attn, values, audio_mask = fixture()
    text_a, audio_a = decompose_attention_contributions(attn, values, audio_mask, torch.ones(1, 4))
    text_b, audio_b = decompose_attention_contributions(attn, values, audio_mask, torch.zeros(1, 4))
    assert torch.allclose(text_a, text_b)
    assert not torch.allclose(audio_a, audio_b)


def test_non_audio_positions_are_never_gated():
    attn, values, audio_mask = fixture()
    gate = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    routed = routed_attention_output(attn, values, audio_mask, gate)
    text_only, _ = decompose_attention_contributions(attn, values, audio_mask, gate)
    assert torch.allclose(routed, text_only)


def test_same_audio_different_query_can_flip_gates():
    router = QACRRouter(query_dim=2, audio_dim=2, router_dim=1)
    with torch.no_grad():
        router.query_proj.weight[:] = torch.tensor([[1.0, 0.0]])
        router.audio_proj.weight[:] = torch.tensor([[1.0, 0.0]])
    audio = torch.tensor([[[2.0, 0.0], [-2.0, 0.0]]])
    mask = torch.tensor([[True, True]])
    q_pos = torch.tensor([[2.0, 0.0]])
    q_neg = torch.tensor([[-2.0, 0.0]])
    g_pos = router(q_pos, audio, mask)
    g_neg = router(q_neg, audio, mask)
    assert g_pos[0, 0] > 0.9 and g_neg[0, 0] < 0.1
    assert g_pos[0, 1] < 0.1 and g_neg[0, 1] > 0.9
