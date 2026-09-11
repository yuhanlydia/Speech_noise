"""Exercise the DAA hook against installed Transformers, without model downloads."""

import pytest
import torch

from sar.models.daa_hook import DAAController, install_daa_on_qwen_attention


@pytest.mark.parametrize("exclude_audio", [False, True])
def test_real_qwen_hook_matches_native_attention_with_equivalent_mask(exclude_audio):
    qwen = pytest.importorskip(
        "transformers.models.qwen2_5_omni.modeling_qwen2_5_omni"
    )
    from transformers import Qwen2_5OmniTextConfig

    torch.manual_seed(0)
    config = Qwen2_5OmniTextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        head_dim=8,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "mrope_section": [1, 1, 2]},
    )
    config._attn_implementation = "eager"
    attention = qwen.Qwen2_5OmniAttention(config, layer_idx=0).eval()
    hidden = torch.randn(1, 5, 32)
    positions = torch.tensor([
        [[0, 1, 2, 3, 4]],
        [[0, 2, 4, 6, 8]],
        [[0, 3, 6, 9, 12]],
    ])
    embeddings = qwen.Qwen2_5OmniRotaryEmbedding(config)(hidden, positions)
    causal = torch.full((1, 1, 5, 5), torch.finfo(hidden.dtype).min).triu(1)
    native_mask = causal.clone()
    audio = torch.tensor([[False, True, True, False, False]])
    allowed = audio.clone()
    if exclude_audio:
        allowed[0, 2] = False
        native_mask[:, :, 3:, 2] = torch.finfo(hidden.dtype).min
    consumer = torch.tensor([[False, False, False, True, True]])
    controller = DAAController()
    controller.set_masks(audio, allowed, consumer)
    with torch.no_grad():
        expected, expected_weights = attention(
            hidden, attention_mask=native_mask, position_embeddings=embeddings
        )
        restore = install_daa_on_qwen_attention(attention, controller)
        try:
            actual, actual_weights = attention(
                hidden, attention_mask=causal, position_embeddings=embeddings
            )
        finally:
            restore()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_weights, expected_weights, atol=1e-6, rtol=1e-6)
