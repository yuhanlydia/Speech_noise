import torch

from sar.models.qwen_hook import QACRController, qacr_eager_attention_forward
from sar.methods.qacr import QACRRouter


class FakeModule:
    num_key_value_groups = 1
    training = False


def test_qacr_eager_attention_gates_audio_without_renormalizing_text():
    module = FakeModule()
    query = torch.tensor([[[[1.0, 0.0]]]])
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    value = torch.tensor([[[[2.0, 0.0], [0.0, 4.0]]]])
    audio_mask = torch.tensor([[False, True]])
    gate = torch.tensor([[1.0, 0.0]])
    out, attn = qacr_eager_attention_forward(module, query, key, value, None, 1.0, audio_mask=audio_mask, gate=gate)
    expected_text_weight = torch.softmax(torch.tensor([1.0, 0.0]), dim=0)[0]
    assert torch.allclose(out[0, 0, 0], torch.tensor([2.0 * expected_text_weight, 0.0]), atol=1e-6)
    assert torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)))


def test_controller_computes_gate_once_on_prefill_and_reuses_for_decode():
    router = QACRRouter(query_dim=2, audio_dim=2, router_dim=1)
    controller = QACRController(router)
    controller.set_masks(
        audio_mask=torch.tensor([[True, False, False]]),
        query_mask=torch.tensor([[False, True, True]]),
    )
    hidden = torch.tensor([[[2.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
    gate1 = controller.compute_prefill_gate(hidden)
    gate2 = controller.gate_for_key_length(5)
    assert gate1.shape == (1, 3)
    assert gate2.shape == (1, 5)
    assert torch.allclose(gate2[:, :3], gate1)
    assert torch.allclose(gate2[:, 3:], torch.ones(1, 2))
