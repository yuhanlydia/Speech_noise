import pytest
import torch

from sar.train import qacr_record_loss


def test_qacr_record_loss_combines_qa_switch_and_identity_terms():
    scores = torch.tensor([0.0, 2.0], requires_grad=True)
    gate = torch.tensor([[1.0, 0.8, 0.2, 1.0]], requires_grad=True)
    audio_mask = torch.tensor([[False, True, True, False]])
    event_mask = torch.tensor([[False, False, True, False]])
    loss, parts = qacr_record_loss(
        scores=scores,
        gold_index=1,
        gate=gate,
        audio_mask=audio_mask,
        event_mask=event_mask,
        role='ignore',
        lambda_switch=1.0,
        lambda_identity=1.0,
    )
    expected_qa = torch.nn.functional.cross_entropy(scores[None, :], torch.tensor([1]))
    expected_switch = torch.nn.functional.binary_cross_entropy(torch.tensor([0.2]), torch.tensor([0.0]))
    expected_identity = torch.tensor((0.8 - 1.0) ** 2)
    assert torch.allclose(loss.detach(), expected_qa.detach() + expected_switch + expected_identity, atol=1e-6)
    assert set(parts) == {'qa', 'switch', 'identity'}
    loss.backward()
    assert scores.grad is not None
    assert gate.grad is not None


def test_use_role_targets_event_gate_to_one():
    scores = torch.tensor([2.0, 0.0])
    gate = torch.tensor([[0.25]])
    mask = torch.tensor([[True]])
    _, parts = qacr_record_loss(scores, 0, gate, mask, mask, 'use', 1.0, 0.0)
    expected = torch.nn.functional.binary_cross_entropy(torch.tensor([0.25]), torch.tensor([1.0]))
    assert parts['switch'] == pytest.approx(float(expected))
