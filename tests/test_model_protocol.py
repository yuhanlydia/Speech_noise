import pytest

from sar.models.base import OptionScores, RoutingContext
from sar.models.qwen_omni import QwenOmniConfig, QwenOmniWrapper


def test_option_scores_predicted_index_and_gold_margin():
    scores = OptionScores(options=["A", "B", "C"], logprobs=[-2.0, -0.25, -1.0])
    assert scores.predicted_index == 1
    assert scores.predicted_option == "B"
    assert scores.gold_margin(1) == pytest.approx(0.75)


def test_routing_context_requires_boolean_audio_mask():
    import torch
    with pytest.raises(ValueError, match="audio_mask"):
        RoutingContext(
            query_repr=torch.zeros(1, 4),
            audio_repr=torch.zeros(1, 3, 4),
            audio_mask=torch.zeros(1, 3, dtype=torch.float32),
        )


def test_qwen_wrapper_fails_actionably_without_gpu_extra(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "transformers", None)
    wrapper = QwenOmniWrapper(QwenOmniConfig())
    with pytest.raises(RuntimeError, match=r"Install sar\[gpu\]"):
        wrapper.load()


def test_qwen_conversation_places_audio_before_query():
    convo = QwenOmniWrapper.build_conversation('a.wav', 'What animal is heard?')
    user = convo[-1]['content']
    assert user[0] == {'type': 'audio', 'path': 'a.wav'}
    assert user[1] == {'type': 'text', 'text': 'What animal is heard?'}


def test_find_subsequence_mask_marks_only_query_tokens():
    import torch
    mask = QwenOmniWrapper.find_subsequence_mask(torch.tensor([[9, 1, 2, 3, 8]]), [1, 2, 3])
    assert torch.equal(mask, torch.tensor([[False, True, True, True, False]]))
