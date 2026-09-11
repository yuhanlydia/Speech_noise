import torch

from sar.models.qwen_omni import QwenOmniWrapper


def test_extend_prefill_mask_appends_false_candidate_tokens():
    mask = torch.tensor([[False, True, True, False]])
    out = QwenOmniWrapper.extend_prefill_mask(mask, 3)
    assert out.tolist() == [[False, True, True, False, False, False, False]]


def test_event_span_maps_only_event_audio_tokens():
    audio_mask = torch.tensor([[False, True, True, True, True, False]])
    event = QwenOmniWrapper.event_token_mask(audio_mask, duration_s=4.0, start_s=1.0, end_s=3.0)
    assert event.tolist() == [[False, False, True, True, False, False]]


def test_append_candidate_extends_input_and_attention_mask():
    base = {
        'input_ids': torch.tensor([[10, 20, 30]]),
        'attention_mask': torch.tensor([[1, 1, 1]]),
        'input_features': torch.tensor([[0.5]]),
        'position_ids': torch.tensor([[0, 1, 2]]),
    }
    option_ids = torch.tensor([[40, 41]])
    out = QwenOmniWrapper.append_candidate_inputs(base, option_ids)
    assert out['input_ids'].tolist() == [[10, 20, 30, 40, 41]]
    assert out['attention_mask'].tolist() == [[1, 1, 1, 1, 1]]
    assert 'position_ids' not in out
    assert out['input_features'] is base['input_features']


def test_routed_scoring_requires_loaded_model():
    from sar.methods.qacr import QACRRouter
    wrapper = QwenOmniWrapper.__new__(QwenOmniWrapper)
    wrapper.model = None; wrapper.processor = None
    wrapper.config = type('Cfg', (), {'routing_layer': 0})()
    router = QACRRouter(4, 4, 2)
    import pytest
    with pytest.raises(RuntimeError, match='load'):
        wrapper.score_options_qacr_tensors('x.wav', 'q', ['A', 'B'], router)


class _ToyTokenResult:
    def __init__(self, ids):
        self.input_ids = ids


class _ToyTokenizer:
    table = {'A':[1], 'B':[2], 'long':[3,4]}
    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = self.table[text]
        if return_tensors == 'pt':
            return _ToyTokenResult(torch.tensor([ids]))
        return _ToyTokenResult(ids)


def test_single_token_option_ids_accepts_canonical_mcq_and_rejects_multitoken():
    tok = _ToyTokenizer()
    ids = QwenOmniWrapper.single_token_option_ids(tok, ['A','B'])
    assert ids == [1,2]
    import pytest
    with pytest.raises(ValueError, match='single token'):
        QwenOmniWrapper.single_token_option_ids(tok, ['A','long'])
