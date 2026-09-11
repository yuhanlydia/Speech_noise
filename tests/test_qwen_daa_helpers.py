from types import SimpleNamespace

import pytest

from sar.data.blocks import AcousticBlock
from sar.models.qwen_omni_daa import QwenOmniDAAWrapper


def test_focus_tag_and_query_include_same_address_table_for_prompt_control():
    assert QwenOmniDAAWrapper.focus_tag(['B1', 'B3']) == '<focus_audio blocks="B1,B3">'
    blocks = [
        AcousticBlock('B1', 0.0, 1.0, 'audio segment 1'),
        AcousticBlock('B2', 1.0, 2.0, 'audio segment 2'),
    ]
    text = QwenOmniDAAWrapper.focused_query('What animal?', blocks, ['B2'])
    assert 'What animal?' in text
    assert 'B1: 0.00-1.00s | audio segment 1' in text
    assert 'B2: 1.00-2.00s | audio segment 2' in text
    assert '<focus_audio blocks="B2">' in text


def test_resolve_empty_layer_list_means_all_layers():
    wrapper = QwenOmniDAAWrapper(config=None)
    wrapper.model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(self_attn='a'),
                SimpleNamespace(self_attn='b'),
            ]
        )
    )
    assert wrapper._resolve_daa_layers([]) == ['a', 'b']


def test_resolve_rejects_invalid_layer():
    wrapper = QwenOmniDAAWrapper(config=None)
    wrapper.model = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn='a')])
    )
    with pytest.raises(IndexError):
        wrapper._resolve_daa_layers([2])
