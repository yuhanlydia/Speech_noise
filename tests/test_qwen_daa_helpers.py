from types import SimpleNamespace

import pytest

from sar.models.qwen_omni_daa import QwenOmniDAAWrapper


def test_focus_tag_and_query_are_explicit():
    assert QwenOmniDAAWrapper.focus_tag(['B1', 'B3']) == '<focus_audio blocks="B1,B3">'
    text = QwenOmniDAAWrapper.focused_query('What animal?', ['B2'])
    assert 'What animal?' in text
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
