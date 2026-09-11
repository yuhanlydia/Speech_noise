import pytest

from sar.data.blocks import AcousticBlock
from sar.methods.daa import format_focus_prompt, parse_focus_declaration


def test_focus_parser_accepts_multiple_valid_blocks():
    ids = parse_focus_declaration(
        '<focus_audio blocks="B2,B3">',
        valid_block_ids={'B1', 'B2', 'B3'},
        max_selected=2,
    )
    assert ids == ['B2', 'B3']


def test_focus_parser_rejects_unknown_block():
    with pytest.raises(ValueError, match='unknown'):
        parse_focus_declaration(
            '<focus_audio blocks="B9">',
            valid_block_ids={'B1', 'B2'},
            max_selected=2,
        )


def test_focus_prompt_contains_query_and_addressable_blocks():
    blocks = [
        AcousticBlock('B1', 0.0, 1.0, 'speech'),
        AcousticBlock('B2', 1.0, 2.0, 'dog bark'),
    ]
    prompt = format_focus_prompt('What animal is audible?', blocks, max_selected=1)
    assert 'B2' in prompt
    assert 'dog bark' in prompt
    assert 'What animal is audible?' in prompt
    assert '<focus_audio blocks=' in prompt
