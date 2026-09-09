import pytest
import torch

from sar.data.blocks import (
    AcousticBlock,
    blocks_to_audio_token_mask,
    fixed_temporal_blocks,
    parse_audio_blocks,
)


def test_parse_declared_blocks_and_validate_duration():
    text = (
        '<audio_blocks>\n'
        'B1|0.00|1.20|target speech\n'
        'B2|1.20|2.00|dog bark\n'
        '</audio_blocks>'
    )
    blocks = parse_audio_blocks(text, duration_s=2.0, max_blocks=4)
    assert [(b.block_id, b.start_s, b.end_s, b.label) for b in blocks] == [
        ('B1', 0.0, 1.2, 'target speech'),
        ('B2', 1.2, 2.0, 'dog bark'),
    ]


def test_parse_declared_blocks_rejects_overlap():
    text = (
        '<audio_blocks>\n'
        'B1|0.00|1.50|speech\n'
        'B2|1.20|2.00|bark\n'
        '</audio_blocks>'
    )
    with pytest.raises(ValueError, match='overlap'):
        parse_audio_blocks(text, duration_s=2.0, max_blocks=4)


def test_fixed_blocks_cover_duration_without_overflow():
    blocks = fixed_temporal_blocks(3.2, block_seconds=1.5, max_blocks=4)
    assert [(b.start_s, b.end_s) for b in blocks] == [
        (0.0, 1.5),
        (1.5, 3.0),
        (3.0, 3.2),
    ]


def test_blocks_map_to_only_selected_audio_tokens():
    audio_mask = torch.tensor(
        [[False, True, True, True, True, True, True, True, True, False]]
    )
    blocks = [
        AcousticBlock('B1', 0.0, 1.0, 'speech'),
        AcousticBlock('B2', 1.0, 2.0, 'bark'),
    ]
    selected = blocks_to_audio_token_mask(
        audio_mask,
        duration_s=2.0,
        blocks=blocks,
        selected_ids=['B2'],
    )
    assert torch.where(selected[0])[0].tolist() == [5, 6, 7, 8]
