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


def test_parse_declared_blocks_rejects_large_uncovered_suffix():
    text = (
        '<audio_blocks>\n'
        'B1|0.00|1.00|speech\n'
        'B2|1.00|1.40|bark\n'
        '</audio_blocks>'
    )
    with pytest.raises(ValueError, match='cover'):
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


def test_scan_displayed_endpoint_maps_to_true_audio_end():
    from sar.methods.daa import format_scan_prompt

    duration = 10.126
    assert '10.13 seconds' in format_scan_prompt(duration_s=duration, max_blocks=8)
    blocks = parse_audio_blocks(
        '<audio_blocks>\nB1|0|10.13|speech\n</audio_blocks>',
        duration_s=duration, max_blocks=8,
    )
    assert blocks[-1].end_s == duration


@pytest.mark.parametrize('duration,end', [(10.124, 10.13), (10.126, 10.131), (10.126, 11)])
def test_scan_endpoint_repair_does_not_accept_other_overflows(duration, end):
    with pytest.raises(ValueError, match='beyond audio duration'):
        parse_audio_blocks(
            f'<audio_blocks>\nB1|0|{end}|speech\n</audio_blocks>',
            duration_s=duration, max_blocks=8,
        )


def test_scan_endpoint_repair_does_not_change_intermediate_boundaries():
    with pytest.raises(ValueError, match='beyond audio duration'):
        parse_audio_blocks(
            '<audio_blocks>\nB1|0|10.13|speech\nB2|10|10.13|bark\n</audio_blocks>',
            duration_s=10.126, max_blocks=8,
        )


def test_scan_endpoint_repair_does_not_create_empty_span():
    with pytest.raises(ValueError, match='beyond audio duration'):
        parse_audio_blocks(
            '<audio_blocks>\nB1|0|10|speech\nB2|10.128|10.13|bark\n</audio_blocks>',
            duration_s=10.126, max_blocks=8,
        )


def test_scan_endpoint_repair_keeps_other_validation():
    with pytest.raises(ValueError, match='overlap'):
        parse_audio_blocks(
            '<audio_blocks>\nB1|0|8|speech\nB2|7|10.13|bark\n</audio_blocks>',
            duration_s=10.126, max_blocks=8,
        )


def test_scan_endpoint_repair_preserves_already_accepted_endpoint():
    blocks = parse_audio_blocks(
        '<audio_blocks>\nB1|0|10.13|speech\n</audio_blocks>',
        duration_s=10.1299995, max_blocks=8,
    )
    assert blocks[-1].end_s == 10.13
