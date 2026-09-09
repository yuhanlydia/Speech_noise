import torch

from sar.models.daa_hook import DAAController, post_audio_consumer_mask


def test_post_audio_consumer_mask_only_marks_positions_after_last_audio_token():
    audio = torch.tensor([[False, True, True, False, False]])
    assert post_audio_consumer_mask(audio).tolist() == [
        [False, False, False, True, True]
    ]


def test_controller_extends_audio_masks_with_non_audio_generated_keys():
    audio = torch.tensor([[False, True, True, False]])
    allowed = torch.tensor([[False, False, True, False]])
    consumer = torch.tensor([[False, False, False, True]])
    ctl = DAAController()
    ctl.set_masks(audio, allowed, consumer)
    amask, allow = ctl.key_masks_for_length(6)
    assert amask.tolist() == [[False, True, True, False, False, False]]
    assert allow.tolist() == [[False, False, True, False, False, False]]


def test_controller_uses_prefill_consumer_mask_then_all_true_for_decode():
    audio = torch.tensor([[False, True, True, False]])
    allowed = torch.tensor([[False, True, False, False]])
    consumer = torch.tensor([[False, False, False, True]])
    ctl = DAAController()
    ctl.set_masks(audio, allowed, consumer)
    assert ctl.consumer_mask_for_query_length(4).tolist() == consumer.tolist()
    assert ctl.consumer_mask_for_query_length(1).tolist() == [[True]]
