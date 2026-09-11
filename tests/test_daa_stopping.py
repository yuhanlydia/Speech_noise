import pytest
import torch

from sar.data.blocks import parse_audio_blocks
from sar.methods.daa import parse_focus_declaration
from sar.models.qwen_omni import QwenOmniConfig
from sar.models.qwen_omni_daa import QwenOmniDAAWrapper

class Characters:
    def decode(self, ids, **kwargs):
        return ''.join(chr(int(i)) for i in ids)

class ScriptedModel:
    def __init__(self, output):
        self.output = output
    def generate(self, input_ids, max_new_tokens, stopping_criteria=None, **kwargs):
        for c in self.output[:max_new_tokens]:
            input_ids = torch.cat([input_ids, torch.tensor([[ord(c)]])], dim=1)
            if stopping_criteria is not None and stopping_criteria(input_ids, None).all():
                break
        return input_ids


def wrapper(output, enabled):
    obj = QwenOmniDAAWrapper(QwenOmniConfig())
    obj.stop_on_complete_declaration = enabled
    obj.processor = type('Processor', (), {'tokenizer': Characters()})()
    obj.model = ScriptedModel(output)
    # A complete example in the prompt must never stop generation.
    obj.prepare_inputs = lambda *args: {'input_ids': torch.tensor([[ord(c) for c in '<focus_audio blocks="B8">']])}
    return obj

@pytest.mark.parametrize('enabled',[False,True])
def test_focus_stops_only_after_new_complete_tag(enabled):
    from sar.data.blocks import AcousticBlock
    tag = '<focus_audio blocks="B2">'
    obj=wrapper(tag+' trailing irrelevant output',enabled)
    ids, raw=obj.select_audio_blocks('audio','question',[AcousticBlock('B2',0,1,'sound')],max_selected=1,max_new_tokens=100)
    assert ids == ['B2']
    assert raw == (tag if enabled else tag+' trailing irrelevant output')


def test_blocks_stopping_preserves_first_table_and_parser_failure():
    from sar.generation_stopping import DeclarationStoppingCriteria
    prompt = '<audio_blocks>prompt example</audio_blocks>'
    for text in ['<audio_blocks>\nB1|0|2|speech\n</audio_blocks>tail', '<audio_blocks>malformed line</audio_blocks>tail']:
        stop=DeclarationStoppingCriteria(Characters(),len(prompt),'blocks')
        stopped=None
        for end in range(1,len(text)+1):
            ids=torch.tensor([[ord(c) for c in prompt+text[:end]]])
            if stop(ids,None).item():
                stopped=text[:end];break
        assert stopped and stopped.endswith('</audio_blocks>')
        def outcome(raw):
            try:return parse_audio_blocks(raw,duration_s=2,max_blocks=8)
            except ValueError as exc:return str(exc)
        assert outcome(stopped)==outcome(text)


def test_no_complete_tag_keeps_original_token_limit():
    obj=wrapper('no declaration here',True)
    with pytest.raises(ValueError,match='missing'):
        obj.select_audio_blocks('audio','question',[],max_selected=1,max_new_tokens=8)


def test_stopping_uses_same_first_match_as_focus_parser():
    from sar.generation_stopping import DeclarationStoppingCriteria
    raw='<focus_audio blocks=""> <focus_audio blocks="B2,B2"> later <focus_audio blocks="B1">'
    stop=DeclarationStoppingCriteria(Characters(),0,'focus')
    for end in range(1,len(raw)+1):
        if stop(torch.tensor([[ord(c) for c in raw[:end]]]),None).item():
            break
    for text in [raw[:end],raw]:
        with pytest.raises(ValueError,match='duplicate'):
            parse_focus_declaration(text,valid_block_ids={'B1','B2'},max_selected=8)


def test_config_supports_opt_in_stopping():
    from sar.config import DAAConfig
    assert DAAConfig(stop_on_complete_declaration=True).stop_on_complete_declaration
    assert not DAAConfig().stop_on_complete_declaration
