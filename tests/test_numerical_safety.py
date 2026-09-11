import importlib.util
import json
import sys
from pathlib import Path

import pytest

from sar.config import ExperimentConfig
from sar.data.schema import RelevancePairRecord
from sar.models.base import OptionScores


def records():
    common=dict(pair_id='p',waveform_path='/audio.wav',waveform_sha256='hash',options=['A','B'],event_type='dog',source_start_s=1.0,source_end_s=2.0,source_mask_valid=True)
    return [RelevancePairRecord(role='ignore',query='speech?',answer='A',**common),RelevancePairRecord(role='use',query='event?',answer='B',**common)]


def config(tmp_path, **model):
    return ExperimentConfig.model_validate(dict(model=model,data={'manifest':'unused'},method={'name':'daa','layers':[0],'daa':{'block_strategy':'declared'}},output_dir=str(tmp_path/'out')))


@pytest.mark.parametrize('value',[float('nan'),float('inf'),float('-inf')])
@pytest.mark.parametrize('position',[0,1])
def test_option_scores_reject_nonfinite(value,position):
    from sar.models.base import NonFiniteScoreError
    values=[-1.,-2.];values[position]=value
    with pytest.raises(NonFiniteScoreError,match='non-finite'):
        OptionScores(['A','B'],values)


def test_base_numerical_error_never_uses_fallback(tmp_path):
    from sar.models.base import NonFiniteScoreError
    from sar.gpu_smoke import evaluate_base
    class Broken:
        def score_single_token_options(self,*args):
            return OptionScores(['A','B'],[float('nan'),-1.])
        def score_options(self,*args):
            pytest.fail('Numerical faults must not trigger fallback scoring')
    with pytest.raises(NonFiniteScoreError):
        evaluate_base(config(tmp_path),Broken(),records())


def test_capability_numerical_error_does_not_write_eligible(tmp_path,monkeypatch):
    from sar.models.base import NonFiniteScoreError
    from sar import validity_smoke
    source=tmp_path/'source.jsonl';source.write_text(json.dumps({'pair_id':'p','target_wav':'target.wav','event_wav':'event.wav'})+'\n')
    class Broken:
        def score_single_token_options(self,*args):
            return OptionScores(['A','B'],[float('nan'),-1.])
    monkeypatch.setattr(validity_smoke,'load_jsonl_records',lambda _:records())
    monkeypatch.setattr(validity_smoke,'_build_wrapper',lambda _:Broken())
    eligible=tmp_path/'eligible.jsonl'
    with pytest.raises(NonFiniteScoreError):
        validity_smoke.run_capability_gate(config(tmp_path),source_manifest=source,eligible_manifest=eligible)
    assert not eligible.exists()
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('stage',['declare','select','score'])
def test_daa_numerical_error_is_not_a_protocol_failure(tmp_path,stage):
    from sar.models.base import NonFiniteScoreError
    from sar.daa_smoke import evaluate_daa_with_wrapper
    from sar.data.blocks import AcousticBlock
    class Broken:
        def declare_audio_blocks(self,*args,**kwargs):
            if stage=='declare':raise NonFiniteScoreError('non-finite generation scores')
            return [AcousticBlock('B1',0,2,'audio')],'blocks'
        def select_audio_blocks(self,*args,**kwargs):
            if stage=='select':raise NonFiniteScoreError('non-finite generation scores')
            return ['B1'],'focus'
        def score_single_token_options_daa(self,*args,**kwargs):
            return OptionScores(['A','B'],[-1.,float('nan')])
    with pytest.raises(NonFiniteScoreError):
        evaluate_daa_with_wrapper(config(tmp_path),Broken(),records(),duration_resolver=lambda _:2.)


def test_identity_rejects_later_nan_after_score_mutation():
    from sar.models.base import NonFiniteScoreError
    from sar.hook_sanity import compare_score_vectors
    reference=OptionScores(['A','B'],[-1.,-2.]);masked=OptionScores(['A','B'],[-1.,-2.])
    masked.logprobs[1]=float('nan')
    with pytest.raises(NonFiniteScoreError):
        compare_score_vectors(reference,masked)


@pytest.mark.parametrize('module_name',['sar.validity_smoke','sar.gpu_smoke','sar.daa_smoke'])
def test_all_builders_pass_requested_compute_dtype(tmp_path,monkeypatch,module_name):
    import importlib
    import sar.models.qwen_omni_daa as daa_module
    module=importlib.import_module(module_name)
    class Fake:
        def __init__(self,cfg):self.config=cfg
        def load(self):pass
    if module_name=='sar.daa_smoke':monkeypatch.setattr(daa_module,'QwenOmniDAAWrapper',Fake)
    else:monkeypatch.setattr(module,'QwenOmniWrapper',Fake)
    cfg=config(tmp_path,torch_dtype='bfloat16')
    assert module._build_wrapper(cfg).config.torch_dtype=='bfloat16'


def test_identity_cli_passes_requested_compute_dtype(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('identity_cli',Path(__file__).parents[1]/'scripts/check_daa_identity.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    captured=[]
    class Fake:
        def __init__(self,cfg):captured.append(cfg)
        def load(self):pass
        def score_single_token_options_daa(self,*args,**kwargs):return OptionScores(['A','B'],[-1.,-2.])
    monkeypatch.setattr(module,'QwenOmniDAAWrapper',Fake)
    monkeypatch.setattr(module,'load_experiment_config',lambda _:config(tmp_path,torch_dtype='bfloat16'))
    monkeypatch.setattr(module,'load_jsonl_records',lambda _:records())
    monkeypatch.setattr(module.sf,'info',lambda _:type('Info',(),{'duration':2.})())
    monkeypatch.setattr(sys,'argv',['check_daa_identity.py','--output',str(tmp_path/'identity.json')])
    module.main()
    assert captured[0].torch_dtype=='bfloat16'


def test_existing_default_precision_is_preserved():
    from sar.config import ModelConfig
    assert ModelConfig().torch_dtype=='float16'


@pytest.mark.parametrize("values", [[float("nan"), 0.], [0., float("nan")], [float("inf"), 0.], [float("-inf"), float("-inf")]])
def test_generation_rejects_invalid_scores(values):
    import torch
    from sar.generation_stopping import FiniteGenerationScores
    from sar.models.base import NonFiniteScoreError
    with pytest.raises(NonFiniteScoreError):
        FiniteGenerationScores()(torch.tensor([[1]]),torch.tensor([values]))


def test_generation_allows_intentionally_suppressed_tokens():
    import torch
    from sar.generation_stopping import FiniteGenerationScores
    values=torch.tensor([[float("-inf"),0.]])
    assert FiniteGenerationScores()(torch.tensor([[1]]),values) is values


def test_free_form_daa_generation_restores_hook_on_numerical_fault(monkeypatch):
    import torch
    from types import SimpleNamespace
    import sar.models.qwen_omni_daa as module
    from sar.models.base import NonFiniteScoreError
    from sar.data.blocks import AcousticBlock
    restored=[]
    class Model:
        def generate(self,input_ids,**kwargs):
            scores=torch.tensor([[float('nan'),0.]])
            for processor in kwargs.get('logits_processor',[]):
                scores=processor(input_ids,scores)
            return input_ids
    w=module.QwenOmniDAAWrapper(config=None);w.model=Model()
    w.processor=SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda *args,**kwargs:'unchecked'))
    monkeypatch.setattr(w,'prepare_inputs',lambda *args:{'input_ids':torch.tensor([[1]])})
    monkeypatch.setattr(w,'_audio_duration',lambda _:2.)
    monkeypatch.setattr(w,'_daa_controller_for_inputs',lambda *args,**kwargs:object())
    monkeypatch.setattr(w,'_resolve_daa_layers',lambda _:[])
    monkeypatch.setattr(module,'install_daa_on_qwen_layers',lambda *args:lambda:restored.append(True))
    with pytest.raises(NonFiniteScoreError):
        w.generate_answer_daa('audio','question',[AcousticBlock('B1',0,2,'audio')],['B1'],layers=[],max_new_tokens=8)
    assert restored==[True]
