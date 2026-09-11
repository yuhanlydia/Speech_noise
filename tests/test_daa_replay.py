import pytest

from sar.config import ExperimentConfig
from sar.daa_pipeline import run_daa_pair
from sar.data.schema import RelevancePairRecord
from sar.models.base import OptionScores
from sar.daa_replay import evaluate_replay_with_wrapper


def inputs():
    source = ExperimentConfig.model_validate({
        "model": {}, "data": {"manifest": "unchanged.jsonl"},
        "method": {"name": "daa", "layers": [], "daa": {
            "block_strategy": "fixed", "block_seconds": 1.0,
            "focus_source": "model", "apply_kv_mask": False,
        }}, "output_dir": "source",
    })
    target = source.model_copy(deep=True)
    target.method.daa.apply_kv_mask = True
    target.output_dir = "target"
    common = dict(pair_id="p", waveform_path="/audio.wav", waveform_sha256="hash",
                  answer="A", options=["A", "B"], event_type="dog",
                  source_mask_valid=True, source_start_s=1.0, source_end_s=2.0)
    records = [RelevancePairRecord(role="ignore", query="speech?", **common),
               RelevancePairRecord(role="use", query="event?", **common)]

    class Source:
        def select_audio_blocks(self, audio_path, query, blocks, **kwargs):
            selected = ["B1"] if query == "speech?" else ["B2"]
            return selected, f'<focus_audio blocks="{selected[0]}">'

        def score_single_token_options_daa(self, *args, **kwargs):
            return OptionScores(["A", "B"], [-2.0, -1.0])

    rows = run_daa_pair(Source(), records, duration_s=2.0, block_strategy="fixed",
                        block_seconds=1.0, max_blocks=8, max_focus_blocks=8,
                        scan_max_new_tokens=192, select_max_new_tokens=48, layers=[],
                        apply_kv_mask=False)
    return source, target, records, rows


class AnswersOnly:
    # No generation/selection methods: replay must only run the real answer pass.
    def score_single_token_options_daa(self, audio_path, query, options, blocks,
                                       selected_ids, *, layers, apply_kv_mask):
        assert selected_ids == (["B1"] if query == "speech?" else ["B2"])
        assert apply_kv_mask is True
        return OptionScores(options, [-0.1, -3.0])


def test_replay_recomputes_answers_with_identical_declarations():
    source, target, records, cached = inputs()
    rows, summary = evaluate_replay_with_wrapper(target, source, AnswersOnly(), records, cached, duration_resolver=lambda _: 2.0)
    assert [r["prediction"] for r in cached] == ["B", "B"]
    assert [r["prediction"] for r in rows] == ["A", "A"]
    assert [r["raw_focus_declaration"] for r in rows] == [r["raw_focus_declaration"] for r in cached]
    assert all(r["apply_kv_mask"] and r["selection_replayed"] for r in rows)
    assert summary["pair_switch_acc"] == 1.0
    assert summary["protocol_failure_pairs"] == 0


@pytest.mark.parametrize("mutation", ["model", "manifest", "blocks", "missing", "hash", "selection"])
def test_replay_rejects_incompatible_or_corrupt_cached_input(mutation):
    source, target, records, cached = inputs()
    if mutation == "model":
        target.model.model_id = "different/model"
    elif mutation == "manifest":
        target.data.manifest = "different.jsonl"
    elif mutation == "blocks":
        target.method.daa.block_seconds = 2.0
    elif mutation == "missing":
        cached.pop()
    elif mutation == "hash":
        cached[0]["waveform_sha256"] = "different"
    else:
        cached[0]["selected_blocks"] = ["B2"]
    with pytest.raises(ValueError):
        evaluate_replay_with_wrapper(target, source, AnswersOnly(), records, cached, duration_resolver=lambda _: 2.0)


def test_replay_preserves_pair_level_declaration_failures():
    source, target, records, cached = inputs()
    for row in cached:
        row.update(prediction=None, correct=False, selected_blocks=[], event_selected=None,
                   target_coverage=None, ignore_selection_valid=None,
                   daa_error_stage="focus_use", daa_error="missing focus declaration")
    rows, summary = evaluate_replay_with_wrapper(target, source, object(), records, cached, duration_resolver=lambda _: 2.0)
    assert all(r["prediction"] is None and not r["correct"] for r in rows)
    assert summary["protocol_failure_pairs"] == 1
    assert summary["protocol_failure_stages"] == {"focus_use": 1}
    assert summary["protocol_completion_rate"] == 0.0


def test_replay_does_not_convert_runtime_errors_to_negative_scores():
    source, target, records, cached = inputs()

    class Broken:
        def score_single_token_options_daa(self, *args, **kwargs):
            raise RuntimeError("broken attention")

    with pytest.raises(RuntimeError, match="broken attention"):
        evaluate_replay_with_wrapper(target, source, Broken(), records, cached, duration_resolver=lambda _: 2.0)


def test_replay_rejects_non_finite_answer_scores():
    source, target, records, cached = inputs()

    class NonFinite:
        def score_single_token_options_daa(self, *args, **kwargs):
            return OptionScores(["A", "B"], [float("nan"), -1.0])

    with pytest.raises(ValueError, match="non-finite"):
        evaluate_replay_with_wrapper(target, source, NonFinite(), records, cached,
                                     duration_resolver=lambda _: 2.0)


@pytest.mark.parametrize("mode", ["declared", "oracle"])
def test_replay_is_restricted_to_model_selected_fixed_blocks(mode):
    source, target, records, cached = inputs()
    if mode == "declared":
        source.method.daa.block_strategy = target.method.daa.block_strategy = mode
        for r in cached:
            r["block_strategy"] = mode
    else:
        source.method.daa.focus_source = target.method.daa.focus_source = mode
        for r in cached:
            r["focus_source"] = mode
    with pytest.raises(ValueError, match="model-selected fixed"):
        evaluate_replay_with_wrapper(target, source, AnswersOnly(), records, cached,
                                     duration_resolver=lambda _: 2.0)


def stored_inputs(tmp_path):
    import hashlib
    import json
    from sar import daa_smoke

    source, target, records, cached = inputs()
    source.output_dir = str(tmp_path / "source")
    target.output_dir = str(tmp_path / "target")
    directory = tmp_path / "source"
    directory.mkdir()
    artifact = directory / "results.jsonl"
    artifact.write_text("".join(json.dumps(r) + "\n" for r in cached))
    metadata = {
        "config": source.model_dump(),
        "records_sha256": daa_smoke.record_manifest_sha256(records),
        "results_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    (directory / "run_metadata.json").write_text(json.dumps(metadata))
    return source, target, records, cached, artifact


@pytest.mark.parametrize("mutation", ["other_results", "forged_model", "changed_query", "changed_results", "output_alias"])
def test_replay_checks_artifact_provenance_and_preserves_source(tmp_path, mutation):
    import json
    from sar import daa_replay

    source, target, records, cached, artifact = stored_inputs(tmp_path)
    original = artifact.read_bytes()
    if mutation == "other_results":
        other = tmp_path / "other.jsonl"
        other.write_bytes(original)
        artifact = other
    elif mutation == "forged_model":
        source.model.model_id = target.model.model_id = "different/model"
    elif mutation == "changed_query":
        records[0].query = "another question?"
    elif mutation == "changed_results":
        cached[0]["raw_focus_declaration"] = '<focus_audio blocks="B2">'
        artifact.write_text("".join(json.dumps(r) + "\n" for r in cached))
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path / "source", target_is_directory=True)
        target.output_dir = str(alias)
    before = artifact.read_bytes()
    with pytest.raises(ValueError):
        daa_replay.load_replay_source(target, source, records, artifact)
    assert artifact.read_bytes() == before


def test_replay_accepts_bound_unchanged_source_artifact(tmp_path):
    from sar import daa_replay

    source, target, records, cached, artifact = stored_inputs(tmp_path)
    assert daa_replay.load_replay_source(target, source, records, artifact) == cached
