"""Recompute and cross-check complete matrix artifacts before reporting."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from sar.metrics import compute_pair_metrics
from sar.report import STAGES, export_v2_report

root = Path('results/v2_full_matrix')
out = Path('reports/run_v2_full_matrix')
out.mkdir(parents=True, exist_ok=True)
conditions = {k:v for k,v in STAGES.items() if k not in ('capability', 'hook_identity')}
all_rows, cohorts, report, hashes = {}, {}, {}, {}
manifest = {(r['pair_id'], r['role']): r for r in [json.loads(line) for line in Path('data/mvp_expanded_480/pairs.jsonl').read_text().splitlines()]}

def load_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]

def group(rows):
    d = {}
    for r in rows:
        p = d.setdefault(r['pair_id'], {})
        assert r['role'] not in p
        p[r['role']] = r
    assert all(set(p) == {'ignore','use'} for p in d.values())
    return d

def transitions(a, b):
    assert a.keys() == b.keys()
    aok = {k:all(r['correct'] for r in p.values()) for k,p in a.items()}
    bok = {k:all(r['correct'] for r in p.values()) for k,p in b.items()}
    rescue = sum(not aok[k] and bok[k] for k in aok)
    harm = sum(aok[k] and not bok[k] for k in aok)
    return dict(num_pairs=len(a), rescued_pairs=rescue, harmed_pairs=harm,
                both_correct=sum(aok[k] and bok[k] for k in aok),
                both_failed=sum(not aok[k] and not bok[k] for k in aok),
                net_gain=(rescue-harm)/len(a))

for model in ('3b','7b'):
    cap = load_rows(root/model/'mvp_capability_gate/results.jsonl')
    cohorts[model] = {r['pair_id'] for r in cap if r['eligible']}
    assert len(cap) == 480 and len({r['pair_id'] for r in cap}) == 480
    cap_summary=json.loads((root/model/'mvp_capability_gate/summary.json').read_text())
    assert cap_summary['eligible_pairs']==len(cohorts[model])
    assert abs(cap_summary['target_only_acc']-sum(r['target_only_correct'] for r in cap)/480)<1e-12
    assert abs(cap_summary['event_only_acc']-sum(r['event_only_correct'] for r in cap)/480)<1e-12
    identity = json.loads((root/model/'daa_identity/summary.json').read_text())
    assert identity['ok'] and identity['predictions_match']
    all_rows[model], report[model] = {}, {}
    for name, stage in conditions.items():
        stage_dir = root/model/stage
        rows = load_rows(stage_dir/'results.jsonl')
        for row in rows:
            expected = manifest[(row['pair_id'], row['role'])]
            assert row['waveform_sha256'] == expected['waveform_sha256']
            assert row['answer'] == expected['answer']
            assert row['correct'] == (row['prediction'] == row['answer'])
        grouped = group(rows)
        assert set(grouped) == cohorts[model]
        all_rows[model][name] = grouped
        actual = compute_pair_metrics(rows)
        stored = json.loads((stage_dir/'summary.json').read_text())
        for key,value in actual.items():
            assert abs(stored[key]-value) < 1e-12, (model,name,key)
        failures = {pid:p['ignore'].get('daa_error_stage') for pid,p in grouped.items() if p['ignore'].get('daa_error_stage')}
        assert all(bool(p['ignore'].get('daa_error_stage')) == bool(p['use'].get('daa_error_stage')) for p in grouped.values())
        valid = [p for p in grouped.values() if p['use'].get('event_selected') is not None and p['ignore'].get('ignore_selection_valid') is not None]
        selected = sum(bool(p['use']['event_selected']) and bool(p['ignore']['ignore_selection_valid']) for p in valid)
        report[model][name] = dict(actual,
            successful_declaration_pairs=None if name=='base' else len(grouped)-len(failures),
            failure_pairs=len(failures),
            failure_messages=dict(Counter(p['ignore'].get('daa_error') for p in grouped.values() if p['ignore'].get('daa_error_stage'))),
            selection_labeled_pairs=len(valid),
            selection_switch_acc_all_pairs=selected/len(grouped) if name != 'base' else None,
            selection_switch_acc_conditional=selected/len(valid) if valid else None,
            ignore_target_coverage_mean=stored.get('ignore_target_coverage_mean'))
        hashes[str(stage_dir/'results.jsonl')] = hashlib.sha256((stage_dir/'results.jsonl').read_bytes()).hexdigest()
    report[model]['paired_changes'] = {
        'oracle_prompt_vs_base':transitions(all_rows[model]['base'],all_rows[model]['oracle_prompt_only']),
        'oracle_kv_vs_base':transitions(all_rows[model]['base'],all_rows[model]['oracle_kv']),
        'oracle_kv_vs_prompt':transitions(all_rows[model]['oracle_prompt_only'],all_rows[model]['oracle_kv']),
        'fixed_vs_self_prompt':transitions(all_rows[model]['prompt_only'],all_rows[model]['fixed_daa']),
        'declared_vs_fixed':transitions(all_rows[model]['fixed_daa'],all_rows[model]['declared_daa']),
    }
    changes=report[model]['paired_changes']
    base_acc=report[model]['base']['pair_switch_acc']
    base_failed=sum(not all(r['correct'] for r in p.values()) for p in all_rows[model]['base'].values())
    report[model]['protocol_gain_metrics']={
        'oracle_total_gain':changes['oracle_kv_vs_base']['net_gain'],
        'oracle_kv_gain':changes['oracle_kv_vs_prompt']['net_gain'],
        'failure_rescue_fraction':changes['oracle_kv_vs_base']['net_gain']/(1-base_acc) if base_acc<1 else None,
        'actual_base_failure_rescue_fraction':changes['oracle_kv_vs_base']['rescued_pairs']/base_failed if base_failed else None,
        'self_kv_gain':changes['fixed_vs_self_prompt']['net_gain'],
    }
    for first, second in [('oracle_prompt_only','oracle_kv'), ('prompt_only','fixed_daa')]:
        fields = ('blocks','selected_blocks','raw_focus_declaration','raw_block_declaration','daa_error_stage')
        mismatches = []
        for pid,pair in all_rows[model][first].items():
            for role,row in pair.items():
                other = all_rows[model][second][pid][role]
                if any(row.get(f) != other.get(f) for f in fields):
                    mismatches.append([pid,role])
        report[model][first+'_declaration_mismatches_with_'+second] = mismatches
    export_v2_report(results_root=root/model, output_dir=out/model)

common = cohorts['3b'] & cohorts['7b']
assert common
shared = {'num_pairs':len(common),'model_eligible_counts':{m:len(c) for m,c in cohorts.items()},'metrics':{}, '7b_vs_3b_paired_changes':{}}
for model in ('3b','7b'):
    shared['metrics'][model] = {name:compute_pair_metrics([r for pid,p in pairs.items() if pid in common for r in p.values()]) for name,pairs in all_rows[model].items()}
for name in conditions:
    shared['7b_vs_3b_paired_changes'][name] = transitions(
        {k:v for k,v in all_rows['3b'][name].items() if k in common},
        {k:v for k,v in all_rows['7b'][name].items() if k in common})
report['shared_eligible'] = shared
report['interpretation'] = 'All metrics are descriptive on a post-hoc dev cohort; no performance-based stopping or selective omission. Selection conditional rates exclude failed declarations, while all-pair rates count those as unsuccessful.'
(out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
(out/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
source = {r['pair_id']:r for r in load_rows(Path('data/mvp_expanded_480/source.jsonl'))}
with (out/'pair_outcomes.csv').open('w') as f:
    fields = ['model','pair_id','shared_eligible','event_type','condition','waveform_sha256','ignore_correct','use_correct','ignore_prediction','use_prediction','failure_stage','ignore_valid_selection','use_event_selected','ignore_target_coverage']
    writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
    for model in ('3b','7b'):
        for name,pairs in all_rows[model].items():
            for pid,pair in pairs.items():
                i,u=pair['ignore'],pair['use']
                writer.writerow(dict(model=model,pair_id=pid,shared_eligible=pid in common,event_type=source[pid]['event_type'],condition=name,waveform_sha256=i['waveform_sha256'],ignore_correct=i['correct'],use_correct=u['correct'],ignore_prediction=i['prediction'],use_prediction=u['prediction'],failure_stage=i.get('daa_error_stage'),ignore_valid_selection=i.get('ignore_selection_valid'),use_event_selected=u.get('event_selected'),ignore_target_coverage=i.get('target_coverage')))
print(json.dumps({'all_stages_verified':True,'eligible_counts':shared['model_eligible_counts'],'shared_eligible':len(common)},indent=2))
