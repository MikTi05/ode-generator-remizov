"""Read-only preparation audit: no solver, model fit, inference or JIT."""
from __future__ import annotations
import argparse,ast,csv,hashlib,json,struct,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnostics.final140_protocol import RELEASE,SOURCE,REVIEW,validate_scope
from diagnostics.prepare_development8 import write_json
from ml.revision_evidence import read,sha,verify_manifest


def verify(out):
    verify_manifest(RELEASE);verify_manifest(REVIEW)
    plan=read(out/'plan.json');validate_scope(plan)
    for name,h in plan['protected_sha256'].items():assert sha(ROOT/name)==h,name
    from diagnostics.prepare_final_selector_experiment import select
    audit=read(out/'independence.json')
    rows=[json.loads(l) for l in (SOURCE/'input_pool/all_methods_equations_v1_metadata.jsonl').open()]
    ds=sha(SOURCE/'input_pool/all_methods_equations_v1.csv')
    chosen=select(list(reversed(rows)),set(audit['excluded_topologies']),ds)
    assert [r['equation_id'] for r in chosen]==[c['equation_id'] for c in plan['cases']]
    assert [r['equation_id'] for r in chosen[::3]]==plan['MC_equation_ids']
    for row,c in zip(chosen,plan['cases']):
        assert row['expression_hash']==c['expression_hash'] and c['dataset_sha256']==ds
        assert c['identity']=='|'.join((ds,c['equation_id'],c['expression_hash']))
    manifest=list(csv.DictReader((out/'manifest.csv').open()))
    subset=list(csv.DictReader((out/'MC_subset20.csv').open()))
    assert [r['key'] for r in manifest]==[j['key'] for j in plan['jobs']]
    assert [r['equation_id'] for r in subset]==plan['MC_equation_ids']
    assert len(subset)==20 and len({(r['complexity_level'],r['regime']) for r in subset})==20
    addresses=set()
    for j in plan['jobs']:
        if j['profile']!='MC1':continue
        for x in j['target_x']:
            address=(j['config']['seed'],hashlib.sha256(b'mc_point_v1\0'+j['equation_id'].encode()+struct.pack('<d',0. if x==0 else x)).digest(),j['config']['stream_id'])
            assert address not in addresses;addresses.add(address)
    assert len(addresses)==660
    freeze=read(RELEASE/'release_freeze.json');prereg=read(SOURCE/'preregistration.json')
    assert freeze['before_final_holdout_inputs'] and prereg['model_frozen_before_inputs']
    assert prereg['release_freeze_sha256']==sha(RELEASE/'release_freeze.json')
    assert freeze['model_registry_sha256']==sha(RELEASE/'model/registry.json')
    assert freeze['model_manifest_sha256']==sha(RELEASE/'model/artifact_sha256.json')
    assert freeze['frozen_utc']<prereg['created_utc']<plan['created_utc']
    assert sha(out/'frozen_predictions.json')==sha(SOURCE/'frozen_predictions.json')
    # New controller uses the existing exact worker implementation.
    def body(path):
        tree=ast.parse(path.read_text());return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker'),include_attributes=False)
    assert body(ROOT/'diagnostics/run_final140.py')==body(ROOT/'diagnostics/run_final_selector_experiment.py')
    a=read(out/'approval_request.json')
    assert not a['approved'] and a['plan_sha256']==sha(out/'plan.json') and a['maximum_calls']==140 and a['series_seconds']==7464
    for folder in (out,SOURCE):
        assert not list((folder/'records').glob('*.json')) and not list((folder/'events').glob('*.request.json'))
        assert not (folder/'budget.json').exists() and not (folder/'execution_authorization.json').exists()
    assert not list((out/'workers').glob('*')) and not list((out/'sessions').glob('*'))
    # Entire pre-existing project snapshot, excluding only new preparation code.
    allowed={'PROJECT_STATUS.md','diagnostics/final140_protocol.py','diagnostics/run_final140.py','diagnostics/evaluate_final140.py'}
    before=read(out/'protected_before.json');changed=[n for n,h in before.items() if not (ROOT/n).exists() or sha(ROOT/n)!=h]
    assert not set(changed)-allowed,changed
    value=dict(passed=True,source_release_files_verified=len(read(RELEASE/'artifact_sha256.json')),
        budget_review_files_verified=len(read(REVIEW/'artifact_sha256.json')),preexisting_files_checked=len(before),allowed_preparation_changes=changed,
        cases=60,primary_R_calls=120,secondary_MC_calls=20,total_calls=140,MC_seed=plan['MC_seed'],
        MC_one_per_cell=True,unique_semantic_point_streams=len(addresses),MC_selection_reproduced_from_original_metadata=True,independence_preserved=True,
        frozen_model_predictions_inputs_unchanged=True,worker_AST_identical_to_original=True,
        profiles=dict(Counter(j['profile'] for j in plan['jobs'])),series_seconds=7464,
        actual_requests=0,actual_records=0,actual_workers=0,new_solver_calls=0,new_training=0,new_inference=0,
        production_blocked=True,execution_authorized=False,plan_sha256=sha(out/'plan.json'))
    write_json(out/'independent_verification.json',value);return value


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    print(json.dumps(verify(ap.parse_args().out.resolve()),ensure_ascii=False))
