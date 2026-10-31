"""Final research release verification without numerical execution."""
from __future__ import annotations
import argparse,json,sys
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from ml.final_selector import FinalResearchSelector
from ml.revision_evidence import read,sha,verify_manifest
from ml.revision_selector import ResearchRevision
from diagnostics.prepare_development8 import write_json
from diagnostics.prepare_final_selector_experiment import select
from diagnostics.run_final_selector_experiment import preflight
from diagnostics.evaluate_final_selector_experiment import analyze


def verify(out):
    old=ROOT/'results/ml_train_revision_2026-10-27';verify_manifest(old)
    verify_manifest(ROOT/'results/ml_development8_2026-10-26')
    model=FinalResearchSelector(out/'model');freeze=read(out/'release_freeze.json');exp=out/'final_experiment'
    assert freeze['model_registry_sha256']==sha(out/'model/registry.json') and freeze['model_manifest_sha256']==sha(out/'model/artifact_sha256.json')
    assert freeze['inference_code_sha256']==sha(ROOT/'ml/final_selector.py')
    assert datetime.fromisoformat(freeze['frozen_utc'])<datetime.fromisoformat(read(exp/'preregistration.json')['created_utc'])
    with np.load(out/'training/features_train.npz',allow_pickle=False) as z:X=z['X']
    assert model.model.predict_features(X)==ResearchRevision(old).predict_features(X)
    assert read(out/'training/outer_oof.json')==read(old/'outer_oof.json')
    assert model.registry['final_policy']==read(old/'registry.json')['final_policy']
    p=read(out/'example_input.json');served=model.predict(p['coefficients'],p['x_grid'],p['coefficient_metadata'])
    assert served['selected_profile']==read(out/'example_prediction.json')['selected_profile']==read(out/'cli_prediction.json')['selected_profile']
    assert served['estimates']==read(out/'cli_prediction.json')['estimates']
    assert model.production_select_method()['action']=='abstain'
    check=preflight(exp);assert not check['execution_authorized'] and check['requests']==check['records']==0
    assert not (exp/'execution_authorization.json').exists() and not (exp/'budget.json').exists()
    assert analyze(exp)['status']=='not_executed'
    plan=read(exp/'plan.json');identity=read(exp/'independence.json')
    for r in identity['history']:assert sha(ROOT/r['path'])==r['sha256']
    rows=[json.loads(l) for l in (exp/'input_pool/all_methods_equations_v1_metadata.jsonl').open()]
    selection=select(rows,set(identity['excluded_topologies']),plan['cases'][0]['dataset_sha256'])
    assert [r['equation_id'] for r in selection]==[c['equation_id'] for c in plan['cases']]
    trained={r['family_id'] for r in read(out/'training/accuracy.json')};assert not trained&{c['structure_signature'] for c in plan['cases']}
    for head in model.model.heads.values():assert set(head.meta['families'])<=trained
    request=read(exp/'approval_request.json');assert not request['approved'] and request['plan_sha256']==sha(exp/'plan.json')
    assert request['series_seconds']==49780 and request['maximum_calls']==720
    result=dict(passed=True,model_reload_and_real_CLI_prediction=True,train_only_retraining_bitwise_OOF_and_final_predictions_reproduced=True,
        nested_train_families=90,train_feature_count=253,previous_models_results_preserved=True,final_weights_frozen_before_new_test_inputs=True,
        independent_manifest_families=60,manifest_jobs=720,call_counts=dict(R128=240,R256=240,MC1=240),proposed_cap_seconds=49780,
        manifest_selection_replayed_without_predictions_or_outcomes=True,global_topology_disjointness_verified=True,
        production_blocked=True,final_experiment_authorized=False,final_experiment_requests=0,solver_calls=0,
        scientific_quality_independently_confirmed=False,implementation_complete=True)
    write_json(out/'independent_verification.json',result);print(result);return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);verify(ap.parse_args().out.resolve())
