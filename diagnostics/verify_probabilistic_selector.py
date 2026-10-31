"""Independent saved-only count, source, split, score and model integrity audit."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnostics.train_method_selector import write,snapshot
from ml.selector_evidence import read,sha,CONTRACT


def verify(out):
    import numpy as np
    from ml.remizov_features import load_inputs
    from all_methods_dataset_loader import load_all_methods_equations
    from ml.probabilistic_selector import ProbabilisticSelector,estimates_from_heads
    contract=read(CONTRACT/'plan.json');data=ROOT/contract['data_dir'];inputs=load_inputs(data)
    eqs={e.equation_id:e for e in load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl')}
    for name,h in read(out/'source_sha256.json').items():assert sha(ROOT/name)==h,name
    accuracy=read(out/'accuracy_targets.json');cost=read(out/'full_cost_targets.json');seen=set();trials=0
    for row in accuracy:
        pid=row['equation_id'];p=row['profile'];e=eqs[pid]
        assert row['split']==e.metadata['split']=='train' and row['family_id']==e.metadata['family_id']
        assert inputs['ids'][row['index']]==pid and inputs['splits'][row['index']]=='train'
        exact=np.asarray(e.y_exact(np.asarray(contract['target_x'])));k=0
        for name in row['sources']:
            r=read(ROOT/name);z=r['result'];assert r['status']=='completed' and z['success']
            assert r['expression_hash']==e.metadata['expression_hash'] and z['problem_id']==pid
            assert np.array_equal(z['x'],contract['target_x']) and np.array_equal(r['y_exact'],exact)
            error=float(np.max(abs(np.asarray(z['f'])-exact)))
            assert error==r.get('E_grid',r.get('E_inf'));k+=int(error<=.01)
            cfg=r['job']['config']
            if p=='MC1':
                key=(pid,p,cfg['seed']);assert cfg['M']==4096 and cfg['dt']==1/128
                assert cfg['seed']==248800000000+r['job']['stage']*10000+100+r['job']['episode']
                assert 'fresh' not in name and 'f0047_observed_timing' not in name
                assert key not in seen;seen.add(key)
            else:
                assert cfg==dict(contract['profiles'][p]['config'],T=z['T'])
        assert k==row['k'] and len(row['sources'])==row['n']
        if p=='MC1':trials+=row['n']
    assert trials==160 and len(seen)==160
    for row in cost:
        times=[read(ROOT/name)['method_wall_seconds'] for name in row['sources']]
        assert all(t>0 for t in times) and float(np.median(times))==row['seconds']
        assert row['scope']=='warm_end_to_end_method_wall_v1'
    with np.load(out/'features_train.npz',allow_pickle=False) as z:
        assert np.array_equal(z['X'],inputs['X'][z['original_indices']])
        assert all(inputs['splits'][i]=='train' for i in z['original_indices'])
        assert list(z['feature_names'])==inputs['feature_names'] and z['X'].shape==(103,253)
    cv=read(out/'family_cv.json');held_all=[]
    for fold in cv['folds']:
        held=set(fold['held_families']);held_all+=list(held)
        for key,h in fold['heads'].items():
            assert not held&set(h['families'])
            assert all(inputs['splits'][i]=='train' and inputs['families'][i] not in held for i in h['indices'])
            np.testing.assert_allclose(h['train_scaler_mean'],inputs['X'][h['indices']].mean(axis=0),rtol=0,atol=1e-12)
    assert len(held_all)==len(set(held_all))==40
    for p in ('R128','R256','MC1'):
        rs=cv['oof'];s=np.array([r['observed'][p]['k']/r['observed'][p]['n'] for r in rs]);pred=np.array([r['estimates'][p]['probability']['point'] for r in rs])
        brier=float(np.mean(s*(1-pred)**2+(1-s)*pred**2));assert brier==cv['metrics'][p]['probability']['brier']
        assert all(r['split']=='train' for r in rs)
    registry=read(out/'registry.json');api=ProbabilisticSelector(out/'registry.json')
    pred=estimates_from_heads(api.heads,inputs['X'][[next(i for i,pid in enumerate(inputs['ids']) if pid=='AMV1_F0047_S00')]],inputs['feature_names'])[0]
    audit=read(out/'seen_family_fresh_seed_audit.json');assert pred==audit['estimates']
    assert audit['registry_sha256_before_outcomes']==sha(out/'registry.json') and not audit['used_for_fit_or_calibration']
    assert audit['fresh_n']==4 and audit['training_MC_counts']=={'k':4,'n':4}
    proposal=read(out/'NEXT_VALIDATION.json');assert not proposal['execution_authorized']
    assert len({r['family_id'] for r in proposal['cases']})==8 and all(eqs[r['equation_id']].metadata['split']=='validation' for r in proposal['cases'])
    assert not registry['deployment_ready'] and not registry['validation_calibrated'] and not registry['old_MC_winner_gate_applied']
    before=read(out/'protected_before.json');after=snapshot(out)
    changed={n for n,h in before.items() if after.get(n)!=h};assert changed<= {'PROJECT_STATUS.md'},changed
    result=dict(passed=True,accuracy_rows=len(accuracy),cost_rows=len(cost),independent_MC_trials=160,unique_semantic_MC_trials=len(seen),
        no_replayed_or_new_seed_targets=True,train_features=103,feature_count=253,held_families=40,folds=3,
        original_validation_or_test_fit=False,test_outcomes_used=False,all_models_loaded_and_hash_checked=True,
        fitted_seed_audit_predictions_reload_exact=True,raw_source_files=len(read(out/'source_sha256.json')),
        previous_paths_checked=len(before),previous_paths_changed=sorted(changed),numerical_calls=0)
    write(out/'independent_verification.json',result);print(json.dumps(result,sort_keys=True),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);verify(ap.parse_args().out.resolve())
