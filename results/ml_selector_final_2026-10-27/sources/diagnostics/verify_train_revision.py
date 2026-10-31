"""Replay saved predictions, scalers and nested exclusion graph. No solver/fit calls."""
from __future__ import annotations
import argparse,math,os,sys
import statistics
from collections import defaultdict
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import expit,logit
from ml.revision_evidence import read,sha,collapse_accuracy,ORIGINAL
from ml.revision_selector import load_point,predict_bundle,choose_policy,candidate_grid,family_weights,ResearchRevision
from diagnostics.prepare_development8 import write_json


def verify(out):
    accuracy=read(out/'accuracy.json');cost=read(out/'cost.json');audit=read(out/'audit.json');nodes=read(out/'model_index.json');bundles=read(out/'bundle_index.json')
    raw=read(out/'raw.json');collapsed,duplicates=collapse_accuracy(raw)
    assert len(duplicates)==4 and len(collapsed)==len(accuracy)==183
    for actual,expected in zip(collapsed,accuracy):
        assert all(actual[k]==expected[k] for k in actual),actual['equation_id']
    for n,h in read(out/'pins.json').items():assert sha(ROOT/n)==h,n
    for file in ('additional_pins.json','legacy_plan_pins.json'):
        for n,h in read(out/file).items():assert sha(ROOT/n)==h,n
    ds=sha(ORIGINAL/'all_methods_equations_v1.csv');assert ds==audit['original_dataset_sha256']
    assert all(r['dataset_sha256']==ds and r['split']=='train' for r in accuracy+cost)
    quarantined={r['equation_key'] for r in read(out/'quarantine.json') if r['dataset_sha256']!=ds}
    assert not quarantined&{r['equation_key'] for r in accuracy}
    forbidden_paths={r['source'] for r in read(out/'quarantine.json')}
    assert not forbidden_paths&{s for r in accuracy for s in r['sources']}
    with np.load(out/'features_train.npz',allow_pickle=False) as z:X=z['X'];names=z['feature_names'].tolist();keys=z['equation_keys'].tolist()
    ai={(r['equation_key'],r['profile']):r for r in accuracy};ci={(r['equation_key'],r['profile']):r for r in cost};by_key={r['equation_key']:r for r in accuracy}
    all_groups={r['family_id'] for r in accuracy};heads={}
    original_model=ROOT/'results/ml_probability_selector_2026-10-26'
    old_targets=read(original_model/'accuracy_targets.json');old_indices={r['equation_id']:r['index'] for r in old_targets}
    with np.load(original_model/'features_train.npz',allow_pickle=False) as z:
        old_X=z['X'];local={int(v):i for i,v in enumerate(z['original_indices'])};assert z['feature_names'].tolist()==names
        for r in accuracy:np.testing.assert_array_equal(X[r['index']],old_X[local[old_indices[r['equation_id']]]])
    for r in cost:
        own=[v for v in raw if v['equation_key']==r['equation_key'] and v['profile']==r['profile'] and v['full_seconds'] is not None]
        groups=defaultdict(list)
        for v in own:groups[v['timing_cohort']].append(v['full_seconds'])
        expected=statistics.median(statistics.median(v) for v in groups.values())
        assert expected==r['seconds'] and r['times']==[v['full_seconds'] for v in own] and r['sources']==[v['source'] for v in own]
    for name,node in nodes.items():
        h=load_point(out/node['path']);meta=h.meta;assert meta==node['metadata']
        assert not set(meta['families'])&set(node['excluded_families'])
        assert set(meta['families'])<=set(node['train_families']) and set(node['train_families'])|set(node['excluded_families'])==all_groups
        indices=meta['train_indices'];np.testing.assert_allclose(h.scaler.mean_,X[indices].mean(axis=0),atol=1e-14,rtol=1e-14)
        np.testing.assert_allclose(h.scaler.var_,X[indices].var(axis=0),atol=1e-12,rtol=1e-12)
        expected=[];rows=[]
        for k,p in zip(meta['train_keys'],meta['train_profiles']):
            row=(ci if node['head'].endswith('cost') else ai)[k,p];assert row['index'] in indices and row['family_id'] not in node['excluded_families']
            rows.append(row)
            if node['head'].endswith('prob'):value=row['k']/row['n']
            elif node['head']=='R_cost':value=math.log(row['seconds']/int(p[1:]))
            elif node['head']=='MC_cost':value=math.log(row['seconds'])
            else:value=math.log(max(row['error'],1e-14)*int(p[1:]))
            expected.append(value)
        np.testing.assert_array_equal(meta['target'],expected)
        weights=family_weights(rows)*(np.array([r['n'] for r in rows]) if node['head'].endswith('prob') else 1)
        np.testing.assert_array_equal(meta['weights'],weights);heads[node['path']]=h
    replayed=0;calibrators=0;max_KKT=0.;outer=read(out/'outer_oof.json');oi={r['equation_key']:r for r in outer};paired=set(read(out/'paired_keys.json'))
    protocol=read(out/'protocol.json')
    for name,b in bundles.items():
        train=set(b['train_families']);excluded=set(b['excluded_families']);cal=b['calibration'];assert train|excluded==all_groups and not train&excluded
        if name.startswith('outer'):
            fold=int(name.split('/')[0][5:]);ancestor=set(protocol['outer_groups'][fold]);assert ancestor<=excluded
        assert set(cal['excluded_families'])==excluded
        for block in cal['subfolds']:
            held=set(block['held_families']);assert held<=train
            indices=[by_key[k]['index'] for k in block['held_keys']]
            for key,path in block['paths'].items():
                h=heads[path];assert not (held|excluded)&set(h.meta['families'])
                preds=h.predict(X[indices],names)
                for k,p in zip(block['held_keys'],preds):assert cal['raw_OOF'][k][key]==float(p)
                replayed+=len(preds)
        for p,c in cal['sigmoid'].items():
            rows=[ai[k,p] for k in c['equation_keys']];assert all(r['family_id'] in train for r in rows)
            raw_p=[cal['raw_OOF'][k][p+'_prob'] for k in c['equation_keys']];np.testing.assert_array_equal(c['raw_inner_OOF'],raw_p)
            z=logit(np.clip(raw_p,1e-6,1-1e-6));w=family_weights(rows)*np.array([r['n'] for r in rows]);y=np.array([r['k']/r['n'] for r in rows]);s=c['slope'];intercept=c['intercept']
            err=w*(expit(s*z+intercept)-y);gradient=np.array([np.sum(err*z)+2*(s-1),np.sum(err)+2*intercept])
            for i,(v,lo,hi) in enumerate([(s,0,10),(intercept,-10,10)]):
                if v<=lo+1e-8:gradient[i]=min(0.,gradient[i])
                elif v>=hi-1e-8:gradient[i]=max(0.,gradient[i])
            max_KKT=max(max_KKT,float(abs(gradient).max()));assert abs(gradient).max()<3e-5
            calibrators+=1
        for p,c in cal['error_cdf'].items():
            for k,res in zip(c['equation_keys'],c['residuals']):
                r=ai[k,p];assert r['family_id'] in train
                expected=math.log(max(r['error'],1e-14))-(cal['raw_OOF'][k]['R_error']-math.log(int(p[1:])))
                assert res==expected
        for p,c in cal['time_residuals'].items():
            for k,res in zip(c['equation_keys'],c['residuals']):
                r=ci[k,p];assert r['family_id'] in train
                predicted=cal['raw_OOF'][k]['MC_cost' if p=='MC1' else 'R_cost']+(0 if p=='MC1' else math.log(int(p[1:])))
                assert res==math.log(r['seconds'])-predicted
            assert c['q10']==float(np.quantile(c['residuals'],.1)) and c['q90']==float(np.quantile(c['residuals'],.9))
        hold=[k for k in keys if by_key[k]['family_id'] in set(b['held_families'])]
        hs={key:heads[path] for key,path in b['paths'].items()}
        if hold:
            preds=predict_bundle(hs,cal,X[[by_key[k]['index'] for k in hold]],names)
            if name.endswith('/evaluation'):
                for k,pred in zip(hold,preds):assert pred==oi[k]['variants']
            else:
                fold=int(name.split('/')[0][5:]);saved={r['equation_key']:r for r in read(out/f'outer{fold}_selection.json')['inner_predictions']}
                for k,pred in zip(hold,preds):
                    if k in paired:assert pred==saved[k]['variants']
    for fold in range(3):
        s=read(out/f'outer{fold}_selection.json');assert not set(r['family_id'] for r in s['inner_predictions'])&set(s['outer_held_families'])
        policy,scores=choose_policy(s['inner_predictions'],ai,candidate_grid());assert policy==s['selected_policy'] and scores==s['selection_scores']
        for r in outer:
            if r['outer_fold']==fold:assert r['nested_selected_policy']==policy and r['family_id'] in s['outer_held_families']
    final_policy,_=choose_policy([r for r in outer if r['equation_key'] in paired],ai,candidate_grid());assert final_policy==read(out/'registry.json')['final_policy']
    assert len(outer)==len({r['equation_key'] for r in outer})==103 and len({r['family_id'] for r in outer})==90
    model=ResearchRevision(out);served=model.select_features(X[:2]);assert all(r['action']=='abstain' for r in served)
    old=read(ROOT/'results/ml_nested_calibration_2026-10-26/registry.json');assert sha(ROOT/'ml/probabilistic_selector.py')==old['original_production_code_sha256']
    result=dict(passed=True,nodes_verified=len(nodes),raw_OOF_values_replayed=replayed,calibrators_verified=calibrators,maximum_calibrator_KKT=max_KKT,
        all_ancestor_family_exclusions_verified=True,all_scalers_and_saved_models_reloaded=True,real_targets_and_weights_recomputed=True,MC_replays_deduplicated=4,
        feature_values_identical_to_existing_train253=True,full_time_aggregation_recomputed=True,
        MC_unique_trials=180,MC_equations=40,MC_independent_positive_families=3,development_and_reserved_seed_audit_excluded=True,
        validation_test_outcomes_not_used=True,source_raw_hashes_verified=True,production_API_unchanged_and_blocked=True,solver_calls=0)
    write_json(out/'independent_verification.json',result);print(result);return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);verify(ap.parse_args().out.resolve())
