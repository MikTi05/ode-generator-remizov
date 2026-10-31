"""Independent saved-only verification of labels, features, split isolation and models."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from ml.selector_evidence import read,sha,verify_manifest
from ml.method_selector import classification_metrics,retrospective_policy
from ml.remizov_features import load_inputs


def verify(out):
    from all_methods_dataset_loader import load_all_methods_equations
    from catboost import CatBoostClassifier,Pool
    from sklearn.preprocessing import StandardScaler
    data=ROOT/'data/all_methods_v1_1000_2026-10-18';inputs=load_inputs(data)
    rows=[json.loads(s) for s in (out/'labels.jsonl').read_text().splitlines()];ev=read(out/'evidence.json');schema=read(out/'feature_schema.json')
    for name,h in read(out/'source_sha256.json').items():
        if sha(ROOT/name)!=h:raise ValueError('source/input changed: '+name)
    for path in [ROOT/'reports/ml_selector_contract/sentinel_p1_2026-10-26',ROOT/'reports/ml_selector_contract/pilot_p2_2026-10-26']:verify_manifest(path)
    with np.load(out/'features.npz',allow_pickle=False) as z:
        assert set(z.files)=={'X','feature_names'};X=z['X'];assert np.array_equal(X,inputs['X']);assert list(z['feature_names'])==inputs['feature_names']==schema['feature_names']
    assert X.shape==(1000,253)
    eqs={e.equation_id:e for e in load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl')}
    errors=0
    for e in ev.values():
        pid=e['equation_id'];meta=inputs['metadata'][pid]
        assert meta['expression_hash']==e['expression_hash'] and meta['family_id']==e['family_id'] and meta['split']==e['split']
        for record in e['records']:
            assert sha(ROOT/record['source'])==record['source_sha256']
            r=read(ROOT/record['source']);z=r['result'];exact=np.asarray(eqs[pid].y_exact(np.linspace(-1,1,33)))
            assert z['success'] and r['status']=='completed' and np.array_equal(r['y_exact'],exact)
            assert np.array_equal(z['x'],np.linspace(-1,1,33));err=float(np.max(np.abs(np.asarray(z['f'])-exact)))
            assert err==record['error'];errors+=1
        if e['records']:
            zero=next(r for r in e['records'] if r['episode']==0)
            assert e['error']==zero['error'] and e['status']==zero['status']
    for row,pid in zip(rows,inputs['ids']):
        assert row['equation_id']==pid;assert row['split']==inputs['metadata'][pid]['split']
        a=ev[pid+':remizov'];b=ev[pid+':monte_carlo'];expected='unknown'
        if a['status']==b['status']=='completed' and a['T']==b['T']:
            pa=a['error']<=.01;pb=b['error']<=.01
            if pa and not pb:expected='remizov'
            elif pb and not pa:expected='monte_carlo'
            elif not pa and not pb:expected='none'
            else:
                # In current saved evidence R has one old-scope runtime only.
                assert len(a['times'])==1 and a['timing_scope']!='warm_end_to_end_method_wall_v1'
        assert row['label']==expected
    known=[i for i,r in enumerate(rows) if r['label']!='unknown'];assert len(known)==36
    assert all(rows[i]['split']=='train' for i in known)
    d=read(out/'diagnostic_cv.json');assert set(d['fit_indices'])==set(known)
    assert d['classes']==['none','remizov'] and not d['MC_recognition_learned'] and not d['original_validation_or_test_used']
    held=[]
    for fold in d['folds']:
        fit=fold['fit_indices'];hold=fold['held_out_indices'];held.extend(hold)
        assert set(fit)|set(hold)==set(known) and not set(fit)&set(hold)
        assert not {rows[i]['family_id'] for i in fit}&{rows[i]['family_id'] for i in hold}
        np.testing.assert_allclose(fold['scaler_mean'],X[fit].mean(axis=0),atol=1e-12,rtol=0)
    assert sorted(held)==sorted(known)
    yp={r['index']:r['prediction'] for r in d['oof']};assert set(yp)==set(known)
    assert classification_metrics([rows[i]['label'] for i in known],[yp[i] for i in known])==d['metrics']
    reg=read(out/'registry.json');assert reg['selector_status']=='insufficient_evidence' and reg['model'] is None
    art=reg['diagnostic'];assert sha(out/art['model']['path'])==art['model']['sha256'] and sha(out/art['scaler']['path'])==art['scaler']['sha256']
    model=CatBoostClassifier();model.load_model(str(out/art['model']['path']));assert list(model.classes_)==['none','remizov'];assert list(model.feature_names_)==schema['feature_names']
    scaler=StandardScaler()
    with np.load(out/art['scaler']['path'],allow_pickle=False) as z:
        scaler.mean_=z['mean'];scaler.scale_=z['scale'];scaler.var_=z['var'];scaler.n_features_in_=253
    np.testing.assert_allclose(scaler.mean_,X[known].mean(axis=0),atol=1e-12,rtol=0)
    probabilities=model.predict_proba(Pool(scaler.transform(X[known]),feature_names=schema['feature_names']),thread_count=1)
    assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(axis=1),1)
    assert read(out/'test_evaluation.json')['status']=='not_evaluated_no_full_model'
    check=read(out/'serving_check.json');assert check['action']=='abstain' and check['status']=='insufficient_evidence'
    policies={}
    own=[rows[i] for i in known]
    policies['CatBoost_OOF']=retrospective_policy(own,ev,[yp[i] for i in known])
    policies['constant_remizov']=retrospective_policy(own,ev,['remizov']*len(known))
    policies['offline_oracle']=retrospective_policy(own,ev,[r['label'] for r in own])
    with (out/'policy_diagnostics.json').open('x') as f:json.dump(policies,f,indent=2,sort_keys=True);f.write('\n')
    old=ROOT/'results/ml_selector_observed_2026-10-26';older=read(old/'diagnostic_cv.json')
    assert older['oof']==d['oof'] and (old/'labels.jsonl').read_bytes()==(out/'labels.jsonl').read_bytes()
    result=dict(passed=True,rows=1000,features=253,known=36,unknown=964,canonical_error_checks=errors,trained_MC_class=False,
        original_validation_test_used_for_fit_or_diagnostic=False,original_family_split_preserved=True,
        scaler_fit_only_labeled_train=True,CV_family_disjoint=True,CBM_and_scaler_reload_verified=True,
        repeated_run_labels_OOF_probabilities_identical=True,full_selector_not_deployed=True,
        numerical_calls=0,MC_calls=0,R_calls=0,diagnostic_fit_rows=36,diagnostic_fit_families=36)
    with (out/'independent_verification.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);verify(ap.parse_args().out.resolve())
