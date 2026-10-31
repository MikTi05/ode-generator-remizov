"""Audit saved MC results, identities, counts, features and reloadable models."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def verify(out):
    import numpy as np
    import pandas as pd
    import joblib
    from scipy.special import expit
    from catboost import CatBoostClassifier,CatBoostRegressor
    from ml.remizov_features import load_inputs
    from diagnostics.mc_production_x0 import collect
    from diagnostics.chernov_batch_control import lock_is_held
    from diagnostics.train_remizov_ml import write
    plan=read(out/'plan.json');records=collect(out,plan);before=read(out/'protected_before.json')
    changed=[n for n,h in before.items() if not (ROOT/n).is_file() or sha(ROOT/n)!=h]
    assert not changed,changed
    for n,h in plan['protected_sha256'].items():assert sha(ROOT/n)==h,n
    eq=pd.read_csv(out/'analysis/equations.csv');att=pd.read_csv(out/'analysis/attempts.csv')
    assert len(eq)==1000 and len(att)==4000
    assert eq.groupby('family_id').split.nunique().max()==1
    assert eq.split.value_counts().to_dict()=={'train':700,'validation':150,'test':150}
    assert eq.groupby('split').family_id.nunique().to_dict()=={'train':140,'validation':30,'test':30}
    pre={r['problem_id']:r for r in plan['preflight']};completed=0
    for k,r in records.items():
        if r['status']!='completed':continue
        z=r['result'];expected=pre[r['problem_id']]
        assert z['x']==[0.] and z['M']==1024 and z['completed_paths']==[1024]
        assert z['T']==expected['T'] and z['n_steps']==expected['N']
        assert z['forcing_abs_upper_bound']==expected['G'] and z['tail_tolerance_met']
        assert z['n_path_steps']==z['n_normal_draws']==1024*z['n_steps']
        assert z['seed_entropy']==plan['seeds'][r['seed_index']] and z['backend']=='numba'
        assert r['task_wall_seconds']<=60 and r['process_peak_rss_mib']<=512
        assert z['stderr'][0]>=0 and not z['total_error_certified']
        assert abs(abs(z['f'][0]-r['y_exact'])-r['abs_error'])<=1e-14
        assert math.isfinite(r['abs_error']);completed+=1
    inputs=load_inputs(ROOT/plan['data_dir'])
    model_audit={};feature_equal=None;metric_checks=0
    ml=out/'ml'
    if ml.exists():
        with np.load(ml/'features.npz',allow_pickle=False) as z:
            assert z['X'].shape==(1000,253)
            np.testing.assert_array_equal(z['X'],inputs['X'])
            np.testing.assert_array_equal(z['equation_id'],inputs['ids'])
            np.testing.assert_array_equal(z['split'],inputs['splits'])
        with np.load(ROOT/'results/remizov_ml_v1_2026-10-25/features.npz',allow_pickle=False) as z:
            feature_equal=bool(np.array_equal(z['X'],inputs['X']));assert feature_equal
        X=pd.DataFrame(inputs['X'],columns=inputs['feature_names']);p=pd.read_csv(ml/'predictions.csv')
        assert p.equation_id.tolist()==list(inputs['ids'])
        schema=read(ml/'feature_schema.json');assert schema['feature_names']==list(X.columns)
        assert not any(any(forbidden in n for forbidden in ('exact','solver_seconds','seed','y_shape','family','split')) for n in X.columns)
        m=read(ml/'metrics.json')
        from diagnostics.train_mc1000_quality import metrics_class
        for target,models in m['classification'].items():
            for model,values in models.items():
                if not isinstance(values,dict) or 'test' not in values:continue
                base=ml/'models'/f'{target}_{model}'
                if model=='catboost':
                    fitted=CatBoostClassifier();fitted.load_model(str(base.with_suffix('.cbm')))
                    probability=fitted.predict_proba(X,thread_count=1)[:,1];info=dict(trees=fitted.tree_count_)
                else:
                    fitted=joblib.load(base.with_suffix('.joblib'));probability=fitted.predict_proba(X)[:,1];info={}
                    if model=='logistic':
                        scaled=fitted[0].transform(X);linear=fitted[1]
                        independent=expit(np.einsum('ij,j->i',scaled,linear.coef_[0],optimize=False)+linear.intercept_[0])
                        info['nonblas_probability_max_difference']=float(np.max(abs(probability-independent)))
                        assert info['nonblas_probability_max_difference']<1e-12
                delta=float(np.max(abs(probability-p[f'p_{target}_{model}'])));assert delta<1e-12
                info['reload_max_difference']=delta;model_audit[target+'_'+model]=info
                if target=='mc_single_seed_accuracy':
                    good=att.status.eq('completed');sub=att.loc[good];lookup=dict(zip(p.equation_id,probability))
                    y=sub.accuracy_met.to_numpy(dtype=int);pred=np.array([lookup[x] for x in sub.equation_id]);split=sub.split.to_numpy()
                else:
                    if target=='remizov_point_accuracy':mask=p.remizov_known.to_numpy(dtype=bool);y=p.loc[mask,'remizov_accuracy0'].to_numpy(dtype=int)
                    else:
                        mask=p.quality_label.isin(['mc_mean_closer_exploratory','remizov_closer_exploratory']).to_numpy()
                        y=p.loc[mask,'quality_label'].eq('mc_mean_closer_exploratory').to_numpy(dtype=int)
                    pred=probability[mask];split=p.loc[mask,'split'].to_numpy()
                for s in ('train','validation','test'):
                    sm=split==s;actual=metrics_class(y[sm],pred[sm])
                    for field in ('logloss','brier','accuracy'):
                        assert abs(actual[field]-values[s][field])<1e-12
                    # An explicit non-sklearn Brier check.
                    assert abs(float(np.mean((pred[sm]-y[sm])**2))-values[s]['brier'])<1e-12;metric_checks+=1
        for target,models in m['regression'].items():
            for model,values in models.items():
                if not isinstance(values,dict) or 'test' not in values:continue
                base=ml/'models'/f'{target}_{model}'
                if model=='catboost':
                    fitted=CatBoostRegressor();fitted.load_model(str(base.with_suffix('.cbm')));pred=fitted.predict(X,thread_count=1)
                else:fitted=joblib.load(base.with_suffix('.joblib'));pred=fitted.predict(X)
                delta=float(np.max(abs(pred-p[f'logerror_{target}_{model}'])));assert delta<1e-12
                model_audit[target+'_'+model]=dict(reload_max_difference=delta)
                if model=='ridge':
                    scaled=fitted[0].transform(X);linear=fitted[1]
                    independent=np.einsum('ij,j->i',scaled,linear.coef_,optimize=False)+linear.intercept_
                    d=float(np.max(abs(pred-independent)));assert d<1e-12
                    model_audit[target+'_'+model]['nonblas_prediction_max_difference']=d
    active=read(out/'active.json');assert active['finished']
    guardians=[read(p) for p in (out/'control').rglob('*.guardian.json')]
    assert all(g['finished'] and g.get('cleanup_success') for g in guardians)
    locks={n:lock_is_held(ROOT/'results'/n) for n in ('.chernov_batch_compute.lock','.chernov_batch_controller.lock')};assert not any(locks.values())
    result=dict(protected_files_checked=len(before),protected_files_changed=changed,terminal_attempts=len(records),completed_attempts=completed,
        pending=4000-len(records),feature_matrix_equal_to_prior_input_only_matrix=feature_equal,
        feature_schema_columns=253,model_audit=model_audit,classification_metric_checks=metric_checks,
        family_split_rows=eq.split.value_counts().to_dict(),all_guardians_finished=True,locks_held=locks,
        source_sha256={n:sha(ROOT/n) for n in plan['protected_sha256'] if not n.startswith('data/')},
        independent_runtime_comparison=False,scope='x=0; no most-efficient-method labels or total error certificate')
    write(out/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('model_audit','source_sha256')},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);verify(ap.parse_args().out.resolve())
