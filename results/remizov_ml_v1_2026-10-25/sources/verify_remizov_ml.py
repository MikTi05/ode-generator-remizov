"""Independent saved-data/model audit, including non-BLAS linear predictions."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys

for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True,ensure_ascii=False)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    out=ap.parse_args().out.resolve()
    import numpy as np
    import pandas as pd
    import joblib
    from scipy.special import expit
    before=read(out/'protected_before.json')
    changed=[n for n,h in before.items() if not (ROOT/n).is_file() or sha(ROOT/n)!=h]
    assert not changed,changed
    targets=pd.read_csv(out/'targets.csv');pred=pd.read_csv(out/'predictions.csv')
    X=pd.read_csv(out/'features.csv').drop(columns='equation_id')
    schema=read(out/'feature_schema.json');audit=read(out/'data_audit.json');metrics=read(out/'metrics.json')
    assert list(X.columns)==schema['feature_names'] and X.shape==(1000,253)
    assert np.isfinite(X.to_numpy()).all()
    assert pred.equation_id.tolist()==targets.equation_id.tolist()
    assert targets.target_known.sum()==997 and targets.accuracy_met.sum()==871
    assert targets.loc[~targets.target_known,'solver_seconds'].isna().all()
    family_splits=targets.groupby('family_id').split.nunique()
    topology_splits=targets.groupby('structure_signature').split.nunique()
    assert family_splits.eq(1).all() and topology_splits.eq(1).all()
    assert targets.groupby('split').family_id.nunique().to_dict()=={'train':140,'validation':30,'test':30}
    workers=read(ROOT/'results/chernov_managed_completion_2026-10-18/plan_v2.json')['pending_ids']
    checked=0
    for row in targets.itertuples():
        result=read(ROOT/row.source_result)
        assert sha(ROOT/row.source_result)==row.source_result_sha256
        assert row.planned_external_workers==(1 if row.equation_id in workers else 8)
        if row.target_known:
            f=result['result']['f'];y=result['y_exact']
            E=max(abs(a-b) for a,b in zip(f,y))
            assert len(f)==len(y)==33
            assert abs(E-row.max_abs_error)<1e-12*max(1,E)
            assert int(E<=.01)==row.accuracy_met
            assert abs(row.solver_seconds-result['solver_seconds'])<1e-12
            checked+=1
    assert checked==997
    masks={s:(targets.split==s)&targets.target_known for s in ['train','validation','test']}
    for name,values in metrics['classification'].items():
        for split,mask in masks.items():
            good=targets.loc[mask,'accuracy_met'].to_numpy().astype(bool)
            p=pred.loc[mask,'p_success_'+name].to_numpy()
            assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
            auc=(np.sum(p[good,None]>p[~good])+0.5*np.sum(p[good,None]==p[~good]))/(sum(good)*sum(~good))
            for operating,m in values[split].items():
                assert abs(auc-m['roc_auc'])<1e-12
                rejected=1-p>=m['failure_threshold']
                assert int(np.sum((~good)&~rejected))==m['false_accept_bad']
                assert int(np.sum(good&rejected))==m['false_reject_good']
    for name,values in metrics['regression'].items():
        for split,mask in masks.items():
            mae=np.mean(np.abs(targets.loc[mask,'solver_seconds']-pred.loc[mask,'seconds_'+name]))
            assert abs(mae-values[split]['mae_seconds'])<1e-12
    model=joblib.load(out/'models/classifier_logistic.joblib')
    Z=model[0].transform(X);coef=model[1].coef_[0]
    independent=expit(np.einsum('ij,j->i',Z,coef,optimize=False)+model[1].intercept_[0])
    tr=masks['train'].to_numpy();y=targets.accuracy_met.to_numpy()
    grad=np.mean(Z[tr]*(independent[tr]-y[tr])[:,None],axis=0)+coef/(model[1].C*sum(tr))
    gnorm=float(max(np.max(np.abs(grad)),abs(np.mean(independent[tr]-y[tr]))))
    difference=float(np.max(np.abs(independent-pred.p_success_logistic)))
    assert difference<1e-12 and gnorm<=model[1].tol
    linear_audit=dict(warning_observed='Initial sklearn fit emitted matmul divide/overflow/invalid RuntimeWarning; cause not established',
        all_features_finite=True,all_models_predictions_finite=bool(np.isfinite(pred.filter(regex='^(p_success_|seconds_)').to_numpy()).all()),
        logistic_prob_max_difference_nonblas_einsum=difference,
        independent_logistic_gradient_inf=gnorm,logistic_tolerance=model[1].tol,
        logistic_iterations=model[1].n_iter_.tolist(),ridge_nonblas_checks={})
    for name in ['ridge_coeff','ridge_context']:
        model=joblib.load(out/f'models/regressor_{name}.joblib')
        frame=X.assign(planned_external_workers=targets.planned_external_workers) if name.endswith('context') else X
        z=model[0].transform(frame)
        p=np.exp(np.einsum('ij,j->i',z,model[1].coef_,optimize=False)+model[1].intercept_)
        delta=float(np.max(np.abs(p-pred['seconds_'+name])))
        assert delta<1e-12
        linear_audit['ridge_nonblas_checks'][name]=delta
    write(out/'numerical_warning_audit.json',linear_audit)
    assert importlib.metadata.version('numpy')=='2.2.6' and importlib.metadata.version('scipy')=='1.15.3'
    broken=[]
    for doc in [out/'REPORT.md',out/'MC_OPTIMIZATION.md',ROOT/'PROJECT_STATUS.md']:
        for link in re.findall(r'\]\(([^)]+)\)',doc.read_text()):
            if '://' in link or link.startswith('#'):continue
            path=link.split('#')[0]
            path=re.sub(r':\d+$','',path)
            if not (doc.parent/path).exists():broken.append({'document':str(doc),'target':link})
    assert not broken,broken
    report=dict(completed=True,protected_files=len(before),changed_protected_files=changed,
        input_features=253,targets=1000,known_targets=997,unknown_targets=3,
        independently_recomputed_saved_errors=checked,family_split_preserved=True,topology_split_preserved=True,
        classifier_metrics_verified=True,runtime_metrics_verified=True,independent_linear_audit=linear_audit,
        numerical_dependencies_preserved=True,new_ode_solves=0,new_mc_runs=0,
        local_links_broken=broken,status_before_sha256=sha(out/'PROJECT_STATUS.before.md'),
        status_after_sha256=sha(ROOT/'PROJECT_STATUS.md'),
        core_sha256={n:sha(ROOT/n) for n in ['mc_solver.py','chernov_solver.py','remizov_solver.py',
            'src/all_methods_expressions.py','src/all_methods_dataset_v1.py','all_methods_dataset_loader.py']})
    write(out/'verification.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('core_sha256','independent_linear_audit')},indent=2))


if __name__=='__main__':main()
