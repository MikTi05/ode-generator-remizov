"""Predict from input coefficients only; never read solver results/targets."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def predict(models,inputs,indices,workers):
    import joblib
    import numpy as np
    import pandas as pd
    from catboost import CatBoostClassifier,CatBoostRegressor
    meta=json.loads((models/'metadata.json').read_text())
    if inputs['feature_names']!=meta['feature_names'] or inputs['grid_sha256']!=meta['input_grid_sha256']:
        raise ValueError('model/input feature schema or grid mismatch')
    frame=pd.DataFrame(inputs['X'][indices],columns=meta['feature_names'])
    answer={}
    for role,key in [('p_success',meta['selected_classifier']),('solver_seconds',meta['selected_regressor'])]:
        spec=meta['models'][key];data=frame.assign(planned_external_workers=workers) if spec['features']=='context' else frame
        path=models/spec['path']
        if spec['kind']=='context_median':
            value=json.loads(path.read_text());p=np.full(len(indices),value[str(workers)])
        else:
            if spec['kind']=='catboost':
                model=CatBoostClassifier() if role=='p_success' else CatBoostRegressor()
                model.load_model(str(path))
                p=model.predict_proba(data,thread_count=1)[:,1] if role=='p_success' else model.predict(data,thread_count=1)
            else:
                model=joblib.load(path)
                p=model.predict_proba(data)[:,1] if role=='p_success' else model.predict(data)
            if spec['transform']=='exp':p=np.exp(p)
        answer[role]=p
    return [dict(equation_id=str(inputs['ids'][i]),p_accuracy_met=float(answer['p_success'][j]),
        predicted_accuracy_met=bool(1-answer['p_success'][j]<meta['failure_threshold']),
        predicted_solver_seconds=float(answer['solver_seconds'][j]),planned_external_workers=workers,
        classifier=meta['selected_classifier'],regressor=meta['selected_regressor'],
        classification_scope='Accuracy conditional on a finite completed solve; worker exceptions are unlabeled',
        profile='ChernovSolver.solve n128/Q513/dx.0125/padding12/cubic_taper/tail_eps1e-8/grid33',
        prediction_is_not_accuracy_certificate=True,runtime_context_warning=meta['runtime_warning']) for j,i in enumerate(indices)]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models',required=True,type=Path)
    ap.add_argument('--dataset-dir',type=Path,default=ROOT/'data/all_methods_v1_1000_2026-10-18')
    ap.add_argument('--equation-id',required=True,nargs='+')
    ap.add_argument('--planned-workers',type=int,required=True,choices=(1,8))
    args=ap.parse_args()
    from ml.remizov_features import load_inputs
    inputs=load_inputs(args.dataset_dir)
    lookup={str(v):i for i,v in enumerate(inputs['ids'])}
    print(json.dumps(predict(args.models,inputs,[lookup[pid] for pid in args.equation_id],args.planned_workers),indent=2))


if __name__=='__main__':main()
