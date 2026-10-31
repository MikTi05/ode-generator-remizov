"""Reload validation-selected quality models; no solver, exact or runtime win."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def predict(folder,dataset,ids):
    import pandas as pd
    import joblib
    from catboost import CatBoostClassifier,CatBoostRegressor
    from ml.remizov_features import load_inputs
    metrics=json.loads((folder/'metrics.json').read_text());schema=json.loads((folder/'feature_schema.json').read_text())
    inputs=load_inputs(dataset);lookup={pid:i for i,pid in enumerate(inputs['ids'])};ix=[lookup[pid] for pid in ids]
    if list(inputs['feature_names'])!=schema['feature_names']:raise ValueError('feature schema mismatch')
    X=pd.DataFrame(inputs['X'][ix],columns=inputs['feature_names']);answer=[dict(equation_id=pid) for pid in ids]
    for task,models in metrics['classification'].items():
        selected=models.get('selected_by_validation_logloss')
        if selected is None:continue
        path=folder/'models'/f'{task}_{selected}'
        if selected=='catboost':
            model=CatBoostClassifier();model.load_model(str(path.with_suffix('.cbm')));p=model.predict_proba(X,thread_count=1)[:,1]
        else:p=joblib.load(path.with_suffix('.joblib')).predict_proba(X)[:,1]
        for row,value in zip(answer,p):row['p_'+task]=float(value)
    for task,models in metrics['regression'].items():
        selected=models['selected_by_validation_mae'];path=folder/'models'/f'{task}_{selected}'
        if selected=='catboost':
            model=CatBoostRegressor();model.load_model(str(path.with_suffix('.cbm')));p=model.predict(X,thread_count=1)
        else:p=joblib.load(path.with_suffix('.joblib')).predict(X)
        for row,value in zip(answer,p):row['predicted_'+task]=float(value)
    return answer


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--models',type=Path,required=True)
    ap.add_argument('--data',type=Path,default=ROOT/'data/all_methods_v1_1000_2026-10-18');ap.add_argument('--ids',nargs='+',required=True)
    a=ap.parse_args();print(json.dumps(predict(a.models,a.data,a.ids),indent=2))
