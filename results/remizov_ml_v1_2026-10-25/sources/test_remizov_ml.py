"""Leakage, labels, held-out groups and serving contract; no ODE solves."""
import ast
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.remizov_features import coefficient_features,load_inputs
from diagnostics.train_remizov_ml import choose_threshold,classification_metrics,regression_metrics

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/all_methods_v1_1000_2026-10-18'
OUT=ROOT/'results/remizov_ml_v1_2026-10-25'


def sample():
    meta=json.loads((DATA/'all_methods_equations_v1_metadata.jsonl').read_text().splitlines()[0])
    with np.load(DATA/'all_methods_equations_v1_ml.npz',allow_pickle=False) as z:
        return z['X'][0],z['x_grid'],meta


def test_features_ignore_exact_outputs_and_generator_labels():
    X,x,m=sample();a=coefficient_features(X,x,m)
    b=copy.deepcopy(m)
    for key in ['y_exact_expr','dy_exact_expr','d2y_exact_expr','y0','v0','target_amplitude',
                'support_left','support_right','complexity_level','regime','tree_metrics',
                'family_id','equation_id','feature_hash','structure_signature','validation']:
        b[key]='forbidden sentinel'
    b['serialized_ast']['y_shape']='forbidden sentinel'
    b['serialized_ast']['construction']='forbidden sentinel'
    assert coefficient_features(X,x,b)==a


def test_input_sign_conventions_and_preflight_horizon():
    X,x,m=sample();f=coefficient_features(X,x,m)
    assert f['gR_sample_00']==-X[3,0]
    assert f['cR_mean']==pytest.approx(np.mean(X[2]+m['remizov_lambda']))
    gap=m['remizov_lambda']-m['remizov_c_abs_cert']
    assert f['pre_T']==pytest.approx(np.log(2/(gap*1e-8))/gap)
    assert all(np.isfinite(v) for v in f.values())
    assert not any(any(t in n for t in ['exact','family','split','error','seconds','y_shape','complexity']) for n in f)


def test_original_family_and_topology_splits_unchanged():
    m=[json.loads(s) for s in (DATA/'all_methods_equations_v1_metadata.jsonl').read_text().splitlines()]
    for key in ['family_id','structure_signature']:
        groups={}
        for row in m:groups.setdefault(row[key],set()).add(row['split'])
        assert all(len(s)==1 for s in groups.values())
    assert {s:len({r['family_id'] for r in m if r['split']==s}) for s in ['train','validation','test']}=={
        'train':140,'validation':30,'test':30}


def test_unknown_worker_exits_do_not_become_negative_math_labels():
    table=pd.read_csv(OUT/'targets.csv')
    unknown=table[~table.target_known]
    assert len(unknown)==3 and unknown.accuracy_met.isna().all() and unknown.solver_seconds.isna().all()
    assert len(table[table.target_known])==997
    assert int(table.accuracy_met.sum())==871
    assert table.groupby('split').size().to_dict()=={'train':700,'validation':150,'test':150}


def test_threshold_and_failure_metrics_use_correct_positive_class():
    y=np.array([1,1,0,0]);p=np.array([.95,.7,.6,.1])
    t=choose_threshold(y,p);assert t==pytest.approx(.4)
    m=classification_metrics(y,p,t)
    assert m['balanced_accuracy']==1. and m['failure_recall']==1.
    assert m['false_accept_bad']==0 and m['false_reject_good']==0


def test_cost_metrics_are_in_seconds_after_inverse_log_transform():
    m=regression_metrics(np.array([10.,20.]),np.array([11.,18.]))
    assert m['mae_seconds']==1.5 and m['rmse_seconds']==pytest.approx(np.sqrt(2.5))
    assert m['mape_percent']==10. and m['r2']==pytest.approx(.9)


def test_serving_reloads_selected_models_and_matches_all_saved_predictions():
    from diagnostics.predict_remizov_ml import predict
    data=load_inputs(DATA);saved=pd.read_csv(OUT/'predictions.csv');meta=json.loads((OUT/'models/metadata.json').read_text())
    for workers in (1,8):
        indices=np.flatnonzero(saved.planned_external_workers.to_numpy()==workers)
        answer=predict(OUT/'models',data,indices,workers)
        np.testing.assert_allclose([r['p_accuracy_met'] for r in answer],
            saved['p_success_'+meta['selected_classifier'].removeprefix('classifier_')].to_numpy()[indices],atol=2e-15,rtol=0)
        np.testing.assert_allclose([r['predicted_solver_seconds'] for r in answer],
            saved['seconds_'+meta['selected_regressor'].removeprefix('regressor_')].to_numpy()[indices],rtol=2e-15)


def test_feature_and_training_modules_do_not_import_solvers():
    for name in ['ml/remizov_features.py','diagnostics/train_remizov_ml.py','diagnostics/predict_remizov_ml.py']:
        tree=ast.parse((ROOT/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):
                assert node.module not in ['remizov_solver','chernov_solver','mc_solver']
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
                assert node.func.id not in ['solve_remizov','solve_monte_carlo','ChernovSolver','eval','exec']
