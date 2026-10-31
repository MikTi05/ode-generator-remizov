"""Reproducible A--E research with a separate, one-shot final test stage.

Usage: python -B diagnostics/run_selector_hypotheses.py {prepare,train,final} --out PATH
No ODE solver or equation generator is imported or executed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from ml.accuracy_selector import CATBOOST_CONFIG, METHODS, sha
from ml.selector_hypotheses import (choices, metrics, selection_key,
                                    stratified_family_split, threshold_scan)
from ml.selector_research_features import additional_coefficient_features, FEATURE_NAMES

SOURCE = ROOT/'results/ml_mean_mc_selector_2026-10-31/run'
DATA = ROOT/'data/all_methods_v1_1000_2026-10-18'
SEED = 2488
CONFIGS = [dict(name='baseline',iterations=120,depth=2,l2_leaf_reg=10),
           dict(name='medium',iterations=240,depth=3,l2_leaf_reg=10),
           dict(name='deeper',iterations=300,depth=4,l2_leaf_reg=20)]
WEIGHTS = [1,2,3,5,8,12]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n')


def csv_write(path, rows):
    if not rows:
        return
    keys = list(rows[0])
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v
                             for k,v in row.items()})


def summary(rows):
    return dict(n=len(rows),classes=dict(Counter(r['best_method'] for r in rows)),
                families=len({r['family_id'] for r in rows}))


def pins():
    files=set()
    for folder in (DATA,SOURCE.parent,ROOT/'results/ml_accuracy_selector_2026-10-31'):
        files.update(p for p in folder.rglob('*') if p.is_file())
    for folder in ('ml','src'):
        files.update(p for p in (ROOT/folder).rglob('*.py'))
    files.update(ROOT/p for p in ('MIGRATION_MANIFEST.json','remizov_solver.py',
                 'chernov_solver.py','mc_solver.py','all_methods_dataset_loader.py') if (ROOT/p).exists())
    # All original numerical provenance, including old benchmark results.
    files.update(ROOT/p for p in read(SOURCE/'evidence.json')['source_sha256'] if (ROOT/p).is_file())
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}


def check_pins(out):
    expected=read(out/'preserved_sha256.json')
    changed=[p for p,h in expected.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    if changed:
        raise ValueError('preserved sources changed: '+str(changed[:5]))
    return dict(checked_files=len(expected),changed_files=changed)


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(ROOT/'PROJECT_STATUS.md',out/'PROJECT_STATUS.before.md')
    write(out/'preserved_sha256.json',pins())
    evidence=read(SOURCE/'evidence.json')
    rows=evidence['main_rows']
    assert len(rows)==997 and len({r['equation_id'] for r in rows})==997
    assert len({r['expression_hash'] for r in rows})==997
    for r in rows:
        assert r['E_R']==abs(r['remizov_f0']-r['y_exact0'])
        assert r['E_MC']==abs(float(np.mean(r['mc_seed_f0']))-r['y_exact0'])
        assert r['best_method']==METHODS[int(r['E_MC']<r['E_R'])]
    new,algorithm=stratified_family_split(rows,SEED)
    with np.load(SOURCE/'input_features.npz',allow_pickle=False) as z:
        lookup={v:i for i,v in enumerate(z['equation_id'].tolist())}
        X=z['X'][[lookup[r['equation_id']] for r in rows]]
        names=z['feature_names'].tolist()
    with np.load(DATA/'all_methods_equations_v1_ml.npz',allow_pickle=False) as z:
        lookup={v:i for i,v in enumerate(z['equation_id'].tolist())}
        raw=z['X']; x=z['x_grid']
        extra=np.array([list(additional_coefficient_features(raw[lookup[r['equation_id']]],x).values()) for r in rows])
    assert X.shape==(997,253) and extra.shape==(997,5)
    np.savez_compressed(out/'features.npz',X=X,augmented=np.c_[X,extra],
                        feature_names=names,augmented_names=names+list(FEATURE_NAMES),
                        equation_id=[r['equation_id'] for r in rows])
    manifests={}
    for scenario,split_map in (('old',{r['equation_id']:r['split'] for r in rows}),('new',new)):
        groups={};topologies={}
        for r in rows:
            s=split_map[r['equation_id']]
            assert groups.setdefault(r['family_id'],s)==s
            assert topologies.setdefault(r['structure_signature'],s)==s
        manifests[scenario]={s:dict(ids=[r['equation_id'] for r in rows if split_map[r['equation_id']]==s],
             **summary([r for r in rows if split_map[r['equation_id']]==s])) for s in ('train','validation','test')}
    assert manifests['old']['test']['ids']==manifests['new']['test']['ids']
    write(out/'splits.json',dict(seed=SEED,algorithm=algorithm,scenarios=manifests,
          identities=[{k:r[k] for k in ('equation_id','expression_hash','dataset_sha256','family_id','structure_signature')} for r in rows]))
    fields=('equation_id','family_id','structure_signature','E_R','E_MC','best_method')
    dev=[{**{k:r[k] for k in fields},'old_split':r['split'],'new_split':new[r['equation_id']],
           'delta':r['E_R']-r['E_MC']} for r in rows if r['split']!='test']
    write(out/'development_rows.json',dev)
    csv_write(out/'development_targets.csv',dev)
    plan=dict(schema='selector_hypotheses_v1',source=str(SOURCE.relative_to(ROOT)),
       scope='retrospective exploratory comparison on previously disclosed test; x=0 only',
       winner='MC iff E_MC < E_R; ties R',
       errors='E_R=abs(saved_R128-exact0); E_MC=abs(mean(four saved MC1024)-exact0)',
       primary='minimum validation mean_regret = minimum mean selected absolute error minus fixed oracle mean',
       tie_breakers=['higher success at epsilon=.05','higher macro_f1','fewer MC choices','stable candidate ID'],
       numerical_calls=0,new_equations=0,seed=SEED,thread_count=1,
       classifier_base=CATBOOST_CONFIG,configs=CONFIGS,class_weights_MC=WEIGHTS,
       features=['original253','augmented258'],scenarios=['new','old'],
       regression_targets=['raw_delta','asinh(delta/train_median_abs_delta)'],
       regression_decision='MC iff predicted_delta > threshold; strict sign at threshold0',
       classifier_thresholds='0.05:0.05:0.50 plus every validation score partition including constant policies',
       regression_thresholds='0 plus every validation score partition including constant policies',
       threshold_selection='same primary criterion, validation only; no fit on validation',
       calibration='descriptive validation Brier/logloss/ECE only; weighted scores not assumed calibrated; no calibration fit',
       final_stage='lock every candidate/threshold and selected policy before one final pass on fixed test; no changes afterwards',
       cv='5 stratified family folds within each train; fixed-default comparisons plus nested inner one-quarter group holdout selection over baseline original253 six weights two regressions and constants; refit chosen spec on outer train',
       fit_budget=dict(main=96,cv_default_and_nested_max=170,upper_total=266),
       resources='one process, CatBoost and BLAS one thread; no universal wall limit',
       split_algorithm=algorithm)
    write(out/'plan.json',plan)
    write(out/'environment.json',dict(python=sys.version,platform=platform.platform(),
          versions={p:importlib.metadata.version(p) for p in ('numpy','scipy','catboost','scikit-learn')},
          thread_env={k:v for k,v in os.environ.items() if 'THREAD' in k}))
    print('PREPARED', {k:{s:v['n'] for s,v in m.items()} for k,m in manifests.items()},flush=True)


def load_dev(out):
    rows=read(out/'development_rows.json')
    with np.load(out/'features.npz',allow_pickle=False) as z:
        ids=z['equation_id'].tolist(); lookup={p:i for i,p in enumerate(ids)}
        matrices={key:z[key][[lookup[r['equation_id']] for r in rows]] for key in ('X','augmented')}
        names={'X':z['feature_names'].tolist(),'augmented':z['augmented_names'].tolist()}
    return rows,matrices,names


def fit_model(kind, config, weight, X, rows, names):
    params=copy.deepcopy(CATBOOST_CONFIG)
    params.update({k:v for k,v in config.items() if k!='name'})
    scale=1.
    if kind=='classifier':
        if weight!=1:
            params['class_weights']=[1,weight]
        model=CatBoostClassifier(**params)
        y=[r['best_method'] for r in rows]
    else:
        params.pop('class_names');params.pop('allow_const_label')
        params['loss_function']='RMSE'
        model=CatBoostRegressor(**params)
        y=np.array([r['E_R']-r['E_MC'] for r in rows])
        if kind=='asinh':
            scale=max(float(np.median(np.abs(y))),1e-12)
            y=np.arcsinh(y/scale)
    model.fit(Pool(X,label=y,feature_names=names))
    return model,scale


def score_model(model,kind,scale,X,names):
    pool=Pool(X,feature_names=names)
    if kind=='classifier':
        return model.predict_proba(pool,thread_count=1)[:,list(model.classes_).index(METHODS[1])]
    scores=model.predict(pool,thread_count=1)
    return scale*np.sinh(scores) if kind=='asinh' else scores


def calibration(rows,scores):
    y=np.array([r['best_method']==METHODS[1] for r in rows],int)
    bins=[]
    for lower in np.arange(0,1,.1):
        own=(scores>=lower)&(scores<(lower+.1) if lower<.9 else scores<=1)
        if own.any():
            bins.append(dict(lower=float(lower),n=int(own.sum()),mean_score=float(scores[own].mean()),
                             MC_fraction=float(y[own].mean())))
    return dict(brier=float(brier_score_loss(y,scores)),logloss=float(log_loss(y,scores,labels=[0,1])),
                ece=float(sum(b['n']*abs(b['mean_score']-b['MC_fraction']) for b in bins)/len(rows)),
                bins=bins,interpretation='descriptive validation only; no calibration or probability guarantee')


def train(out):
    if (out/'selection_lock.json').exists():
        raise ValueError('selection already locked; use a new output for another study')
    if (out/'candidates.json').exists():
        raise ValueError('partial study exists: preserve it and start a separate output')
    plan=read(out/'plan.json'); assert plan['configs']==CONFIGS and plan['class_weights_MC']==WEIGHTS
    check_pins(out)
    rows,matrices,names=load_dev(out)
    candidates=[]; policy_rows=[]; validation_predictions=[]; importance=[]; calibration_rows=[]
    (out/'models').mkdir(exist_ok=False)
    (out/'thresholds').mkdir(exist_ok=False)
    started=time.perf_counter()
    # A/B/C on original features; D ablation; E repeats the full frozen grid on old split.
    for scenario in ('new','old'):
        ti=[i for i,r in enumerate(rows) if r[scenario+'_split']=='train']
        vi=[i for i,r in enumerate(rows) if r[scenario+'_split']=='validation']
        tr=[rows[i] for i in ti]; va=[rows[i] for i in vi]
        for constant,t in (('always_remizov',1.),('always_monte_carlo',-1.)):
            result=metrics(va,[METHODS[int(t<0)]]*len(va))
            cid=f'{scenario}__{constant}'
            c=dict(id=cid,scenario=scenario,kind='constant',constant=METHODS[int(t<0)],
                   feature_set='X',threshold=0.,config={},weight=1,scale=1.,model=None)
            candidates.append(c)
            policy_rows.append(dict(candidate_id=cid,scenario=scenario,policy='constant',threshold=0.,**result))
        for feature_set in ('X','augmented'):
            X=matrices[feature_set]; own_names=names[feature_set]
            for kind in ('classifier','raw','asinh'):
                for weight in (WEIGHTS if kind=='classifier' else [1]):
                    for config in CONFIGS:
                        cid=f'{scenario}__{feature_set}__{kind}__w{weight}__{config["name"]}'
                        fit_start=time.perf_counter()
                        model,scale=fit_model(kind,config,weight,X[ti],tr,own_names)
                        model_path=out/'models'/f'{cid}.cbm';model.save_model(str(model_path))
                        scores=score_model(model,kind,scale,X[vi],own_names)
                        scan,best=threshold_scan(va,scores,kind=='classifier')
                        csv_write(out/'thresholds'/f'{cid}.csv',scan)
                        default=.5 if kind=='classifier' else 0.
                        fixed=metrics(va,choices(scores,default))
                        c=dict(id=cid,scenario=scenario,kind=kind,feature_set=feature_set,
                               config=config,weight=weight,scale=scale,threshold=best['threshold'],
                               model=str(model_path.relative_to(out)),model_sha256=sha(model_path),
                               fit_seconds=time.perf_counter()-fit_start)
                        candidates.append(c)
                        for label,t,result in (('default',default,fixed),('validation_optimal',best['threshold'],best)):
                            policy_rows.append(dict(candidate_id=cid,scenario=scenario,policy=label,threshold=t,
                                                    **{k:v for k,v in result.items() if k not in ('threshold','fixed_grid')}))
                        if kind=='classifier':
                            for p in scan:
                                if p['fixed_grid'] and p['threshold']!=default:
                                    policy_rows.append(dict(candidate_id=cid,scenario=scenario,
                                        policy=f'grid_{p["threshold"]:.2f}',
                                        **{k:v for k,v in p.items() if k!='fixed_grid'}))
                        for r,score in zip(va,scores):
                            validation_predictions.append(dict(candidate_id=cid,equation_id=r['equation_id'],
                                scenario=scenario,score=float(score),best_method=r['best_method'],
                                E_R=r['E_R'],E_MC=r['E_MC']))
                        for name,value in zip(own_names,model.get_feature_importance()):
                            importance.append(dict(candidate_id=cid,feature=name,importance=float(value)))
                        if kind=='classifier':
                            calibration_rows.append(dict(candidate_id=cid,**calibration(va,scores)))
                        print('FIT',cid,'val_regret',best['mean_regret'],'MC',best['mc_correct'],best['mc_false'],flush=True)
    write(out/'candidates.json',candidates)
    write(out/'validation_metrics.json',policy_rows)
    csv_write(out/'validation_metrics.csv',policy_rows)
    csv_write(out/'validation_predictions.csv',validation_predictions)
    csv_write(out/'feature_importance.csv',importance)
    write(out/'calibration.json',calibration_rows)
    # Freeze selection BEFORE any test prediction or evaluation; CV cannot change it.
    selected={}
    for scenario in ('new','old'):
        selected[scenario]=min((p for p in policy_rows if p['scenario']==scenario),
                              key=lambda p:(selection_key(p),p['candidate_id'],p['policy']))
    source_files=['diagnostics/run_selector_hypotheses.py','ml/selector_hypotheses.py',
                  'ml/selector_research_features.py','ml/accuracy_selector.py']
    (out/'sources').mkdir()
    for file in source_files:
        dest=out/'sources'/file;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/file,dest)
    write(out/'selection_lock.json',dict(plan_sha256=sha(out/'plan.json'),
          candidates_sha256=sha(out/'candidates.json'),splits_sha256=sha(out/'splits.json'),
          validation_metrics_sha256=sha(out/'validation_metrics.json'),
          selected=selected,test_evaluated=False,test_used_for_selection=False,
          fit_count=len([c for c in candidates if c['kind']!='constant']),
          source_sha256={p:sha(ROOT/p) for p in source_files},
          main_fit_seconds=time.perf_counter()-started))
    print('LOCKED', {s:(v['candidate_id'],v['threshold']) for s,v in selected.items()},flush=True)


def final(out):
    if (out/'test_metrics.json').exists() or (out/'FINAL_STARTED.json').exists():
        raise ValueError('one-shot final stage already started; no automatic rerun')
    lock=read(out/'selection_lock.json')
    for file,key in (('plan.json','plan_sha256'),('candidates.json','candidates_sha256'),
                     ('splits.json','splits_sha256'),('validation_metrics.json','validation_metrics_sha256')):
        assert sha(out/file)==lock[key]
    assert (out/'cv_summary.json').exists() and (out/'feature_analysis.json').exists()
    check_pins(out)
    write(out/'FINAL_STARTED.json',dict(selection_lock_sha256=sha(out/'selection_lock.json'),
          note='All test variants fixed before this marker. No subsequent tuning is authorized by this protocol.'))
    evidence=read(SOURCE/'evidence.json')
    test=[r for r in evidence['main_rows'] if r['split']=='test']
    assert [r['equation_id'] for r in test]==read(out/'splits.json')['scenarios']['new']['test']['ids']
    with np.load(out/'features.npz',allow_pickle=False) as z:
        lookup={p:i for i,p in enumerate(z['equation_id'].tolist())}
        ti=[lookup[r['equation_id']] for r in test]
        matrices={k:z[k][ti] for k in ('X','augmented')}
        names={'X':z['feature_names'].tolist(),'augmented':z['augmented_names'].tolist()}
    val=read(out/'validation_metrics.json')
    output=[];predictions=[]
    for c in read(out/'candidates.json'):
        own=[p for p in val if p['candidate_id']==c['id']]
        if c['kind']=='constant':
            scores=np.full(len(test),1. if c['constant']==METHODS[1] else -1.)
        else:
            assert sha(out/c['model'])==c['model_sha256']
            model=CatBoostClassifier() if c['kind']=='classifier' else CatBoostRegressor()
            model.load_model(str(out/c['model']))
            scores=score_model(model,c['kind'],c['scale'],matrices[c['feature_set']],names[c['feature_set']])
        for policy in own:
            selected=choices(scores,policy['threshold'])
            result=metrics(test,selected)
            is_selected=all(policy[k]==lock['selected'][c['scenario']][k] for k in ('candidate_id','policy','threshold'))
            output.append(dict(candidate_id=c['id'],scenario=c['scenario'],policy=policy['policy'],
                threshold=policy['threshold'],selected_on_validation=is_selected,**result))
            for r,score,pred in zip(test,scores,selected):
                error=r['E_R'] if pred==METHODS[0] else r['E_MC']
                predictions.append(dict(candidate_id=c['id'],policy=policy['policy'],
                    equation_id=r['equation_id'],family_id=r['family_id'],score=float(score),
                    threshold=policy['threshold'],best_method=r['best_method'],selected_method=pred,
                    E_R=r['E_R'],E_MC=r['E_MC'],E_selected=error,regret=error-min(r['E_R'],r['E_MC'])))
    oracle=metrics(test,[r['best_method'] for r in test])
    write(out/'test_metrics.json',dict(policies=output,oracle=oracle,
          selected={s:next(p for p in output if p['scenario']==s and p['selected_on_validation']) for s in ('new','old')},
          exploratory=True,independent_test=False,test_evaluation_passes=1))
    csv_write(out/'test_metrics.csv',output)
    csv_write(out/'test_predictions.csv',predictions)
    write(out/'preservation_verification.json',check_pins(out))
    print('FINAL', {p['scenario']:(p['candidate_id'],p['mc_correct'],p['mc_false'],p['mean_regret'])
                    for p in output if p['selected_on_validation']},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','train','final'])
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    globals()[args.stage](args.out.resolve())
