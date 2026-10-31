"""First input-only ML surrogates using saved 1000 Chernov solve results.

No numerical solver imports or calls. Families/splits remain unchanged. All
preprocessing fits train only; validation chooses stopping/threshold/model.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time

for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
DATA=ROOT/'data/all_methods_v1_1000_2026-10-18'
SAVED=ROOT/'results/chernov_solve1000_2026-10-18'
MANAGED=ROOT/'results/chernov_managed_completion_2026-10-18/plan_v2.json'
SEED=20261025


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())


def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,float):return v if math.isfinite(v) else None
    if hasattr(v,'tolist'):return clean(v.tolist())
    if hasattr(v,'item'):return clean(v.item())
    return v


def write(p,v):Path(p).write_text(json.dumps(clean(v),indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')


def make_targets(inputs):
    """Outputs are targets/audit only; context comes from plans written pre-run."""
    import numpy as np
    import pandas as pd
    initial=read(SAVED/'plan.json');managed=read(MANAGED)
    for n,h in initial['protected_sha256'].items():
        if n.startswith('data/') and sha(ROOT/n)!=h:raise ValueError('historical input dataset mismatch')
    sequential=set(managed['pending_ids'])
    if len(sequential)!=300:raise ValueError('expected 300 planned serial tasks')
    by_id={}
    for p in (SAVED/'solutions').glob('*.json'):
        r=read(p);pid=r['problem_id']
        if pid in by_id:raise ValueError('duplicate saved answer')
        by_id[pid]=(p,r)
    if set(by_id)!=set(inputs['ids']):raise ValueError('NPZ/result IDs differ')
    rows=[]
    for pid in inputs['ids']:
        pid=str(pid);m=inputs['metadata'][pid];path,r=by_id[pid]
        row=dict(equation_id=pid,family_id=m['family_id'],split=m['split'],expression_hash=m['expression_hash'],
            structure_signature=m['structure_signature'],complexity=m['complexity_level'],regime=m['regime'],
            status=r['status'],source_result=str(path.relative_to(ROOT)),source_result_sha256=sha(path),
            planned_external_workers=managed['max_compute_workers'] if pid in sequential else initial['workers'],
            execution_cohort='managed_serial1' if pid in sequential else 'initial_parallel8',
            accuracy_met=None,solver_seconds=None,max_abs_error=None,target_known=False)
        if r.get('result') and r['result']['success']:
            if r['metadata']['expression_hash']!=m['expression_hash']:raise ValueError('expression identity mismatch')
            if r['config']!=initial['config']:raise ValueError('mixed numerical configurations')
            if r['implementation']!='chernov_solver.ChernovSolver.solve':raise ValueError('wrong implementation')
            if r['target_x']!=[-1+j/16 for j in range(33)]:raise ValueError('wrong target grid')
            f=np.asarray(r['result']['f']);y=np.asarray(r['y_exact'])
            E=float(np.max(np.abs(f-y)))
            if len(f)!=33 or E!=r['max_abs_error'] or not math.isfinite(E):raise ValueError('stored error mismatch')
            if (E<=.01)!=(r['status']=='accuracy_pass'):raise ValueError('label mismatch')
            if not math.isfinite(r['solver_seconds']) or r['solver_seconds']<=0:raise ValueError('bad runtime')
            prior=managed['created_utc'] if pid in sequential else initial['created_utc']
            if r['finished_utc']<=prior:raise ValueError('execution context was not planned before result')
            row.update(accuracy_met=int(E<=.01),solver_seconds=r['solver_seconds'],max_abs_error=E,target_known=True)
        elif r['status']!='worker_exception':raise ValueError('unexpected unknown target')
        rows.append(row)
    table=pd.DataFrame(rows)
    if (int(table.target_known.sum()),int(table.accuracy_met.sum()))!=(997,871):raise ValueError('target census mismatch')
    return table,initial['config']


def classification_metrics(y,p_success,t_failure):
    import numpy as np
    from sklearn.metrics import (accuracy_score,balanced_accuracy_score,roc_auc_score,
        average_precision_score,f1_score,log_loss,brier_score_loss,confusion_matrix)
    fail=1-np.asarray(y,dtype=int);pf=1-np.asarray(p_success);pred=(pf>=t_failure).astype(int)
    tn,fp,fn,tp=confusion_matrix(fail,pred,labels=[0,1]).ravel()
    return dict(accuracy=accuracy_score(fail,pred),balanced_accuracy=balanced_accuracy_score(fail,pred),
        failure_f1=f1_score(fail,pred,zero_division=0),success_f1=f1_score(1-fail,1-pred,zero_division=0),
        failure_recall=tp/(tp+fn) if tp+fn else None,
        accepted_success_precision=tn/(tn+fn) if tn+fn else None,
        false_accept_bad=int(fn),false_reject_good=int(fp),
        roc_auc=roc_auc_score(fail,pf),failure_average_precision=average_precision_score(fail,pf),
        logloss=log_loss(y,np.column_stack([pf,1-pf]),labels=[0,1]),
        brier=brier_score_loss(y,p_success),confusion_good_bad=confusion_matrix(fail,pred,labels=[0,1]),
        failure_threshold=t_failure,rows=len(y))


def choose_threshold(y,p):
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score,recall_score
    fail=1-np.asarray(y,dtype=int);pf=1-np.asarray(p)
    candidates=np.unique(np.r_[0.,pf,np.nextafter(1.,2.)])
    return float(max(candidates,key=lambda t:(balanced_accuracy_score(fail,pf>=t),
                       recall_score(fail,pf>=t,zero_division=0),t)))


def regression_metrics(y,p):
    import numpy as np
    from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
    return dict(mae_seconds=mean_absolute_error(y,p),rmse_seconds=math.sqrt(mean_squared_error(y,p)),
        mape_percent=float(100*np.mean(np.abs((np.asarray(y)-p)/np.asarray(y)))),
        r2=r2_score(y,p),rows=len(y))


def train(out):
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.dummy import DummyClassifier,DummyRegressor
    from sklearn.linear_model import LogisticRegression,Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from catboost import CatBoostClassifier,CatBoostRegressor
    from ml.remizov_features import load_inputs,VERSION
    from diagnostics.plan_joint_benchmark import protected_snapshot
    began=time.perf_counter()
    out.mkdir(parents=True,exist_ok=False);(out/'models').mkdir();(out/'sources').mkdir()
    shutil.copy2(ROOT/'PROJECT_STATUS.md',out/'PROJECT_STATUS.before.md')
    write(out/'protected_before.json',protected_snapshot(out))
    inputs=load_inputs(DATA);targets,config=make_targets(inputs)
    names=inputs['feature_names'];X=pd.DataFrame(inputs['X'],columns=names)
    Xcontext=X.assign(planned_external_workers=targets.planned_external_workers.to_numpy())
    known=targets.target_known.to_numpy();split=inputs['splits']
    masks={s:(split==s)&known for s in ('train','validation','test')}
    y=targets.accuracy_met.to_numpy(dtype=float);seconds=targets.solver_seconds.to_numpy(dtype=float)
    np.savez_compressed(out/'features.npz',X=inputs['X'],feature_names=np.array(names),equation_id=inputs['ids'],
        family_id=inputs['families'],split=split)
    X.assign(equation_id=inputs['ids']).to_csv(out/'features.csv',index=False)
    targets.to_csv(out/'targets.csv',index=False)
    audit=dict(total=1000,known=997,unknown=3,classes={'accuracy_met':871,'accuracy_not_met':126},
        splits={s:dict(rows=int(np.sum(split==s)),labeled=int(masks[s].sum()),
                      families=len(set(inputs['families'][split==s])),
                      accuracy_met=int(np.nansum(y[masks[s]]))) for s in masks},
        topology_split_overlap=False,dataset_sha256={p.name:sha(p) for p in DATA.glob('all_methods_equations_v1*')},
        timing_context_sources={str(SAVED/'plan.json'):sha(SAVED/'plan.json'),str(MANAGED):sha(MANAGED)},
        context_counts=targets.execution_cohort.value_counts().to_dict(),solver_config=config,
        historical_test_exposure='Prior numerical audits inspected test outcomes; no test-guided tuning in this ML run')
    write(out/'data_audit.json',audit)
    schema=dict(version=VERSION,feature_names=names,runtime_context_feature='planned_external_workers',
        input_grid_sha256=inputs['grid_sha256'],input_grid_points=len(inputs['x_grid']),
        target_grid_points=33,coefficient_channels=['a','b','c_R=c+lambda','g_R=-g'],
        allowed_metadata=['a_lower_cert','a_upper_cert','b_abs_cert','remizov_c_abs_cert','remizov_lambda',
                          'serialized_ast.a_raw','serialized_ast.b_raw','serialized_ast.c_raw'],
        forbidden=['y_exact','dy_exact','d2y_exact','y0','v0','y_shape','support_left','support_right',
                   'target_amplitude','complexity_level','regime','tree_metrics','equation_id','family_id',
                   'structure_signature','seed','split','max_abs_error','solver_seconds','boundary_state_max_abs'],
        note='Identity/splits and diagnostic groups retained outside model matrices. g is the given ODE forcing, not the solution.')
    write(out/'feature_schema.json',schema)
    cb_common=dict(iterations=1200,depth=4,learning_rate=.03,l2_leaf_reg=10.,random_seed=SEED,
                   thread_count=1,allow_writing_files=False,verbose=False)
    protocol=dict(seed=SEED,created_utc=datetime.now(timezone.utc).isoformat(),config=config,
        classifier_catboost={**cb_common,'loss_function':'Logloss','eval_metric':'Logloss'},
        regressor_catboost={**cb_common,'loss_function':'RMSE','target':'log(solver_seconds)'},
        early_stopping_rounds=80,logistic=dict(C=.1,max_iter=3000),ridge=dict(alpha=100.),
        selection='validation ROC-AUC for classifier, validation seconds-MAE for regression; test only after freeze',
        thresholds='learned classifiers: validation maximum balanced accuracy, then failure recall; dummy fixed .5',
        uncertainty='500 bootstrap resamples of whole held-out families, not individual equations',
        training_scope='saved Chernov solve cubic/taper profile only; not zero, adaptive Remizov, or method selection',
        no_new_ode_solutions=True)
    write(out/'training_protocol.json',protocol)
    write(out/'environment.json',dict(python=sys.version,platform=platform.platform(),
        versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','pandas','scikit-learn','catboost','joblib','matplotlib')},
        numerical_dependencies_unchanged={'numpy':'2.2.6','scipy':'1.15.3'},threads=1))
    metrics=dict(classification={},regression={});predictions=targets.copy();files={};thresholds={};timings={};fitted={}
    trainmask,valmask=masks['train'],masks['validation']
    print(json.dumps(dict(event='data_ready',features=len(names),audit=audit['splits'])),flush=True)
    classifiers={'dummy_prior':DummyClassifier(strategy='prior'),
        'logistic':make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=3000,random_state=SEED)),
        'catboost':CatBoostClassifier(**cb_common,loss_function='Logloss',eval_metric='Logloss')}
    for name,model in classifiers.items():
        t=time.perf_counter()
        if name=='catboost':model.fit(X.loc[trainmask],y[trainmask].astype(int),
            eval_set=(X.loc[valmask],y[valmask].astype(int)),early_stopping_rounds=80,use_best_model=True)
        else:model.fit(X.loc[trainmask],y[trainmask].astype(int))
        timings['classifier_'+name]=time.perf_counter()-t
        p=model.predict_proba(X,**({'thread_count':1} if name=='catboost' else {}))[:,1]
        threshold=.5 if name=='dummy_prior' else choose_threshold(y[valmask],p[valmask])
        thresholds[name]=threshold
        metrics['classification'][name]={s:{'validation_threshold':classification_metrics(y[m],p[m],threshold),
            'default_threshold':classification_metrics(y[m],p[m],.5)} for s,m in masks.items()}
        predictions['p_success_'+name]=p;predictions['predicted_success_'+name]=(1-p<threshold).astype(int)
        path=out/'models'/('classifier_'+name+('.cbm' if name=='catboost' else '.joblib'))
        if name=='catboost':model.save_model(str(path))
        else:joblib.dump(model,path)
        files['classifier_'+name]=dict(path=path.name,kind='catboost' if name=='catboost' else 'joblib',features='coefficients',transform='identity')
        fitted['classifier_'+name]=model
        print(json.dumps(dict(event='classifier_fitted',name=name,seconds=timings['classifier_'+name],
            validation_auc=metrics['classification'][name]['validation']['validation_threshold']['roc_auc'])),flush=True)
    regressors={'dummy_median':DummyRegressor(strategy='median'),
        'ridge_coeff':make_pipeline(StandardScaler(),Ridge(alpha=100.)),
        'ridge_context':make_pipeline(StandardScaler(),Ridge(alpha=100.)),
        'catboost_coeff':CatBoostRegressor(**cb_common,loss_function='RMSE'),
        'catboost_context':CatBoostRegressor(**cb_common,loss_function='RMSE')}
    for name,model in regressors.items():
        frame=Xcontext if name.endswith('_context') else X
        logarithmic=name!='dummy_median';target=np.log(seconds) if logarithmic else seconds
        t=time.perf_counter()
        if name.startswith('catboost'):
            model.fit(frame.loc[trainmask],target[trainmask],eval_set=(frame.loc[valmask],target[valmask]),
                      early_stopping_rounds=80,use_best_model=True)
        else:model.fit(frame.loc[trainmask],target[trainmask])
        timings['regressor_'+name]=time.perf_counter()-t
        p=model.predict(frame,**({'thread_count':1} if name.startswith('catboost') else {}))
        if logarithmic:p=np.exp(p)
        metrics['regression'][name]={s:regression_metrics(seconds[m],p[m]) for s,m in masks.items()}
        predictions['seconds_'+name]=p
        path=out/'models'/('regressor_'+name+('.cbm' if name.startswith('catboost') else '.joblib'))
        if name.startswith('catboost'):model.save_model(str(path))
        else:joblib.dump(model,path)
        files['regressor_'+name]=dict(path=path.name,kind='catboost' if name.startswith('catboost') else 'joblib',
            features='context' if name.endswith('_context') else 'coefficients',transform='exp' if logarithmic else 'identity')
        fitted['regressor_'+name]=model
    medians={int(w):float(np.median(seconds[trainmask & (targets.planned_external_workers.to_numpy()==w)]))
             for w in (1,8)}
    p=np.asarray([medians[int(w)] for w in targets.planned_external_workers])
    metrics['regression']['dummy_context']={s:regression_metrics(seconds[m],p[m]) for s,m in masks.items()}
    predictions['seconds_dummy_context']=p
    write(out/'models/regressor_dummy_context.json',medians)
    files['regressor_dummy_context']=dict(path='regressor_dummy_context.json',kind='context_median',features='context',transform='identity')
    selected_c=max(classifiers,key=lambda n:(metrics['classification'][n]['validation']['validation_threshold']['roc_auc'],
                                            -metrics['classification'][n]['validation']['validation_threshold']['logloss']))
    selected_r=min(metrics['regression'],key=lambda n:metrics['regression'][n]['validation']['mae_seconds'])
    metadata=dict(feature_version=VERSION,feature_names=names,runtime_context_feature='planned_external_workers',
        input_grid_sha256=inputs['grid_sha256'],solver_config=config,
        selected_classifier='classifier_'+selected_c,selected_regressor='regressor_'+selected_r,
        failure_threshold=thresholds[selected_c],classifier_thresholds=thresholds,models=files,
        runtime_warning='Historical wall time mixes planned external concurrency 8/1; not isolated CPU cost or a counterfactual single-worker timing.',
        fit_trees={n:int(m.tree_count_) for n,m in fitted.items() if hasattr(m,'tree_count_')})
    write(out/'models/metadata.json',metadata)
    write(out/'metrics.json',metrics);write(out/'fit_seconds.json',timings)
    predictions.to_csv(out/'predictions.csv',index=False)
    # Interpretation only on validation, never used for feature selection/refitting.
    importance=[]
    for key,model in fitted.items():
        if hasattr(model,'feature_importances_'):
            columns=list(Xcontext.columns) if key.endswith('_context') else names
            importance.extend(dict(model=key,feature=n,importance=float(v)) for n,v in zip(columns,model.feature_importances_))
    pd.DataFrame(importance).sort_values(['model','importance'],ascending=[True,False]).to_csv(out/'feature_importance.csv',index=False)
    validation_permutation(out,fitted,X,Xcontext,y,seconds,valmask,metrics)
    error_analysis(out,predictions,selected_c,selected_r)
    bootstrap(out,inputs,targets,predictions,selected_c,selected_r)
    write(out/'run_summary.json',dict(status='completed',training_wall_seconds=time.perf_counter()-began,
        selected_classifier=selected_c,selected_regressor=selected_r,new_numerical_solves=0,
        feature_count=len(names),test_metrics={
            'classification':metrics['classification'][selected_c]['test'],
            'regression':metrics['regression'][selected_r]['test']}))
    for source in ('ml/remizov_features.py','diagnostics/train_remizov_ml.py'):
        shutil.copy2(ROOT/source,out/'sources'/Path(source).name)
    print(json.dumps(clean(read(out/'run_summary.json')),ensure_ascii=False,indent=2),flush=True)


def validation_permutation(out,fitted,X,Xcontext,y,seconds,mask,metrics):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score,mean_absolute_error
    rng=np.random.default_rng(SEED+1);rows=[]
    def group(n):
        if n.startswith('ast_'):return '_'.join(n.split('_')[:3])
        return n.split('_')[0]
    for key in ('classifier_catboost','regressor_catboost_context'):
        frame=X if key.startswith('classifier') else Xcontext
        data=frame.loc[mask].reset_index(drop=True);model=fitted[key]
        groups={g:[c for c in data.columns if group(c)==g] for g in sorted({group(c) for c in data.columns})}
        base=(metrics['classification']['catboost']['validation']['validation_threshold']['roc_auc']
              if key.startswith('classifier') else metrics['regression']['catboost_context']['validation']['mae_seconds'])
        for name,columns in groups.items():
            delta=[]
            for _ in range(10):
                modified=data.copy();modified.loc[:,columns]=data[columns].to_numpy()[rng.permutation(len(data))]
                if key.startswith('classifier'):
                    p=model.predict_proba(modified,thread_count=1)[:,1];delta.append(base-roc_auc_score(y[mask],p))
                else:
                    p=np.exp(model.predict(modified,thread_count=1));delta.append(mean_absolute_error(seconds[mask],p)-base)
            rows.append(dict(model=key,feature_group=name,mean_validation_score_degradation=np.mean(delta),
                             std=np.std(delta,ddof=1),repeats=10))
    pd.DataFrame(rows).to_csv(out/'validation_group_permutation.csv',index=False)


def error_analysis(out,table,classifier,regressor):
    test=table[(table.split=='test')&table.target_known].copy()
    test['false_accept_bad']=(test.accuracy_met==0)&(test['predicted_success_'+classifier]==1)
    test['false_reject_good']=(test.accuracy_met==1)&(test['predicted_success_'+classifier]==0)
    test['runtime_abs_error']=(test.solver_seconds-test['seconds_'+regressor]).abs()
    test.sort_values('runtime_abs_error',ascending=False).to_csv(out/'test_error_analysis.csv',index=False)
    records=[]
    for column in ('regime','complexity','execution_cohort'):
        for name,sub in test.groupby(column):
            records.append(dict(group_type=column,group=name,rows=len(sub),
                known_failures=int((sub.accuracy_met==0).sum()),false_accept_bad=int(sub.false_accept_bad.sum()),
                false_reject_good=int(sub.false_reject_good.sum()),runtime_mae_seconds=sub.runtime_abs_error.mean()))
    import pandas as pd
    pd.DataFrame(records).to_csv(out/'test_subgroups.csv',index=False)


def bootstrap(out,inputs,targets,predictions,classifier,regressor):
    import numpy as np
    from sklearn.metrics import roc_auc_score,balanced_accuracy_score,mean_absolute_error
    mask=(inputs['splits']=='test')&targets.target_known.to_numpy()
    family=inputs['families'][mask];unique=np.unique(family);rng=np.random.default_rng(SEED+2)
    y=targets.accuracy_met.to_numpy()[mask].astype(int);p=predictions['p_success_'+classifier].to_numpy()[mask]
    pred=predictions['predicted_success_'+classifier].to_numpy()[mask];sec=targets.solver_seconds.to_numpy()[mask]
    rt=predictions['seconds_'+regressor].to_numpy()[mask]
    values=[]
    for _ in range(500):
        ii=np.concatenate([np.flatnonzero(family==g) for g in rng.choice(unique,len(unique),replace=True)])
        if len(np.unique(y[ii]))<2:continue
        values.append([roc_auc_score(y[ii],p[ii]),balanced_accuracy_score(y[ii],pred[ii]),mean_absolute_error(sec[ii],rt[ii])])
    a=np.asarray(values)
    write(out/'test_family_bootstrap.json',dict(unit='whole original family, five equations',families=len(unique),
        repetitions=500,valid=len(values),percentile95={n:np.quantile(a[:,i],[.025,.975]) for i,n in
            enumerate(('roc_auc','balanced_accuracy','runtime_mae_seconds'))},
        caveat='Descriptive uncertainty conditional on trained models and the original split; test previously exposed in numerical audits.'))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args();out=args.out.resolve()
    if not out.is_relative_to(ROOT/'results'):ap.error('new output directory under results required')
    train(out)


if __name__=='__main__':main()
