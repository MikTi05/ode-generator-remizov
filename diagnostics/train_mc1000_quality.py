"""Experimental quality surrogates for frozen MC x0 and saved Chernov x0.

No numerical solver imports. MC seed replicas share family/split and features.
Accuracy of a single 1024-path run and of a 4096-path aggregate are distinct.
No model predicts a runtime winner from incomparable point/grid timings.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
SEED=2026102502


def metrics_class(y,p):
    import numpy as np
    from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss,balanced_accuracy_score,accuracy_score
    y=np.asarray(y,dtype=int);p=np.asarray(p);q=np.clip(p,1e-15,1-1e-15)
    two=len(set(y))==2
    return dict(n=len(y),successes=int(sum(y)),success_rate=float(np.mean(y)),
        roc_auc=roc_auc_score(y,p) if two else None,
        failure_average_precision=average_precision_score(1-y,1-p) if two else None,
        logloss=log_loss(y,np.c_[1-q,q],labels=[0,1]),brier=brier_score_loss(y,p),
        accuracy=accuracy_score(y,p>=.5),balanced_accuracy=balanced_accuracy_score(y,p>=.5) if two else None)


def fit_classification(name,X,y,eqidx,frame,out,protocol):
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from catboost import CatBoostClassifier
    from diagnostics.train_remizov_ml import write
    split=frame.split.to_numpy()[eqidx];train=split=='train';val=split=='validation'
    if len(set(y[train]))<2 or len(set(y[val]))<2:
        return dict(skipped='insufficient classes'),{},{}
    cb=dict(protocol['models']['catboost']);stop=cb.pop('early_stopping_rounds')
    models=dict(dummy=DummyClassifier(strategy='prior'),
        logistic=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=3000,random_state=SEED)),
        catboost=CatBoostClassifier(**cb,allow_writing_files=False,verbose=False,loss_function='Logloss'))
    pred={};summary={};fitted={}
    for kind,model in models.items():
        t=time.perf_counter()
        if kind=='catboost':model.fit(X.iloc[eqidx[train]],y[train],eval_set=(X.iloc[eqidx[val]],y[val]),early_stopping_rounds=stop,use_best_model=True)
        else:model.fit(X.iloc[eqidx[train]],y[train])
        p=model.predict_proba(X,**({'thread_count':1} if kind=='catboost' else {}))[:,1]
        if not np.isfinite(p).all():raise ValueError('nonfinite probabilities')
        summary[kind]={s:metrics_class(y[split==s],p[eqidx[split==s]]) for s in ('train','validation','test')}
        summary[kind]['fit_seconds']=time.perf_counter()-t
        path=out/'models'/f'{name}_{kind}'
        if kind=='catboost':model.save_model(str(path.with_suffix('.cbm')))
        else:joblib.dump(model,path.with_suffix('.joblib'))
        pred[kind]=p;fitted[kind]=model
    selected=min(summary,key=lambda k:summary[k]['validation']['logloss'])
    summary['selected_by_validation_logloss']=selected
    if 'catboost' in fitted:
        pd.DataFrame(dict(feature=X.columns,importance=fitted['catboost'].feature_importances_)).sort_values('importance',ascending=False).to_csv(out/f'importance_{name}.csv',index=False)
    calibration=[]
    for kind,p in pred.items():
        for s in ('validation','test'):
            mask=split==s;ys=y[mask];ps=p[eqidx[mask]]
            bins=np.minimum((ps*10).astype(int),9)
            for b in range(10):
                z=bins==b
                if z.any():calibration.append(dict(model=kind,split=s,bin=b,n=int(z.sum()),mean_probability=float(np.mean(ps[z])),observed_success=float(np.mean(ys[z]))))
    pd.DataFrame(calibration).to_csv(out/f'calibration_{name}.csv',index=False)
    write(out/f'classification_{name}.json',summary)
    return summary,pred,fitted


def train(out):
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import mean_absolute_error,r2_score
    from catboost import CatBoostRegressor
    from ml.remizov_features import load_inputs,VERSION
    from diagnostics.train_remizov_ml import read,write,sha
    began=time.perf_counter();folder=out/'ml';folder.mkdir(exist_ok=False);(folder/'models').mkdir()
    protocol=read(out/'analysis_protocol.json');inputs=load_inputs(ROOT/'data/all_methods_v1_1000_2026-10-18')
    eq=pd.read_csv(out/'analysis/equations.csv').set_index('equation_id').loc[inputs['ids']].reset_index()
    att=pd.read_csv(out/'analysis/attempts.csv');X=pd.DataFrame(inputs['X'],columns=inputs['feature_names'])
    if not np.array_equal(eq.split.to_numpy(),inputs['splits']) or not np.array_equal(eq.family_id.to_numpy(),inputs['families']):raise ValueError('split/family mismatch')
    if eq.groupby('family_id').split.nunique().max()!=1:raise ValueError('family leakage')
    np.savez_compressed(folder/'features.npz',X=X.to_numpy(),feature_names=np.asarray(X.columns),equation_id=inputs['ids'],family_id=inputs['families'],split=inputs['splits'])
    schema=read(ROOT/'results/remizov_ml_v1_2026-10-25/feature_schema.json')
    schema.update(runtime_context_feature=None,target_scope='fixed numerical profiles at x=0; input-only coefficients')
    write(folder/'feature_schema.json',schema)
    predictions=eq.copy();summary=dict(classification={},regression={},comparative={});fitted={}
    index={pid:i for i,pid in enumerate(eq.equation_id)}
    rmask=eq.remizov_known.to_numpy(dtype=bool);valid=att.status.eq('completed').to_numpy()
    seedrows=att.loc[valid];mc_indices=np.asarray([index[p] for p in seedrows.equation_id],dtype=int)
    targets={'remizov_point_accuracy':(eq.loc[rmask,'remizov_accuracy0'].to_numpy(dtype=int),np.flatnonzero(rmask)),
             'mc_single_seed_accuracy':(seedrows.accuracy_met.to_numpy(dtype=int),mc_indices)}
    labels=eq.quality_label.to_numpy();cmask=np.isin(labels,['mc_mean_closer_exploratory','remizov_closer_exploratory'])
    gate={s:dict(n=int(sum(cmask&(eq.split.to_numpy()==s))),
        mc=int(sum(cmask&(eq.split.to_numpy()==s)&(labels=='mc_mean_closer_exploratory'))),
        remizov=int(sum(cmask&(eq.split.to_numpy()==s)&(labels=='remizov_closer_exploratory')))) for s in ('train','validation','test')}
    train_gate=cmask&(eq.split.to_numpy()=='train')
    adequate=gate['train']['n']>=100 and eq.loc[train_gate,'family_id'].nunique()>=40 and min(gate['train']['mc'],gate['train']['remizov'])>=30 and all(min(gate[s]['mc'],gate[s]['remizov'])>=10 for s in ('validation','test'))
    summary['comparative']=dict(gate=gate,trained=adequate,
        limitation='Exploratory fixed-profile mean-error label, approximate t95 from 4 seeds; no fastest-method or 95%-reliability certificate',
        reason='predeclared sample/class/family thresholds met' if adequate else 'insufficient separated labels/classes; separate quality models are primary')
    if adequate:targets['exploratory_comparative_quality']=((labels[cmask]=='mc_mean_closer_exploratory').astype(int),np.flatnonzero(cmask))
    bootstrap_data={}
    for name,(y,ix) in targets.items():
        sm,pred,models=fit_classification(name,X,y,ix,eq,folder,protocol);summary['classification'][name]=sm
        for k,p in pred.items():predictions[f'p_{name}_{k}']=p
        fitted[name]=models;bootstrap_data[name]=(y,ix,pred)
        print(json.dumps(dict(event='classified',target=name,selected=sm.get('selected_by_validation_logloss'))),flush=True)
    common=dict(protocol['models']['catboost']);stop=common.pop('early_stopping_rounds')
    for name,col,mask in [('remizov_point_log_error','remizov_error0',rmask),('mc_aggregate_log_error','mc_aggregate_error',eq.mc_all_four.to_numpy(dtype=bool))]:
        y=np.log10(np.maximum(eq[col].to_numpy(dtype=float),1e-12));tr=mask&eq.split.eq('train').to_numpy();va=mask&eq.split.eq('validation').to_numpy()
        regressors=dict(dummy=DummyRegressor(strategy='median'),ridge=make_pipeline(StandardScaler(),Ridge(alpha=100.)),
            catboost=CatBoostRegressor(**common,loss_function='RMSE',allow_writing_files=False,verbose=False))
        sm={}
        for kind,model in regressors.items():
            if kind=='catboost':model.fit(X.loc[tr],y[tr],eval_set=(X.loc[va],y[va]),early_stopping_rounds=stop,use_best_model=True)
            else:model.fit(X.loc[tr],y[tr])
            p=model.predict(X,**({'thread_count':1} if kind=='catboost' else {}))
            if not np.isfinite(p).all():raise ValueError('nonfinite log error')
            sm[kind]={s:dict(n=int(sum(m)),mae_log10=mean_absolute_error(y[m],p[m]),
                mae_error=mean_absolute_error(eq.loc[m,col],10.**p[m]),r2_log10=r2_score(y[m],p[m])) for s in ('train','validation','test') if (m:=mask&eq.split.eq(s).to_numpy()).any()}
            predictions[f'logerror_{name}_{kind}']=p
            path=folder/'models'/f'{name}_{kind}'
            if kind=='catboost':
                model.save_model(str(path.with_suffix('.cbm')))
                pd.DataFrame(dict(feature=X.columns,importance=model.feature_importances_)).sort_values('importance',ascending=False).to_csv(folder/f'importance_{name}.csv',index=False)
            else:joblib.dump(model,path.with_suffix('.joblib'))
        sm['selected_by_validation_mae']=min(regressors,key=lambda k:sm[k]['validation']['mae_log10']);summary['regression'][name]=sm
    # Family bootstrap preserves all equations and all seed replicas together.
    rng=np.random.default_rng(SEED);uncertainty={}
    for name,(y,ix,preds) in bootstrap_data.items():
        if not preds:continue
        tm=eq.split.to_numpy()[ix]=='test';yt=y[tm];ids=ix[tm];groups=eq.family_id.to_numpy()[ids];families=np.unique(groups)
        vals={k:[] for k in preds};deltas=[]
        for _ in range(500):
            sampled=rng.choice(families,len(families),replace=True);chosen=np.concatenate([np.flatnonzero(groups==f) for f in sampled])
            losses={k:float(np.mean(-(yt[chosen]*np.log(np.clip(p[ids[chosen]],1e-15,1-1e-15))+(1-yt[chosen])*np.log(np.clip(1-p[ids[chosen]],1e-15,1-1e-15))))) for k,p in preds.items()}
            for k,v in losses.items():vals[k].append(v)
            if 'catboost' in losses:deltas.append(losses['catboost']-losses['dummy'])
        uncertainty[name]=dict(test_logloss_95={k:np.quantile(v,[.025,.975]).tolist() for k,v in vals.items()},
            catboost_minus_dummy_logloss_95=np.quantile(deltas,[.025,.975]).tolist() if deltas else None,
            scope='conditional on these observed seed outcomes; 500 whole-family resamples')
    summary['uncertainty']=uncertainty;summary['elapsed_seconds']=time.perf_counter()-began
    summary['features']=len(X.columns);summary['split_rows']=eq.split.value_counts().to_dict()
    summary['training_scope']='Preliminary quality of fixed profiles at x=0. MC success probability is for one 1024-path seed, aggregate error for 4x1024. No runtime winner.'
    predictions.to_csv(folder/'predictions.csv',index=False);write(folder/'metrics.json',summary)
    write(folder/'reproducibility.json',dict(seed=SEED,protocol_sha256=sha(out/'analysis_protocol.json'),
        feature_code_sha256=sha(ROOT/'ml/remizov_features.py'),feature_version=VERSION,
        analysis_equations_sha256=sha(out/'analysis/equations.csv'),analysis_attempts_sha256=sha(out/'analysis/attempts.csv'),
        historical_test_exposure='Original family split preserved; test outcomes previously audited numerically; not prospective pristine holdout',
        label_censoring='Only completed MC outcomes and successful saved Chernov; resource failures and pending excluded'))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);train(ap.parse_args().out.resolve())
