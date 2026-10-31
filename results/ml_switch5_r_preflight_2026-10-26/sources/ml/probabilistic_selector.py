"""Accuracy-probability and full-wall heads, with explicit uncalibrated uncertainty.

Both methods remain candidate actions. No winner labels or numerical solvers.
"""
from __future__ import annotations
import hashlib,json,math,time
from pathlib import Path
import numpy as np

PROFILES=('R128','R256','MC1')
VERSION='risk_constrained_typical_cost_v1'


def probability_scores(k,n,p):
    k=np.asarray(k,float);n=np.asarray(n,float);p=np.asarray(p,float)
    if np.any(n<=0) or np.any(k<0) or np.any(k>n) or np.any(~np.isfinite(p)) or np.any((p<0)|(p>1)):raise ValueError('invalid binomial evidence')
    s=k/n;clipped=np.clip(p,1e-10,1-1e-10)
    return dict(brier=float(np.mean(s*(1-p)**2+(1-s)*p**2)),
        logloss=float(np.mean(-s*np.log(clipped)-(1-s)*np.log(1-clipped))),
        mean_prediction=float(np.mean(p)),observed_rate_equal_equation_weight=float(np.mean(s)),equations=len(k),trials=int(np.sum(n)))


def choose(estimates,q,*,robust=False):
    """Uncertainty-aware policy, never an accuracy guarantee or a winner classifier."""
    if not 0<q<1:raise ValueError('q must be explicitly supplied in (0,1)')
    for p,e in estimates.items():
        a=e['probability'];t=e['cost']
        if not 0<=a['low']<=a['point']<=a['high']<=1 or not 0<t['low']<=t['point']<=t['high']<math.inf:raise ValueError('invalid estimates')
    eligible=[p for p,e in estimates.items() if e['probability']['low' if robust else 'point']>=q]
    if robust and any(e['probability']['high']>=q and not e['cost']['supported'] for e in estimates.values()):
        return dict(action='abstain',reason='potential_candidate_time_head_insufficient',q=q)
    if not eligible:return dict(action='abstain',reason='probability_requirement_unresolved' if any(e['probability']['high']>=q for e in estimates.values()) else 'no_predicted_feasible_method',q=q)
    ranked=sorted(eligible,key=lambda p:(estimates[p]['cost']['point'],p=='MC1',p=='R256'))
    fastest=ranked[0]
    if not robust:
        rs=[p for p in eligible if p!='MC1'];bestR=min(rs,key=lambda p:(estimates[p]['cost']['point'],p=='R256')) if rs else None
        if bestR and fastest=='MC1' and estimates['MC1']['cost']['point']>.9*estimates[bestR]['cost']['point']:
            fastest=bestR
        return dict(action=fastest,reason='point_estimate_risk_constraint_and10pct_tie_policy',q=q)
    # A possibly feasible faster alternative prevents a confident selection.
    for p,e in estimates.items():
        if p==fastest or e['probability']['high']<q:continue
        if e['probability']['low']<q and e['cost']['low']<=estimates[fastest]['cost']['high']/0.9:
            return dict(action='abstain',reason='competing_method_probability_unresolved',q=q)
        if estimates[fastest]['cost']['high']>.9*e['cost']['low']:
            return dict(action='abstain',reason='cost_order_unresolved',q=q)
    return dict(action=fastest,reason='exploratory_model_spread_dominance_NOT_a_certificate',q=q)


class Head:
    def __init__(self,kind,scaler,models,metadata):self.kind=kind;self.scaler=scaler;self.models=models;self.metadata=metadata
    def predict(self,X,names):
        from catboost import Pool
        Z=self.scaler.transform(X);columns=[]
        for model in self.models:
            if isinstance(model,float):pred=np.full(len(X),model)
            elif self.kind=='probability':pred=model.predict_proba(Pool(Z,feature_names=names),thread_count=1)[:,1]
            else:pred=model.predict(Pool(Z,feature_names=names),thread_count=1)
            columns.append(np.exp(pred) if self.kind=='cost' else pred)
        values=np.asarray(columns);point=values[0]
        low=np.minimum(point,np.quantile(values[1:],.1,axis=0));high=np.maximum(point,np.quantile(values[1:],.9,axis=0))
        return dict(point=point,low=low,high=high)


def fit_head(X,rows,names,kind,*,seed=2488,replicas=10,iterations=120):
    from sklearn.preprocessing import StandardScaler
    from catboost import CatBoostClassifier,CatBoostRegressor,Pool
    if not rows or any(r['split']!='train' for r in rows):raise ValueError('observed original-train rows required')
    if kind not in ('probability','cost'):raise ValueError('unknown head')
    indices=np.array([r['index'] for r in rows]);groups=np.array([r['family_id'] for r in rows]);unique=np.unique(groups)
    scaler=StandardScaler().fit(X[indices]);Z=scaler.transform(X[indices])
    if kind=='probability':
        n=np.array([r['n'] for r in rows],float);k=np.array([r['k'] for r in rows],float)
        if np.any(n<=0) or np.any(k<0) or np.any(k>n):raise ValueError('invalid real binomial counts')
        y=k/n;weights=n
    else:
        y=np.log([r['seconds'] for r in rows]);weights=np.ones(len(rows))
        if not np.isfinite(y).all():raise ValueError('finite positive full times required')
    rng=np.random.default_rng(seed);models=[];constant_count=0;started=time.perf_counter()
    for b in range(replicas+1):
        multiplicity={g:1 for g in unique} if b==0 else {g:0 for g in unique}
        if b:
            for g in rng.choice(unique,size=len(unique),replace=True):multiplicity[g]+=1
        w=weights*np.array([multiplicity[g] for g in groups]);active=w>0
        if len(np.unique(y[active]))==1 or (kind=='cost' and len(unique)<8):
            model=float(np.average(y[active],weights=w[active]));constant_count+=1
        else:
            cls=CatBoostClassifier if kind=='probability' else CatBoostRegressor
            model=cls(loss_function='CrossEntropy' if kind=='probability' else 'RMSE',iterations=iterations,depth=2,
                learning_rate=.05,l2_leaf_reg=10.,bootstrap_type='No',random_strength=0,random_seed=seed+b,
                thread_count=1,allow_writing_files=False,verbose=False)
            model.fit(Pool(Z[active],label=y[active],weight=w[active],feature_names=names))
        models.append(model)
    meta=dict(kind=kind,indices=indices.tolist(),families=sorted(set(groups)),rows=len(rows),replicas=replicas,
        iterations=iterations,fit_seconds=time.perf_counter()-started,constant_replicas=constant_count,
        train_scaler_mean=scaler.mean_.tolist(),supported_for_regression=kind!='cost' or len(unique)>=8,
        uncertainty='10/90 percentiles of family-bootstrap model predictions; UNCALIBRATED model spread, not CI or reliability lower bound')
    return Head(kind,scaler,models,meta)


def save_head(head,out,prefix,names,X):
    from sklearn.preprocessing import StandardScaler
    from catboost import CatBoostClassifier,CatBoostRegressor
    out=Path(out);scalepath=out/f'{prefix}_scaler.npz'
    np.savez_compressed(scalepath,mean=head.scaler.mean_,scale=head.scaler.scale_,var=head.scaler.var_)
    entries=[]
    for b,model in enumerate(head.models):
        if isinstance(model,float):entries.append(dict(constant=model));continue
        path=out/f'{prefix}_{b}.cbm';model.save_model(str(path));entries.append(dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    entry=dict(kind=head.kind,metadata=head.metadata,models=entries,scaler=dict(path=scalepath.name,sha256=hashlib.sha256(scalepath.read_bytes()).hexdigest()))
    restored=load_head(entry,out)
    a=head.predict(X,names);b=restored.predict(X,names)
    if any(not np.array_equal(a[k],b[k]) for k in a):raise ValueError('reload changed predictions')
    entry['reload_exact']=True;return entry


def load_head(entry,folder):
    from sklearn.preprocessing import StandardScaler
    from catboost import CatBoostClassifier,CatBoostRegressor
    folder=Path(folder);s=entry['scaler'];path=folder/s['path']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=s['sha256']:raise ValueError('scaler hash mismatch')
    scaler=StandardScaler()
    with np.load(path,allow_pickle=False) as z:scaler.mean_=z['mean'];scaler.scale_=z['scale'];scaler.var_=z['var'];scaler.n_features_in_=len(z['mean'])
    models=[]
    for item in entry['models']:
        if 'constant' in item:models.append(float(item['constant']));continue
        path=folder/item['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('model hash mismatch')
        cls=CatBoostClassifier if entry['kind']=='probability' else CatBoostRegressor
        model=cls();model.load_model(str(path));models.append(model)
    return Head(entry['kind'],scaler,models,entry['metadata'])


def estimates_from_heads(heads,X,names):
    predicted={key:head.predict(X,names) for key,head in heads.items()};rows=[]
    for i in range(len(X)):
        own={}
        for p in PROFILES:
            own[p]={kind:{k:float(predicted[p+':'+kind][k][i]) for k in ('point','low','high')} for kind in ('probability','cost')}
            own[p]['cost']['supported']=heads[p+':cost'].metadata['supported_for_regression']
        rows.append(own)
    return rows


class ProbabilisticSelector:
    """Expose both-method estimates; deployment remains blocked without validation."""
    def __init__(self,registry_path):
        self.path=Path(registry_path);self.registry=json.loads(self.path.read_text());r=self.registry
        from ml.remizov_features import VERSION as FV
        if r['schema']!=VERSION or r['feature_version']!=FV:raise ValueError('schema mismatch')
        if hashlib.sha256(Path(__file__).with_name('remizov_features.py').read_bytes()).hexdigest()!=r['feature_code_sha256']:raise ValueError('feature code changed')
        self.heads={key:load_head(e,self.path.parent/'models') for key,e in r['heads'].items()}
    def select_method(self,coefficients,x_grid,metadata,*,q,profile_context,timing_context,domain='all_methods_v1'):
        from ml.remizov_features import coefficient_features,CERT_FIELDS
        r=self.registry;result=dict(schema=VERSION,action='abstain',status='research_only',accuracy_certificate=False,reliability_status='not_certified')
        if domain!='all_methods_v1' or profile_context!=r['profile_context'] or timing_context!=r['timing_context']:
            return dict(result,status='unsupported_context')
        x=np.asarray(x_grid,float)
        if x.shape!=(256,) or hashlib.sha256(x.tobytes()).hexdigest()!=r['input_grid_sha256']:return dict(result,status='invalid_input_grid')
        cert={k:float(metadata[k]) for k in CERT_FIELDS}
        if not all(math.isfinite(v) for v in cert.values()) or not 0<cert['a_lower_cert']<=cert['a_upper_cert'] or cert['b_abs_cert']<0 or cert['remizov_c_abs_cert']<0 or cert['remizov_lambda']<=cert['remizov_c_abs_cert']:raise ValueError('invalid certificates')
        f=coefficient_features(coefficients,x,metadata)
        if list(f)!=r['feature_names']:raise ValueError('feature order mismatch')
        est=estimates_from_heads(self.heads,np.asarray([[f[n] for n in r['feature_names']]]),r['feature_names'])[0]
        point=choose(est,q);robust=choose(est,q,robust=True)
        return dict(result,estimates=est,q=q,diagnostic_point_choice=point,diagnostic_uncertainty_choice=robust,
            reason='no_original_validation_calibration;cost_context_unvalidated;R128_time_only4families',
            uncertainty_status='uncalibrated_family_bootstrap_model_spread')
