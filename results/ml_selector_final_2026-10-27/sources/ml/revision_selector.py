"""Research-only train revision: real binomial heads, full costs, nested calibration.

The continuous-error challenger is an ML assumption, not a changed ODE scheme:
it models log(n*E_grid) jointly, with profile-specific held-family residual CDFs.
"""
from __future__ import annotations
import json,math,time
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.special import expit,logit
from scipy.optimize import minimize
from ml.revision_evidence import read,sha,fingerprint

PROFILES=('R128','R256','MC1')
MODES=('binary_raw','binary_sigmoid','continuous_error')


def family_weights(rows):
    counts=Counter(r['family_id'] for r in rows)
    return np.asarray([1/counts[r['family_id']] for r in rows])


def check_fit(rows,excluded):
    if not rows or any(r['split']!='train' for r in rows):raise ValueError('only nonempty permitted train rows')
    if set(excluded)&{r['family_id'] for r in rows}:raise ValueError('held family reached model or calibrator')


def family_blocks(groups,n=3,salt='revision_v1'):
    import hashlib
    groups=sorted(set(groups),key=lambda g:hashlib.sha256((salt+'|'+g).encode()).hexdigest())
    if len(groups)<n:raise ValueError('insufficient independent families')
    return [set(groups[i::n]) for i in range(n)]


class PointHead:
    def __init__(self,kind,scaler,model,meta):self.kind=kind;self.scaler=scaler;self.model=model;self.meta=meta
    def predict(self,X,names):
        from catboost import Pool
        Z=self.scaler.transform(X)
        if isinstance(self.model,float):return np.full(len(X),self.model)
        pool=Pool(Z,feature_names=names)
        return self.model.predict_proba(pool,thread_count=1)[:,1] if self.kind=='probability' else self.model.predict(pool,thread_count=1)


def fit_point(X,rows,names,kind,excluded,node,seed=248827):
    from sklearn.preprocessing import StandardScaler
    from catboost import CatBoostClassifier,CatBoostRegressor,Pool
    check_fit(rows,excluded);indices=[r['index'] for r in rows];scaler=StandardScaler().fit(X[indices]);Z=scaler.transform(X[indices])
    if kind=='probability':
        y=np.array([r['k']/r['n'] for r in rows]);weights=family_weights(rows)*np.array([r['n'] for r in rows])
    else:y=np.array([r['target'] for r in rows]);weights=family_weights(rows)
    assert np.isfinite(y).all();groups=sorted({r['family_id'] for r in rows})
    if len(set(y))==1 or (kind=='regression' and len(groups)<8):model=float(np.average(y,weights=weights))
    else:
        cls=CatBoostClassifier if kind=='probability' else CatBoostRegressor
        model=cls(loss_function='CrossEntropy' if kind=='probability' else 'RMSE',iterations=120,depth=2,learning_rate=.05,l2_leaf_reg=10.,
            bootstrap_type='No',random_strength=0,random_seed=seed,thread_count=1,allow_writing_files=False,verbose=False)
        model.fit(Pool(Z,label=y,weight=weights,feature_names=names))
    node=Path(node);node.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(node/'scaler.npz',mean=scaler.mean_,scale=scaler.scale_,var=scaler.var_)
    if isinstance(model,float):entry=dict(constant=model)
    else:model.save_model(str(node/'model.cbm'));entry=dict(path='model.cbm',sha256=sha(node/'model.cbm'))
    meta=dict(kind=kind,model=entry,scaler_sha256=sha(node/'scaler.npz'),train_keys=[r['equation_key'] for r in rows],train_indices=indices,
        train_profiles=[r['profile'] for r in rows],families=groups,excluded_families=sorted(excluded),seed=seed,
        weights=weights.tolist(),target=y.tolist(),constant=isinstance(model,float),rows=len(rows),supported=len(groups)>=8,
        family_weighting='equal total equation weight within topology; MC binomial likelihood additionally weighted by real n trials')
    (node/'metadata.json').write_text(json.dumps(meta,sort_keys=True))
    head=PointHead(kind,scaler,model,meta);restored=load_point(node)
    np.testing.assert_array_equal(head.predict(X,names),restored.predict(X,names))
    return head


def load_point(node):
    from sklearn.preprocessing import StandardScaler
    from catboost import CatBoostClassifier,CatBoostRegressor
    node=Path(node);meta=read(node/'metadata.json');assert sha(node/'scaler.npz')==meta['scaler_sha256']
    scaler=StandardScaler()
    with np.load(node/'scaler.npz',allow_pickle=False) as z:
        scaler.mean_=z['mean'];scaler.scale_=z['scale'];scaler.var_=z['var'];scaler.n_features_in_=len(z['mean'])
    if 'constant' in meta['model']:model=float(meta['model']['constant'])
    else:
        assert sha(node/meta['model']['path'])==meta['model']['sha256']
        cls=CatBoostClassifier if meta['kind']=='probability' else CatBoostRegressor;model=cls();model.load_model(str(node/meta['model']['path']))
    return PointHead(meta['kind'],scaler,model,meta)


def fit_binomial_sigmoid(rows,p,excluded):
    check_fit(rows,excluded);p=np.asarray(p,float);z=logit(np.clip(p,1e-6,1-1e-6));y=np.array([r['k']/r['n'] for r in rows]);w=family_weights(rows)*np.array([r['n'] for r in rows])
    def loss(parameters):
        a,b=parameters;v=a*z+b;err=w*(expit(v)-y)
        return float(np.sum(w*(np.logaddexp(0,v)-y*v))+((a-1)**2+b*b)),np.array([np.sum(err*z)+2*(a-1),np.sum(err)+2*b])
    result=minimize(loss,[1.,0.],jac=True,bounds=((0,10),(-10,10)),method='L-BFGS-B',options=dict(ftol=1e-12,gtol=1e-8,maxiter=500))
    if not result.success:raise RuntimeError(result.message)
    return dict(slope=float(result.x[0]),intercept=float(result.x[1]),L2=2.,families=sorted({r['family_id'] for r in rows}),
        excluded_families=sorted(excluded),equation_keys=[r['equation_key'] for r in rows],raw_inner_OOF=p.tolist(),objective=float(result.fun),n_trials=int(sum(r['n'] for r in rows)))


def sigmoid(cal,p):return expit(cal['slope']*logit(np.clip(p,1e-6,1-1e-6))+cal['intercept'])


def weighted_cdf(cal,threshold):
    vals=np.asarray(cal['residuals']);weights=np.asarray(cal['weights'])
    return np.asarray([float(np.sum(weights[vals<=v])/np.sum(weights)) for v in np.atleast_1d(threshold)])


def candidate_grid():
    candidates=[dict(name=p,rule=p,mode='binary_sigmoid') for p in ('constant_R128','constant_R256')]
    for mode in MODES:
        candidates.extend([dict(name=mode+'_q95_default',mode=mode,rule='q95_default',q=.95),dict(name=mode+'_q95_risk',mode=mode,rule='q95_risk',q=.95)])
        for delta in (0.,.02,.05):candidates.append(dict(name=mode+'_parity_'+str(delta),mode=mode,rule='parity',delta=delta))
        candidates.append(dict(name=mode+'_cost_per_success',mode=mode,rule='cost_per_success'))
    return candidates


def action(estimates,policy):
    rule=policy['rule']
    if rule.startswith('constant_'):return rule[len('constant_'):]
    p={k:estimates[k]['p'] for k in PROFILES};t={k:estimates[k]['seconds'] for k in PROFILES}
    if rule=='q95_default':available=[x for x in PROFILES if x=='R256' or p[x]>=.95]
    elif rule=='q95_risk':available=[x for x in PROFILES if p[x]>=.95]
    elif rule=='parity':available=[x for x in PROFILES if p[x]>=max(p.values())-policy['delta']]
    elif rule=='cost_per_success':available=list(PROFILES)
    else:raise ValueError('unknown policy')
    if not available:return 'abstain'
    cost={x:(t[x]/max(p[x],1e-10) if rule=='cost_per_success' else t[x]) for x in available}
    chosen=min(available,key=lambda x:(cost[x],x=='MC1',x=='R256'))
    rs=[x for x in available if x!='MC1']
    if chosen=='MC1' and rs:
        best=min(rs,key=lambda x:cost[x])
        if cost['MC1']>.9*cost[best]:chosen=best
    return chosen


def choose_policy(paired_predictions,accuracy_index,candidates):
    """Inner-only empirical nonloss gate; does not establish population safety."""
    scores=[]
    for policy in candidates:
        lost=0.;correct=0.;predicted_cost=0.;actions=[]
        for row in paired_predictions:
            key=row['equation_key'];a=action(row['variants'][policy['mode']],policy);actions.append(a)
            base=accuracy_index[key,'R256'];baseline=base['k']/base['n']
            obs=accuracy_index.get((key,a));s=obs['k']/obs['n'] if a!='abstain' else 0.
            lost+=max(0.,baseline-s);correct+=s
            if a!='abstain':predicted_cost+=row['variants'][policy['mode']][a]['seconds']
        scores.append(dict(policy=policy,lost_vs_R256=lost,correct=correct,predicted_method_cost=predicted_cost,actions=Counter(actions)))
    feasible=[s for s in scores if s['lost_vs_R256']<=1e-12]
    # Accuracy first; then predicted (not imputed observed) cost. Baseline always feasible.
    selected=min(feasible,key=lambda s:(-s['correct'],s['predicted_method_cost'],s['policy']['name']))
    return selected['policy'],scores


class ResearchRevision:
    """Loaded frozen estimates only. Production dispatch always remains blocked."""
    def __init__(self,out):
        self.out=Path(out);self.registry=read(self.out/'registry.json')
        if self.registry['deployment_ready'] or not self.registry['production_blocked']:raise ValueError('research-only registry required')
        self.heads={k:load_point(self.out/v) for k,v in self.registry['final_heads'].items()}
    def predict_features(self,X):
        return predict_bundle(self.heads,self.registry['calibration'],np.asarray(X),self.registry['feature_names'])
    def select_features(self,X):
        est=self.predict_features(X);policy=self.registry['final_policy']
        return [dict(action='abstain',status='research_only',diagnostic_choice=action(v[policy['mode']],policy),accuracy_certificate=False,variants=v) for v in est]


def predict_bundle(heads,calibration,X,names):
    values={k:h.predict(X,names) for k,h in heads.items()};out=[]
    for i in range(len(X)):
        variants={m:{} for m in MODES}
        for p in PROFILES:
            raw=float(values[p+'_prob'][i]);sig=float(sigmoid(calibration['sigmoid'][p],raw))
            if p=='MC1':log_cost=values['MC_cost'][i];error_p=sig
            else:
                n=int(p[1:]);log_cost=values['R_cost'][i]+math.log(n)
                predicted_log_error=values['R_error'][i]-math.log(n)
                error_p=float(weighted_cdf(calibration['error_cdf'][p],math.log(.01)-predicted_log_error)[0])
            cost=float(np.exp(log_cost));res=calibration['time_residuals'][p]
            low=cost*math.exp(res['q10']);high=cost*math.exp(res['q90'])
            for mode,prob in [('binary_raw',raw),('binary_sigmoid',sig),('continuous_error',error_p)]:
                variants[mode][p]=dict(p=prob,seconds=cost,time_low=min(low,cost),time_high=max(high,cost),
                    uncertainty='family-OOF residual spread, not a calibrated coverage or per-input reliability certificate')
        out.append(variants)
    return out
