"""Train-family calibration utilities; no ODE or production API changes."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
import numpy as np

Q_VALUES=(.5,.8,.95)
CALIBRATION_L2=2.


def require_disjoint(train_rows,held_families):
    if not train_rows or any(r['split']!='train' for r in train_rows):
        raise ValueError('nonempty original-train observations required')
    if {r['family_id'] for r in train_rows}&set(held_families):
        raise ValueError('held family reached model/scaler/calibrator fit')


def outer_blocks(r128_rows,old_folds):
    """Keep the original40 paired folds; partition the other50 families separately."""
    all_families={r['family_id'] for r in r128_rows};paired=set()
    blocks=[]
    for f in old_folds:
        held=set(f['held_families'])
        if paired&held:raise ValueError('overlapping original folds')
        paired|=held
        blocks.append(dict(fold=f['fold'],held_families=sorted(held),kind='original_paired40'))
    extra=sorted(all_families-paired)
    if not paired<=all_families:raise ValueError('unknown paired families')
    # Fixed input-family hash only. No held labels or calibration diagnostics select splits.
    extra.sort(key=lambda g:hashlib.sha256(('full_R128_outer_v1|'+g).encode()).hexdigest())
    for k in range(3):
        blocks.append(dict(fold=3+k,held_families=sorted(extra[k::3]),kind='additional_R128'))
    flattened=[g for b in blocks for g in b['held_families']]
    if len(flattened)!=len(set(flattened)) or set(flattened)!=all_families:
        raise ValueError('outer folds are not a family partition')
    return blocks


def inner_blocks(rows,seed):
    from sklearn.model_selection import StratifiedGroupKFold
    groups=np.asarray([r['family_id'] for r in rows]);y=np.asarray([r['k'] for r in rows])
    split=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=seed)
    return [dict(train_indices=t.tolist(),held_indices=h.tolist(),
                 held_families=sorted(set(groups[h])))
            for t,h in split.split(np.zeros((len(rows),1)),y,groups)]


def family_weights(rows):
    counts=Counter(r['family_id'] for r in rows)
    return np.asarray([1/counts[r['family_id']] for r in rows])


def apply_sigmoid(calibrator,p):
    from scipy.special import expit,logit
    p=np.asarray(p,float)
    if np.any(~np.isfinite(p)) or np.any((p<0)|(p>1)):
        raise ValueError('invalid base probabilities')
    return expit(calibrator['slope']*logit(np.clip(p,1e-6,1-1e-6))+calibrator['intercept'])


def calibration_objective(parameters,z,y,weights,penalty=CALIBRATION_L2):
    from scipy.special import expit
    slope,intercept=parameters;logits=slope*z+intercept
    loss=float(np.sum(weights*(np.logaddexp(0.,logits)-y*logits))+
               penalty/2*((slope-1)**2+intercept**2))
    residual=weights*(expit(logits)-y)
    gradient=np.asarray([np.sum(residual*z)+penalty*(slope-1),
                         np.sum(residual)+penalty*intercept])
    return loss,gradient


def fit_sigmoid(rows,p,held_families):
    from scipy.optimize import minimize
    from scipy.special import logit
    require_disjoint(rows,held_families)
    if any(r['n']!=1 or r['k'] not in (0,1) for r in rows):
        raise ValueError('R calibration requires real deterministic labels')
    p=np.asarray(p,float)
    if p.shape!=(len(rows),) or np.any(~np.isfinite(p)) or np.any((p<0)|(p>1)):
        raise ValueError('invalid inner OOF probabilities')
    y=np.asarray([r['k'] for r in rows],float);z=logit(np.clip(p,1e-6,1-1e-6));weights=family_weights(rows)
    result=minimize(calibration_objective,[1.,0.],args=(z,y,weights),jac=True,
        method='L-BFGS-B',bounds=((0.,10.),(-10.,10.)),options=dict(ftol=1e-12,gtol=1e-8,maxiter=500))
    if not result.success or not np.isfinite(result.fun):raise RuntimeError(str(result.message))
    return dict(schema='family_weighted_regularized_sigmoid_v1',slope=float(result.x[0]),intercept=float(result.x[1]),
        L2=CALIBRATION_L2,identity_prior=[1.,0.],bounds=[[0.,10.],[-10.,10.]],optimizer_success=True,
        objective=float(result.fun),iterations=int(result.nit),families=sorted({r['family_id'] for r in rows}),
        equation_ids=[r['equation_id'] for r in rows],rows=len(rows),positive_rows=int(y.sum()),
        row_weights=weights.tolist(),training_probabilities='inner family OOF only',
        uncertainty='no per-input confidence bound; regularization is not synthetic observations')


def metric_summary(rows,p):
    y=np.asarray([r['k']/r['n'] for r in rows]);p=np.asarray(p,float)
    clipped=np.clip(p,1e-10,1-1e-10)
    brier=y*(1-p)**2+(1-y)*p*p
    logloss=-y*np.log(clipped)-(1-y)*np.log1p(-clipped)
    weights=family_weights(rows)
    def summary(w):return dict(brier=float(np.average(brier,weights=w)),logloss=float(np.average(logloss,weights=w)),
        mean_prediction=float(np.average(p,weights=w)),observed_rate=float(np.average(y,weights=w)))
    return dict(equations=len(rows),families=len({r['family_id'] for r in rows}),
        equation_weighted=summary(np.ones(len(rows))),family_weighted=summary(weights))


def paired_uncertainty(rows,raw,calibrated,seed=2488):
    y=np.asarray([r['k']/r['n'] for r in rows]);raw=np.clip(raw,1e-10,1-1e-10);cal=np.clip(calibrated,1e-10,1-1e-10)
    def losses(p):return np.column_stack((y*(1-p)**2+(1-y)*p*p,-y*np.log(p)-(1-y)*np.log1p(-p)))
    delta=losses(cal)-losses(raw);grouped=defaultdict(list)
    for r,d in zip(rows,delta):grouped[r['family_id']].append(d)
    matrix=np.asarray([np.mean(grouped[g],axis=0) for g in sorted(grouped)])
    rng=np.random.default_rng(seed);draws=np.mean(matrix[rng.integers(0,len(matrix),size=(4000,len(matrix)))],axis=1)
    return dict(family_weighted_delta_cal_minus_raw=matrix.mean(axis=0).tolist(),
        descriptive_fixed_prediction_family_bootstrap95=np.quantile(draws,[.025,.975],axis=0).T.tolist(),
        columns=['Brier','logloss'],families=len(matrix),resamples=4000,
        scope='conditional fixed outer-OOF predictions; shared training and model/refit uncertainty not captured; not prospective proof')


def baseline_default_choice(estimates,q):
    """Research challenger: R256 fallback; switch to a cheaper predicted-feasible profile.

    This avoids abstention losses, but cannot guarantee no R128/MC accuracy losses.
    The q constraint applies to switches, not to the unconditional R256 fallback.
    """
    if not 0<q<1:raise ValueError('invalid risk threshold')
    candidates={p:e for p,e in estimates.items() if p!='R256' and e['probability']['point']>=q
                and e['cost']['point']<estimates['R256']['cost']['point']}
    if not candidates:return 'R256'
    available=dict(candidates,R256=estimates['R256'])
    fastest=min(available,key=lambda p:(available[p]['cost']['point'],p=='MC1',p=='R256'))
    best_R=min((p for p in available if p!='MC1'),key=lambda p:available[p]['cost']['point'])
    if fastest=='MC1' and available['MC1']['cost']['point']>.9*available[best_R]['cost']['point']:
        return best_R
    return fastest
