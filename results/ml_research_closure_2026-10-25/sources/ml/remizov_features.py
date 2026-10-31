"""Strictly whitelisted input-only features for the frozen Chernov solve profile.

X contains the *given equation coefficients* a,b,c,g, never solution channels.
No solver import, result access, formula evaluation or exact-solution AST use.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

VERSION='remizov_input_features_v1'
CERT_FIELDS=('a_lower_cert','a_upper_cert','b_abs_cert','remizov_c_abs_cert','remizov_lambda')
OPS=('constant','sin_affine','cos_affine','tanh_affine','gaussian','lorentz',
     'saturating','cos','sin','tanh','exp_neg_square','rational_saturate','lincomb','mul')


def ast_features(node):
    counts=Counter();depth=0
    def walk(n,d):
        nonlocal depth
        counts[n['op']]+=1;depth=max(depth,d)
        for child in n['children']:walk(child,d+1)
    walk(node,1)
    if set(counts)-set(OPS):raise ValueError('unsupported coefficient AST')
    return dict(nodes=sum(counts.values()),depth=depth,**{f'op_{op}':counts[op] for op in OPS})


def coefficient_features(values,x,meta):
    """Only X[a,b,c,g], coefficient certificates, and a/b/c coefficient trees."""
    X=np.asarray(values,dtype=float);x=np.asarray(x,dtype=float)
    if X.shape!=(4,len(x)) or len(x)<32 or not np.all(np.isfinite(X)):
        raise ValueError('four finite input coefficient channels required')
    if not np.all(np.diff(x)>0):raise ValueError('input grid must increase')
    f={f'cert_{key}':float(meta[key]) for key in CERT_FIELDS}
    lam=f['cert_remizov_lambda'];C=f['cert_remizov_c_abs_cert'];gap=lam-C
    if gap<=0 or f['cert_a_lower_cert']<=0:raise ValueError('invalid input certificates')
    T=max(0.,(math.log(2)-math.log(gap)-math.log(1e-8))/gap)
    A=f['cert_a_upper_cert'];B=f['cert_b_abs_cert']
    f.update(pre_gap=gap,pre_T=T,pre_T_over_n=T/128,
             pre_C_T=C*T,pre_lambda_T=lam*T,pre_a_contrast=A/f['cert_a_lower_cert'],
             pre_diffusive_shift=2*math.sqrt(A*T/128),pre_drift_shift=2*B*T/128,
             pre_diffusive_length=math.sqrt(2*A*T),pre_drift_length=B*T)
    channels=dict(a=X[0],b=X[1],cR=X[2]+lam,gR=-X[3])
    indices=np.linspace(0,len(x)-1,32,dtype=int)
    for name,v in channels.items():
        d1=np.diff(v)/np.diff(x);d2=np.diff(d1)/np.diff((x[1:]+x[:-1])*.5)
        centered=v-np.mean(v);power=np.abs(np.fft.rfft(centered))**2;energy=float(np.sum(power))
        stats=dict(min=np.min(v),max=np.max(v),mean=np.mean(v),std=np.std(v),
            rms=np.sqrt(np.mean(v*v)),absmax=np.max(np.abs(v)),median=np.median(v),
            total_variation=np.sum(np.abs(np.diff(v))),d1_rms=np.sqrt(np.mean(d1*d1)),
            d1_absmax=np.max(np.abs(d1)),d2_rms=np.sqrt(np.mean(d2*d2)),
            negative_fraction=np.mean(v<0),zero_crossings=np.count_nonzero(v[:-1]*v[1:]<0),
            spectral_high_fraction=0. if energy==0 else np.sum(power[8:])/energy)
        f.update({name+'_'+k:float(z) for k,z in stats.items()})
        f.update({f'{name}_sample_{i:02d}':float(v[j]) for i,j in enumerate(indices)})
    names=list(channels)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            u=channels[a]-np.mean(channels[a]);v=channels[b]-np.mean(channels[b])
            denom=math.sqrt(float(np.sum(u*u))*float(np.sum(v*v)))
            f[f'corr_{a}_{b}']=0. if denom==0 else float(np.sum(u*v))/denom
    for key in ('a_raw','b_raw','c_raw'):
        f.update({f'ast_{key}_{k}':float(v) for k,v in ast_features(meta['serialized_ast'][key]).items()})
    if not all(math.isfinite(v) for v in f.values()):raise ValueError('nonfinite pre-solve feature')
    return dict(sorted(f.items()))


def load_inputs(dataset_dir):
    """Access only whitelisted NPZ entries, despite other entries in the archive."""
    dataset_dir=Path(dataset_dir)
    meta=[json.loads(s) for s in (dataset_dir/'all_methods_equations_v1_metadata.jsonl').read_text().splitlines()]
    by_id={m['equation_id']:m for m in meta}
    with np.load(dataset_dir/'all_methods_equations_v1_ml.npz',allow_pickle=False) as z:
        if list(z['channel_order'])!=['a','b','c','g']:raise ValueError('unknown coefficient channel order')
        ids=z['equation_id'].copy();X=z['X'].copy();x=z['x_grid'].copy()
        families=z['family_id'].copy();splits=z['split'].copy()
    if len(set(ids))!=len(ids) or set(ids)!=set(by_id):raise ValueError('input identity mismatch')
    rows=[]
    for i,pid in enumerate(ids):
        m=by_id[str(pid)]
        if m['family_id']!=families[i] or m['split']!=splits[i]:raise ValueError('metadata/NPZ split mismatch')
        rows.append(coefficient_features(X[i],x,m))
    names=list(rows[0])
    matrix=np.asarray([[r[n] for n in names] for r in rows])
    groups={}
    for pid,family,split in zip(ids,families,splits):
        previous=groups.setdefault(str(family),str(split))
        if previous!=split:raise ValueError('family split leakage')
    signatures={}
    for m in meta:
        previous=signatures.setdefault(m['structure_signature'],m['split'])
        if previous!=m['split']:raise ValueError('topology split leakage')
    return dict(X=matrix,feature_names=names,ids=ids,families=families,splits=splits,
                metadata=by_id,x_grid=x,grid_sha256=hashlib.sha256(x.tobytes()).hexdigest())
