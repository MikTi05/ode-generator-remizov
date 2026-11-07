"""Separate exact-AST MC backend; reference mc_solver.py is never modified.

NumPy: coefficient value-only tape + fused evaluation, necessary y jets retained.
Numba: fused scalar value/jet tapes and blocked serial EM/trapezoids. NumPy
PCG64 draws stay outside JIT, one vector per original step, in original order.
Only safe all_methods_v1 metadata and two explicit analytical fixtures accepted.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import importlib
import math
import time
from typing import Mapping

import numpy as np

from remizov_solver import RemizovProblem,_problem_conditions_valid,_validate_target_x
from mc_solver import (MCConfig,MCResult,MCForcingBound,_Run,_Moments,_MCFailure,
    _real,_integer,_spawn_key,_gap_lower,_tail_bound,auto_T)

OPS=('constant','sin_affine','cos_affine','tanh_affine','gaussian','lorentz','saturating',
     'sin','cos','tanh','exp_neg_square','rational_saturate','lincomb','mul')
_NUMBA=None


def _readonly(a,dtype=float):
    z=np.ascontiguousarray(a,dtype=dtype);z.setflags(write=False);return z


def flatten_trees(nodes):
    rows=[];roots=[]
    def add(node):
        children=[add(child) for child in node.children];op=node.op;p=node.params
        if op=='constant':v,w=p['value'],0.
        elif op in ('sin_affine','cos_affine','tanh_affine'):v,w=p['omega'],p['phi']
        elif op in ('gaussian','lorentz'):v,w=p['alpha'],p['mu']
        elif op=='saturating':v,w=p['mu'],0.
        elif op=='lincomb':v,w=p['alpha'],p['beta']
        else:v,w=0.,0.
        rows.append((OPS.index(op),children[0] if children else -1,
                     children[1] if len(children)>1 else -1,v,w))
        return len(rows)-1
    for node in nodes:roots.append(add(node))
    return _readonly(np.asarray(rows).reshape(-1,5)),_readonly(roots,np.int64)


@dataclass(frozen=True)
class PreparedMC:
    problem: RemizovProblem
    forcing_bound: MCForcingBound
    coeff_tape: np.ndarray
    coeff_roots: np.ndarray
    y_tape: np.ndarray
    parameters: np.ndarray
    kind: int
    y_shape: object
    preparation_seconds: float
    source: str


def prepare_all_methods_mc(metadata: Mapping) -> PreparedMC:
    """AST is authoritative, reconstructed exactly as the safe loader does.

    No arbitrary user callbacks are silently substituted or compiled. The
    returned RemizovProblem carries the existing all-real analytic certificates.
    """
    from all_methods_dataset_loader import _decode_bundle,_validate_metadata_shape,_validate_construction
    from all_methods_bounds import certify_all_methods_forcing
    began=time.perf_counter();meta=dict(metadata);pid=meta['equation_id']
    _validate_metadata_shape(meta,pid);bundle=_decode_bundle(meta,pid);_validate_construction(meta,bundle,pid)
    p=RemizovProblem(pid,bundle.a,bundle.b,bundle.c_remizov,bundle.g_remizov,
        meta['remizov_lambda'],meta['a_lower_cert'],meta['a_upper_cert'],meta['b_abs_cert'],
        meta['remizov_c_abs_cert'],'analytic','all_real_x',meta['x0'],meta['x1'])
    proof=certify_all_methods_forcing(p,meta);c=bundle.parameters
    coeff,roots=flatten_trees((bundle.a_raw,bundle.b_raw,bundle.c_raw));y,_=flatten_trees((bundle.y_shape,))
    params=_readonly([c.a_mid,c.a_half,c.b_scale,c.c_scale,c.lambda_value,
                      c.solution_scale,c.bump_mu,c.bump_radius])
    return PreparedMC(p,proof.forcing_bound,coeff,roots,y,params,0,bundle.y_shape,
                      time.perf_counter()-began,proof.forcing_bound.source)


def prepare_analytic_mc(kind='constant') -> PreparedMC:
    if kind not in ('constant','gaussian'):raise ValueError('only the two explicit analytic fixtures are supported')
    began=time.perf_counter();gaussian=kind=='gaussian'
    g=(lambda x:(4-4*x*x)*np.exp(-x*x)) if gaussian else (lambda x:np.ones_like(x))
    p=RemizovProblem('MC_'+kind,lambda x:np.ones_like(x),lambda x:np.zeros_like(x),
        lambda x:np.zeros_like(x),g,2.,1.,1.,0.,0.,'analytic','all_real_x',-2.,2.)
    bound=MCForcingBound(4. if gaussian else 1.,'analytic Gaussian envelope' if gaussian else 'g_R=1')
    return PreparedMC(p,bound,_readonly(np.empty((0,5))),_readonly([0,0,0],np.int64),
        _readonly(np.empty((0,5))),_readonly([0.,0.,0.,0.,2.,0.,0.,1.]),
        2 if gaussian else 1,None,time.perf_counter()-began,bound.source)


def value_tape_numpy(tape,x):
    v=np.empty((len(tape),len(x)))
    with np.errstate(all='ignore'):
        for i,row in enumerate(tape):
            op,l,r=int(row[0]),int(row[1]),int(row[2]);p,q=row[3],row[4]
            if op==0:z=np.full_like(x,p)
            elif op==1:z=np.sin(p*x+q)
            elif op==2:z=np.cos(p*x+q)
            elif op==3:z=np.tanh(p*x+q)
            elif op==4:
                delta=x-q;z=np.exp(-p*delta*delta)
            elif op==5:
                delta=x-q;z=1./(1.+p*delta*delta)
            elif op==6:
                delta=x-p;z=delta/np.sqrt(1.+delta*delta)
            elif op==7:z=np.sin(v[l])
            elif op==8:z=np.cos(v[l])
            elif op==9:z=np.tanh(v[l])
            elif op==10:z=np.exp(-(v[l]*v[l]))
            elif op==11:z=v[l]/(1.+v[l]*v[l])
            elif op==12:z=p*v[l]+q*v[r]
            else:z=v[l]*v[r]
            if not np.all(np.isfinite(z)):
                raise _MCFailure('coefficient_evaluation_failure','nonfinite AST intermediate')
            v[i]=z
    return v


def fused_numpy(prepared,x):
    if prepared.kind==1:return np.ones_like(x),np.zeros_like(x),np.zeros_like(x),np.ones_like(x)
    if prepared.kind==2:return np.ones_like(x),np.zeros_like(x),np.zeros_like(x),(4-4*x*x)*np.exp(-x*x)
    from src.all_methods_expressions import evaluate_solution
    p=prepared.parameters;v=value_tape_numpy(prepared.coeff_tape,x);ra,rb,rc=prepared.coeff_roots
    a=p[0]+p[1]*np.tanh(v[ra]);b=p[2]*np.tanh(v[rb]);c=p[3]*np.tanh(v[rc])
    from src.all_methods_expressions import ExpressionEvaluationError
    try:y=evaluate_solution(prepared.y_shape,p[6],p[7],p[5],x)
    except ExpressionEvaluationError as exc:
        raise _MCFailure('coefficient_evaluation_failure',str(exc)) from exc
    return a,b,c,-(a*y.d2+b*y.d1+(c-p[4])*y.value)


def compile_backend():
    """Warm the two generic signatures once per process; no ODE solve or RNG."""
    global _NUMBA
    t=time.perf_counter()
    imported=0.
    if _NUMBA is None:
        _NUMBA=importlib.import_module('mc_ast_kernels');imported=time.perf_counter()-t
    if _NUMBA.evaluate_batch.signatures and _NUMBA.advance_block.signatures:
        return dict(numba_import_seconds=imported,compilation_seconds=0.)
    t=time.perf_counter()
    p=prepare_analytic_mc();args=(p.kind,p.coeff_tape,p.coeff_roots,p.y_tape,p.parameters)
    # Arrays match the read-only tape / writable state signature of real runs.
    _NUMBA.evaluate_batch(np.zeros(1),*args)
    _NUMBA.advance_block(np.zeros(1),np.ones(1),np.zeros(1),np.zeros(1),np.zeros(1),
        np.zeros(1),np.empty((0,1)),.01,0,1,*args,_readonly([1.,1.,0.,0.,2.,1.]))
    return dict(numba_import_seconds=imported,compilation_seconds=time.perf_counter()-t)


def fused_values(prepared,x,engine='numpy'):
    raw=np.asarray(x)
    if np.iscomplexobj(raw) or raw.dtype.kind=='b':raise ValueError('real vector required')
    x=np.asarray(x,dtype=float)
    if x.ndim!=1 or not np.all(np.isfinite(x)):raise ValueError('finite vector required')
    if engine=='numpy':return np.asarray(fused_numpy(prepared,x))
    if engine!='numba':raise ValueError('engine must be numpy or numba')
    compile_backend()
    return _NUMBA.evaluate_batch(x,prepared.kind,prepared.coeff_tape,prepared.coeff_roots,
                                 prepared.y_tape,prepared.parameters)


@dataclass
class OptimizedMCResult(MCResult):
    backend: str=''
    ast_preparation_seconds: float=0.
    numba_import_seconds: float=0.
    compilation_seconds: float=0.
    numerical_seconds: float=0.
    rng_seconds: float=0.
    fused_point_evaluations: int=0
    numerical_blocks: int=0
    block_steps: int=0
    coefficient_counter_kind: str='completed-chunk logical reference-role counts; physical fused work reported separately'
    random_prefetch: str='same per-step NumPy PCG64 vectors; failure can discard prefetched draws'


def _check_fused(p,bound,a,b,c,g,check_ab=True):
    if not all(np.all(np.isfinite(v)) for v in (a,b,c,g)):
        raise _MCFailure('coefficient_evaluation_failure','nonfinite fused AST value/jet')
    if check_ab:
        if np.any(a<=0):raise _MCFailure('nonpositive_diffusion','a(x) must be strictly positive')
        if np.any(a<p.a_lower_bound) or np.any(a>p.a_upper_bound) or np.any(np.abs(b)>p.b_abs_upper_bound):
            raise _MCFailure('certified_bound_violation','a or b exceeds global certificate')
    if np.any(np.abs(c)>p.c_abs_upper_bound) or np.any(np.abs(g)>bound.abs_upper_bound):
        raise _MCFailure('certified_bound_violation','c_R or g_R exceeds global certificate')


def path_contributions(prepared,x0,T,N,m,rng,run,engine='numba',block_steps=64,diagnostics=None):
    """Production RNG or a replay object; returns original per-path integrals."""
    d=diagnostics if diagnostics is not None else dict(numerical_seconds=0.,rng_seconds=0.,fused_point_evaluations=0,numerical_blocks=0)
    p=prepared.problem;h=T/N;x=np.full(m,x0,dtype=float);lw=np.zeros(m)
    t=time.perf_counter()
    if engine=='numba':
        values=_NUMBA.evaluate_batch(np.array([x0],dtype=float),prepared.kind,prepared.coeff_tape,
            prepared.coeff_roots,prepared.y_tape,prepared.parameters)
        a,b,c,g=(np.full(m,v[0]) for v in values)
        d['fused_point_evaluations']+=1
    else:
        a,b,c,g=fused_numpy(prepared,x);d['fused_point_evaluations']+=m
    _check_fused(p,prepared.forcing_bound,a,b,c,g)
    total=(.5*h)*g;d['numerical_seconds']+=time.perf_counter()-t
    run.peak_batch_paths=max(run.peak_batch_paths,m)
    bounds=_readonly([p.a_lower_bound,p.a_upper_bound,p.b_abs_upper_bound,p.c_abs_upper_bound,p.lambda_value,prepared.forcing_bound.abs_upper_bound])
    offset=0
    while offset<N:
        run.check_time();count=min(block_steps,N-offset)
        t=time.perf_counter();normals=np.empty((count,m))
        for i in range(count):normals[i]=rng.standard_normal(m)
        run.n_normal_draws+=count*m;d['rng_seconds']+=time.perf_counter()-t
        t=time.perf_counter()
        if engine=='numba':
            code,updates=_NUMBA.advance_block(x,a,b,c,total,lw,normals,h,offset,N,prepared.kind,
                prepared.coeff_tape,prepared.coeff_roots,prepared.y_tape,prepared.parameters,bounds)
            run.n_path_steps+=updates;d['fused_point_evaluations']+=updates
        else:
            code=0
            with np.errstate(over='raise',invalid='raise',divide='raise',under='ignore'):
                for k in range(count):
                    x=x+b*h+((np.sqrt(a)*math.sqrt(h))*math.sqrt(2.))*normals[k]
                    run.n_path_steps+=m
                    if not np.all(np.isfinite(x)):raise _MCFailure('nonfinite_state','nonfinite trajectory')
                    an,bn,cn,gn=fused_numpy(prepared,x);d['fused_point_evaluations']+=m
                    _check_fused(p,prepared.forcing_bound,an,bn,cn,gn)
                    lw+=h*(.5*(c-p.lambda_value)+.5*(cn-p.lambda_value))
                    weight=.5*h if offset+k+1==N else h
                    total+=weight*np.exp(lw)*gn;a,b,c=an,bn,cn
        d['numerical_seconds']+=time.perf_counter()-t;d['numerical_blocks']+=1
        if code:
            reason={1:'nonpositive_diffusion',2:'certified_bound_violation',3:'nonfinite_state',
                    4:'coefficient_evaluation_failure',5:'numerical_overflow'}[code]
            raise _MCFailure(reason,'compiled kernel rejected invalid value/bound')
        offset+=count
    if not np.all(np.isfinite(total)):raise _MCFailure('nonfinite_state','nonfinite path integral')
    for role,count in [('a',N),('b',N),('c_R',N+1),('g_R',N+1)]:
        run.calls[role]+=count;run.values[role]+=count*m
    return total


def solve_monte_carlo_optimized(prepared: PreparedMC,target_x,config: MCConfig|None=None,*,engine='numba',block_steps=64):
    """Same MCResult contract, plus honest setup/JIT/RNG/fused-work diagnostics.

    Global certificates come from safe AST preparation. All elapsed solve time,
    including a first JIT compilation, counts toward max_runtime_seconds.
    External AST preparation is explicitly reported, never hidden as free work.
    """
    began=time.perf_counter();cfg=config or MCConfig();cfg.validate()
    if not isinstance(prepared,PreparedMC):raise TypeError('use a validated PreparedMC factory')
    if engine not in ('numpy','numba'):raise ValueError('engine must be numpy or numba')
    _integer(block_steps,'block_steps',1)
    p=prepared.problem;bound=prepared.forcing_bound;bound.validate()
    raw=np.asarray(target_x)
    if np.iscomplexobj(raw) or raw.dtype.kind=='b':raise ValueError('real target coordinates required')
    x=np.asarray(target_x,dtype=float)
    if x.ndim!=1 or not x.size or not np.all(np.isfinite(x)) or np.any(np.diff(x)<=0):
        raise ValueError('finite nonempty strictly increasing target vector required')
    for name in ('lambda_value','a_lower_bound','a_upper_bound','b_abs_upper_bound','c_abs_upper_bound','x_target_min','x_target_max'):
        _real(getattr(p,name),name)
    conditions=_problem_conditions_valid(p)
    if conditions:x=_validate_target_x(p,x)
    entropy=int(np.random.SeedSequence(cfg.seed).entropy)
    keys=[_spawn_key(p.problem_id,float(v),int(cfg.stream_id)) for v in x]
    run=_Run(deadline=None if cfg.max_runtime_seconds is None else began+cfg.max_runtime_seconds)
    complete=np.zeros(x.size,dtype=np.int64);f=[];errors=[];failure='';message=''
    T=cfg.T;N=0;h=math.nan;gap=None;tail=None;prep=0.;jit=dict(numba_import_seconds=0.,compilation_seconds=0.)
    diag=dict(numerical_seconds=0.,rng_seconds=0.,fused_point_evaluations=0,numerical_blocks=0)
    applicability='bounds_certified_regularity_assumed' if conditions else 'invalid_global_bounds'
    try:
        if not conditions:raise _MCFailure('invalid_remizov_conditions','invalid all-real coefficient certificates')
        gap=_gap_lower(p.lambda_value,p.c_abs_upper_bound)
        if T is None:
            try:T=auto_T(p.lambda_value,p.c_abs_upper_bound,bound.abs_upper_bound,cfg.tail_eps)
            except OverflowError as exc:raise _MCFailure('work_limit_exceeded',str(exc)) from exc
        T=float(T);tail=_tail_bound(bound.abs_upper_bound,gap,T);ratio=T/cfg.dt
        if not math.isfinite(ratio) or ratio>cfg.max_steps:raise _MCFailure('work_limit_exceeded','T/dt exceeds max_steps')
        N=max(1,math.ceil(ratio)) if T else 0
        if N and T/N>cfg.dt:N+=1
        if N>cfg.max_steps or int(cfg.M)*N*x.size>cfg.max_path_steps:raise _MCFailure('work_limit_exceeded','planned path steps exceed budget')
        if N and min(cfg.M,cfg.chunk)>cfg.max_batch_paths:raise _MCFailure('work_limit_exceeded','batch size exceeds budget')
        h=T/N if N else 0.
        if N and .5*h==0:raise _MCFailure('numerical_underflow','half step unrepresentable')
        run.check_time()
        if engine=='numba' and N:jit=compile_backend()
        prep=time.perf_counter()-began;run.check_time()
        for i,point in enumerate(x):
            if N==0:f.append(0.);errors.append(0.);continue
            rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy,spawn_key=keys[i])))
            moments=_Moments()
            while moments.count<cfg.M:
                run.check_time();m=min(int(cfg.chunk),int(cfg.M)-moments.count);run.n_batches+=1
                values=path_contributions(prepared,float(point),T,N,m,rng,run,engine,block_steps,diag)
                moments.add(values);complete[i]=moments.count
            f.append(moments.mean);errors.append(moments.stderr())
        run.check_time();f=np.asarray(f);errors=np.asarray(errors)
        with np.errstate(over='raise',invalid='raise'):
            half=1.96*errors;lower=f-half;upper=f+half
        if not all(np.all(np.isfinite(v)) for v in (f,errors,half,lower,upper)):
            raise _MCFailure('nonfinite_statistics','nonfinite sample estimates')
    except _MCFailure as exc:failure=exc.reason;message=str(exc)
    except (FloatingPointError,OverflowError) as exc:failure='numerical_overflow';message=str(exc)
    if failure:
        f,errors,half,lower,upper=(np.array([],dtype=float) for _ in range(5))
        if failure in ('nonpositive_diffusion','certified_bound_violation'):applicability='violated_on_visited_points'
    else:message='analytical tail available; intervals exclude time bias and tail; total error not certified'
    runtime=time.perf_counter()-began
    if prep==0:prep=runtime
    return OptimizedMCResult(p.problem_id,not failure,failure,message,x.copy(),f,errors,half,lower,upper,
        int(cfg.M),complete,float(cfg.dt),h,T,N,gap,bound.abs_upper_bound,bound.source,cfg.tail_eps,tail,
        None if tail is None else bool(tail<=cfg.tail_eps),entropy,int(cfg.stream_id),keys,runtime,prep,
        run.n_path_steps,run.n_normal_draws,run.n_batches,run.calls,run.values,run.peak_batch_paths,applicability,
        backend=engine,ast_preparation_seconds=prepared.preparation_seconds,**jit,**diag,block_steps=block_steps)


__all__=['PreparedMC','OptimizedMCResult','prepare_all_methods_mc','prepare_analytic_mc',
         'compile_backend','solve_monte_carlo_optimized','fused_values']
