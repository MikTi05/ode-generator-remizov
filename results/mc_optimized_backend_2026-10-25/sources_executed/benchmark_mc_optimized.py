"""Frozen small backend comparison. No parameter selection or 1000 MC solves.

Run in one guardian-owned worker; renew owner lease using the existing control
CLI. Benchmark output is append-free, and reruns require a new output folder.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')
for name in THREADS:os.environ[name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)


def freeze(out):
    from diagnostics.compare_chernov import write_json
    from all_methods_dataset_loader import load_all_methods_equations
    from mc_optimized import prepare_all_methods_mc
    from mc_solver import auto_T
    base=json.loads((out/'protocol.json').read_text());perf=base['performance']
    d=ROOT/'data/generated'
    equations={e.equation_id:e for e in load_all_methods_equations(d/'all_methods_equations_v1.csv',d/'all_methods_equations_v1_metadata.jsonl')}
    tasks=[]
    for pid in perf['ids']:
        p=prepare_all_methods_mc(equations[pid].metadata)
        T=math.ceil(auto_T(p.problem.lambda_value,p.problem.c_abs_upper_bound,p.forcing_bound.abs_upper_bound,1e-4)*8)/8
        tasks.append(dict(problem_id=pid,T=T,N=int(T*128),G=p.forcing_bound.abs_upper_bound,
            coefficient_nodes=len(p.coeff_tape),solution_nodes=len(p.y_tape)))
    grid_ids=[perf['ids'][i] for i in (0,1,3)]
    updates=sum(t['N']*(3*3*1024+3*64+(33*1024 if t['problem_id'] in grid_ids else 0)) for t in tasks)+8192*(1024+128)
    plan=dict(protocol_sha256=sha(out/'protocol.json'),tasks=tasks,grid_ids=grid_ids,
        grid_M=1024,grid_repeats=1,grid_seed=2026102520,replay_M=64,replay_seed=2026102521,
        analytical_seed=2026102511,block_steps=64,maximum_path_updates=updates,
        numerical_calls=45+3+2,owner_lease_seconds=300.,rss_limit_mib=768.,
        resource_sample_seconds=.2,telemetry_stale_seconds=30.,
        stop='first numerical failure/equivalence violation, RSS limit, stale telemetry, owner/controller loss or explicit stop; no retry',
        estimated_seconds_before_run=[120,360],
        note='Finite work ~1 billion updates, no accuracy tuning; 3 actual grids calibrate point-to-grid extrapolation')
    write_json(out/'execution_plan.json',plan)
    print(json.dumps(plan),flush=True)


class Replay:
    def __init__(self,z):self.z=z;self.index=0
    def standard_normal(self,m):
        row=self.z[self.index];self.index+=1;assert row.size==m
        return row.copy()


def worker(out):
    import numpy as np
    from all_methods_dataset_loader import load_all_methods_equations
    from mc_solver import MCConfig,solve_monte_carlo,_path_contributions,_Run
    from mc_optimized import (prepare_all_methods_mc,prepare_analytic_mc,compile_backend,
        solve_monte_carlo_optimized,path_contributions,fused_values)
    from diagnostics.compare_chernov import write_json
    plan=json.loads((out/'execution_plan.json').read_text());perf=json.loads((out/'protocol.json').read_text())['performance']
    if sha(out/'protocol.json')!=plan['protocol_sha256']:raise ValueError('changed protocol')
    (out/'measurements').mkdir(exist_ok=False);(out/'replay').mkdir(exist_ok=False)
    stop=threading.Event();began=time.perf_counter()
    def monitor():
        while not stop.is_set():
            # Atomic heartbeat even while the kernel runs; externally monitored.
            from diagnostics.chernov_batch_control import atomic
            atomic(out/'resource_latest.json',dict(monotonic=time.monotonic(),seconds=time.perf_counter()-began,peak_rss_mib=rss()))
            stop.wait(plan['resource_sample_seconds'])
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        write_json(out/'environment.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
            processor=platform.processor(),versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','numba','llvmlite','pytest')},
            threads={n:os.environ[n] for n in THREADS}))
        t=time.perf_counter();d=ROOT/'data/generated'
        equations={e.equation_id:e for e in load_all_methods_equations(d/'all_methods_equations_v1.csv',d/'all_methods_equations_v1_metadata.jsonl')}
        load_seconds=time.perf_counter()-t
        t=time.perf_counter();compilation=compile_backend();compilation['cold_total_seconds']=time.perf_counter()-t
        compilation['small_dataset_load_seconds']=load_seconds;compilation['peak_rss_mib']=rss()
        write_json(out/'compilation.json',compilation);print(json.dumps(compilation),flush=True)
        for task in plan['tasks']:
            pid=task['problem_id'];e=equations[pid];p=prepare_all_methods_mc(e.metadata)
            x=np.linspace(-5,5,101);old=e.remizov_problem
            reference=np.array([old.a(x),old.b(x),old.c_remizov(x),old.g_remizov(x)])
            coef_error=float(np.max(abs(fused_values(p,x,'numba')-reference)))
            assert coef_error<=1e-12
            m=plan['replay_M'];z=np.random.default_rng(plan['replay_seed']).standard_normal((task['N'],m))
            expected=_path_contributions(old,0.,task['T'],task['N'],m,Replay(z),p.forcing_bound,_Run())
            arrays=dict(reference=expected);maxdiff={}
            for engine in ('numpy','numba'):
                arrays[engine]=path_contributions(p,0.,task['T'],task['N'],m,Replay(z),_Run(),engine,plan['block_steps'])
                maxdiff[engine]=float(np.max(abs(arrays[engine]-expected)))
                assert maxdiff[engine]<=1e-8
            np.savez_compressed(out/'replay'/f'{pid}.npz',**arrays)
            write_json(out/'replay'/f'{pid}.json',dict(task=task,M=m,normal_sha256=hashlib.sha256(z.tobytes()).hexdigest(),
                normal_seed=plan['replay_seed'],coefficient_max_abs_difference=coef_error,
                path_max_abs_difference=maxdiff,reference_mean=float(np.mean(expected)),reference_std=float(np.std(expected,ddof=1))))
            del z
            for repeat in range(perf['repeats']):
                cfg=MCConfig(M=perf['M'],chunk=perf['chunk'],dt=perf['dt'],T=task['T'],seed=perf['seed_base']+repeat,
                    max_path_steps=300000000,max_batch_paths=1024)
                order=['reference','numpy','numba'] if repeat%2==0 else ['numba','numpy','reference']
                results={}
                for engine in order:
                    t=time.perf_counter()
                    r=solve_monte_carlo(old,[0.],cfg,forcing_bound=p.forcing_bound) if engine=='reference' else solve_monte_carlo_optimized(p,[0.],cfg,engine=engine)
                    external=time.perf_counter()-t;results[engine]=r
                    record=dict(task=task,repeat=repeat,engine=engine,config=asdict(cfg),result=asdict(r),
                        ast_preparation_seconds=p.preparation_seconds,external_solve_seconds=external,process_peak_rss_mib=rss(),
                        y_exact=e.y_exact(r.x),error_infinity=float(np.max(abs(r.f-e.y_exact(r.x)))) if r.success else None)
                    write_json(out/'measurements'/f'{pid}_{repeat}_{engine}.json',record)
                    print(json.dumps(dict(id=pid,repeat=repeat,engine=engine,success=r.success,seconds=external,rss=rss())),flush=True)
                    if not r.success:raise RuntimeError(r.failure_reason)
                for engine in ('numpy','numba'):
                    for field in ('f','stderr'):
                        if np.max(abs(getattr(results[engine],field)-getattr(results['reference'],field)))>1e-8:
                            raise RuntimeError(f'{pid}: paired {field} discrepancy')
        for pid in plan['grid_ids']:
            task=next(t for t in plan['tasks'] if t['problem_id']==pid);e=equations[pid];p=prepare_all_methods_mc(e.metadata)
            cfg=MCConfig(M=plan['grid_M'],chunk=1024,dt=1/128,T=task['T'],seed=plan['grid_seed'],max_path_steps=300000000,max_batch_paths=1024)
            t=time.perf_counter();r=solve_monte_carlo_optimized(p,np.linspace(-1,1,33),cfg);external=time.perf_counter()-t
            write_json(out/'measurements'/f'{pid}_grid_numba.json',dict(task=task,engine='numba',series='grid',config=asdict(cfg),
                result=asdict(r),external_solve_seconds=external,process_peak_rss_mib=rss(),y_exact=e.y_exact(r.x),
                error_infinity=float(np.max(abs(r.f-e.y_exact(r.x)))) if r.success else None))
            print(json.dumps(dict(id=pid,grid=True,success=r.success,seconds=external,rss=rss())),flush=True)
            if not r.success:raise RuntimeError(r.failure_reason)
        analytics=[]
        for kind in ('constant','gaussian'):
            p=prepare_analytic_mc(kind);cfg=MCConfig(M=8192,chunk=1024,dt=1/128 if kind=='constant' else 1/32,T=8. if kind=='constant' else 4.,seed=plan['analytical_seed'])
            r=solve_monte_carlo_optimized(p,[0.],cfg)
            t=np.arange(r.n_steps+1)*r.dt_actual
            f=np.exp(-2*t) if kind=='constant' else np.exp(-2*t)*4*(1+2*t)/(1+4*t)**1.5
            discrete=r.dt_actual*(.5*(f[0]+f[-1])+sum(f[1:-1]));exact=.5 if kind=='constant' else 1.
            analytics.append(dict(kind=kind,config=asdict(cfg),result=asdict(r),exact=exact,discrete_expectation=discrete,
                observed_error=float(abs(r.f[0]-exact)),discretization_and_tail=discrete-exact))
        write_json(out/'analytics.json',analytics)
    finally:
        stop.set();thread.join()


def run(out):
    from diagnostics.chernov_batch_control import controller_locks,new_control,GuardedProcess,finish_control,atomic
    plan=json.loads((out/'execution_plan.json').read_text());began=time.perf_counter();reason=None;peak=0.
    with controller_locks(out) as fds:
        active=new_control(out,'backend_benchmark',lease_seconds=plan['owner_lease_seconds'])
        with (out/'worker.log').open('x') as log:
            proc=GuardedProcess([sys.executable,'-B',str(Path(__file__).resolve()),'worker','--out',str(out)],
                log,out/'control/backend_benchmark/worker.guardian.json',active,fds,heartbeat_timeout=15.)
            try:
                while proc.poll() is None:
                    sample=out/'resource_latest.json'
                    if sample.exists():
                        v=json.loads(sample.read_text());peak=max(peak,v['peak_rss_mib'])
                        if peak>plan['rss_limit_mib']:reason='external_memory_limit'
                        elif time.monotonic()-v['monotonic']>plan['telemetry_stale_seconds']:reason='resource_telemetry_stale'
                    if reason:proc.terminate();break
                    time.sleep(.2)
                proc.wait(timeout=10.)
            finally:
                if proc.poll() is None:proc.terminate();proc.wait(timeout=10.)
                finish_control(out,active)
            if proc.stop_reason!='worker_exit' and not reason:reason=proc.stop_reason
            guardian=json.loads(proc.guardian_path.read_text())
            if guardian.get('worker_returncode') and not reason:reason='worker_exception'
            atomic(out/'run_summary.json',dict(reason=reason,wall_seconds=time.perf_counter()-began,
                peak_rss_mib=peak,guardian=guardian,measurements=len(list((out/'measurements').glob('*.json')))))
            print((out/'run_summary.json').read_text(),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=('freeze','run','worker'));ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();globals()[a.action](a.out.resolve())
