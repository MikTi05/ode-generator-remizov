#!/usr/bin/env python3
"""One bounded, preregistered development pilot: 60 tasks, RK4/PI, four N."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime,timezone
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from diagnostics.cost60_common import (DATA,SPLITS,OTHER_TEST,METHODS,NS,EPS,REPEATS,POLICY,FEATURES,
    atomic_json,configuration,descriptor,digest_key,read,require,select_metadata,sha)
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')


def utc():return datetime.now(timezone.utc).isoformat()


def write_csv(path,rows):
    import csv
    with path.open('x',newline='',encoding='utf-8') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def prepare(out):
    require(not out.exists(),'choose a new output directory')
    metadata=[json.loads(line) for line in (ROOT/DATA/'all_methods_equations_v1_metadata.jsonl').read_text().splitlines()]
    splits=read(ROOT/SPLITS)
    other_test=[json.loads(line) for line in (ROOT/OTHER_TEST).read_text().splitlines()]
    chosen,selection,pool,audit=select_metadata(metadata,splits,other_test)
    paths=[SPLITS,OTHER_TEST,Path('reports/ml_selector_contract/CONTRACT.md'),Path('all_methods_dataset_loader.py'),
           Path('src/problem_types.py'),Path('src/all_methods_expressions.py'),Path('src/all_methods_dataset_v1.py'),
           Path('src/numerical_methods/__init__.py'),Path('src/numerical_methods/rk4_solver.py'),
           Path('src/numerical_methods/product_integral_solver.py'),Path('MIGRATION_MANIFEST.json'),
           *[Path(n) for n in ('diagnostics/cost60_common.py','diagnostics/run_cost60.py','diagnostics/report_cost60.py',
                               'diagnostics/verify_cost60.py','tests/test_cost60.py')]]
    paths.extend(p.relative_to(ROOT) for p in sorted((ROOT/DATA).iterdir()) if p.is_file())
    out.mkdir(parents=True,exist_ok=False);(out/'records').mkdir();(out/'inputs').mkdir()
    write_csv(out/'selection.csv',selection);write_csv(out/'candidate_pool.csv',pool)
    atomic_json(out/'plan.json',dict(schema='cost60_v1',created_utc=utc(),date='2026-11-04',dataset_dir=str(DATA),
        equation_ids=[m['equation_id'] for m in chosen],selection='three per complexity/regime; hashed representative; min-max input certificates/ranges; farthest-point families',
        selection_features=list(FEATURES),selection_seed=2488,selection_audit=audit,
        selection_sha256=sha(out/'selection.csv'),candidate_pool_sha256=sha(out/'candidate_pool.csv'),
        methods=list(METHODS),N=list(NS),epsilon=EPS,target_x=[-1+i/16 for i in range(33)],a_min=.001,
        warmup_per_configuration=1,measured_repeats=REPEATS,expected_warmup_calls=480,expected_measured_calls=2400,maximum_solver_calls=2880,
        timing='perf_counter_ns around unmodified public solver wrapper; asdict/scoring/IO excluded; median five measured calls',
        timing_order='warmup then five complete rounds; hash permutation of equations and all eight configurations per round',
        gc_policy='GC disabled for solver phase; collect before each equation group, outside timer',
        accuracy_policy='all five deterministic repeats must be identical and successful; E33 from exactly nested nodes; <=.01',
        baseline_policy='fastest constant configuration accurate on every task; also report max-coverage fallback if none; RK4-only fastest feasible and minimum-N references use hindsight',
        practical_policy='MAD/median<=.20 in both; median fast<=.9 slow and max fast<min slow; marginal PI switches revert to RK4 in conservative oracle',
        GO_policy=POLICY,GO_scope='development pilot feasibility gate only; never evidence of learnability/generalization',
        source_sha256={str(p):sha(ROOT/p) for p in paths},
        resources=dict(compute_workers=1,thread_env={k:'1' for k in THREADS},
                       expected_RK4_RHS_evals=691200,expected_PI_expm_evals=172800,
                       estimated_machine_minutes=[4,12],estimate_basis='old fixed128 series ~308s/256000 aggregate steps; extrapolation only; no hard time cutoff'),
        historical_test_used_for_selection=False,ML_training=False,MC_calls=0,remizov_calls=0,
        no_automatic_expansion=True,no_post_result_repeats=True))
    print(json.dumps(dict(prepared=str(out),families=len(chosen),cells=dict(Counter(m['complexity_level']+'/'+m['regime'] for m in chosen)),maximum_solver_calls=2880),ensure_ascii=False),flush=True)


def system_snapshot():
    def command(args):
        p=subprocess.run(args,capture_output=True,text=True)
        return dict(returncode=p.returncode,stdout=p.stdout.strip(),stderr=p.stderr.strip())
    return dict(utc=utc(),platform=platform.platform(),python=sys.version,pid=os.getpid(),
        hardware={name:command(['/usr/sbin/sysctl','-n',name]) for name in ('machdep.cpu.brand_string','hw.model','hw.physicalcpu','hw.memsize')},
        power=command(['/usr/bin/pmset','-g','batt']),thermal=command(['/usr/bin/pmset','-g','therm']),
        threads={k:os.environ.get(k) for k in THREADS},
        versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','sympy')})


def run(out):
    plan=read(out/'plan.json')
    require(plan['methods']==list(METHODS) and plan['N']==list(NS) and plan['GO_policy']==POLICY,'plan policy mismatch')
    require(plan['epsilon']==EPS and plan['measured_repeats']==REPEATS and len(plan['equation_ids'])==60,'plan size mismatch')
    require(all(sha(ROOT/p)==digest for p,digest in plan['source_sha256'].items()),'pinned source changed')
    require(sha(out/'selection.csv')==plan['selection_sha256'] and sha(out/'candidate_pool.csv')==plan['candidate_pool_sha256'],'selection changed')
    require(all(os.environ.get(k)=='1' for k in THREADS),'set all five thread environment variables to1')
    atomic_json(out/'started.json',dict(utc=utc(),plan_sha256=sha(out/'plan.json'),single_run=True))
    started=time.perf_counter();import_start=time.perf_counter()
    import numpy as np
    import scipy
    from threadpoolctl import threadpool_info
    from all_methods_dataset_loader import load_all_methods_equations
    from src.numerical_methods import solve_problem_rk4,solve_problem_product_integral_left_exponential
    import_seconds=time.perf_counter()-import_start
    solvers=dict(rk4=solve_problem_rk4,product_integral=solve_problem_product_integral_left_exponential)
    load_start=time.perf_counter()
    equations=load_all_methods_equations(ROOT/DATA/'all_methods_equations_v1.csv',ROOT/DATA/'all_methods_equations_v1_metadata.jsonl')
    load_seconds=time.perf_counter()-load_start
    by_id={e.equation_id:e for e in equations};chosen={pid:by_id[pid] for pid in plan['equation_ids']}
    prep_start=time.perf_counter();exact={};preparation=[];grid=np.linspace(-1,1,33)
    for pid,e in chosen.items():
        t=time.perf_counter();p=e.cauchy_problem
        require(e.metadata['split']=='train' and (p.x0,p.x1)==(-1.,1.),'nondevelopment or wrong interval')
        require(all(getattr(p,k)==e.metadata[k] for k in ('x0','x1','y0','v0')),'IVP input mismatch')
        exact[pid]=np.asarray(e.y_exact(grid),dtype=float)
        require(exact[pid].shape==(33,) and np.isfinite(exact[pid]).all(),'invalid exact')
        samples={name:[float(getattr(p,name)(float(x))) for x in grid] for name in ('a','b','c','g')}
        record=dict(equation_id=pid,family_id=e.metadata['family_id'],expression_hash=e.metadata['expression_hash'],
            structure_signature=e.metadata['structure_signature'],split=e.metadata['split'],
            inputs={k:getattr(p,k) for k in ('x0','x1','y0','v0')},target_x=grid.tolist(),exact33=exact[pid].tolist(),
            coefficient_samples=samples,preparation_seconds=time.perf_counter()-t)
        atomic_json(out/'inputs'/(pid+'.json'),record);preparation.append(record['preparation_seconds'])
    prep_seconds=time.perf_counter()-prep_start
    environment=system_snapshot();environment.update(threadpools=threadpool_info(),clock=vars(time.get_clock_info('perf_counter')))
    require(all(p['num_threads']==1 for p in environment['threadpools']),'numeric threadpool is not single-threaded')
    atomic_json(out/'environment.before.json',environment)
    atomic_json(out/'preparation.json',dict(import_seconds=import_seconds,dataset_load_seconds=load_seconds,
        selected_input_exact_preparation_and_IO_seconds=prep_seconds,per_task_preparation_seconds=preparation,
        note='loader validates all1000 records but solvers receive only selected60; all scoring/inputs outside solver timer'))
    plan_sha=sha(out/'plan.json');calls=0;warm=0;measured=0;phase_start=time.perf_counter()
    gc_was_enabled=gc.isenabled();gc.disable()
    try:
        for episode in range(-1,REPEATS):
            phase='warmup' if episode==-1 else 'measured'
            ordered=sorted(chosen,key=lambda pid:digest_key('round',episode,pid))
            for pid in ordered:
                gc.collect()
                p=chosen[pid].cauchy_problem;m=chosen[pid].metadata
                configs=sorted(((method,n) for method in METHODS for n in NS),key=lambda c:digest_key('config',episode,pid,*c))
                for method,n in configs:
                    require(calls<plan['maximum_solver_calls'],'solver call budget exhausted')
                    solve=solvers[method];result=None;exception=None
                    start_ns=time.perf_counter_ns()
                    try:result=solve(p,n_steps=n,a_min=.001)
                    except Exception:exception=traceback.format_exc()
                    elapsed_ns=time.perf_counter_ns()-start_ns
                    calls+=1;warm+=episode==-1;measured+=episode!=-1
                    result_dict=asdict(result) if result is not None else None
                    status='solver_exception' if exception else 'solver_failure';usable=False;error=None
                    scoring_start=time.perf_counter()
                    if result is not None and result.success:
                        yy=np.asarray(result.y);xx=np.asarray(result.x);dy=np.asarray(result.dy)
                        require(len(xx)==len(yy)==len(dy)==n+1,'incomplete successful trajectory')
                        require(np.array_equal(xx,np.linspace(-1,1,n+1)) and np.array_equal(xx[::(n//32)],grid),'wrong nested grid')
                        require(np.isfinite(yy).all() and np.isfinite(dy).all(),'nonfinite successful trajectory')
                        require(yy[0]==p.y0 and dy[0]==p.v0,'initial state mismatch')
                        error=float(np.max(np.abs(yy[::(n//32)]-exact[pid])))
                        usable=True;status='success'
                    record=dict(schema='cost60_attempt_v1',plan_sha256=plan_sha,equation_id=pid,family_id=m['family_id'],
                        expression_hash=m['expression_hash'],method=method,N=n,phase=phase,episode=episode,sequence=calls,
                        solver_nanoseconds=elapsed_ns,solver_seconds=elapsed_ns/1e9,usable=usable,status=status,
                        E33=error,accuracy_met=error<=EPS if usable else None,result=result_dict,exception=exception,
                        scoring_seconds=time.perf_counter()-scoring_start,finished_utc=utc())
                    atomic_json(out/'records'/f'{phase}_{episode}_{pid}_{method}_N{n}.json',record)
                if calls%120==0:
                    print(json.dumps(dict(calls=calls,budget=2880,warmup=warm,measured=measured,phase=phase,episode=episode,
                        elapsed_seconds=round(time.perf_counter()-phase_start,2),last=pid),ensure_ascii=False),flush=True)
            atomic_json(out/f'round_{episode}.json',dict(phase=phase,episode=episode,completed_calls=calls,snapshot=system_snapshot()))
    finally:
        if gc_was_enabled:gc.enable()
    require(calls==2880 and warm==480 and measured==2400,'incomplete fixed experiment')
    atomic_json(out/'environment.after.json',system_snapshot())
    require(all(sha(ROOT/p)==digest for p,digest in plan['source_sha256'].items()),'source changed during experiment')
    atomic_json(out/'session.json',dict(complete=True,finished_utc=utc(),solver_calls=calls,warmup_calls=warm,measured_calls=measured,
        solver_phase_wall_seconds=time.perf_counter()-phase_start,total_wall_seconds=time.perf_counter()-started,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024),
        forbidden_calls=dict(MC=0,remizov=0,fit=0,inference=0),input_sources_unchanged=True))
    print(json.dumps(read(out/'session.json')),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    if args.action=='prepare':prepare(args.out.resolve())
    else:
        import fcntl
        with (args.out/'run.lock').open('a') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);run(args.out.resolve())
