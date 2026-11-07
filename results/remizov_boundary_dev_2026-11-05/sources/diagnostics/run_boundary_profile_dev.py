#!/usr/bin/env python3
"""One fixed development intervention; only missing cubic/zero solves are run."""
from __future__ import annotations
import argparse
from dataclasses import asdict,replace
from datetime import datetime,timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from diagnostics.boundary_profile_common import (DATA,BASE,GRID_RELEASE,OLD_DIAG,SPLITS,OTHER_TEST,
    CONFIG,EPS,CAT,atomic_json,csv_write,read,require,select,sha,table,max_error)
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')


def utc():return datetime.now(timezone.utc).isoformat()


def clean(value):
    import numpy as np
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.bool_):return bool(value)
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,(float,np.floating)):return float(value) if math.isfinite(value) else None
    return value


def numerical_inputs(problem,config,before):
    """Reproduce only input/grid/quadrature preparation, with no S applications."""
    import numpy as np
    from chernov_solver import _State,_simpson_weights
    from remizov_solver import _as_values,_evaluate_coefficients,compute_truncation_T
    T=compute_truncation_T(problem,config.tail_eps) if config.T is None else float(config.T)
    require(T==before['result']['T'],'effective T changed')
    domain=(problem.x_target_min-config.padding,problem.x_target_max+config.padding)
    intervals=max(3,math.ceil((domain[1]-domain[0])/config.dx));grid=np.linspace(*domain,intervals+1)
    a,b,c=_evaluate_coefficients(problem,grid);g=_as_values(problem.g_remizov,grid,'nonfinite_state')
    state=_State(grid,a,b,c,g,problem.lambda_value,config.n,config.interp,config.boundary,
                 config.taper_width or .1*(domain[1]-domain[0]))
    s=np.linspace(0,1,config.num_t+(config.num_t%2==0));times=T*(s**2 if config.t_grading=='quadratic' else s)
    weights=_simpson_weights(times)
    arrays=dict(grid=grid,a=a,b=b,c_R=c,g_R=g,t=times,weights=weights,target_x=np.asarray(before['target_x']),exact=np.asarray(before['y_exact']))
    fingerprints={name:__import__('hashlib').sha256(np.ascontiguousarray(value,dtype=np.float64).tobytes()).hexdigest() for name,value in arrays.items()}
    require(list(domain)==before['result']['work_domain'] and len(grid)==before['result']['work_grid_points'],'work grid changed')
    return state,times,weights,fingerprints


def check_saved_zero(record,before,fingerprints):
    require(record['config']==dict(CONFIG,boundary='zero'),'cached zero config mismatch')
    require(record['problem_id']==before['problem_id'] and record['production_result_sha256']==before['_source_sha256'],'cached zero baseline identity mismatch')
    require(record['y_exact']==before['y_exact'],'cached exact differs')
    require(all(fingerprints[k]==v for k,v in record['fingerprints'].items()),'cached numerical inputs differ')
    q=record['result'];require(q['scheme']=='solve' and q['x']==before['target_x'],'cached target/scheme')
    if q['success']:
        require(max_error(q['f'],record['y_exact'])==record['max_abs_error'],'cached error mismatch')
    for k in ('T','n','n_steps','time_nodes','actual_dx','work_grid_points','work_domain','n_operator_applications','n_spline_builds','outside_query_values'):
        require(q[k]==before['result'][k],'cached geometry/work changed: '+k)


def prepare(out):
    require(not out.exists(),'choose a NEW output directory')
    from all_methods_dataset_loader import load_all_methods_equations
    from chernov_solver import ChernovConfig
    import numpy as np
    md=[json.loads(line) for line in (ROOT/DATA/'all_methods_equations_v1_metadata.jsonl').read_text().splitlines()]
    other=[json.loads(line) for line in (ROOT/OTHER_TEST).read_text().splitlines()]
    release=read(ROOT/GRID_RELEASE/'verification.json')
    require(release['passed'] and sha(ROOT/GRID_RELEASE/'grid33_all1000.csv')==release['artifacts_sha256']['grid33_all1000.csv'],'grid release changed')
    selection,audit=select(table(ROOT/GRID_RELEASE/'grid33_all1000.csv'),md,read(ROOT/SPLITS),other)
    require(audit['case_count']==22 and audit['control_count']==30,'unexpected fixed sample size')
    old_plan=read(ROOT/OLD_DIAG/'plan.json')
    require(old_plan['baseline_config']==CONFIG,'old baseline configuration differs')
    require(all(sha(ROOT/p)==digest for p,digest in old_plan['source_data_sha256'].items()),'old diagnostic source/data no longer match')
    equations={e.equation_id:e for e in load_all_methods_equations(ROOT/DATA/'all_methods_equations_v1.csv',ROOT/DATA/'all_methods_equations_v1_metadata.jsonl')}
    source_names={str(SPLITS),str(OTHER_TEST),str(GRID_RELEASE/'grid33_all1000.csv'),str(OLD_DIAG/'plan.json'),
        'chernov_solver.py','remizov_solver.py','all_methods_dataset_loader.py','src/all_methods_expressions.py',
        'src/all_methods_dataset_v1.py','MIGRATION_MANIFEST.json','diagnostics/boundary_profile_common.py',
        'diagnostics/run_boundary_profile_dev.py','diagnostics/report_boundary_profile_dev.py','diagnostics/verify_boundary_profile_dev.py',
        'tests/test_boundary_profile_dev.py'}
    source_names.update(str(p.relative_to(ROOT)) for p in (ROOT/DATA).iterdir() if p.is_file())
    tasks=[];config=ChernovConfig(**CONFIG)
    for row in selection:
        pid=row['equation_id'];name=BASE/(pid+'.json');before=read(ROOT/name)
        before['_source_sha256']=sha(ROOT/name);source_names.add(str(name))
        require(before['config']==CONFIG and before['computation_status']=='success','baseline unavailable/config changed')
        require(before['metadata']==equations[pid].metadata,'canonical identity mismatch')
        require(before['target_x']==[-1+i/16 for i in range(33)],'baseline target changed')
        require(max_error(before['result']['f'],before['y_exact'])==row['E_before']==before['max_abs_error'],'baseline error mismatch')
        require(np.array_equal(equations[pid].y_exact(np.asarray(before['target_x'])),before['y_exact']),'canonical exact changed')
        state,t,w,fingerprints=numerical_inputs(equations[pid].remizov_problem,config,before)
        zero_state,zt,zw,zfingerprints=numerical_inputs(equations[pid].remizov_problem,replace(config,boundary='zero'),before)
        require(fingerprints==zfingerprints,'intervention changed inputs beyond boundary')
        cached=OLD_DIAG/'runs'/(pid+'_cubic_zero.json');cached_path=None
        if (ROOT/cached).exists():
            raw=read(ROOT/cached);check_saved_zero(raw,before,fingerprints);cached_path=str(cached);source_names.add(cached_path)
        tasks.append(dict(**row,before_source=str(name),before_sha256=before['_source_sha256'],
            cached_zero_source=cached_path,effective_T=before['result']['T'],fingerprints=fingerprints,
            initial_boundary_values=[float(state.g[0]),float(state.g[-1])],
            initial_forcing_support_inside_window=bool(equations[pid].metadata['support_left']>state.x[0] and equations[pid].metadata['support_right']<state.x[-1])))
    out.mkdir(parents=True,exist_ok=False);(out/'records').mkdir()
    csv_write(out/'selection.csv',selection)
    inventory=Path('/private/tmp/boundary_dev_reuse_inventory_2026-11-05.json')
    if inventory.exists():atomic_json(out/'reuse_inventory.json',read(inventory))
    plan=dict(schema='boundary_profile_dev_v1',created_utc=utc(),date='2026-11-05',tasks=tasks,selection_audit=audit,
        baseline_config=CONFIG,zero_config=dict(CONFIG,boundary='zero'),epsilon=EPS,catastrophe_threshold=CAT,
        comparison_tolerance='1e-10*max(1,E_before); numerical comparison convention, not an error certificate',
        GO_policy=['all zero usable','no material regressions','no tolerance losses','no new E33>1',
                   'all former E33>1 reduced to <=1','strict increase of pass count','at least one material improvement'],
        profile_scope='experimental only; existing production taper stays unchanged; no theory certification',
        source_sha256={name:sha(ROOT/name) for name in sorted(source_names)},selection_sha256=sha(out/'selection.csv'),
        new_zero_call_budget=sum(t['cached_zero_source'] is None for t in tasks),new_taper_call_budget=0,
        reuse_count=sum(t['cached_zero_source'] is not None for t in tasks),
        resources=dict(compute_workers=1,n_jobs=1,thread_env={name:'1' for name in THREADS},
            expected_operator_applications=sum(t['cached_zero_source'] is None for t in tasks)*128*512,
            expected_work_grid_points=2081,estimated_minutes=[8,15],estimate_basis='50 calls at historical cubic/zero runtimes~12s; not a time cutoff'),
        environment=dict(python=platform.python_version(),platform=platform.platform(),versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','sympy')}),
        no_automatic_expansion=True,no_new_taper=True,no_linear_or_other_continuation=True,MC_calls=0,ML_calls=0)
    atomic_json(out/'plan.json',plan)
    print(json.dumps(dict(prepared=str(out),tasks=len(tasks),cases=audit['case_count'],controls=audit['control_count'],new_zero=plan['new_zero_call_budget'],reused_zero=plan['reuse_count'],new_taper=0),ensure_ascii=False),flush=True)


def run(out):
    import numpy as np
    from all_methods_dataset_loader import load_all_methods_equations
    from chernov_solver import ChernovConfig,ChernovSolver
    plan=read(out/'plan.json');plan_sha=sha(out/'plan.json')
    require(plan['zero_config']==dict(CONFIG,boundary='zero') and plan['baseline_config']==CONFIG,'fixed intervention changed')
    require(all(sha(ROOT/p)==v for p,v in plan['source_sha256'].items()),'pinned source changed')
    require(sha(out/'selection.csv')==plan['selection_sha256'],'selection changed')
    require(all(os.environ.get(name)=='1' for name in THREADS),'set all planned thread variables=1')
    atomic_json(out/'started.json',dict(utc=utc(),plan_sha256=plan_sha))
    began=time.perf_counter();load_start=time.perf_counter()
    equations={e.equation_id:e for e in load_all_methods_equations(ROOT/DATA/'all_methods_equations_v1.csv',ROOT/DATA/'all_methods_equations_v1_metadata.jsonl')}
    atomic_json(out/'environment.json',dict(**plan['environment'],threads={name:os.environ.get(name) for name in THREADS},pid=os.getpid(),dataset_load_seconds=time.perf_counter()-load_start))
    config=ChernovConfig(**plan['zero_config']);calls=0;reused=0
    with (out/'call_journal.jsonl').open('x') as journal:
        for task in plan['tasks']:
            pid=task['equation_id'];before=read(ROOT/task['before_source']);before['_source_sha256']=sha(ROOT/task['before_source'])
            _,_,_,fingerprints=numerical_inputs(equations[pid].remizov_problem,config,before)
            require(fingerprints==task['fingerprints'],'pre-solve fingerprint changed')
            cached=task['cached_zero_source'];exception=None;wall=None
            if cached:
                old=read(ROOT/cached);check_saved_zero(old,before,fingerprints);result=old['result'];reused+=1
            else:
                require(calls<plan['new_zero_call_budget'],'zero call budget exhausted')
                journal.write(json.dumps(dict(event='start',problem_id=pid,call_index=calls+1,utc=utc()))+'\n');journal.flush()
                start=time.perf_counter();calls+=1
                try:result=clean(asdict(ChernovSolver(equations[pid].remizov_problem,config).solve(before['target_x'])))
                except Exception:result=None;exception=traceback.format_exc()
                wall=time.perf_counter()-start
            usable=bool(result and result['success'] and result['failure_reason']=='')
            if usable:
                require(result['scheme']=='solve' and result['problem_id']==pid and result['x']==before['target_x'],'wrong result identity/target')
                for field in ('T','n','n_steps','time_nodes','actual_dx','work_grid_points','work_domain','n_operator_applications','n_spline_builds','outside_query_values'):
                    require(result[field]==before['result'][field],'result geometry/work mismatch: '+field)
            error=max_error(result['f'],before['y_exact']) if usable else None
            record=dict(schema='boundary_profile_result_v1',plan_sha256=plan_sha,problem_id=pid,
                expression_hash=task['expression_hash'],category=task['category'],config=plan['zero_config'],
                before_source=task['before_source'],before_sha256=task['before_sha256'],zero_source=cached or str((out/'records'/(pid+'.json')).relative_to(ROOT)),
                reused=bool(cached),fingerprints=fingerprints,y_exact=before['y_exact'],result=result,
                max_abs_error=error,accuracy_met=error<=EPS if usable else None,exception=exception,
                driver_solver_wall_seconds=wall,finished_utc=utc(),peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024))
            atomic_json(out/'records'/(pid+'.json'),record)
            journal.write(json.dumps(dict(event='saved',problem_id=pid,new_call=not bool(cached),usable=usable,utc=utc()))+'\n');journal.flush()
            print(json.dumps(dict(saved=calls+reused,total=len(plan['tasks']),new_calls=calls,reused=reused,last=pid,seconds=round(time.perf_counter()-began,2)),ensure_ascii=False),flush=True)
    require(calls==plan['new_zero_call_budget'] and reused==plan['reuse_count'],'incomplete fixed series')
    require(all(sha(ROOT/p)==v for p,v in plan['source_sha256'].items()),'source changed during experiment')
    atomic_json(out/'session.json',dict(complete=True,finished_utc=utc(),new_zero_calls=calls,reused_zero=reused,new_taper_calls=0,records=calls+reused,
        wall_seconds=time.perf_counter()-began,MC_calls=0,ML_calls=0,other_continuations_called=0,source_unchanged=True))
    print(json.dumps(read(out/'session.json')),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','run'));parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.out.resolve())
    else:
        import fcntl
        with (args.out/'run.lock').open('a') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);run(args.out.resolve())
