"""New resumable MC1000 x=0 series. One guarded persistent compute worker.

Terminal results (including resource failures) are immutable. Interrupted
requests remain pending. Absolute series deadline survives every resume.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import threading
import time
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')
for name in THREADS:os.environ[name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from diagnostics.run_chernov_batch import write_json,read_json,digest
from diagnostics.chernov_batch_control import (controller_locks,new_control,finish_control,
    GuardedProcess,CONTROL_STOPS)


def utc():return datetime.now(timezone.utc).isoformat()
def rss():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)
def key(pid,index):return f'{pid}_seed{index}'


def initialize(out):
    import numpy as np
    from all_methods_dataset_loader import load_all_methods_equations
    from mc_optimized import prepare_all_methods_mc
    from mc_solver import MCConfig,auto_T
    from diagnostics.plan_joint_benchmark import protected_snapshot
    data=ROOT/'data/all_methods_v1_1000_2026-10-18'
    equations=load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl')
    if len(equations)!=1000:raise ValueError('expected production 1000')
    expected=read_json(ROOT/'results/dataset1000_2026-10-18/artifacts_sha256.json')
    if any(digest(data/n)!=h for n,h in expected.items()):raise ValueError('production SHA mismatch')
    rows=[]
    for e in equations:
        p=prepare_all_methods_mc(e.metadata);T=auto_T(p.problem.lambda_value,p.problem.c_abs_upper_bound,p.forcing_bound.abs_upper_bound,1e-4)
        rows.append(dict(problem_id=e.equation_id,family_id=e.metadata['family_id'],split=e.metadata['split'],
            expression_hash=e.metadata['expression_hash'],G=p.forcing_bound.abs_upper_bound,T=T,N=int(np.ceil(T*128)),
            ast_nodes=len(p.coeff_tape)+3*len(p.y_tape)))
    cfg=MCConfig(M=1024,chunk=1024,dt=1/128,T=None,tail_eps=1e-4,seed=None,
        max_steps=100000,max_path_steps=100000000,max_batch_paths=1024,max_runtime_seconds=60.)
    cfg.validate();seeds=[202610250101+i for i in range(4)]
    sources=['mc_optimized.py','mc_ast_kernels.py','mc_solver.py','remizov_solver.py',
        'all_methods_bounds.py','all_methods_dataset_loader.py','src/all_methods_expressions.py',
        'src/all_methods_dataset_v1.py','diagnostics/mc_production_x0.py',
        'diagnostics/chernov_batch_control.py','diagnostics/run_chernov_batch.py']
    hashes={n:digest(ROOT/n) for n in sources};hashes.update({str((data/n).relative_to(ROOT)):h for n,h in expected.items()})
    ids=[e.equation_id for e in equations]
    plan=dict(schema='optimized_mc1000_x0_v1',created_utc=utc(),data_dir=str(data.relative_to(ROOT)),
        implementation='mc_optimized.solve_monte_carlo_optimized',engine='numba',block_steps=64,
        config=asdict(cfg),target_x=[0.],seeds=seeds,threshold=.01,equation_ids=ids,
        batches=[ids[i:i+20] for i in range(0,1000,20)],preflight=rows,
        resources=dict(attempt_seconds=60.,series_seconds=14400.,rss_mib=512.,compute_workers=1,
            startup_seconds=60.,owner_lease_seconds=300.,poll_seconds=.025,monitor_seconds=.25),
        total_attempts=4000,total_path_updates=sum(r['N']*1024*4 for r in rows),
        seeds_independent=True,parameter_policy='fixed inputs, T=None analytic auto_T; no exact-guided choices',
        exact_policy='evaluate only after solve; no exact access for T/parameters/features',
        completion_policy='immutable terminal records, interrupted stays pending; no retry of resource failures',
        smoke=dict(ids=[ids[0],ids[-1]],seed=202610250199,adopt_into_mass=False),
        protected_sha256=hashes)
    out.mkdir(parents=True,exist_ok=False)
    for folder in ('attempts','events','sessions','workers','batches','sources','smoke'):(out/folder).mkdir()
    shutil.copy2(ROOT/'PROJECT_STATUS.md',out/'PROJECT_STATUS.before.md')
    write_json(out/'protected_before.json',protected_snapshot(out),exclusive=True)
    for n in sources:
        dest=out/'sources'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    write_json(out/'plan.json',plan,exclusive=True)
    write_json(out/'environment.json',dict(python=sys.version,platform=sys.platform,
        versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','numba','llvmlite','pandas','scikit-learn','catboost')},threads={n:os.environ[n] for n in THREADS}),exclusive=True)
    print(json.dumps(dict(initialized=str(out),tasks=4000,path_updates=plan['total_path_updates'],max_T=max(r['T'] for r in rows))),flush=True)


def validate_record(record,plan,plan_sha):
    pid=record['problem_id'];i=record['seed_index'];cfg=dict(plan['config'],seed=plan['seeds'][i])
    if (record['plan_sha256']!=plan_sha or pid not in plan['equation_ids'] or record['config']!=cfg
        or record['target_x']!=[0.] or record['implementation']!=plan['implementation']):
        raise ValueError('saved attempt identity/config mismatch')
    expected=next(r['expression_hash'] for r in plan['preflight'] if r['problem_id']==pid)
    if record['expression_hash']!=expected:raise ValueError('expression mismatch')


def collect(out,plan):
    result={};plan_sha=digest(out/'plan.json')
    for p in (out/'attempts').glob('*.json'):
        r=read_json(p);validate_record(r,plan,plan_sha)
        k=key(r['problem_id'],r['seed_index'])
        if p.stem!=k:raise ValueError('wrong attempt filename')
        result[k]=r
    return result


def worker(out,state_path,parent_pid):
    state=dict(pid=os.getpid(),phase='loading',key=None);stop=threading.Event();plan=read_json(out/'plan.json')
    def monitor():
        with state_path.with_suffix('.rss.jsonl').open('x') as f:
            while not stop.is_set():
                if os.getppid()!=parent_pid:os._exit(75)
                sample=dict(state,monotonic=time.monotonic(),peak_rss_mib=rss())
                write_json(state_path,sample,durable=False);f.write(json.dumps(sample)+'\n');f.flush()
                stop.wait(plan['resources']['monitor_seconds'])
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        import numpy as np
        from all_methods_dataset_loader import load_all_methods_equations
        from mc_optimized import prepare_all_methods_mc,compile_backend,solve_monte_carlo_optimized
        from mc_solver import MCConfig
        t=time.perf_counter();d=ROOT/plan['data_dir']
        equations={e.equation_id:e for e in load_all_methods_equations(d/'all_methods_equations_v1.csv',d/'all_methods_equations_v1_metadata.jsonl')}
        load_seconds=time.perf_counter()-t;compile_info=compile_backend()
        state.update(phase='ready',load_seconds=load_seconds,compilation=compile_info)
        write_json(state_path,dict(state,monotonic=time.monotonic(),peak_rss_mib=rss()),durable=False)
        for line in sys.stdin:
            request=read_json(Path(json.loads(line)['request_path']));k=request['key'];pid=request['problem_id']
            state.update(phase='solving',key=k);began=time.perf_counter()
            r={n:request[n] for n in ('problem_id','seed_index','key','plan_sha256','config','expression_hash','target_x','session')}
            r.update(implementation=plan['implementation'],worker_pid=os.getpid(),started_utc=utc(),
                shared_dataset_load_seconds=load_seconds,shared_compilation=compile_info)
            try:
                e=equations[pid];p=prepare_all_methods_mc(e.metadata)
                result=solve_monte_carlo_optimized(p,[0.],MCConfig(**request['config']),engine='numba',block_steps=64)
                # First exact invocation in a numerical attempt is AFTER solve.
                exact=float(e.y_exact(np.array([0.]))[0])
                error=float(abs(result.f[0]-exact)) if result.success else None
                resource_status='external_time_limit' if result.failure_reason=='time_limit_exceeded' else None
                r.update(result=asdict(result),y_exact=exact,abs_error=error,
                    accuracy_met=(error<=.01) if error is not None else None,
                    status=resource_status or ('completed' if result.success else 'solver_failure'),
                    failure_reason=result.failure_reason,forcing_bound=asdict(p.forcing_bound),
                    metadata_identity=dict(family_id=e.metadata['family_id'],split=e.metadata['split'],expression_hash=e.metadata['expression_hash']))
            except Exception:
                r.update(status='worker_exception',result=None,y_exact=None,abs_error=None,accuracy_met=None,traceback=traceback.format_exc())
            elapsed=time.perf_counter()-began
            r.update(task_wall_seconds=elapsed,finished_utc=utc(),process_peak_rss_mib=rss())
            if elapsed>60:r['status']='external_time_limit'
            if rss()>512:r['status']='external_memory_limit'
            write_json(Path(request['result_path']),r,exclusive=True)
            state.update(phase='ready',key=None)
    finally:stop.set();thread.join()


def wait(child,path,state_path,deadline,limits,series_deadline,stop_file):
    peak=0.
    while True:
        child.poll() # controller heartbeat to independent guardian
        if path is not None and path.exists():return None,peak
        state=read_json(state_path) if state_path.exists() else {}
        peak=max(peak,state.get('peak_rss_mib',0.))
        if path is None and state.get('phase')=='ready':return None,peak
        reason=None
        if time.time()>=series_deadline:reason='series_budget_exhausted'
        elif stop_file.exists():reason='interrupted'
        elif peak>limits['rss_mib']:reason='external_memory_limit'
        elif time.monotonic()>=deadline:reason='external_time_limit'
        elif state and time.monotonic()-state.get('monotonic',0)>15:reason='telemetry_stale'
        elif child.poll() is not None:reason='interrupted' if child.stop_reason in CONTROL_STOPS else 'worker_exception'
        if reason:return reason,peak
        time.sleep(limits['poll_seconds'])


def stop_child(child):
    if child is not None and child.poll() is None:
        child.terminate();child.wait(timeout=10.)


def run(out,smoke=False):
    plan=read_json(out/'plan.json');limits=plan['resources'];plan_sha=digest(out/'plan.json')
    for n,h in plan['protected_sha256'].items():
        if digest(ROOT/n)!=h:raise ValueError(f'protected source/data changed: {n}')
    if not smoke and not (out/'smoke/check.json').exists():raise ValueError('smoke must pass first')
    if not smoke and not read_json(out/'smoke/check.json')['passed']:raise ValueError('smoke failed')
    completed={} if smoke else collect(out,plan)
    if smoke:
        jobs=[(pid,0,plan['smoke']['seed']) for pid in plan['smoke']['ids']]
        if (out/'smoke/check.json').exists():raise FileExistsError('smoke already exists')
    else:jobs=[(pid,i,seed) for batch in plan['batches'] for pid in batch for i,seed in enumerate(plan['seeds']) if key(pid,i) not in completed]
    with controller_locks(out) as fds:
        if not smoke and not (out/'budget.json').exists():
            start=time.time();write_json(out/'budget.json',dict(started_epoch=start,deadline_epoch=start+14400.,started_utc=utc()),exclusive=True)
        series_deadline=time.time()+180. if smoke else read_json(out/'budget.json')['deadline_epoch']
        session=('smoke_' if smoke else 'mass_')+uuid.uuid4().hex[:10]
        active=new_control(out,session,lease_seconds=limits['owner_lease_seconds']);control=Path(active['control_dir'])
        session_data=dict(session=session,smoke=smoke,started_utc=utc(),planned=len(jobs),new_terminal=0,reason=None,startups=[])
        began=time.perf_counter();child=None;log=None;state_path=None;reason=None
        def progress():
            terminal=completed if not smoke else {p.stem:read_json(p) for p in (out/'smoke').glob('*_seed*.json')}
            write_json(out/('smoke/progress.json' if smoke else 'progress.json'),dict(session=session,updated_utc=utc(),
                terminal=len(terminal),pending=(len(jobs)-len(terminal) if smoke else 4000-len(terminal)),
                outcomes=dict(Counter(r['status'] for r in terminal.values())),
                remaining_seconds=max(0.,series_deadline-time.time()),session_elapsed_seconds=time.perf_counter()-began))
        old={s:signal.signal(s,lambda *_:write_json(control/'stop.json',dict(requested_utc=utc()))) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
        try:
            progress()
            for pid,i,seed in jobs:
                if time.time()>=series_deadline:reason='series_budget_exhausted';break
                if (control/'stop.json').exists():reason='interrupted';break
                if child is None or child.poll() is not None:
                    if log is not None:log.close()
                    token=session+'_'+uuid.uuid4().hex[:6];state_path=out/'workers'/f'{token}.json'
                    log=(out/'workers'/f'{token}.log').open('x')
                    child=GuardedProcess([sys.executable,'-B',str(Path(__file__).resolve()),'worker','--out',str(out),
                        '--state',str(state_path),'--parent-pid','{GUARD_PID}'],log,control/f'{token}.guardian.json',active,fds,heartbeat_timeout=5.)
                    startup,peak=wait(child,None,state_path,time.monotonic()+limits['startup_seconds'],limits,series_deadline,control/'stop.json')
                    session_data['startups'].append(dict(status=startup or 'ready',state=str(state_path.relative_to(out)),peak_rss_mib=peak))
                    if startup:reason='startup_'+startup;break
                k=key(pid,i);result_path=out/('smoke' if smoke else 'attempts')/(k+'.json')
                spec=next(r for r in plan['preflight'] if r['problem_id']==pid)
                request=dict(problem_id=pid,seed_index=i,key=k,session=session,plan_sha256=plan_sha,
                    config=dict(plan['config'],seed=seed),expression_hash=spec['expression_hash'],target_x=[0.],result_path=str(result_path.resolve()))
                req=out/'events'/f'{session}_{k}.request.json';write_json(req,request,exclusive=True)
                dispatched=time.perf_counter();child.stdin.write(json.dumps(dict(request_path=str(req.resolve())))+'\n');child.stdin.flush()
                outcome,peak=wait(child,result_path,state_path,time.monotonic()+60.,limits,series_deadline,control/'stop.json')
                if outcome:
                    stop_child(child)
                    # Preserve an already durable result if it completed during shutdown.
                    if result_path.exists():outcome=None
                if outcome in ('interrupted','series_budget_exhausted','telemetry_stale'):
                    write_json(out/'events'/f'{session}_{k}.interrupted.json',dict(reason=outcome,wall_seconds=time.perf_counter()-dispatched,peak_rss_mib=peak),exclusive=True)
                    reason=outcome;break
                if outcome:
                    r={n:request[n] for n in ('problem_id','seed_index','key','session','plan_sha256','config','expression_hash','target_x')}
                    r.update(implementation=plan['implementation'],status=outcome,result=None,abs_error=None,y_exact=None,
                        accuracy_met=None,task_wall_seconds=time.perf_counter()-dispatched,process_peak_rss_mib=peak,finished_utc=utc())
                    write_json(result_path,r,exclusive=True)
                r=read_json(result_path)
                if not smoke:validate_record(r,plan,plan_sha);completed[k]=r
                session_data['new_terminal']+=1;progress()
                if smoke or session_data['new_terminal']%20==0:
                    print(json.dumps(dict(terminal=len(completed),new=session_data['new_terminal'],id=pid,status=r['status'],elapsed_seconds=time.perf_counter()-began)),flush=True)
                if not smoke:
                    batch_index=plan['equation_ids'].index(pid)//20
                    batch_ids=plan['batches'][batch_index]
                    if all(key(p,j) in completed for p in batch_ids for j in range(4)):
                        bp=out/'batches'/f'batch{batch_index:02d}.json'
                        if not bp.exists():write_json(bp,dict(batch=batch_index,ids=batch_ids,attempts=80,finished_utc=utc()),exclusive=True)
                if child.poll() is not None:child=None
            if smoke:
                records=[read_json(p) for p in (out/'smoke').glob('*_seed*.json')]
                passed=len(records)==2 and all(r['status']=='completed' and r['result']['tail_tolerance_met'] and r['result']['backend']=='numba' for r in records)
                if not passed:reason=reason or 'smoke_failed'
                write_json(out/'smoke/check.json',dict(passed=passed,reason=reason,count=len(records),plan_sha256=plan_sha),exclusive=True)
        finally:
            stop_child(child)
            if log is not None:log.close()
            for sig,handler in old.items():signal.signal(sig,handler)
            session_data.update(reason=reason,finished_utc=utc(),wall_seconds=time.perf_counter()-began)
            write_json(out/'sessions'/f'{session}.json',session_data,exclusive=True);progress();finish_control(out,active)
            if not smoke:
                pending=[key(pid,i) for pid in plan['equation_ids'] for i in range(4) if key(pid,i) not in completed]
                write_json(out/'pending.json',dict(status='pending',keys=pending,reason=reason,deadline_epoch=series_deadline))
        print(json.dumps(session_data),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=('init','smoke','run','worker'));ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--state',type=Path);ap.add_argument('--parent-pid',type=int)
    a=ap.parse_args();out=a.out.resolve()
    if a.action=='init':initialize(out)
    elif a.action=='worker':worker(out,a.state,a.parent_pid)
    else:run(out,a.action=='smoke')
