"""Execute ONLY the 140 frozen MC1 P2 jobs; no R, benchmark or ML.

Numerical methods are imported unchanged. A separately guarded worker warms
Numba once, then times AST/G preparation + config + the full MC call. Exact
scoring and persistence happen after the measured interval.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import sys
import threading
import time
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
         'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS')
for name in THREADS:os.environ[name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from diagnostics.run_chernov_batch import read_json,write_json,digest
from diagnostics.chernov_batch_control import (
    controller_locks,new_control,finish_control,GuardedProcess,lock_is_held)
from diagnostics.mc_production_x0 import wait,stop_child,rss
from diagnostics.joint_benchmark_pilot import thermal

CONTRACT=ROOT/'reports/ml_selector_contract'
AUTHORIZATION=ROOT/'diagnostics/p2_authorization_2026-10-26.txt'
P1=CONTRACT/'sentinel_p1_2026-10-26'
NUMERICAL=('mc_solver.py','mc_optimized.py','mc_ast_kernels.py','remizov_solver.py',
           'all_methods_bounds.py','all_methods_dataset_loader.py',
           'src/all_methods_expressions.py','src/all_methods_dataset_v1.py')


def utc():return datetime.now(timezone.utc).isoformat()


def scope_jobs(contract,jobs):
    selected=[j for j in jobs if j['stage']==2 and j['profile']=='MC1']
    import csv
    manifest=list(csv.DictReader((CONTRACT/'pilot_manifest.csv').open()))
    ids=[r['equation_id'] for r in manifest if r['equation_id'] not in contract['pilot']['sentinels']]
    assert len(ids)==len(set(ids))==35 and len(selected)==140
    assert len({j['key'] for j in selected})==140
    assert Counter(j['equation_id'] for j in selected)=={p:4 for p in ids}
    assert len({j['family_id'] for j in selected})==35
    for j in selected:
        assert j['split']=='train' and j['stage']==2 and j['profile']=='MC1'
        assert j['target_x']==[-1+i/16 for i in range(33)]
        cfg=j['config'];episode=j['episode'];assert episode in range(4)
        assert cfg['seed']==248800020100+episode
        assert cfg['M']==4096 and cfg['dt']==1/128
        assert cfg['chunk']==cfg['max_batch_paths']==1024 and cfg['T'] is None
        assert cfg['tail_eps']==1e-4 and cfg['stream_id']==0
    index={(j['equation_id'],j['episode']):j for j in selected}
    return [index[(pid,episode)] for pid in ids for episode in range(4)]


def snapshot(out):
    result={}
    for p in ROOT.rglob('*'):
        if not p.is_file() or out in p.parents or any(part in
            ('.venv','.git','__pycache__','.pytest_cache','.DS_Store') for part in p.parts):continue
        result[str(p.relative_to(ROOT))]=digest(p)
    return result


def measured_call(metadata,job,prepare,config_factory,solve,clock=time.perf_counter_ns):
    """No exact, metric evaluation or result serialization inside this timer."""
    start=clock()
    try:
        prepared=prepare(metadata)
        config=config_factory(**job['config'])
        result=solve(prepared,job['target_x'],config,engine='numba',block_steps=64)
    except Exception as exc:
        end=clock();exc.p1_timer=(start,end)
        raise
    end=clock()
    return prepared,result,(end-start)*1e-9,start,end


def initialize(out):
    import numpy as np
    from all_methods_dataset_loader import load_all_methods_equations
    from mc_optimized import prepare_all_methods_mc
    from mc_solver import MCConfig,auto_T
    contract=read_json(CONTRACT/'plan.json')
    original=[json.loads(s) for s in (CONTRACT/'pilot_jobs.jsonl').read_text().splitlines()]
    jobs=scope_jobs(contract,original)
    for name,h in read_json(P1/'artifact_sha256.json').items():
        assert digest(P1/name)==h, 'P1 artifact changed: '+name
    previous=read_json(P1/'analysis/summary.json')
    assert previous['decision']['selected']=='MC1' and previous['completed']==60
    assert previous['decision']['profiles']['MC1']['eligible']
    p1_ids=set(contract['pilot']['sentinels'])
    assert not p1_ids.intersection(j['equation_id'] for j in jobs)
    verification=read_json(CONTRACT/'verification.json')
    # The status file was intentionally updated after planning. This exception
    # is already recorded in the contract verification; numerical pins are exact.
    for name,h in verification['artifact_sha256'].items():
        assert digest(CONTRACT/name)==h,name
    sources=read_json(CONTRACT/'sources_sha256.json')
    prior_snapshot=read_json(CONTRACT/'protected_before.json')
    for name in NUMERICAL:
        expected=sources[name] if name in sources else prior_snapshot[name]
        assert digest(ROOT/name)==expected,name
    data=ROOT/contract['data_dir']
    for name,h in contract['dataset_sha256'].items():assert digest(data/name)==h,name
    eqs={e.equation_id:e for e in load_all_methods_equations(
        data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl')}
    checks=[]
    for pid in dict.fromkeys(j['equation_id'] for j in jobs):
        e=eqs[pid];metadata=e.metadata;prepared=prepare_all_methods_mc(metadata)
        own=[j for j in jobs if j['equation_id']==pid]
        assert metadata['split']=='train'
        exact=np.asarray(e.y_exact(np.asarray(contract['target_x'])))
        assert exact.shape==(33,) and np.isfinite(exact).all()
        p=prepared.problem;G=prepared.forcing_bound.abs_upper_bound
        T=auto_T(p.lambda_value,p.c_abs_upper_bound,G,1e-4)
        for j in own:
            MCConfig(**j['config']).validate()
            assert j['expression_hash']==metadata['expression_hash']
            assert j['family_id']==metadata['family_id']
            assert j['expected_T']==T and j['forcing_G']==G
            n=math.ceil(T/j['config']['dt'])
            if T/n>j['config']['dt']:n+=1
            assert n==j['expected_steps']
            assert j['numerical_work']==j['config']['max_path_steps']==33*j['config']['M']*n
        checks.append(dict(equation_id=pid,expression_hash=metadata['expression_hash'],
            family_id=metadata['family_id'],split='train',reference_valid=True,T=T,G=G))
    assert len({r['family_id'] for r in checks})==35
    for name in ('.chernov_batch_controller.lock','.chernov_batch_compute.lock'):
        assert not lock_is_held(ROOT/'results'/name),'another managed compute is active'
    sensor=thermal()
    assert sensor['thermal_state'] is None or sensor['thermal_state']<2
    disk=shutil.disk_usage(ROOT);assert disk.free>=256*1024**2
    out.mkdir(parents=True,exist_ok=False)
    for name in ('records','events','workers','sessions','sources'):(out/name).mkdir()
    shutil.copy2(ROOT/'PROJECT_STATUS.md',out/'PROJECT_STATUS.before.md')
    shutil.copy2(AUTHORIZATION,out/'AUTHORIZATION.txt')
    write_json(out/'protected_before.json',snapshot(out),exclusive=True)
    pins=list(NUMERICAL)+['diagnostics/mc_pilot_p2.py','diagnostics/mc_sentinel_p1.py','diagnostics/p2_authorization_2026-10-26.txt',
        'diagnostics/chernov_batch_control.py','diagnostics/mc_production_x0.py',
        'diagnostics/joint_benchmark_pilot.py','diagnostics/run_chernov_batch.py']
    hashes={n:digest(ROOT/n) for n in pins}
    hashes.update({str((data/n).relative_to(ROOT)):h for n,h in contract['dataset_sha256'].items()})
    hashes.update({str(p.relative_to(ROOT)):digest(p) for p in CONTRACT.iterdir() if p.is_file()})
    for name in pins:
        dest=out/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    plan=dict(schema='MC_P2_execution_v1',created_utc=utc(),
        authorization='Current user request explicitly authorizes ONLY140 MC1 stage2 calls; no R0/benchmark/ML',
        authorization_sha256=digest(AUTHORIZATION),execution_authorized=True,
        source_contract_sha256=digest(CONTRACT/'plan.json'),source_jobs_sha256=digest(CONTRACT/'pilot_jobs.jsonl'),
        data_dir=contract['data_dir'],target_x=contract['target_x'],threshold=.01,jobs=jobs,
        sentinels=contract['pilot']['sentinels'],gates=contract['gates'],preflight=checks,
        protected_sha256=hashes,total_new_calls=140,total_path_updates=sum(j['numerical_work'] for j in jobs),
        timing=dict(clock='perf_counter_ns',scope='prepare_all_methods_mc + MCConfig + solve_monte_carlo_optimized',
            excludes=['shared_loading','JIT_warmup','exact_scoring','identity_validation_after_return','serialization']),
        resources=dict(compute_workers=1,rss_mib=512.,series_seconds=18599.,owner_lease_seconds=300.,
            startup_seconds=90.,poll_seconds=.05,monitor_seconds=.25,thermal_seconds=10.,thermal_stop_at=2),
        order='remaining manifest order;episode0..3;MC1 only',
        retry_policy='no retries within140 physical requests;resume can dispatch only never-requested keys',
        scenario_solver_seconds=9149.472926866389,
        no_R=True,no_benchmark1000=True,no_ML=True,
        scope_note='P2 user-authorized; cap ceil(2*9149.472926866389+300)=18599, calibrated from P1; estimate not guarantee')
    write_json(out/'plan.json',plan,exclusive=True)
    write_json(out/'preflight.json',dict(passed=True,checks=checks,train_only=True,unique_families=35,
        MC_calls=140,exact_grid_points=33,disk_free_bytes=disk.free,initial_thermal=sensor,
        numeric_contract_discrepancies=[],authorization_overrides_inert_planned_flags=True),exclusive=True)
    write_json(out/'environment.json',dict(platform=platform.platform(),machine=platform.machine(),
        python=sys.version,executable=sys.executable,cpu_count=os.cpu_count(),
        versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','sympy','numba','llvmlite','pytest')},
        thread_environment={n:os.environ[n] for n in THREADS},thermal=sensor),exclusive=True)
    print(json.dumps(dict(preflight='passed',jobs=140,ids=list(dict.fromkeys(j['equation_id'] for j in jobs)),work=plan['total_path_updates'])),flush=True)


def validate_result(job,prepared,z):
    import numpy as np
    assert z.problem_id==job['equation_id'] and z.backend=='numba'
    assert z.compilation_seconds==0. and z.numba_import_seconds==0.,'unexpected in-call compilation'
    assert z.T==job['expected_T'] and prepared.forcing_bound.abs_upper_bound==job['forcing_G']
    assert z.seed_entropy==job['config']['seed'] and z.rng_scheme=='PCG64/SeedSequence/point_address_v1'
    np.testing.assert_array_equal(z.x,job['target_x'])
    if z.success:
        assert z.failure_reason=='' and len(z.f)==33 and np.isfinite(z.f).all()
        assert z.completed_paths.tolist()==[job['config']['M']]*33
        assert z.n_path_steps==z.n_normal_draws==job['numerical_work']
        assert z.n_steps==job['expected_steps'] and z.tail_tolerance_met


def worker(out,state_path,parent_pid):
    import numpy as np
    from all_methods_dataset_loader import load_all_methods_equations
    from mc_optimized import prepare_all_methods_mc,solve_monte_carlo_optimized,compile_backend
    from mc_solver import MCConfig
    plan=read_json(out/'plan.json');stop=threading.Event()
    state=dict(pid=os.getpid(),phase='loading',key=None)
    def monitor():
        with state_path.with_suffix('.rss.jsonl').open('x') as stream:
            while not stop.is_set():
                if os.getppid()!=parent_pid:os._exit(75)
                sample=dict(state,monotonic=time.monotonic(),peak_rss_mib=rss())
                write_json(state_path,sample,durable=False);stream.write(json.dumps(sample)+'\n');stream.flush()
                stop.wait(.25)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        began=time.perf_counter();data=ROOT/plan['data_dir']
        eqs={e.equation_id:e for e in load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl')}
        loading=time.perf_counter()-began
        warm_started=time.perf_counter_ns();jit=compile_backend();warm_wall=(time.perf_counter_ns()-warm_started)*1e-9
        import numba
        write_json(state_path.with_suffix('.warmup.json'),dict(jit=jit,warmup_wall_seconds=warm_wall,
            numba_threads=numba.get_num_threads(),shared_load_seconds=loading,no_MC_calls_in_warmup=True),exclusive=True)
        state.update(phase='ready',load_seconds=loading,jit=jit)
        write_json(state_path,dict(state,monotonic=time.monotonic(),peak_rss_mib=rss()),durable=False)
        lookup={j['key']:j for j in plan['jobs']}
        for line in sys.stdin:
            request=read_json(Path(json.loads(line)['request_path']));job=request['job']
            assert job==lookup[job['key']]
            e=eqs[job['equation_id']];m=e.metadata
            assert m['split']=='train' and m['expression_hash']==job['expression_hash']
            state.update(phase='solving',key=job['key'])
            r=dict(job=job,equation_id=e.equation_id,expression_hash=m['expression_hash'],family_id=m['family_id'],
                split=m['split'],profile=job['profile'],config=job['config'],seed=job['config']['seed'],
                plan_sha256=request['plan_sha256'],session=request['session'],worker_pid=os.getpid(),
                started_utc=utc(),shared_load_seconds=loading,shared_compilation=jit,solver_called=False)
            started=time.perf_counter_ns()
            def counted_solve(*args,**kwargs):
                r['solver_called']=True
                return solve_monte_carlo_optimized(*args,**kwargs)
            try:
                prepared,z,wall,t0,t1=measured_call(m,job,prepare_all_methods_mc,MCConfig,counted_solve)
                r.update(solver_called=True,method_wall_seconds=wall,timer_start_ns=t0,timer_stop_ns=t1,
                    result=asdict(z),status='completed' if z.success else 'solver_failure')
                validate_result(job,prepared,z)
                # No exact access for parameter choice or during timing.
                score_start=time.perf_counter_ns();exact=np.asarray(e.y_exact(np.asarray(plan['target_x'])))
                error=float(np.max(abs(z.f-exact))) if z.success else None
                r.update(y_exact=exact,E_grid=error,accuracy_met=None if error is None else error<=.01,
                    scoring_seconds=(time.perf_counter_ns()-score_start)*1e-9,
                    accuracy_status='unknown_error' if error is None else ('observed_pass' if error<=.01 else 'observed_fail'))
            except AssertionError as exc:
                r.update(status='invalid_result',traceback=traceback.format_exc(),accuracy_met=None,E_grid=None)
                if hasattr(exc,'p1_timer'):
                    t0,t1=exc.p1_timer;r.update(method_wall_seconds=(t1-t0)*1e-9,timer_start_ns=t0,timer_stop_ns=t1)
            except Exception as exc:
                r.update(status='worker_exception',traceback=traceback.format_exc(),accuracy_met=None,E_grid=None)
                if hasattr(exc,'p1_timer'):
                    t0,t1=exc.p1_timer;r.update(method_wall_seconds=(t1-t0)*1e-9,timer_start_ns=t0,timer_stop_ns=t1)
            if 'method_wall_seconds' not in r:r['method_wall_seconds']=None
            r.update(finished_utc=utc(),peak_rss_mib=rss())
            write_json(out/'records'/f"{job['key']}.json",r,exclusive=True)
            state.update(phase='ready',key=None)
    finally:stop.set();thread.join()


def run(out):
    plan=read_json(out/'plan.json');ph=digest(out/'plan.json');jobs=plan['jobs'];limits=plan['resources']
    assert plan['execution_authorized'] and len(jobs)==140 and all(j['stage']==2 and j['profile']=='MC1' for j in jobs)
    for name,h in plan['protected_sha256'].items():assert digest(ROOT/name)==h,name
    done={p.stem:read_json(p) for p in (out/'records').glob('*.json')}
    assert set(done)<=set(j['key'] for j in jobs)
    if len(done)==140:
        print(json.dumps(dict(new_calls=0,worker_started=False,reason='already_complete')),flush=True);return
    # Do not retry an interrupted physical invocation under a140-call ceiling.
    dispatched={read_json(p)['job']['key'] for p in (out/'events').glob('*.request.json')}
    pending=[j for j in jobs if j['key'] not in done and j['key'] not in dispatched]
    with controller_locks(out) as fds:
        if not (out/'budget.json').exists():
            start=time.time();write_json(out/'budget.json',dict(started_epoch=start,deadline_epoch=start+limits['series_seconds'],started_utc=utc()),exclusive=True)
        deadline=read_json(out/'budget.json')['deadline_epoch'];session='p2_'+uuid.uuid4().hex[:10]
        active=new_control(out,session,lease_seconds=300.);control=Path(active['control_dir'])
        began=time.perf_counter();child=None;log=None;state_path=None;reason=None;new=0;last_thermal=0.;sensor={}
        def checkpoint():
            write_json(out/'progress.json',dict(session=session,terminal=len(done),pending=140-len(done),
                statuses=dict(Counter(r['status'] for r in done.values())),new_terminal=new,
                session_wall_seconds=time.perf_counter()-began,remaining_seconds=max(0.,deadline-time.time()),thermal=sensor))
        def wait_thermal(path,attempt_deadline):
            nonlocal last_thermal,sensor
            peak=0.
            while True:
                if time.monotonic()-last_thermal>=10.:
                    sensor=thermal();last_thermal=time.monotonic()
                    with (out/'thermal.jsonl').open('a') as stream:stream.write(json.dumps(dict(utc=utc(),**sensor))+'\n')
                    if sensor['thermal_state'] is not None and sensor['thermal_state']>=2:
                        return 'thermal_state_serious_or_critical',peak
                # Existing wait maintains guardian heartbeats and resource checks.
                short=min(attempt_deadline,time.monotonic()+.5)
                result,used=wait(child,path,state_path,short,limits,deadline,control/'stop.json');peak=max(peak,used)
                if result=='external_time_limit' and short<attempt_deadline:continue
                return result,peak
        old={s:signal.signal(s,lambda *_:write_json(control/'stop.json',dict(requested_utc=utc()))) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
        try:
            checkpoint()
            for job in pending:
                if time.time()>=deadline:reason='series_budget_exhausted';break
                if (control/'stop.json').exists():reason='interrupted';break
                if child is None or child.poll() is not None:
                    if log:log.close()
                    token=session+'_'+uuid.uuid4().hex[:6];state_path=out/'workers'/f'{token}.json';log=(out/'workers'/f'{token}.log').open('x')
                    child=GuardedProcess([sys.executable,'-B',str(Path(__file__).resolve()),'worker','--out',str(out),
                        '--state',str(state_path),'--parent-pid','{GUARD_PID}'],log,control/f'{token}.guardian.json',active,fds)
                    startup,peak=wait_thermal(None,time.monotonic()+90.)
                    if startup:reason='startup_'+startup;break
                path=out/'records'/f"{job['key']}.json";request_path=out/'events'/f"{session}_{job['key']}.request.json"
                write_json(request_path,dict(job=job,session=session,plan_sha256=ph),exclusive=True);dispatched.add(job['key'])
                assert len(dispatched)<=140
                clock=time.perf_counter();child.stdin.write(json.dumps(dict(request_path=str(request_path.resolve())))+'\n');child.stdin.flush()
                outcome,peak=wait_thermal(path,time.monotonic()+job['proposed_watchdog_seconds'])
                if outcome:
                    stop_child(child)
                    if path.exists():outcome=None
                if outcome in ('interrupted','series_budget_exhausted','telemetry_stale','thermal_state_serious_or_critical'):
                    reason=outcome
                    write_json(out/'events'/f"{session}_{job['key']}.interrupted.json",dict(reason=reason,wall_seconds=time.perf_counter()-clock,
                        peak_rss_mib=peak,last_worker_state=read_json(state_path) if state_path.exists() else {}),exclusive=True)
                    break
                if outcome:
                    write_json(path,dict(job=job,equation_id=job['equation_id'],profile=job['profile'],expression_hash=job['expression_hash'],
                        family_id=job['family_id'],split='train',config=job['config'],seed=job['config']['seed'],
                        plan_sha256=ph,status=outcome,result=None,E_grid=None,accuracy_met=None,accuracy_status='unknown_error',
                        method_wall_seconds=None,external_elapsed_seconds=time.perf_counter()-clock,
                        peak_rss_mib=peak,worker_exit_reason=child.stop_reason,finished_utc=utc()),exclusive=True)
                r=read_json(path);done[job['key']]=r;new+=1;checkpoint()
                print(json.dumps(dict(terminal=len(done),key=job['key'],status=r['status'],E_grid=r['E_grid'],
                    wall=r.get('method_wall_seconds'),session_seconds=time.perf_counter()-began)),flush=True)
                if r['status']=='invalid_result':reason='invalid_result';break
        finally:
            stop_child(child)
            if log:log.close()
            for s,handler in old.items():signal.signal(s,handler)
            write_json(out/'sessions'/f'{session}.json',dict(session=session,new_terminal=new,reason=reason,
                wall_seconds=time.perf_counter()-began,finished_utc=utc()),exclusive=True)
            checkpoint();finish_control(out,active)
            write_json(out/'pending.json',dict(keys=[j['key'] for j in jobs if j['key'] not in done],reason=reason))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=('init','run','worker'))
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--state',type=Path);ap.add_argument('--parent-pid',type=int)
    a=ap.parse_args();out=a.out.resolve()
    if a.action=='init':initialize(out)
    elif a.action=='run':run(out)
    else:worker(out,a.state,a.parent_pid)
