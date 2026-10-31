"""Execute the authorized small joint pilot, with a separately renewed owner lease.

No numerical scheme changes. Transparent observers count completed calls to the
original MC batch/Chernov node kernels. A guardian owns and reaps each worker
session. The foreground controller NEVER renews its own external owner lease.
No attempt/phase/session duration cutoff; only resource and liveness guards.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
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
import statistics
import subprocess
import sys
import threading
import time
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
THREAD_VARS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')
for key in THREAD_VARS:
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
PROPOSAL=ROOT/'results/joint_benchmark_plan_2026-10-25'
SOURCES=('remizov_solver.py','chernov_solver.py','mc_solver.py',
         'all_methods_dataset_loader.py','all_methods_bounds.py',
         'src/all_methods_expressions.py','src/all_methods_dataset_v1.py',
         'diagnostics/chernov_batch_control.py','diagnostics/joint_benchmark_pilot.py',
         'tests/test_joint_benchmark_pilot.py')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [clean(x) for x in v]
    if isinstance(v,float):return v if math.isfinite(v) else None
    if hasattr(v,'tolist'):return clean(v.tolist())
    if hasattr(v,'item'):return clean(v.item())
    return v


def write(path,value):
    from diagnostics.chernov_batch_control import atomic
    atomic(Path(path),clean(value))


def rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)


def cpu_seconds():
    r=resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime+r.ru_stime


def thermal():
    if sys.platform!='darwin':
        return dict(thermal_state=None,temperature_celsius=None,measurement='unavailable')
    code='ObjC.import("Foundation"); JSON.stringify({thermalState:Number($.NSProcessInfo.processInfo.thermalState)})'
    p=subprocess.run(['/usr/bin/osascript','-l','JavaScript','-e',code],
                     capture_output=True,text=True,timeout=4)
    if p.returncode:raise RuntimeError('thermal telemetry unavailable: '+p.stderr.strip())
    state=json.loads(p.stdout)['thermalState']
    if type(state) is not int or state not in range(4):raise ValueError('bad thermalState')
    return dict(thermal_state=state,temperature_celsius=None,
                measurement='NSProcessInfo.thermalState; not degrees Celsius')


def project_workers():
    p=subprocess.run(['/bin/ps','-axo','pid=,command='],capture_output=True,text=True,timeout=4)
    if p.returncode:raise RuntimeError('process inventory unavailable: '+p.stderr.strip())
    return [line.strip() for line in p.stdout.splitlines()
            if str(ROOT) in line and any(flag in line for flag in
                (' --worker-request ',' --worker-plan ',' --pilot-worker '))]


def safeguards(sample, elapsed, idle_cpu, idle_progress, stale, limits):
    """Duration alone is never a stop reason: active 10-minute calls may finish."""
    if sample.get('peak_rss_mib',0)>limits['rss_limit_mib']:return 'external_memory_limit'
    if stale>limits['telemetry_stale_seconds']:return 'worker_telemetry_stale'
    if idle_cpu>limits['cpu_idle_seconds']:return 'worker_cpu_stalled'
    if idle_progress>limits['progress_stall_seconds']:return 'worker_progress_stalled'
    return None


@contextmanager
def observe(method,state):
    """Pass original arguments/results through; no arithmetic/RNG/path change."""
    import mc_solver
    import chernov_solver
    if method=='monte_carlo':
        original=mc_solver._path_contributions
        def wrapped(*args,**kwargs):
            result=original(*args,**kwargs)
            run=args[-1] if args else kwargs['run']
            state['completed_units']=run.n_path_steps
            state['completed_batches']=run.n_batches
            return result
        mc_solver._path_contributions=wrapped
        try:yield
        finally:mc_solver._path_contributions=original
    else:
        original=chernov_solver._at_t
        def wrapped(*args,**kwargs):
            result=original(*args,**kwargs)
            state['completed_units']+=1
            return result
        chernov_solver._at_t=wrapped
        try:yield
        finally:chernov_solver._at_t=original


def _identity(eq,fingerprint):
    m=eq.metadata
    return dict(dataset_sha256=fingerprint,equation_id=eq.equation_id,
                expression_hash=m['expression_hash'],family_id=m['family_id'],
                structure_signature=m['structure_signature'],original_split=m['split'],
                complexity=m['complexity_level'],regime=m['regime'],
                policy_development_exposed=True,training_eligible=False)


def build_attempts(proposal,tasks):
    attempts=[]
    for phase in proposal['phases']:
        mode=phase['name']
        for rep in range(max(phase['mc_repeats'],phase['remizov_repeats'])):
            # Balanced cell rounds, then deterministic order within each level.
            ids=sorted(phase['ids'],key=lambda p:hashlib.sha256(f'{mode}|{rep}|{p}'.encode()).digest())
            for pid in ids:
                methods=['solve','monte_carlo'] if rep<phase['remizov_repeats'] else ['monte_carlo']
                if int(hashlib.sha256(f'{mode}|{rep}|{pid}|order'.encode()).hexdigest(),16)%2:
                    methods.reverse()
                for method in methods:
                    cfg=dict(proposal['base_configs']['remizov' if method=='solve' else 'monte_carlo'])
                    cfg['T']=tasks[pid]['T']
                    if method=='monte_carlo':
                        cfg.update(M=phase['M'],seed=phase['root_seeds'][rep],max_runtime_seconds=None)
                    attempts.append(dict(attempt_id=f'{len(attempts):03d}_{mode}_{pid}_{method}_r{rep:02d}',
                        mode=mode,problem_id=pid,method=method,repeat=rep,
                        target_x=phase['target_x'],config=cfg,identity=tasks[pid]['identity']))
    return attempts


def prepare(out):
    from all_methods_dataset_loader import load_all_methods_equations
    from all_methods_bounds import certify_all_methods_forcing
    from mc_solver import MCConfig,auto_T,_tail_bound,_gap_lower
    from chernov_solver import ChernovConfig
    from diagnostics.plan_joint_benchmark import dataset_digest,SMALL,protected_snapshot,mc_cost
    proposal=read(PROPOSAL/'proposal.json')
    if (len(proposal['phases'][0]['ids']),len(proposal['phases'][1]['ids']))!=(20,6):
        raise ValueError('only the authorized 20+6 pilot may run')
    fingerprint=dataset_digest(SMALL)
    equations=load_all_methods_equations(ROOT/SMALL/'all_methods_equations_v1.csv',
                                       ROOT/SMALL/'all_methods_equations_v1_metadata.jsonl')
    if len(equations)!=20:raise ValueError('small20 dataset required')
    saved={r['problem_id']:r for r in read(ROOT/'results/forcing_bounds_2026-10-24/survey/certificates.json')}
    tasks={}
    for eq in equations:
        p=eq.remizov_problem
        proof=certify_all_methods_forcing(p,eq.metadata)
        G=proof.forcing_bound.abs_upper_bound
        Tstar=auto_T(p.lambda_value,p.c_abs_upper_bound,G,1e-4)
        T=math.ceil(Tstar*8)/8
        tail=_tail_bound(G,_gap_lower(p.lambda_value,p.c_abs_upper_bound),T)
        if G!=saved[eq.equation_id]['G'] or T!=math.ceil(saved[eq.equation_id]['T']*8)/8 or tail>1e-4:
            raise ValueError('certificate does not match frozen proposal')
        tasks[eq.equation_id]=dict(T=T,certified_T=Tstar,tail_error_bound=tail,
            certificate=clean(asdict(proof)),identity=_identity(eq,fingerprint))
    attempts=build_attempts(proposal,tasks)
    work=0
    for a in attempts:
        if a['method']=='solve':
            ChernovConfig(**a['config']).validate()
            a['work']=dict(operator_applications=128*512,quadrature_nodes=513,work_grid_points=2081)
        else:
            MCConfig(**a['config']).validate()
            a['work']=mc_cost(a['config']['T'],a['config']['M'],len(a['target_x']))
            work+=a['work']['path_steps']
            if a['work']['path_steps']>a['config']['max_path_steps']:raise ValueError('MC work cap too small')
    if len(attempts)!=262 or work!=4450484224:raise ValueError('pilot scope/work changed')
    sensor=thermal()
    if sensor['thermal_state'] is not None and sensor['thermal_state']>=2:
        raise RuntimeError('unsafe starting thermal state')
    out.mkdir(parents=True,exist_ok=False)
    (out/'trials').mkdir();(out/'sources').mkdir()
    shutil.copy2(ROOT/'PROJECT_STATUS.md',out/'PROJECT_STATUS.before.md')
    shutil.copy2(PROPOSAL/'proposal.json',out/'proposal_snapshot.json')
    for name in SOURCES:
        if name.startswith(('diagnostics/','tests/')):
            shutil.copy2(ROOT/name,out/'sources'/Path(name).name)
    write(out/'protected_before.json',protected_snapshot(out))
    plan=dict(schema_version='joint_pilot_execution_v1',frozen_utc=now(),
        authorization='User explicitly requested numerical completion of the 20+6 pilot; removes 60s/90min cutoffs',
        proposal_sha256=digest(PROPOSAL/'proposal.json'),execution_authorized=True,
        scope='small20 point + selected6 grid33, no production1000 solves or ML',
        epsilon=.01,reliability_target=.95,confidence=.95,practical_time_margin=.10,
        tasks=tasks,attempts=attempts,source_sha256={n:digest(ROOT/n) for n in SOURCES},
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'data/generated').glob('all_methods_equations_v1*')},
        resources=dict(workers=1,blas_threads=1,rss_limit_mib=512,
            attempt_wall_seconds=None,phase_wall_seconds=None,total_wall_seconds=None,
            telemetry_stale_seconds=30,cpu_idle_seconds=120,progress_stall_seconds=600,
            guardian_controller_heartbeat_seconds=15,external_owner_lease_seconds=120,
            resource_sample_seconds=1,thermal_sample_seconds=10,
            thermal_stop_at=2,minimum_free_disk_bytes=256*1024**2,
            automatic_retries=0,stop_on_safety_condition=True),
        maximum_work=dict(attempts=262,mc_path_steps=work,remizov_operator_applications=78*128*512),
        watchdog_rationale='No duration cutoff. Progress means a completed original MC chunk or Chernov node. '
            '600s without a completed work unit is >150x the measured 3.858s longest-horizon F0002 chunk; '
            'all ASTs are bounded finite v1 expressions with fixed chunk<=1024,N<=8144. '
            'Telemetry/CPU stalls indicate loss of responsiveness, not slow but advancing computation. '
            '512MiB is >2.8x the prior 179.5MiB sequential peak; one worker. '
            'Guardian and external lease enforce cleanup after controller/operator loss.',
        scoring='Exact after prediction+hash+ready written, never used in parameter choice or scheduling',
        statistic='One independent production answer per seed; no pooling as a free answer; point/grid distinct policies')
    write(out/'plan.json',plan)
    write(out/'environment.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
        versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','sympy','matplotlib','pytest')},
        thread_environment={n:os.environ[n] for n in THREAD_VARS},initial_thermal=sensor,
        free_disk_bytes=shutil.disk_usage(out).free))
    write(out/'summary.json',dict(plan_sha256=digest(out/'plan.json'),status='prepared',
        runs=[dict(attempt_id=a['attempt_id'],status='not_run',reason='not_started') for a in attempts],sessions=[]))
    print(json.dumps(dict(prepared=True,attempts=len(attempts),mc_path_steps=work,
                         cutoff_seconds=None,output=str(out))),flush=True)


def worker(request):
    trial=request.parent
    spec=read(request);out=trial.parents[1];plan=read(out/'plan.json')
    state=dict(stage='imports',completed_units=0,total_units=0,completed_batches=0)
    stop=threading.Event()
    def sample():
        with (trial/'worker.resources.jsonl').open('x') as log:
            while not stop.is_set():
                value=dict(monotonic=time.monotonic(),cpu_seconds=cpu_seconds(),peak_rss_mib=rss_mib(),**state)
                log.write(json.dumps(value)+'\n');log.flush()
                write(trial/'telemetry.json',value)
                stop.wait(1)
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    try:
        _calculate(spec,trial,plan,state)
    except BaseException as exc:
        write(trial/'exception.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise
    finally:
        stop.set();thread.join()
        write(trial/'worker.final.json',dict(peak_rss_mib=rss_mib(),cpu_seconds=cpu_seconds(),
                                           monotonic=time.monotonic(),**state))


def _calculate(spec,trial,plan,state):
    # Same heavy imports for both arms, included in cold response time.
    import numpy as np
    from all_methods_dataset_loader import load_all_methods_equation
    from all_methods_bounds import certify_all_methods_forcing
    from mc_solver import MCConfig,solve_monte_carlo,_tail_bound,_gap_lower
    from chernov_solver import ChernovConfig,ChernovSolver
    state['stage']='load'
    t=time.perf_counter()
    eq=load_all_methods_equation(spec['problem_id'],
        ROOT/'data/generated/all_methods_equations_v1.csv',
        ROOT/'data/generated/all_methods_equations_v1_metadata.jsonl')
    load_seconds=time.perf_counter()-t
    p=eq.remizov_problem
    state['stage']='certificate';t=time.perf_counter()
    proof=certify_all_methods_forcing(p,eq.metadata)
    task=plan['tasks'][p.problem_id]
    if clean(asdict(proof))!=task['certificate']:raise RuntimeError('forcing proof changed')
    T=spec['config']['T']
    tail=_tail_bound(proof.forcing_bound.abs_upper_bound,_gap_lower(p.lambda_value,p.c_abs_upper_bound),T)
    if tail>1e-4 or T!=task['T']:raise RuntimeError('uncontrolled or changed common horizon')
    cert_seconds=time.perf_counter()-t
    x=np.asarray(spec['target_x'],dtype=float)
    state.update(stage='solve',completed_units=0,
        total_units=spec['work']['path_steps'] if spec['method']=='monte_carlo' else spec['work']['quadrature_nodes'])
    t=time.perf_counter()
    with observe(spec['method'],state):
        if spec['method']=='monte_carlo':
            cfg=MCConfig(**spec['config'])
            result=solve_monte_carlo(p,x,cfg,forcing_bound=proof.forcing_bound)
        else:
            cfg=ChernovConfig(**spec['config'])
            result=ChernovSolver(p,cfg).solve(x)
    solve_seconds=time.perf_counter()-t
    state['stage']='prediction'
    prediction=dict(spec=spec,config=asdict(cfg),result=asdict(result),identity=spec['identity'],
        load_seconds=load_seconds,certification_seconds=cert_seconds,solver_call_seconds=solve_seconds,
        common_tail_bound=tail,common_forcing_certificate=asdict(proof),total_error_certified=False)
    write(trial/'prediction.json',prediction)
    frozen=dict(prediction_sha256=digest(trial/'prediction.json'),prediction_frozen_utc=now(),
                worker_response_seconds=time.monotonic()-spec['launch_monotonic'],
                cpu_seconds_at_prediction=cpu_seconds(),peak_rss_mib_at_prediction=rss_mib())
    write(trial/'ready.json',frozen)
    # Exact is deliberately first evaluated for scoring after the frozen answer.
    state['stage']='score';t=time.perf_counter()
    y=np.asarray(eq.y_exact(x),dtype=float)
    complete=bool(result.success and result.f.shape==x.shape and np.all(np.isfinite(result.f)))
    err=result.f-y if complete else None
    E=float(np.max(np.abs(err))) if complete else None
    amplitude=float(np.max(np.abs(y)))
    score=dict(prediction_sha256=frozen['prediction_sha256'],scored_utc=now(),y_exact=y,
        error_vector=err,max_abs_error=E,relative_error=None if E is None or amplitude==0 else E/amplitude,
        exact_amplitude=amplitude,finite_complete=complete,
        observed_accuracy_pass=bool(complete and E<=plan['epsilon']),
        target_mode=spec['mode'],metric='E0' if x.size==1 else 'Einf_grid33')
    if spec['method']=='monte_carlo' and complete:
        z=statistics.NormalDist().inv_cdf(1-.05/(2*x.size))
        half=z*result.stderr
        score.update(statistical_grid_z=z,simultaneous_sampling_halfwidth=half,
            simultaneous_sampling_lower=result.f-half,simultaneous_sampling_upper=result.f+half,
            interval_kind='approximate normal for discrete truncated mean; excludes bias and tail')
    score['scoring_seconds']=time.perf_counter()-t
    write(trial/'score.json',score)
    state['stage']='finished'


def _safe_read(path,default=None):
    try:return read(path)
    except (FileNotFoundError,json.JSONDecodeError):return default


def _check_sources(plan):
    changed=[n for group in ('source_sha256','input_sha256') for n,h in plan[group].items() if digest(ROOT/n)!=h]
    if changed:raise RuntimeError('frozen execution/numerical source changed: '+str(changed))


def run(out):
    from diagnostics.chernov_batch_control import (controller_locks,new_control,
        finish_control,GuardedProcess,lock_is_held)
    plan=read(out/'plan.json');_check_sources(plan)
    summary=read(out/'summary.json')
    if summary['plan_sha256']!=digest(out/'plan.json'):raise RuntimeError('plan hash changed')
    if summary['status']=='completed':raise RuntimeError('pilot already completed; no retries')
    others=project_workers()
    if others:raise RuntimeError('another numerical worker is active: '+str(others))
    limits=plan['resources'];signalled=[]
    def on_signal(signum,frame):signalled.append(signum)
    for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(s,on_signal)
    session=uuid.uuid4().hex
    began=time.monotonic();child=None;active=None;reason=None
    with controller_locks(out) as lock_fds:
        active=new_control(out,session,limits['external_owner_lease_seconds'])
        summary['status']='running'
        summary['sessions'].append(dict(session=session,started_utc=now(),controller_pid=os.getpid()))
        write(out/'summary.json',summary)
        sensor=thermal();last_thermal=0
        try:
            for index,spec in enumerate(plan['attempts']):
                if not summary['runs'][index]['status'].startswith('not_run'):continue
                if signalled or (out/'STOP').exists():reason='stop_requested';break
                _check_sources(plan)
                trial=out/'trials'/spec['attempt_id']
                if trial.exists():raise RuntimeError('started trial must be audited, never automatically retried: '+trial.name)
                trial.mkdir()
                request=dict(spec,launch_monotonic=time.monotonic(),plan_sha256=summary['plan_sha256'])
                write(trial/'request.json',request)
                summary['runs'][index]=dict(attempt_id=spec['attempt_id'],status='started',started_utc=now())
                write(out/'summary.json',summary)
                guardian_path=Path(active['control_dir'])/(spec['attempt_id']+'.guardian.json')
                heartbeat=dict(monotonic=time.monotonic(),controller_pid=os.getpid(),attempt_id=spec['attempt_id'],index=index,
                               completed=sum(r['status'] in ('completed','solver_failure') for r in summary['runs']),stage='worker')
                write(out/'controller.heartbeat.json',heartbeat)
                print(json.dumps(dict(event='started',attempt_id=spec['attempt_id'],index=index+1,total=262)),flush=True)
                attempt_began=time.monotonic();last_cpu_time=attempt_began;last_progress=attempt_began
                last_cpu=0.;progress_key=None;peak=0.;stop_reason=None
                with (trial/'worker.log').open('x') as log,(trial/'controller.resources.jsonl').open('x') as samples:
                    child=GuardedProcess([sys.executable,'-B',str(Path(__file__).resolve()),
                        '--pilot-worker',str(trial/'request.json')],log,guardian_path,active,lock_fds,
                        heartbeat_timeout=limits['guardian_controller_heartbeat_seconds'])
                    while child.poll() is None:
                        clock=time.monotonic();sample=_safe_read(trial/'telemetry.json',{})
                        if sample.get('cpu_seconds',0)>last_cpu+.01:
                            last_cpu=sample['cpu_seconds'];last_cpu_time=clock
                        key=(sample.get('stage'),sample.get('completed_units'))
                        if key!=progress_key:progress_key=key;last_progress=clock
                        stale=clock-sample.get('monotonic',attempt_began)
                        peak=max(peak,sample.get('peak_rss_mib',0))
                        stop_reason=safeguards(sample,clock-attempt_began,clock-last_cpu_time,
                                               clock-last_progress,stale,limits)
                        if clock-last_thermal>=limits['thermal_sample_seconds']:
                            sensor=thermal();last_thermal=clock
                            if len(project_workers())>1:stop_reason='multiple_numeric_workers'
                            if shutil.disk_usage(out).free<limits['minimum_free_disk_bytes']:stop_reason='disk_space_low'
                        if sensor['thermal_state'] is not None and sensor['thermal_state']>=limits['thermal_stop_at']:
                            stop_reason='thermal_state_serious_or_critical'
                        if signalled or (out/'STOP').exists():stop_reason='stop_requested'
                        heartbeat.update(monotonic=time.monotonic(),worker_telemetry=sample,thermal=sensor)
                        write(out/'controller.heartbeat.json',heartbeat)
                        samples.write(json.dumps(dict(elapsed=clock-attempt_began,**sample,**sensor))+'\n');samples.flush()
                        if stop_reason:
                            child.terminate();break
                        time.sleep(limits['resource_sample_seconds'])
                    child.wait(timeout=8)
                g=read(guardian_path)
                if not g.get('cleanup_success'):raise RuntimeError('guardian cleanup failed')
                if g.get('reason')!='worker_exit' and not stop_reason:stop_reason=g.get('reason','guardian_stopped')
                ready=_safe_read(trial/'ready.json',{})
                pred=_safe_read(trial/'prediction.json',{})
                score=_safe_read(trial/'score.json',{})
                final=_safe_read(trial/'worker.final.json',{})
                record=dict(attempt_id=spec['attempt_id'],finished_utc=now(),
                    worker_process_seconds=time.monotonic()-attempt_began,
                    worker_response_seconds=ready.get('worker_response_seconds'),
                    solver_seconds=pred.get('result',{}).get('runtime_seconds'),
                    peak_rss_mib=max(peak,final.get('peak_rss_mib',0)),
                    worker_cpu_seconds=final.get('cpu_seconds'),guardian=str(guardian_path.relative_to(out)),
                    guardian_reason=g.get('reason'),cleanup_success=g.get('cleanup_success'),
                    worker_returncode=g.get('worker_returncode'),
                    max_abs_error=score.get('max_abs_error'),observed_accuracy_pass=score.get('observed_accuracy_pass'),
                    status='completed' if score.get('finite_complete') else 'solver_failure',reason=stop_reason)
                if stop_reason:record['status']='safety_stopped'
                elif g.get('worker_returncode')!=0 or not score:
                    record.update(status='worker_exception',reason='missing score or nonzero worker returncode')
                elif not score['finite_complete']:record['reason']=pred['result']['failure_reason']
                write(trial/'terminal.json',record)
                summary['runs'][index]=record;write(out/'summary.json',summary)
                child=None
                print(json.dumps(dict(event='finished',**record)),flush=True)
                if record['status'] in ('safety_stopped','worker_exception'):
                    reason=record['reason'];break
            else:
                summary['status']='completed'
        except BaseException as exc:
            reason=f'{type(exc).__name__}: {exc}'
            write(out/'controller.exception.json',dict(message=reason,traceback=traceback.format_exc()))
            raise
        finally:
            if child is not None:
                child.terminate();child.wait(timeout=8)
            finish_control(out,active)
            if summary['status']!='completed':
                summary['status']='stopped_safely'
                for r in summary['runs']:
                    if r['status']=='not_run':r.update(reason='not_run_after_safety_stop')
            summary['sessions'][-1].update(finished_utc=now(),wall_seconds=time.monotonic()-began,stop_reason=reason)
            summary.update(finished_utc=now(),outcomes=dict(Counter(r['status'] for r in summary['runs'])))
            write(out/'summary.json',summary)
            write(out/'controller.heartbeat.json',dict(monotonic=time.monotonic(),stage='finished',status=summary['status']))
            print(json.dumps(dict(event='series_finished',status=summary['status'],outcomes=summary['outcomes'],reason=reason)),flush=True)


def renew_checked(out):
    """External caller only; never invoked by run(). An expired lease stays dead."""
    from diagnostics.chernov_batch_control import atomic
    active=read(out/'active.json')
    if active.get('finished'):return dict(renewed=False,reason='finished',summary=read(out/'summary.json')['outcomes'])
    control=Path(active['control_dir']);lease=read(control/'lease.json')
    clock=time.monotonic();heartbeat=read(out/'controller.heartbeat.json')
    if clock-lease['renewed_monotonic']>active['owner_lease_seconds']:raise RuntimeError('expired lease cannot be revived')
    if clock-heartbeat['monotonic']>15:raise RuntimeError('controller is stale; do not renew')
    if (control/'stop.json').exists():raise RuntimeError('stop requested; do not renew')
    os.kill(active['controller_pid'],0)
    atomic(control/'lease.json',dict(token=active['token'],renewed_monotonic=clock))
    return dict(renewed=True,remaining_seconds=active['owner_lease_seconds'],
                completed=heartbeat.get('completed'),attempt_id=heartbeat.get('attempt_id'),
                progress=heartbeat.get('worker_telemetry'),thermal=heartbeat.get('thermal'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',nargs='?',choices=('prepare','run','renew','status','stop'))
    parser.add_argument('--out',type=Path)
    parser.add_argument('--pilot-worker',type=Path)
    args=parser.parse_args()
    if args.pilot_worker:worker(args.pilot_worker.resolve());return
    if args.out is None:parser.error('--out required')
    out=args.out.resolve()
    if not out.is_relative_to(ROOT/'results'):parser.error('new run must be under results')
    if args.action=='prepare':prepare(out)
    elif args.action=='run':run(out)
    elif args.action=='renew':print(json.dumps(renew_checked(out)),flush=True)
    elif args.action=='stop':
        from diagnostics.chernov_batch_control import request_stop
        (out/'STOP').touch();print(json.dumps(request_stop(out)),flush=True)
    elif args.action=='status':print(json.dumps(read(out/'summary.json')),flush=True)
    else:parser.error('action required')


if __name__=='__main__':main()
