"""Approved R-only40 execution. No MC/JIT/ML or other-equation calls."""
from __future__ import annotations
import argparse,hashlib,importlib.metadata,json,os,platform,shutil,signal,subprocess,sys,threading,time,traceback,uuid
from collections import Counter
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnostics.remizov_r0 import (measured_r_call,utc,read_json,write_json,digest,snapshot,THREADS,
    controller_locks,new_control,finish_control,GuardedProcess,lock_is_held,wait,stop_child,rss,thermal)
SOURCE=ROOT/'results/ml_switch5_r_preflight_2026-10-26'
CAP=1766;CALLS=40


def validate_scope(plan):
    source=read_json(SOURCE/'plan.json');jobs=plan['jobs'];limits=plan['resources']
    assert plan['execution_authorized'] and jobs==source['jobs']
    assert len(jobs)==len({j['key'] for j in jobs})==CALLS
    assert Counter(j['profile'] for j in jobs)=={'R128':20,'R256':20}
    assert {j['equation_id'] for j in jobs}=={c['equation_id'] for c in source['cases']}
    assert all(j['split']=='train' and j['seed'] is None and j['profile'] in ('R128','R256') for j in jobs)
    assert limits['series_seconds']==CAP and limits['rss_mib']==512 and limits['compute_workers']==1
    assert limits.get('owner_lease_seconds',300)==300 and limits.get('thermal_stop_at',2)==2


def initialize(out):
    import numpy as np
    from all_methods_dataset_loader import load_all_methods_equations
    from all_methods_bounds import certify_all_methods_forcing
    from mc_solver import auto_T
    from chernov_solver import ChernovConfig
    assert not out.exists()
    for name,h in read_json(SOURCE/'artifact_sha256.json').items():assert digest(SOURCE/name)==h,name
    source=read_json(SOURCE/'plan.json')
    for name,h in {**source['protected_sha256'],**read_json(SOURCE/'additional_protected_sha256.json')}.items():assert digest(ROOT/name)==h,name
    assert read_json(SOURCE/'independent_verification.json')['passed']
    assert read_json(SOURCE/'approval_request.json')['proposed_cap_seconds']==CAP
    data=SOURCE/'training_inputs'
    eqs={e.equation_id:e for e in load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl')}
    assert len(eqs)==5
    for c in source['cases']:
        e=eqs[c['equation_id']];m=e.metadata;p=e.remizov_problem
        assert m['split']=='train' and m['expression_hash']==c['expression_hash'] and m['family_id']==c['family_id']
        cert=certify_all_methods_forcing(p,m);G=cert.forcing_bound.abs_upper_bound;T=auto_T(p.lambda_value,p.c_abs_upper_bound,G,1e-4)
        assert G==c['G'] and T==c['T']
        exact=np.asarray(e.y_exact(np.asarray(source['target_x'])))
        for prof,ref in c['references'].items():
            r=read_json(ROOT/ref['source']);z=r['result'];ChernovConfig(**ref['config']).validate()
            np.testing.assert_array_equal(exact,r['y_exact']);np.testing.assert_array_equal(z['x'],source['target_x'])
            assert float(np.max(abs(np.asarray(z['f'])-exact)))==ref['E_grid']
            assert hashlib.sha256(np.asarray(z['f'],dtype='<f8').tobytes()).hexdigest()==ref['f_float64_sha256']
    for name in ('.chernov_batch_controller.lock','.chernov_batch_compute.lock'):assert not lock_is_held(ROOT/'results'/name)
    sensor=thermal();assert sensor['thermal_state'] is not None and sensor['thermal_state']<2
    assert shutil.disk_usage(ROOT).free>=512*1024**2
    out.mkdir(parents=True,exist_ok=False)
    for n in ('records','events','workers','sessions','sources'):(out/n).mkdir()
    shutil.copyfile(ROOT/'PROJECT_STATUS.md',out/'PROJECT_STATUS.before.md')
    write_json(out/'protected_before.json',snapshot(out),exclusive=True)
    shutil.copytree(data,out/'training_inputs');shutil.copyfile(SOURCE/'frozen_OOF_decisions.json',out/'frozen_OOF_decisions.json')
    pins={**source['protected_sha256'],**read_json(SOURCE/'additional_protected_sha256.json')}
    for n,h in read_json(SOURCE/'artifact_sha256.json').items():pins[str((SOURCE/n).relative_to(ROOT))]=h
    pins[str((SOURCE/'artifact_sha256.json').relative_to(ROOT))]=digest(SOURCE/'artifact_sha256.json')
    for n in ('diagnostics/run_switch5_r_timing.py','diagnostics/remizov_r0.py','diagnostics/mc_pilot_p2.py',
              'diagnostics/mc_production_x0.py','diagnostics/chernov_batch_control.py','diagnostics/run_chernov_batch.py'):
        pins[n]=digest(ROOT/n);dest=out/'sources'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dest)
    for f in (out/'training_inputs').iterdir():pins[str(f.relative_to(ROOT))]=digest(f)
    auth='User explicitly approves1766 seconds, maximum40 calls:4 R128+4 R256 on each frozen5 TRAIN cases,1 worker/thread,RSS<=512MiB,guardian/thermal/resume; no MC/ML/benchmark1000/validation/test.'
    (out/'AUTHORIZATION.txt').write_text(auth+'\n')
    plan=dict(schema='switch5_R40_execution_v1',execution_authorized=True,authorization=auth,created_utc=utc(),
        source_plan_sha256=digest(SOURCE/'plan.json'),jobs=source['jobs'],cases=source['cases'],target_x=source['target_x'],
        protected_sha256=pins,resources=dict(compute_workers=1,rss_mib=512,series_seconds=CAP,owner_lease_seconds=300,
            startup_seconds=90,poll_seconds=.05,monitor_seconds=.25,thermal_seconds=10,thermal_stop_at=2),
        evaluation=source['evaluation'],timing=source['timing'],order=source['order'],resumption=source['resumption'],
        source_proposed_budget_unchanged=True,no_MC=True,no_ML=True,no_validation_test=True,numerical_warmup_calls=0)
    validate_scope(plan);write_json(out/'plan.json',plan,exclusive=True)
    write_json(out/'preflight.json',dict(passed=True,identities=5,saved_reference_errors=10,G_T_configs_verified=True,
        source_manifest_verified=True,thermal=sensor,approved_cap_seconds=CAP,maximum_calls=CALLS,known_failure='F0172 both profiles inaccurate',
        model_policy_sources_unchanged=True),exclusive=True)
    write_json(out/'environment.json',dict(python=sys.version,platform=platform.platform(),machine=platform.machine(),
        executable=sys.executable,versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','sympy')},
        thread_environment={n:os.environ[n] for n in THREADS}),exclusive=True)
    power=dict(captured_utc=utc())
    for n,cmd in [('settings',['pmset','-g','custom']),('source',['pmset','-g','batt'])]:
        r=subprocess.run(cmd,capture_output=True,text=True);power[n]=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
    write_json(out/'hardware_power_before.json',power,exclusive=True)
    print(json.dumps(dict(preflight='passed',calls=CALLS,cap=CAP,worker_not_started=True)),flush=True)


def worker(out,state_path,parent_pid):
    plan=read_json(out/'plan.json');validate_scope(plan);stop=threading.Event();state=dict(pid=os.getpid(),phase='loading',key=None)
    def monitor():
        with state_path.with_suffix('.rss.jsonl').open('x') as stream:
            while not stop.is_set():
                if os.getppid()!=parent_pid:os._exit(75)
                sample=dict(state,monotonic=time.monotonic(),peak_rss_mib=rss());write_json(state_path,sample,durable=False)
                stream.write(json.dumps(sample)+'\n');stream.flush();stop.wait(.25)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        t=time.perf_counter()
        import numpy as np
        from all_methods_dataset_loader import load_all_methods_equations
        from all_methods_bounds import certify_all_methods_forcing
        from chernov_solver import ChernovConfig,ChernovSolver
        from mc_solver import auto_T  # Analytic formula only; no MC execution.
        imports=time.perf_counter()-t;t=time.perf_counter();data=out/'training_inputs'
        eqs={e.equation_id:e for e in load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl')}
        assert len(eqs)==5 and all(e.metadata['split']=='train' for e in eqs.values())
        loading=time.perf_counter()-t
        write_json(state_path.with_suffix('.setup.json'),dict(shared_import_seconds=imports,shared_load_seconds=loading,
            extra_solver_calls=0,JIT_calls=0,MC_calls=0),exclusive=True)
        lookup={j['key']:j for j in plan['jobs']};repeats={};state.update(phase='ready')
        write_json(state_path,dict(state,monotonic=time.monotonic(),peak_rss_mib=rss()),durable=False)
        for line in sys.stdin:
            request=read_json(Path(json.loads(line)['request_path']));job=request['job'];assert job==lookup[job['key']]
            e=eqs[job['equation_id']];m=e.metadata;assert m['expression_hash']==job['expression_hash'] and m['split']=='train'
            state.update(phase='solving',key=job['key'])
            r=dict(job=job,plan_sha256=request['plan_sha256'],session=request['session'],equation_id=e.equation_id,
                family_id=m['family_id'],expression_hash=m['expression_hash'],split='train',profile=job['profile'],config=job['config'],seed=None,
                started_utc=utc(),worker_pid=os.getpid(),solver_called=False)
            try:
                class CountedSolver:
                    def __init__(self,*args,**kw):self.solver=ChernovSolver(*args,**kw)
                    def solve(self,x):r['solver_called']=True;return self.solver.solve(x)
                cert,T,z,wall,t0,t1=measured_r_call(e,job,certify_all_methods_forcing,auto_T,ChernovConfig,CountedSolver)
                r.update(result=asdict(z),method_wall_seconds=wall,timer_start_ns=t0,timer_stop_ns=t1,
                    status='completed' if z.success else 'solver_failure',forcing_G=cert.forcing_bound.abs_upper_bound,effective_T=T)
                assert z.success and z.failure_reason=='' and z.n==job['config']['n'] and z.scheme=='solve'
                assert z.T==job['expected_T'] and z.n_operator_applications==job['numerical_work'] and z.n_spline_builds==job['numerical_work']+1
                assert z.time_nodes==513 and z.work_grid_points==2081 and z.actual_dx==.0125 and tuple(z.work_domain)==(-13.,13.)
                np.testing.assert_array_equal(z.x,job['target_x']);assert z.f.shape==(33,) and np.isfinite(z.f).all()
                tscore=time.perf_counter_ns();exact=e.y_exact(np.asarray(job['target_x']));err=float(np.max(abs(z.f-exact)))
                r.update(y_exact=exact,E_grid=err,accuracy_met=err<=.01,scoring_seconds=(time.perf_counter_ns()-tscore)*1e-9)
                assert err==job['reference_E_grid'] and r['accuracy_met']==job['expected_accurate']
                assert hashlib.sha256(np.asarray(z.f,dtype='<f8').tobytes()).hexdigest()==job['expected_f_float64_sha256']
                ref=read_json(ROOT/job['reference_path']);assert digest(ROOT/job['reference_path'])==job['reference_sha256']
                np.testing.assert_array_equal(z.f,ref['result']['f']);np.testing.assert_array_equal(exact,ref['y_exact'])
                r['reference_array_identical']=True
                key=(e.equation_id,job['profile']);f=z.f.tolist()
                if key in repeats:assert repeats[key]==f
                else:repeats[key]=f
                assert rss()<=512
            except Exception as exc:
                r.update(status='invalid_result' if isinstance(exc,AssertionError) else 'worker_exception',traceback=traceback.format_exc())
                r.setdefault('E_grid',None);r.setdefault('accuracy_met',None)
                timer=getattr(exc,'r0_timer',None)
                if timer:r.update(method_wall_seconds=(timer[1]-timer[0])*1e-9,timer_start_ns=timer[0],timer_stop_ns=timer[1])
            r.update(finished_utc=utc(),peak_rss_mib=rss());write_json(out/'records'/f"{job['key']}.json",r,exclusive=True)
            state.update(phase='ready',key=None)
    finally:stop.set();thread.join()

def run(out):
    plan=read_json(out/'plan.json');ph=digest(out/'plan.json');jobs=plan['jobs'];limits=plan['resources']
    validate_scope(plan)
    for name,h in plan['protected_sha256'].items():assert digest(ROOT/name)==h,name
    done={p.stem:read_json(p) for p in (out/'records').glob('*.json')}
    assert set(done)<=set(j['key'] for j in jobs)
    for key,r in done.items():
        assert r['job']==next(j for j in jobs if j['key']==key) and r['plan_sha256']==ph,'saved result does not match plan'
    assert all(r.get('status')=='completed' for r in done.values()), 'resume blocked after critical result'
    if len(done)==40:
        print(json.dumps(dict(new_calls=0,worker_started=False,reason='already_complete')),flush=True);return
    # Do not retry an interrupted physical invocation under a40-call ceiling.
    requests=[read_json(p) for p in (out/'events').glob('*.request.json')]
    dispatched={r['job']['key'] for r in requests}
    assert len(requests)==len(dispatched)<=40
    assert all(r['job'] in jobs and r['plan_sha256']==ph for r in requests)
    pending=[j for j in jobs if j['key'] not in done and j['key'] not in dispatched]
    if not pending:
        print(json.dumps(dict(new_calls=0,worker_started=False,reason='all_missing_keys_already_requested',pending=40-len(done))),flush=True);return
    if (out/'budget.json').exists():
        old_budget=read_json(out/'budget.json')
        if time.time()>=old_budget['deadline_epoch'] or time.monotonic()>=old_budget['deadline_monotonic']:
            print(json.dumps(dict(new_calls=0,worker_started=False,reason='original_budget_exhausted')),flush=True);return
    with controller_locks(out) as fds:
        if not (out/'budget.json').exists():
            start=time.time();startmono=time.monotonic();write_json(out/'budget.json',dict(started_epoch=start,deadline_epoch=start+limits['series_seconds'],started_monotonic=startmono,deadline_monotonic=startmono+limits['series_seconds'],started_utc=utc(),approved_cap_seconds=1766,maximum_physical_calls=40),exclusive=True)
        budget=read_json(out/'budget.json');deadline=budget['deadline_epoch'];deadline_mono=budget['deadline_monotonic'];session='switch5_R40_'+uuid.uuid4().hex[:10]
        active=new_control(out,session,lease_seconds=300.);control=Path(active['control_dir'])
        began=time.perf_counter();child=None;log=None;state_path=None;reason=None;new=0;last_thermal=0.;sensor={}
        def checkpoint():
            write_json(out/'progress.json',dict(session=session,terminal=len(done),pending=40-len(done),
                statuses=dict(Counter(r['status'] for r in done.values())),new_terminal=new,
                session_wall_seconds=time.perf_counter()-began,remaining_seconds=max(0.,min(deadline-time.time(),deadline_mono-time.monotonic())),thermal=sensor))
        def wait_thermal(path,attempt_deadline):
            nonlocal last_thermal,sensor
            peak=0.
            while True:
                if time.monotonic()>=deadline_mono:return 'series_budget_exhausted',peak
                if time.monotonic()-last_thermal>=10.:
                    sensor=thermal();last_thermal=time.monotonic()
                    with (out/'thermal.jsonl').open('a') as stream:stream.write(json.dumps(dict(utc=utc(),**sensor))+'\n')
                    if sensor['thermal_state'] is None or sensor['thermal_state']>=2:
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
                if time.time()>=deadline or time.monotonic()>=deadline_mono:reason='series_budget_exhausted';break
                if (control/'stop.json').exists():reason='interrupted';break
                if child is not None and child.poll() is not None:reason='worker_exited_between_jobs';break
                if child is None:
                    if log:log.close()
                    token=session+'_'+uuid.uuid4().hex[:6];state_path=out/'workers'/f'{token}.json';log=(out/'workers'/f'{token}.log').open('x')
                    child=GuardedProcess([sys.executable,'-B',str(Path(__file__).resolve()),'worker','--out',str(out),
                        '--state',str(state_path),'--parent-pid','{GUARD_PID}'],log,control/f'{token}.guardian.json',active,fds)
                    startup,peak=wait_thermal(None,time.monotonic()+90.)
                    if startup:reason='startup_'+startup;break
                path=out/'records'/f"{job['key']}.json";request_path=out/'events'/f"{session}_{job['key']}.request.json"
                write_json(request_path,dict(job=job,session=session,plan_sha256=ph),exclusive=True);dispatched.add(job['key'])
                assert len(dispatched)<=40
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
                        family_id=job['family_id'],split='train',config=job['config'],seed=None,
                        plan_sha256=ph,status=outcome,result=None,E_grid=None,accuracy_met=None,accuracy_status='unknown_error',
                        method_wall_seconds=None,external_elapsed_seconds=time.perf_counter()-clock,
                        peak_rss_mib=peak,worker_exit_reason=child.stop_reason,finished_utc=utc()),exclusive=True)
                r=read_json(path);done[job['key']]=r;new+=1;checkpoint()
                print(json.dumps(dict(terminal=len(done),key=job['key'],status=r['status'],E_grid=r['E_grid'],
                    wall=r.get('method_wall_seconds'),session_seconds=time.perf_counter()-began)),flush=True)
                if r['status']!='completed':reason=r['status'];break
        finally:
            stop_child(child)
            if log:log.close()
            for s,handler in old.items():signal.signal(s,handler)
            write_json(out/'sessions'/f'{session}.json',dict(session=session,new_terminal=new,reason=reason,
                wall_seconds=time.perf_counter()-began,finished_utc=utc()),exclusive=True)
            checkpoint();finish_control(out,active)
            write_json(out/'pending.json',dict(keys=[j['key'] for j in jobs if j['key'] not in done],reason=reason))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=('init','run','worker'));ap.add_argument('--out',type=Path,required=True);ap.add_argument('--state',type=Path);ap.add_argument('--parent-pid',type=int)
    a=ap.parse_args();out=a.out.resolve()
    if a.action=='init':initialize(out)
    elif a.action=='run':run(out)
    else:worker(out,a.state,a.parent_pid)
