"""Execute the approved frozen validation96 jobs only; no ML fit or test evaluation."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnostics.confirm_f0047_timing import *
SOURCE=ROOT/'results/ml_probability_validation_preflight_2026-10-26'
MODEL=ROOT/'results/ml_probability_selector_2026-10-26'


def validate_scope(plan):
    jobs=plan['jobs'];assert plan['execution_authorized']
    assert len(jobs)==len({j['key'] for j in jobs})==96
    assert jobs==read_json(SOURCE/'plan.json')['jobs']
    assert Counter(j['profile'] for j in jobs)=={'R128':32,'R256':32,'MC1':32}
    assert len({j['equation_id'] for j in jobs})==8 and len({j['family_id'] for j in jobs})==8
    assert all(j['split']=='validation' and j['episode'] in range(4) for j in jobs)
    assert plan['resources']['series_seconds']==5691 and plan['resources']['rss_mib']==512
    assert plan['resources']['compute_workers']==1


def initialize(out):
    import numpy as np
    from all_methods_dataset_loader import load_all_methods_equations
    from all_methods_bounds import certify_all_methods_forcing
    from mc_solver import MCConfig,auto_T
    from chernov_solver import ChernovConfig
    assert not out.exists()
    for folder in (SOURCE,MODEL):
        for n,h in read_json(folder/'artifact_sha256.json').items():assert digest(folder/n)==h,(folder,n)
    requested=read_json(SOURCE/'approval_request.json');source=read_json(SOURCE/'plan.json');contract=read_json(CONTRACT/'plan.json')
    assert digest(SOURCE/'plan.json')==requested['plan_sha256'] and digest(MODEL/'registry.json')==requested['model_registry_sha256']
    assert digest(SOURCE/'frozen_predictions.json')==requested['frozen_predictions_sha256']
    for n,h in read_json(MODEL/'source_sha256.json').items():assert digest(ROOT/n)==h,n
    old=read_json(ROOT/'results/remizov_r0_2026-10-26/plan.json')
    for n in NUMERICAL:assert digest(ROOT/n)==old['protected_sha256'][n],n
    data=ROOT/contract['data_dir']
    for n,h in contract['dataset_sha256'].items():assert digest(data/n)==h,n
    eqs={e.equation_id:e for e in load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl')}
    checks=[];jobs=source['jobs'];registry=read_json(MODEL/'registry.json');fitted=set(g for h in registry['heads'].values() for g in h['metadata']['families'])
    for case in source['cases']:
        e=eqs[case['equation_id']];m=e.metadata;assert m['family_id']==case['family_id'] and m['split']=='validation' and m['expression_hash']==case['expression_hash']
        assert m['family_id'] not in fitted
        cert=certify_all_methods_forcing(e.remizov_problem,m);G=cert.forcing_bound.abs_upper_bound;p=e.remizov_problem
        T=auto_T(p.lambda_value,p.c_abs_upper_bound,G,1e-4)
        assert G==case['G'] and T==case['T'] and math.ceil(T*128)==case['N']
        checks.append(dict(equation_id=e.equation_id,expression_hash=m['expression_hash'],family_id=m['family_id'],split='validation',G=G,T=T,N=case['N']))
    for j in jobs:
        cfg=j['config'];assert j['target_x']==contract['target_x']
        if j['profile']=='MC1':
            expected=dict(contract['profiles']['MC1']['config'],seed=248800060100+j['episode'],max_path_steps=j['numerical_work'],max_runtime_seconds=j['proposed_watchdog_seconds'])
            assert cfg==expected and j['numerical_work']==33*4096*j['expected_steps'];MCConfig(**cfg).validate()
        else:
            assert cfg==dict(contract['profiles'][j['profile']]['config'],T=j['expected_T']);ChernovConfig(**cfg).validate()
    pids={c['equation_id'] for c in checks};seedset={248800060100+i for i in range(4)}
    paths=list((ROOT/'results').glob('*/records/*.json'))+list(CONTRACT.glob('*/records/*.json'))+list((ROOT/'results').glob('*/events/*.request.json'))+list(CONTRACT.glob('*/events/*.request.json'))
    for path in paths:
        r=read_json(path);j=r.get('job',{});pid=r.get('equation_id',j.get('equation_id',j.get('problem_id')))
        if pid not in pids:continue
        assert j.get('config',r.get('config',{})).get('seed') not in seedset,'proposed semantic seed already requested'
        z=r.get('result') or {};cfg=j.get('config',{})
        if z.get('x')==contract['target_x'] and (cfg.get('M')==4096 or (cfg.get('n') in (128,256) and cfg.get('boundary')=='zero')):
            raise AssertionError('unexpected saved validation result')
    for n in ('.chernov_batch_controller.lock','.chernov_batch_compute.lock'):assert not lock_is_held(ROOT/'results'/n)
    sensor=thermal();assert sensor['thermal_state'] is not None and sensor['thermal_state']<2
    assert shutil.disk_usage(ROOT).free>=512*1024**2
    out.mkdir(parents=True,exist_ok=False)
    for n in ('records','events','workers','sessions','sources'):(out/n).mkdir()
    shutil.copy2(ROOT/'PROJECT_STATUS.md',out/'PROJECT_STATUS.before.md');write_json(out/'protected_before.json',snapshot(out),exclusive=True)
    sources=list(NUMERICAL)+['diagnostics/run_probability_validation.py','diagnostics/confirm_f0047_timing.py','diagnostics/remizov_r0.py','diagnostics/mc_pilot_p2.py','diagnostics/chernov_batch_control.py','diagnostics/mc_production_x0.py','diagnostics/joint_benchmark_pilot.py','diagnostics/run_chernov_batch.py','ml/probabilistic_selector.py']
    pins={n:digest(ROOT/n) for n in sources}
    for folder in (SOURCE,MODEL):
        for n,h in read_json(folder/'artifact_sha256.json').items():pins[str((folder/n).relative_to(ROOT))]=h
        pins[str((folder/'artifact_sha256.json').relative_to(ROOT))]=digest(folder/'artifact_sha256.json')
    pins.update({str((data/n).relative_to(ROOT)):h for n,h in contract['dataset_sha256'].items()})
    pins.update({str(f.relative_to(ROOT)):digest(f) for f in CONTRACT.iterdir() if f.is_file()})
    for n in sources:
        dest=out/'sources'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    authorization='User approves5691s wall cap, maximum96 calls,1worker/thread,RSS<=512MiB; frozen8 validation families,R128/R256/MC1x4; no test/benchmark1000/ML fit.'
    (out/'AUTHORIZATION.txt').write_text(authorization+'\n')
    plan=dict(schema='probability_validation96_execution_v1',created_utc=utc(),execution_authorized=True,authorization=authorization,
        source_plan_sha256=digest(SOURCE/'plan.json'),frozen_predictions_sha256=digest(SOURCE/'frozen_predictions.json'),registry_sha256=digest(MODEL/'registry.json'),
        cases=source['cases'],jobs=jobs,data_dir=contract['data_dir'],target_x=contract['target_x'],protected_sha256=pins,
        resources=dict(compute_workers=1,rss_mib=512,series_seconds=5691,owner_lease_seconds=300,startup_seconds=90,poll_seconds=.05,monitor_seconds=.25,thermal_seconds=10,thermal_stop_at=2),
        evaluation=source['evaluation'],scenario_seconds=source['budget']['total_scenario_seconds'],
        resume='Completed immutable; never redispatch any requested key under96 physical-call ceiling; original deadline never reset; interrupted keys remain pending',
        numerical_warmup_calls=0,no_test=True,no_benchmark1000=True,no_ML_fit=True)
    validate_scope(plan);write_json(out/'plan.json',plan,exclusive=True)
    write_json(out/'preflight.json',dict(passed=True,checks=checks,training_family_overlap=[],new_seed_conflicts=[],models_policy_predictions_unchanged=True,budget_approved=True,wall_cap_seconds=5691,maximum_calls=96,thermal=sensor),exclusive=True)
    shutil.copy2(SOURCE/'frozen_predictions.json',out/'frozen_predictions.json')
    write_json(out/'environment.json',dict(platform=platform.platform(),machine=platform.machine(),cpu_count=os.cpu_count(),python=sys.version,executable=sys.executable,versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','sympy','numba','llvmlite')},thread_environment={n:os.environ[n] for n in THREADS}),exclusive=True)
    power=dict(captured_utc=utc(),before_first_call=True)
    for n,cmd in [('settings',['pmset','-g','custom']),('source',['pmset','-g','batt'])]:
        r=subprocess.run(cmd,capture_output=True,text=True);power[n]=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
    write_json(out/'hardware_power_before.json',power,exclusive=True)
    print(json.dumps(dict(preflight='passed',jobs=96,cap=5691,scenario=plan['scenario_seconds'])),flush=True)


def worker(out,state_path,parent_pid):
    plan=read_json(out/'plan.json');stop=threading.Event();state=dict(pid=os.getpid(),phase='loading',key=None)
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
        from mc_solver import MCConfig,auto_T
        from mc_optimized import prepare_all_methods_mc,solve_monte_carlo_optimized,compile_backend
        imports=time.perf_counter()-t;t=time.perf_counter();data=ROOT/plan['data_dir'];allowed={c['equation_id'] for c in plan['cases']}
        eqs={e.equation_id:e for e in load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl') if e.equation_id in allowed};assert len(eqs)==8
        loading=time.perf_counter()-t;t=time.perf_counter();jit=compile_backend();warm=time.perf_counter()-t
        import numba
        assert numba.get_num_threads()==1
        write_json(state_path.with_suffix('.warmup.json'),dict(shared_import_seconds=imports,shared_load_seconds=loading,jit=jit,warmup_wall_seconds=warm,numba_threads=1,extra_solver_calls=0,no_RNG_in_warmup=True),exclusive=True)
        lookup={j['key']:j for j in plan['jobs']};repeats={};state.update(phase='ready');write_json(state_path,dict(state,monotonic=time.monotonic(),peak_rss_mib=rss()),durable=False)
        for line in sys.stdin:
            request=read_json(Path(json.loads(line)['request_path']));job=request['job'];assert job==lookup[job['key']]
            e=eqs[job['equation_id']];m=e.metadata;assert m['split']=='validation' and m['expression_hash']==job['expression_hash']
            state.update(phase='solving',key=job['key'])
            r=dict(job=job,plan_sha256=request['plan_sha256'],session=request['session'],equation_id=e.equation_id,expression_hash=m['expression_hash'],family_id=m['family_id'],split='validation',profile=job['profile'],config=job['config'],seed=job['config'].get('seed'),started_utc=utc(),worker_pid=os.getpid(),solver_called=False)
            try:
                if job['profile']!='MC1':
                    class CountedSolver:
                        def __init__(self,*a,**kw):self.solver=ChernovSolver(*a,**kw)
                        def solve(self,x):r['solver_called']=True;return self.solver.solve(x)
                    cert,T,z,wall,t0,t1=measured_r_call(e,job,certify_all_methods_forcing,auto_T,ChernovConfig,CountedSolver)
                else:
                    def counted_solve(*a,**kw):r['solver_called']=True;return solve_monte_carlo_optimized(*a,**kw)
                    prepared,z,wall,t0,t1=measured_call(m,job,prepare_all_methods_mc,MCConfig,counted_solve)
                r.update(result=asdict(z),method_wall_seconds=wall,timer_start_ns=t0,timer_stop_ns=t1,status='completed' if z.success else 'solver_failure')
                if job['profile']=='MC1':validate_result(job,prepared,z)
                else:
                    assert z.n==job['config']['n'] and z.scheme=='solve' and z.T==job['expected_T']
                    if z.success:
                        assert z.n_operator_applications==job['config']['n']*512 and z.n_spline_builds==job['config']['n']*512+1
                        assert z.time_nodes==513 and z.work_grid_points==2081 and z.actual_dx==.0125 and tuple(z.work_domain)==(-13.,13.)
                np.testing.assert_array_equal(z.x,job['target_x'])
                if z.success:assert z.failure_reason=='' and np.isfinite(z.f).all() and z.f.shape==(33,)
                tscore=time.perf_counter_ns();exact=e.y_exact(np.asarray(job['target_x']));assert exact.shape==(33,) and np.isfinite(exact).all()
                error=float(np.max(abs(z.f-exact))) if z.success else None
                r.update(y_exact=exact,E_grid=error,accuracy_met=error<=.01 if error is not None else None,scoring_seconds=(time.perf_counter_ns()-tscore)*1e-9)
                if job['profile']!='MC1' and z.success:
                    key=(e.equation_id,job['profile']);f=z.f.tolist()
                    if key in repeats:assert repeats[key]==f,'R deterministic repeat mismatch'
                    else:repeats[key]=f
            except Exception as exc:
                r.update(status='invalid_result' if isinstance(exc,AssertionError) else 'worker_exception',traceback=traceback.format_exc());r.setdefault('E_grid',None);r.setdefault('accuracy_met',None)
                timer=getattr(exc,'r0_timer',getattr(exc,'p1_timer',None))
                if timer:
                    t0,t1=timer;r.update(method_wall_seconds=(t1-t0)*1e-9,timer_start_ns=t0,timer_stop_ns=t1)
            r.update(finished_utc=utc(),peak_rss_mib=rss());write_json(out/'records'/f"{job['key']}.json",r,exclusive=True);state.update(phase='ready',key=None)
    finally:stop.set();thread.join()

def run(out):
    plan=read_json(out/'plan.json');ph=digest(out/'plan.json');jobs=plan['jobs'];limits=plan['resources']
    validate_scope(plan)
    for name,h in plan['protected_sha256'].items():assert digest(ROOT/name)==h,name
    done={p.stem:read_json(p) for p in (out/'records').glob('*.json')}
    assert set(done)<=set(j['key'] for j in jobs)
    for key,r in done.items():
        assert r['job']==next(j for j in jobs if j['key']==key) and r['plan_sha256']==ph,'saved result does not match plan'
    if len(done)==96:
        print(json.dumps(dict(new_calls=0,worker_started=False,reason='already_complete')),flush=True);return
    # Do not retry an interrupted physical invocation under a96-call ceiling.
    requests=[read_json(p) for p in (out/'events').glob('*.request.json')]
    dispatched={r['job']['key'] for r in requests}
    assert len(requests)==len(dispatched)<=96
    assert all(r['job'] in jobs and r['plan_sha256']==ph for r in requests)
    pending=[j for j in jobs if j['key'] not in done and j['key'] not in dispatched]
    if not pending:
        print(json.dumps(dict(new_calls=0,worker_started=False,reason='all_missing_keys_already_requested',pending=96-len(done))),flush=True);return
    if (out/'budget.json').exists():
        old_budget=read_json(out/'budget.json')
        if time.time()>=old_budget['deadline_epoch'] or time.monotonic()>=old_budget['deadline_monotonic']:
            print(json.dumps(dict(new_calls=0,worker_started=False,reason='original_budget_exhausted')),flush=True);return
    with controller_locks(out) as fds:
        if not (out/'budget.json').exists():
            start=time.time();startmono=time.monotonic();write_json(out/'budget.json',dict(started_epoch=start,deadline_epoch=start+limits['series_seconds'],started_monotonic=startmono,deadline_monotonic=startmono+limits['series_seconds'],started_utc=utc(),approved_cap_seconds=5691,maximum_physical_calls=96),exclusive=True)
        budget=read_json(out/'budget.json');deadline=budget['deadline_epoch'];deadline_mono=budget['deadline_monotonic'];session='validation96_'+uuid.uuid4().hex[:10]
        active=new_control(out,session,lease_seconds=300.);control=Path(active['control_dir'])
        began=time.perf_counter();child=None;log=None;state_path=None;reason=None;new=0;last_thermal=0.;sensor={}
        def checkpoint():
            write_json(out/'progress.json',dict(session=session,terminal=len(done),pending=96-len(done),
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
                assert len(dispatched)<=96
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
                        family_id=job['family_id'],split='validation',config=job['config'],seed=None,
                        plan_sha256=ph,status=outcome,result=None,E_grid=None,accuracy_met=None,accuracy_status='unknown_error',
                        method_wall_seconds=None,external_elapsed_seconds=time.perf_counter()-clock,
                        peak_rss_mib=peak,worker_exit_reason=child.stop_reason,finished_utc=utc()),exclusive=True)
                r=read_json(path);done[job['key']]=r;new+=1;checkpoint()
                print(json.dumps(dict(terminal=len(done),key=job['key'],status=r['status'],E_grid=r['E_grid'],
                    wall=r.get('method_wall_seconds'),session_seconds=time.perf_counter()-began)),flush=True)
                if r['status'] not in ('completed','solver_failure'):reason=r['status'];break
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
