"""Execute only the frozen38 R0 cells; no MC, benchmark1000 or ML fit.

Numerical implementations/configs are imported unchanged. An independent
project guardian enforces RSS, deadline, external owner lease and cleanup.
"""
from __future__ import annotations
import argparse,csv,json,math,os,platform,shutil,signal,subprocess,sys,threading,time,traceback,uuid
from collections import Counter
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnostics.mc_pilot_p2 import (utc,read_json,write_json,digest,snapshot,THREADS,
    controller_locks,new_control,finish_control,GuardedProcess,lock_is_held,wait,stop_child,rss,thermal)
CONTRACT=ROOT/'reports/ml_selector_contract'
NUMERICAL=('chernov_solver.py','remizov_solver.py','all_methods_bounds.py','all_methods_dataset_loader.py',
    'mc_solver.py','mc_optimized.py','mc_ast_kernels.py','src/all_methods_expressions.py','src/all_methods_dataset_v1.py')


def scope_jobs(contract,jobs):
    selected=[j for j in jobs if j['stage']==0]
    assert len(selected)==38 and len({j['key'] for j in selected})==38
    assert Counter(j['profile'] for j in selected)=={'R128':3,'R256':35}
    assert len({j['equation_id'] for j in selected})==35
    for j in selected:
        cfg=j['config'];n=int(j['profile'][1:])
        assert j['split']=='train' and j['episode']==0 and j['target_x']==[-1+i/16 for i in range(33)]
        assert cfg==dict(contract['profiles'][j['profile']]['config'],T=j['expected_T'])
        assert cfg['n']==n and j['numerical_work']==n*512 and j['work_unit']=='S_applications'
    return selected


def measured_r_call(e,job,certify,auto_T,config_factory,solver_factory,clock=time.perf_counter_ns):
    start=clock()
    try:
        cert=certify(e.remizov_problem,e.metadata);G=cert.forcing_bound.abs_upper_bound
        p=e.remizov_problem;T=auto_T(p.lambda_value,p.c_abs_upper_bound,G,1e-4)
        assert G==job['forcing_G'] and T==job['expected_T']==job['config']['T']
        config=config_factory(**job['config']);solver=solver_factory(p,config)
        result=solver.solve(job['target_x'])
    except Exception as exc:
        end=clock();exc.r0_timer=(start,end);raise
    end=clock();return cert,T,result,(end-start)*1e-9,start,end


def initialize(out):
    import numpy as np,importlib.metadata
    from all_methods_dataset_loader import load_all_methods_equations
    from all_methods_bounds import certify_all_methods_forcing
    from chernov_solver import ChernovConfig
    from mc_solver import auto_T
    contract=read_json(CONTRACT/'plan.json');jobs=scope_jobs(contract,[json.loads(s) for s in (CONTRACT/'pilot_jobs.jsonl').read_text().splitlines()])
    for folder in [CONTRACT/'sentinel_p1_2026-10-26',CONTRACT/'pilot_p2_2026-10-26',ROOT/'results/ml_selector_observed_2026-10-26_v2']:
        for name,h in read_json(folder/'artifact_sha256.json').items():assert digest(folder/name)==h,str(folder/name)
    sourceplan=read_json(ROOT/'results/joint_grid100_2026-10-25/plan.json')
    for name in NUMERICAL:assert digest(ROOT/name)==sourceplan['protected_sha256'][name],name
    data=ROOT/contract['data_dir']
    for name,h in contract['dataset_sha256'].items():assert digest(data/name)==h,name
    eqs={e.equation_id:e for e in load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl')}
    manifest=list(csv.DictReader((CONTRACT/'pilot_manifest.csv').open()));checks=[];oldpins={}
    missing=[]
    for row in manifest:
        pid=row['equation_id'];e=eqs[pid];m=e.metadata
        assert m['expression_hash']==row['expression_hash'] and m['family_id']==row['family_id'] and m['split']=='train'
        cert=certify_all_methods_forcing(e.remizov_problem,m);G=cert.forcing_bound.abs_upper_bound;p=e.remizov_problem
        T=auto_T(p.lambda_value,p.c_abs_upper_bound,G,1e-4)
        assert T==float(row['T']) and G==float(row['G'])
        exact=np.asarray(e.y_exact(np.asarray(contract['target_x'])));assert exact.shape==(33,) and np.isfinite(exact).all()
        for profile,folder,suffix in [('R128','joint_grid100_2026-10-25','R'),('R256','final_grid5_2026-10-25','R256')]:
            path=ROOT/'results'/folder/'records'/f'{pid}_{suffix}.json'
            if not path.exists():missing.append((pid,profile));continue
            r=read_json(path);z=r['result'];cfg=r['job']['config'];expected=dict(contract['profiles'][profile]['config'],T=T)
            assert cfg==expected and r['status']=='completed' and z['success'] and z['failure_reason']==''
            assert r['expression_hash']==m['expression_hash'] and r['family_id']==m['family_id'] and r['split']=='train'
            assert z['problem_id']==pid and z['x']==contract['target_x'] and z['T']==T and z['n']==expected['n'] and z['time_nodes']==513
            assert np.array_equal(r['y_exact'],exact) and np.isfinite(z['f']).all() and len(z['f'])==33
            error=float(np.max(abs(np.asarray(z['f'])-exact)));assert error==r['E_inf'] and (error<=.01)==r['accuracy_met']
            oldpins[str(path.relative_to(ROOT))]=digest(path)
        for j in [j for j in jobs if j['equation_id']==pid]:
            assert j['expected_T']==T and j['forcing_G']==G and j['expression_hash']==m['expression_hash']
            ChernovConfig(**j['config']).validate()
        checks.append(dict(equation_id=pid,family_id=m['family_id'],expression_hash=m['expression_hash'],split='train',G=G,T=T,work_grid_points=2081))
    assert len(checks)==40 and len({r['family_id'] for r in checks})==40
    assert set(missing)=={(j['equation_id'],j['profile']) for j in jobs} and len(oldpins)==42
    for name in ('.chernov_batch_controller.lock','.chernov_batch_compute.lock'):assert not lock_is_held(ROOT/'results'/name)
    sensor=thermal();assert sensor['thermal_state'] is None or sensor['thermal_state']<2
    disk=shutil.disk_usage(ROOT);assert disk.free>=256*1024**2
    out.mkdir(parents=True,exist_ok=False)
    for name in ('records','events','workers','sessions','sources'):(out/name).mkdir()
    shutil.copy2(ROOT/'PROJECT_STATUS.md',out/'PROJECT_STATUS.before.md')
    write_json(out/'protected_before.json',snapshot(out),exclusive=True)
    sourcefiles=list(NUMERICAL)+['diagnostics/remizov_r0.py','diagnostics/mc_pilot_p2.py','diagnostics/chernov_batch_control.py','diagnostics/run_chernov_batch.py','diagnostics/mc_production_x0.py','diagnostics/joint_benchmark_pilot.py']
    pins={n:digest(ROOT/n) for n in sourcefiles};pins.update(oldpins)
    pins.update({str((data/n).relative_to(ROOT)):h for n,h in contract['dataset_sha256'].items()})
    pins.update({str(f.relative_to(ROOT)):digest(f) for f in CONTRACT.iterdir() if f.is_file()})
    for name in sourcefiles:
        dest=out/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    scenario=sum(j['scenario_solver_seconds'] for j in jobs);assert math.isclose(scenario,contract['pilot']['missing_R_scenario_seconds'],rel_tol=1e-12)
    cap=math.ceil(2*scenario+300)
    plan=dict(schema='R0_execution_v1',created_utc=utc(),execution_authorized=True,
        authorization='Current user explicitly requests exactly38 missing R0 cells; no MC, benchmark1000 or new ML training',
        source_jobs_sha256=digest(CONTRACT/'pilot_jobs.jsonl'),source_contract_sha256=digest(CONTRACT/'plan.json'),
        data_dir=contract['data_dir'],target_x=contract['target_x'],threshold=.01,jobs=jobs,preflight=checks,
        protected_sha256=pins,total_new_calls=38,total_S_applications=sum(j['numerical_work'] for j in jobs),
        scenario_solver_seconds=scenario,budget_rationale='R0-only cap=ceil(2*saved901.161973scenario+300); shared old R0+P1 cap5437 not reused; scenario not CI/bound',
        resources=dict(compute_workers=1,rss_mib=512.,series_seconds=cap,owner_lease_seconds=300.,startup_seconds=90.,poll_seconds=.05,monitor_seconds=.25,thermal_seconds=10.,thermal_stop_at=2),
        timing=dict(clock='perf_counter_ns',scope='certify_G + auto_T + ChernovConfig + ChernovSolver constructor + solve',excludes=['shared_imports','shared_loader','canonical_exact_scoring','identity_validation_after_return','serialization']),
        retry_policy='immutable completed records; resume never dispatches an already-requested key under38-call ceiling; no budget reset',
        order='original frozen stage0 job order',no_MC=True,no_ML=True,no_benchmark1000=True)
    write_json(out/'plan.json',plan,exclusive=True)
    write_json(out/'preflight.json',dict(passed=True,missing_cells=missing,existing_numeric_cells=42,verified_train_equations=40,unique_families=40,initial_thermal=sensor,disk_free_bytes=disk.free,work_grid_points=2081,guard_max_points=100000,discrepancies=[]),exclusive=True)
    env=dict(platform=platform.platform(),machine=platform.machine(),python=sys.version,executable=sys.executable,cpu_count=os.cpu_count(),versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','sympy','numba','llvmlite')},thread_environment={k:os.environ[k] for k in THREADS},thermal=sensor,captured_utc=utc())
    write_json(out/'environment.json',env,exclusive=True)
    power=dict(captured_utc=utc(),before_first_R0_call=True)
    for name,cmd in [('settings',['pmset','-g','custom']),('source',['pmset','-g','batt'])]:
        r=subprocess.run(cmd,capture_output=True,text=True);power[name]=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
    write_json(out/'hardware_power_before.json',power,exclusive=True)
    print(json.dumps(dict(preflight='passed',jobs=38,scenario_seconds=scenario,cap_seconds=cap,work=plan['total_S_applications'])),flush=True)


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
        from mc_solver import auto_T
        from chernov_solver import ChernovConfig,ChernovSolver
        imports=time.perf_counter()-t;t=time.perf_counter();data=ROOT/plan['data_dir']
        eqs={e.equation_id:e for e in load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl')};loading=time.perf_counter()-t
        write_json(state_path.with_suffix('.startup.json'),dict(shared_import_seconds=imports,shared_load_seconds=loading,extra_solver_calls=0,JIT_not_required=True),exclusive=True)
        lookup={j['key']:j for j in plan['jobs']};state.update(phase='ready');write_json(state_path,dict(state,monotonic=time.monotonic(),peak_rss_mib=rss()),durable=False)
        for line in sys.stdin:
            request=read_json(Path(json.loads(line)['request_path']));job=request['job'];assert job==lookup[job['key']]
            e=eqs[job['equation_id']];m=e.metadata;assert m['expression_hash']==job['expression_hash'] and m['split']=='train'
            state.update(phase='solving',key=job['key'])
            r=dict(job=job,plan_sha256=request['plan_sha256'],session=request['session'],equation_id=e.equation_id,expression_hash=m['expression_hash'],family_id=m['family_id'],split='train',profile=job['profile'],config=job['config'],seed=None,started_utc=utc(),worker_pid=os.getpid(),shared_import_seconds=imports,shared_load_seconds=loading,solver_called=False)
            class CountedSolver:
                def __init__(self,*args,**kwargs):self.solver=ChernovSolver(*args,**kwargs)
                def solve(self,x):r['solver_called']=True;return self.solver.solve(x)
            try:
                cert,T,z,wall,t0,t1=measured_r_call(e,job,certify_all_methods_forcing,auto_T,ChernovConfig,CountedSolver)
                r.update(result=asdict(z),method_wall_seconds=wall,timer_start_ns=t0,timer_stop_ns=t1,forcing_G=cert.forcing_bound.abs_upper_bound,effective_T=T,status='completed' if z.success else 'solver_failure')
                assert z.problem_id==job['equation_id'] and z.T==job['expected_T'] and z.n==job['config']['n'] and z.scheme=='solve'
                np.testing.assert_array_equal(z.x,job['target_x'])
                if z.success:
                    assert z.failure_reason=='' and z.f.shape==(33,) and np.isfinite(z.f).all()
                    assert z.time_nodes==513 and z.work_grid_points==2081 and z.actual_dx==.0125 and tuple(z.work_domain)==(-13.,13.)
                    assert z.n_operator_applications==job['numerical_work']
                    assert z.n_spline_builds==job['numerical_work']+1
                began=time.perf_counter_ns();exact=e.y_exact(np.asarray(job['target_x']));error=float(np.max(abs(z.f-exact))) if z.success else None
                r.update(y_exact=exact,E_grid=error,accuracy_met=error<=.01 if error is not None else None,scoring_seconds=(time.perf_counter_ns()-began)*1e-9)
            except Exception as exc:
                r.update(status='invalid_result' if isinstance(exc,AssertionError) else 'worker_exception',traceback=traceback.format_exc(),E_grid=None,accuracy_met=None)
                if hasattr(exc,'r0_timer'):
                    t0,t1=exc.r0_timer;r.update(method_wall_seconds=(t1-t0)*1e-9,timer_start_ns=t0,timer_stop_ns=t1)
            r.update(finished_utc=utc(),peak_rss_mib=rss());write_json(out/'records'/f"{job['key']}.json",r,exclusive=True);state.update(phase='ready',key=None)
    finally:stop.set();thread.join()


def run(out):
    plan=read_json(out/'plan.json');ph=digest(out/'plan.json');jobs=plan['jobs'];limits=plan['resources']
    assert plan['execution_authorized'] and len(jobs)==38 and all(j['stage']==0 and j['profile'] in ('R128','R256') for j in jobs)
    for name,h in plan['protected_sha256'].items():assert digest(ROOT/name)==h,name
    done={p.stem:read_json(p) for p in (out/'records').glob('*.json')}
    assert set(done)<=set(j['key'] for j in jobs)
    for key,r in done.items():
        assert r['job']==next(j for j in jobs if j['key']==key) and r['plan_sha256']==ph,'saved result does not match plan'
    if len(done)==38:
        print(json.dumps(dict(new_calls=0,worker_started=False,reason='already_complete')),flush=True);return
    # Do not retry an interrupted physical invocation under a38-call ceiling.
    dispatched={read_json(p)['job']['key'] for p in (out/'events').glob('*.request.json')}
    pending=[j for j in jobs if j['key'] not in done and j['key'] not in dispatched]
    with controller_locks(out) as fds:
        if not (out/'budget.json').exists():
            start=time.time();write_json(out/'budget.json',dict(started_epoch=start,deadline_epoch=start+limits['series_seconds'],started_utc=utc()),exclusive=True)
        deadline=read_json(out/'budget.json')['deadline_epoch'];session='r0_'+uuid.uuid4().hex[:10]
        active=new_control(out,session,lease_seconds=300.);control=Path(active['control_dir'])
        began=time.perf_counter();child=None;log=None;state_path=None;reason=None;new=0;last_thermal=0.;sensor={}
        def checkpoint():
            write_json(out/'progress.json',dict(session=session,terminal=len(done),pending=38-len(done),
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
                assert len(dispatched)<=38
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
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=('init','run','worker'));ap.add_argument('--out',type=Path,required=True);ap.add_argument('--state',type=Path);ap.add_argument('--parent-pid',type=int)
    a=ap.parse_args();out=a.out.resolve()
    if a.action=='init':initialize(out)
    elif a.action=='run':run(out)
    else:worker(out,a.state,a.parent_pid)
