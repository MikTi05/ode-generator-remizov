"""Only four R256 + four MC1 replay calls on the frozen F0047/S00 task."""
from __future__ import annotations
import argparse,hashlib,importlib.metadata,json,math,os,platform,shutil,signal,subprocess,sys,threading,time,traceback,uuid
from collections import Counter
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnostics.remizov_r0 import (NUMERICAL,measured_r_call,utc,read_json,write_json,digest,snapshot,THREADS,
    controller_locks,new_control,finish_control,GuardedProcess,lock_is_held,wait,stop_child,rss,thermal)
from diagnostics.mc_pilot_p2 import measured_call,validate_result
PID='AMV1_F0047_S00';CONTRACT=ROOT/'reports/ml_selector_contract';R0=ROOT/'results/remizov_r0_2026-10-26'
P1=CONTRACT/'sentinel_p1_2026-10-26';P2=CONTRACT/'pilot_p2_2026-10-26'


def make_jobs(proposal,original):
    mc={j['episode']:j for j in original if j['equation_id']==PID and j['stage']==1 and j['profile']=='MC1'}
    assert len(mc)==4 and proposal['R_profile']=='R256' and proposal['MC_profile']=='MC1'
    assert proposal['equation_id']==PID and proposal['calls']==8 and proposal['target_x']==[-1+i/16 for i in range(33)]
    assert proposal['MC_original_jobs']==[mc[i] for i in range(4)]
    jobs=[]
    for i in range(4):
        old=mc[i];assert old['config']['seed']==248800010100+i
        m=dict(old,key=f'{PID}_MC1_timing_episode{i}',reference_path=str((P1/'records'/f"{old['key']}.json").relative_to(ROOT)))
        r=dict(old,key=f'{PID}_R256_timing_episode{i}',profile='R256',config=proposal['R_config'],episode=i,
            numerical_work=256*512,work_unit='S_applications',expected_steps=0,proposed_watchdog_seconds=164,
            reference_path=f'results/final_grid5_2026-10-25/records/{PID}_R256.json')
        shift=int(hashlib.sha256(f'{PID}|{i}'.encode()).hexdigest(),16)%2
        jobs.extend((r,m) if shift==0 else (m,r))
    assert Counter(j['profile'] for j in jobs)=={'R256':4,'MC1':4}
    return jobs


def initialize(out):
    import numpy as np
    from all_methods_dataset_loader import load_all_methods_equations
    from all_methods_bounds import certify_all_methods_forcing
    from chernov_solver import ChernovConfig
    from mc_optimized import prepare_all_methods_mc
    from mc_solver import MCConfig,auto_T
    assert not out.exists()
    for folder in (R0,P1,P2):
        for name,h in read_json(folder/'artifact_sha256.json').items():assert digest(folder/name)==h,(folder,name)
    contract=read_json(CONTRACT/'plan.json');proposal=read_json(R0/'NEXT_STEP.json')
    original=[json.loads(s) for s in (CONTRACT/'pilot_jobs.jsonl').read_text().splitlines()]
    jobs=make_jobs(proposal,original)
    r0plan=read_json(R0/'plan.json')
    for name in NUMERICAL:assert digest(ROOT/name)==r0plan['protected_sha256'][name],name
    data=ROOT/contract['data_dir']
    for name,h in contract['dataset_sha256'].items():assert digest(data/name)==h,name
    e=next(e for e in load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl') if e.equation_id==PID)
    m=e.metadata;assert m['split']=='train' and m['family_id']=='F0047'
    cert=certify_all_methods_forcing(e.remizov_problem,m);G=cert.forcing_bound.abs_upper_bound;p=e.remizov_problem
    T=auto_T(p.lambda_value,p.c_abs_upper_bound,G,1e-4);exact=np.asarray(e.y_exact(np.asarray(proposal['target_x'])))
    assert exact.shape==(33,) and np.isfinite(exact).all()
    assert proposal['R_config']==dict(contract['profiles']['R256']['config'],T=T)
    assert read_json(R0/'analysis.json')['selected_R_profile']=='R256'
    assert read_json(P1/'analysis/summary.json')['decision']['selected']=='MC1'
    prepared=prepare_all_methods_mc(m);assert prepared.forcing_bound.abs_upper_bound==G
    for j in jobs:
        assert j['equation_id']==PID and j['expression_hash']==m['expression_hash'] and j['family_id']==m['family_id'] and j['split']=='train'
        assert j['target_x']==proposal['target_x'] and j['expected_T']==T and j['forcing_G']==G
        ref=read_json(ROOT/j['reference_path']);z=ref['result'];assert ref['status']=='completed' and z['success']
        assert ref['job']['config']==j['config'] and ref['expression_hash']==m['expression_hash']
        assert z['problem_id']==PID and z['T']==T and np.array_equal(z['x'],proposal['target_x'])
        assert np.array_equal(ref['y_exact'],exact) and np.isfinite(z['f']).all()
        assert float(np.max(abs(np.asarray(z['f'])-exact)))==ref.get('E_grid',ref.get('E_inf'))
        if j['profile']=='R256':ChernovConfig(**j['config']).validate()
        else:
            MCConfig(**j['config']).validate()
            assert j['numerical_work']==j['config']['max_path_steps']==33*4096*j['expected_steps']
            assert math.ceil(T/j['config']['dt'])==j['expected_steps']
    for name in ('.chernov_batch_controller.lock','.chernov_batch_compute.lock'):assert not lock_is_held(ROOT/'results'/name)
    sensor=thermal();assert sensor['thermal_state'] is not None and sensor['thermal_state']<2
    assert shutil.disk_usage(ROOT).free>=256*1024**2
    out.mkdir(parents=True,exist_ok=False)
    for name in ('records','events','workers','sessions','sources'):(out/name).mkdir()
    shutil.copy2(ROOT/'PROJECT_STATUS.md',out/'PROJECT_STATUS.before.md')
    write_json(out/'protected_before.json',snapshot(out),exclusive=True)
    sources=list(NUMERICAL)+['diagnostics/confirm_f0047_timing.py','diagnostics/remizov_r0.py','diagnostics/mc_pilot_p2.py','diagnostics/chernov_batch_control.py','diagnostics/mc_production_x0.py','diagnostics/joint_benchmark_pilot.py','diagnostics/run_chernov_batch.py','ml/selector_labels.py']
    pins={n:digest(ROOT/n) for n in sources}
    pins.update({j['reference_path']:digest(ROOT/j['reference_path']) for j in jobs})
    pins.update({str((data/n).relative_to(ROOT)):h for n,h in contract['dataset_sha256'].items()})
    pins.update({str(f.relative_to(ROOT)):digest(f) for f in CONTRACT.iterdir() if f.is_file()})
    pins.update({str((R0/n).relative_to(ROOT)):digest(R0/n) for n in ('NEXT_STEP.json','labels_r256_pilot40.json','selected_labels_1000.json')})
    for name in sources:
        dest=out/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    scenario=proposal['scenario_method_seconds'];cap=math.ceil(2*scenario+300)
    plan=dict(schema='F0047_matched_timing_v1',execution_authorized=True,created_utc=utc(),authorization='Current user authorizes exactly4 R256 +4 MC1 on AMV1_F0047_S00',
        equation_id=PID,expression_hash=m['expression_hash'],family_id=m['family_id'],split=m['split'],data_dir=contract['data_dir'],target_x=proposal['target_x'],jobs=jobs,
        protected_sha256=pins,scenario_method_seconds=scenario,budget_rationale='ceil(2*saved124.274862scenario+300); not a bound/CI; no numerical smoke calls',
        resources=dict(compute_workers=1,rss_mib=512.,series_seconds=cap,owner_lease_seconds=300.,startup_seconds=90.,poll_seconds=.05,monitor_seconds=.25,thermal_seconds=10.,thermal_stop_at=2),
        order='episode0..3; within episode SHA256(ID|episode)%2:0 R/MC,1 MC/R; contract precedence over illustrative proposal order',
        timing=dict(clock='perf_counter_ns',scope='warm_end_to_end_method_wall_v1',R='certify_G + auto_T + config + constructor + solve',MC='prepare_all_methods_mc(AST/G) + config + optimized_solve(auto_T/RNG)',excludes=['shared_imports','shared_loading','JIT_warmup','exact_scoring','replay_validation','serialization']),
        total_calls=8,R_calls=4,MC_calls=4,no_other_equations=True,no_benchmark1000=True,no_ML=True,
        repeat_policy='replay same4 semantic MC seeds; no adaptive extra repeats; completed immutable; interrupted calls require recorded new attempt within remaining budget')
    write_json(out/'plan.json',plan,exclusive=True)
    write_json(out/'preflight.json',dict(passed=True,equation_id=PID,expression_hash=m['expression_hash'],family_id=m['family_id'],split=m['split'],G=G,T=T,grid_points=33,R_work_grid_points=2081,MC_steps=jobs[0]['expected_steps'] if jobs[0]['profile']=='MC1' else jobs[1]['expected_steps'],independent_MC_seeds=[248800010100+i for i in range(4)],existing_references_verified=5,discrepancies=[]),exclusive=True)
    write_json(out/'environment.json',dict(platform=platform.platform(),machine=platform.machine(),processor=platform.processor(),cpu_count=os.cpu_count(),python=sys.version,executable=sys.executable,
        versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','sympy','numba','llvmlite')},thread_environment={n:os.environ[n] for n in THREADS},thermal=sensor,captured_utc=utc()),exclusive=True)
    power=dict(captured_utc=utc(),before_first_call=True)
    for name,cmd in [('settings',['pmset','-g','custom']),('source',['pmset','-g','batt'])]:
        q=subprocess.run(cmd,capture_output=True,text=True);power[name]=dict(returncode=q.returncode,stdout=q.stdout,stderr=q.stderr)
    write_json(out/'hardware_power_before.json',power,exclusive=True)
    print(json.dumps(dict(preflight='passed',calls=8,scenario_seconds=scenario,cap_seconds=cap,order=[j['profile']+str(j['episode']) for j in jobs])),flush=True)


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
        imports=time.perf_counter()-t;t=time.perf_counter();data=ROOT/plan['data_dir']
        e=next(e for e in load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl') if e.equation_id==PID)
        loading=time.perf_counter()-t;t=time.perf_counter();jit=compile_backend();warm=time.perf_counter()-t
        import numba
        assert numba.get_num_threads()==1
        write_json(state_path.with_suffix('.warmup.json'),dict(shared_import_seconds=imports,shared_load_seconds=loading,jit=jit,warmup_wall_seconds=warm,numba_threads=numba.get_num_threads(),extra_solver_calls=0,no_RNG_in_warmup=True),exclusive=True)
        lookup={j['key']:j for j in plan['jobs']};state.update(phase='ready');write_json(state_path,dict(state,monotonic=time.monotonic(),peak_rss_mib=rss()),durable=False)
        for line in sys.stdin:
            request=read_json(Path(json.loads(line)['request_path']));job=request['job'];assert job==lookup[job['key']]
            assert e.metadata['expression_hash']==job['expression_hash'] and e.equation_id==PID
            state.update(phase='solving',key=job['key'])
            r=dict(job=job,plan_sha256=request['plan_sha256'],session=request['session'],equation_id=PID,expression_hash=e.metadata['expression_hash'],family_id=e.metadata['family_id'],split='train',profile=job['profile'],config=job['config'],seed=job['config'].get('seed'),started_utc=utc(),worker_pid=os.getpid(),solver_called=False)
            try:
                if job['profile']=='R256':
                    class CountedSolver:
                        def __init__(self,*a,**kw):self.solver=ChernovSolver(*a,**kw)
                        def solve(self,x):r['solver_called']=True;return self.solver.solve(x)
                    cert,T,z,wall,t0,t1=measured_r_call(e,job,certify_all_methods_forcing,auto_T,ChernovConfig,CountedSolver)
                    assert z.n==256 and z.scheme=='solve' and z.T==job['expected_T']
                    assert z.n_operator_applications==131072 and z.n_spline_builds==131073
                    assert z.time_nodes==513 and z.work_grid_points==2081 and z.actual_dx==.0125
                else:
                    def counted_solve(*a,**kw):r['solver_called']=True;return solve_monte_carlo_optimized(*a,**kw)
                    prepared,z,wall,t0,t1=measured_call(e.metadata,job,prepare_all_methods_mc,MCConfig,counted_solve)
                    validate_result(job,prepared,z)
                r.update(result=asdict(z),method_wall_seconds=wall,timer_start_ns=t0,timer_stop_ns=t1,status='completed' if z.success else 'solver_failure')
                assert z.success and z.failure_reason=='' and np.isfinite(z.f).all() and z.f.shape==(33,)
                np.testing.assert_array_equal(z.x,job['target_x'])
                score_start=time.perf_counter_ns();exact=e.y_exact(np.asarray(job['target_x']));error=float(np.max(abs(z.f-exact)))
                r.update(y_exact=exact,E_grid=error,accuracy_met=error<=.01,scoring_seconds=(time.perf_counter_ns()-score_start)*1e-9)
                ref=read_json(ROOT/job['reference_path']);r['reference_sha256']=digest(ROOT/job['reference_path'])
                r['array_identical_to_reference']=bool(np.array_equal(z.f,ref['result']['f']))
                assert r['array_identical_to_reference'],'replay changed numerical output'
            except Exception as exc:
                r.update(status='invalid_result' if isinstance(exc,AssertionError) else 'worker_exception',traceback=traceback.format_exc())
                r.setdefault('E_grid',None);r.setdefault('accuracy_met',None)
                timer=getattr(exc,'r0_timer',getattr(exc,'p1_timer',None))
                if timer:
                    t0,t1=timer;r.update(method_wall_seconds=(t1-t0)*1e-9,timer_start_ns=t0,timer_stop_ns=t1)
            r.update(finished_utc=utc(),peak_rss_mib=rss());write_json(out/'records'/f"{job['key']}.json",r,exclusive=True);state.update(phase='ready',key=None)
    finally:stop.set();thread.join()

def run(out):
    plan=read_json(out/'plan.json');ph=digest(out/'plan.json');jobs=plan['jobs'];limits=plan['resources']
    assert plan['execution_authorized'] and len(jobs)==8 and all(j['equation_id']==PID and j['profile'] in ('R256','MC1') for j in jobs)
    for name,h in plan['protected_sha256'].items():assert digest(ROOT/name)==h,name
    done={p.stem:read_json(p) for p in (out/'records').glob('*.json')}
    assert set(done)<=set(j['key'] for j in jobs)
    for key,r in done.items():
        assert r['job']==next(j for j in jobs if j['key']==key) and r['plan_sha256']==ph,'saved result does not match plan'
    if len(done)==8:
        print(json.dumps(dict(new_calls=0,worker_started=False,reason='already_complete')),flush=True);return
    # Do not retry an interrupted physical invocation under an8-call ceiling.
    dispatched={read_json(p)['job']['key'] for p in (out/'events').glob('*.request.json')}
    pending=[j for j in jobs if j['key'] not in done and j['key'] not in dispatched]
    with controller_locks(out) as fds:
        if not (out/'budget.json').exists():
            start=time.time();write_json(out/'budget.json',dict(started_epoch=start,deadline_epoch=start+limits['series_seconds'],started_utc=utc()),exclusive=True)
        deadline=read_json(out/'budget.json')['deadline_epoch'];session='f0047_'+uuid.uuid4().hex[:10]
        active=new_control(out,session,lease_seconds=300.);control=Path(active['control_dir'])
        began=time.perf_counter();child=None;log=None;state_path=None;reason=None;new=0;last_thermal=0.;sensor={}
        def checkpoint():
            write_json(out/'progress.json',dict(session=session,terminal=len(done),pending=8-len(done),
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
                assert len(dispatched)<=8
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
