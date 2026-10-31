"""One final720-call test. No execution without a separate matching approval."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnostics.confirm_f0047_timing import *
SEEDS=[248800100100+i for i in range(4)]


def validate_scope(plan):
    assert plan['schema']=='final_independent60_v1' and plan['execution_authorized'] is False
    jobs=plan['jobs'];assert len(jobs)==len({j['key'] for j in jobs})==720
    assert Counter(j['profile'] for j in jobs)=={'R128':240,'R256':240,'MC1':240}
    assert len(plan['cases'])==len({c['structure_signature'] for c in plan['cases']})==60
    assert len({j['identity'] for j in jobs})==60 and all(j['split']=='test' for j in jobs)
    assert plan['MC_seed_entropy']==SEEDS
    resources=plan['resources'];assert resources['compute_workers']==1 and resources['rss_mib']==512 and resources['series_seconds']>0
    contract=read_json(CONTRACT/'plan.json')
    for c in plan['cases']:
        js=[j for j in jobs if j['identity']==c['identity']]
        assert Counter((j['profile'],j['episode']) for j in js)==Counter((p,e) for p in ('R128','R256','MC1') for e in range(4))
        assert c['identity']==f"{c['dataset_sha256']}|{c['equation_id']}|{c['expression_hash']}"
        for j in js:
            assert j['equation_id']==c['equation_id'] and j['expression_hash']==c['expression_hash'] and j['family_id']==c['family_id']
            assert j['expected_T']==c['T'] and j['forcing_G']==c['G'] and j['target_x']==contract['target_x']
            cfg=dict(contract['profiles'][j['profile']]['config'])
            if j['profile']=='MC1':
                cfg.update(seed=SEEDS[j['episode']],max_path_steps=33*4096*c['N'],max_runtime_seconds=j['proposed_watchdog_seconds'])
                assert j['expected_steps']==c['N'] and j['numerical_work']==33*4096*c['N']
            else:cfg['T']=c['T'];assert j['numerical_work']==int(j['profile'][1:])*512
            assert j['config']==cfg


def preflight(out):
    plan=read_json(out/'plan.json');validate_scope(plan)
    for name,h in plan['protected_sha256'].items():assert digest(ROOT/name)==h,name
    budget=read_json(out/'budget_proposal.json');assert not budget['approved'] and budget['proposed_cap_seconds']==plan['resources']['series_seconds']
    identity=read_json(out/'independence.json');assert not set(identity['excluded_topologies'])&set(identity['selected_topologies'])
    requests=[read_json(p) for p in (out/'events').glob('*.request.json')];assert len(requests)==len({r['job']['key'] for r in requests})<=720
    for r in requests:assert r['job'] in plan['jobs'] and r['plan_sha256']==digest(out/'plan.json')
    for name in ('.chernov_batch_controller.lock','.chernov_batch_compute.lock'):assert not lock_is_held(ROOT/'results'/name)
    sensor=thermal();assert sensor['thermal_state'] is not None and sensor['thermal_state']<2,'thermal preflight failed'
    assert shutil.disk_usage(ROOT).free>=512*1024**2
    from all_methods_dataset_loader import load_all_methods_equations
    from all_methods_bounds import certify_all_methods_forcing
    from mc_solver import auto_T
    inputs=ROOT/plan['data_dir'];eqs={e.equation_id:e for e in load_all_methods_equations(inputs/'all_methods_equations_v1.csv',inputs/'all_methods_equations_v1_metadata.jsonl')}
    assert len(eqs)==60
    for c in plan['cases']:
        e=eqs[c['equation_id']];p=e.remizov_problem;assert e.metadata['split']=='test' and e.metadata['expression_hash']==c['expression_hash']
        G=certify_all_methods_forcing(p,e.metadata).forcing_bound.abs_upper_bound;T=auto_T(p.lambda_value,p.c_abs_upper_bound,G,1e-4)
        assert G==c['G'] and T==c['T'] and math.ceil(T*128)==c['N']
    authorized=bool((out/'execution_authorization.json').exists())
    if authorized:check_authorization(out)
    return dict(passed=True,execution_authorized=authorized,maximum_calls=720,families=60,proposed_cap_seconds=budget['proposed_cap_seconds'],requests=len(requests),records=len(list((out/'records').glob('*.json'))),solver_started=bool(requests),
        thermal=sensor,note='read-only preflight; separate user approval required')


def check_authorization(out):
    path=out/'execution_authorization.json'
    if not path.exists():raise PermissionError('No separate user approval: final numerical experiment must not run')
    approval=read_json(path);plan=read_json(out/'plan.json')
    if approval.get('approved') is not True or approval.get('plan_sha256')!=digest(out/'plan.json') or approval.get('maximum_calls')!=720 or approval.get('series_seconds')!=plan['resources']['series_seconds'] or not approval.get('source_instruction'):
        raise PermissionError('Approval does not match the frozen final experiment')
    return approval


def execute(out,authorization=None):
    preflight(out)
    if authorization is not None:
        a=read_json(authorization);target=out/'execution_authorization.json'
        if target.exists():assert read_json(target)==a,'cannot replace an existing authorization'
        else:
            plan=read_json(out/'plan.json')
            if a.get('approved') is not True or a.get('plan_sha256')!=digest(out/'plan.json') or a.get('maximum_calls')!=720 or a.get('series_seconds')!=plan['resources']['series_seconds'] or not a.get('source_instruction'):
                raise PermissionError('Separate user approval must match plan, call cap and budget')
            write_json(target,a,exclusive=True)
    check_authorization(out);return run_authorized(out)


def worker(out,state_path,parent_pid):
    check_authorization(out)
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
        eqs={e.equation_id:e for e in load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl') if e.equation_id in allowed};assert len(eqs)==60
        loading=time.perf_counter()-t;t=time.perf_counter();jit=compile_backend();warm=time.perf_counter()-t
        import numba
        assert numba.get_num_threads()==1
        write_json(state_path.with_suffix('.warmup.json'),dict(shared_import_seconds=imports,shared_load_seconds=loading,jit=jit,warmup_wall_seconds=warm,numba_threads=1,extra_solver_calls=0,no_RNG_in_warmup=True),exclusive=True)
        lookup={j['key']:j for j in plan['jobs']};repeats={};state.update(phase='ready');write_json(state_path,dict(state,monotonic=time.monotonic(),peak_rss_mib=rss()),durable=False)
        for line in sys.stdin:
            request=read_json(Path(json.loads(line)['request_path']));job=request['job'];assert job==lookup[job['key']]
            e=eqs[job['equation_id']];m=e.metadata;assert m['split']=='test' and m['expression_hash']==job['expression_hash']
            state.update(phase='solving',key=job['key'])
            r=dict(job=job,plan_sha256=request['plan_sha256'],session=request['session'],equation_id=e.equation_id,expression_hash=m['expression_hash'],family_id=m['family_id'],split='test',profile=job['profile'],config=job['config'],seed=job['config'].get('seed'),started_utc=utc(),worker_pid=os.getpid(),solver_called=False)
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

def run_authorized(out):
    check_authorization(out)
    plan=read_json(out/'plan.json');ph=digest(out/'plan.json');jobs=plan['jobs'];limits=plan['resources']
    validate_scope(plan)
    for name,h in plan['protected_sha256'].items():assert digest(ROOT/name)==h,name
    done={p.stem:read_json(p) for p in (out/'records').glob('*.json')}
    assert set(done)<=set(j['key'] for j in jobs)
    for key,r in done.items():
        assert r['job']==next(j for j in jobs if j['key']==key) and r['plan_sha256']==ph,'saved result does not match plan'
    if len(done)==720:
        assert all(r['status'] in ('completed','solver_failure') for r in done.values()),'incomplete or failed returns require analysis'
        print(json.dumps(dict(new_calls=0,worker_started=False,reason='already_complete')),flush=True);return
    # Do not retry an interrupted physical invocation under a720-call ceiling.
    requests=[read_json(p) for p in (out/'events').glob('*.request.json')]
    dispatched={r['job']['key'] for r in requests}
    assert len(requests)==len(dispatched)<=720
    assert all(r['job'] in jobs and r['plan_sha256']==ph for r in requests)
    pending=[j for j in jobs if j['key'] not in done and j['key'] not in dispatched]
    if not pending:
        print(json.dumps(dict(new_calls=0,worker_started=False,reason='all_missing_keys_already_requested',pending=720-len(done))),flush=True);return
    if (out/'budget.json').exists():
        old_budget=read_json(out/'budget.json')
        if time.time()>=old_budget['deadline_epoch'] or time.monotonic()>=old_budget['deadline_monotonic']:
            print(json.dumps(dict(new_calls=0,worker_started=False,reason='original_budget_exhausted')),flush=True);return
    with controller_locks(out) as fds:
        if not (out/'budget.json').exists():
            start=time.time();startmono=time.monotonic();write_json(out/'budget.json',dict(started_epoch=start,deadline_epoch=start+limits['series_seconds'],started_monotonic=startmono,deadline_monotonic=startmono+limits['series_seconds'],started_utc=utc(),approved_cap_seconds=limits['series_seconds'],maximum_physical_calls=720),exclusive=True)
        budget=read_json(out/'budget.json');deadline=budget['deadline_epoch'];deadline_mono=budget['deadline_monotonic'];session='final60_'+uuid.uuid4().hex[:10]
        active=new_control(out,session,lease_seconds=300.);control=Path(active['control_dir'])
        began=time.perf_counter();child=None;log=None;state_path=None;reason=None;new=0;last_thermal=0.;sensor={}
        def checkpoint():
            write_json(out/'progress.json',dict(session=session,terminal=len(done),pending=720-len(done),
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
                assert len(dispatched)<=720
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
                        family_id=job['family_id'],split='test',config=job['config'],seed=None,
                        plan_sha256=ph,status=outcome,result=None,E_grid=None,accuracy_met=None,accuracy_status='unknown_error',
                        method_wall_seconds=None,external_elapsed_seconds=time.perf_counter()-clock,
                        peak_rss_mib=peak,worker_exit_reason=child.stop_reason,finished_utc=utc()),exclusive=True)
                r=read_json(path);done[job['key']]=r;new+=1;checkpoint()
                print(json.dumps(dict(terminal=len(done),key=job['key'],status=r['status'],E_grid=r['E_grid'],
                    wall=r.get('method_wall_seconds'),session_seconds=time.perf_counter()-began)),flush=True)
                if r['status'] not in ('completed','solver_failure'):reason=r['status'];break
                if (r.get('result') or {}).get('failure_reason') in ('time_limit_exceeded','work_limit_exceeded'):reason='resource_censored';break
        finally:
            stop_child(child)
            if log:log.close()
            for s,handler in old.items():signal.signal(s,handler)
            write_json(out/'sessions'/f'{session}.json',dict(session=session,new_terminal=new,reason=reason,
                wall_seconds=time.perf_counter()-began,finished_utc=utc()),exclusive=True)
            checkpoint();finish_control(out,active)
            write_json(out/'pending.json',dict(keys=[j['key'] for j in jobs if j['key'] not in done],reason=reason))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=('check','run','worker'));ap.add_argument('--out',type=Path,required=True);ap.add_argument('--authorization',type=Path);ap.add_argument('--state',type=Path);ap.add_argument('--parent-pid',type=int)
    a=ap.parse_args();out=a.out.resolve()
    if a.action=='check':print(json.dumps(preflight(out)))
    elif a.action=='run':execute(out,a.authorization)
    else:worker(out,a.state,a.parent_pid)
