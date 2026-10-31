"""Fixed final140: explicit start approval, guardian, immutable reserved slots."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnostics.confirm_f0047_timing import *
from diagnostics.final140_protocol import validate_scope

TERMINAL={'completed','solver_failure','interrupted_unknown','external_time_limit','external_memory_limit','worker_exception','invalid_result'}


def resume_state(plan,records,requests,budget,plan_sha,now_epoch,now_monotonic):
    lookup={j['key']:j for j in plan['jobs']};requested={}
    for r in requests:
        key=r['job']['key']
        if key in requested or r['job']!=lookup.get(key) or r['plan_sha256']!=plan_sha:raise ValueError('duplicate/mismatched reserved request')
        requested[key]=r
    if len(requested)>140:raise ValueError('physical call cap exceeded')
    for key,r in records.items():
        if key not in requested or r['job']!=lookup.get(key) or r['plan_sha256']!=plan_sha or r['status'] not in TERMINAL:raise ValueError('saved record/request/plan mismatch')
    if budget:
        if budget['plan_sha256']!=plan_sha or budget['approved_cap_seconds']!=7464 or budget['maximum_physical_calls']!=140:raise ValueError('wrong persistent budget')
        if abs(budget['deadline_epoch']-budget['started_epoch']-7464)>1e-6 or abs(budget['deadline_monotonic']-budget['started_monotonic']-7464)>1e-6:raise ValueError('deadline reset or extension')
    return dict(requested=len(requested),records=len(records),orphan_keys=sorted(set(requested)-set(records)),
        pending_keys=[j['key'] for j in plan['jobs'] if j['key'] not in requested],
        expired=bool(budget and (now_epoch>=budget['deadline_epoch'] or now_monotonic>=budget['deadline_monotonic'] or now_monotonic<budget['started_monotonic'])),
        critical=any(r['status'] in ('invalid_result','worker_exception') for r in records.values()))


def collect(out,plan):
    records={p.stem:read_json(p) for p in (out/'records').glob('*.json')}
    requests=[read_json(p) for p in sorted((out/'events').glob('*.request.json'))]
    budget=read_json(out/'budget.json') if (out/'budget.json').exists() else None
    if requests and budget is None:raise ValueError('original budget missing; refusing deadline reset')
    state=resume_state(plan,records,requests,budget,digest(out/'plan.json'),time.time(),time.monotonic())
    return records,requests,budget,state


def check_authorization(out):
    file=out/'execution_authorization.json'
    if not file.exists():raise PermissionError('Separate start authorization is absent; preparation is not execution permission')
    a=read_json(file)
    if a.get('approved') is not True or a.get('plan_sha256')!=digest(out/'plan.json') or a.get('maximum_calls')!=140 or a.get('series_seconds')!=7464 or not a.get('source_instruction'):
        raise PermissionError('Start authorization does not match the frozen140-call plan')
    return a


def preflight(out):
    plan=read_json(out/'plan.json');validate_scope(plan)
    for n,h in plan['protected_sha256'].items():assert digest(ROOT/n)==h,n
    pred=read_json(out/'frozen_predictions.json');assert pred['choices']=={'R128':21,'R256':39}
    assert all(r['prediction']['selected_profile'] in ('R128','R256') for r in pred['rows'])
    identity=read_json(out/'independence.json');assert not set(identity['excluded_topologies'])&{c['structure_signature'] for c in plan['cases']}
    budget=read_json(out/'budget_proposal.json');assert budget['series_seconds']==7464 and budget['maximum_calls']==140 and not budget['execution_authorized']
    for n in ('.chernov_batch_controller.lock','.chernov_batch_compute.lock'):assert not lock_is_held(ROOT/'results'/n),'active compute/controller'
    if (out/'active.json').exists():
        active=read_json(out/'active.json');guards=[read_json(p) for p in Path(active['control_dir']).glob('*.guardian.json')]
        assert not guards or all(g.get('finished') and g.get('cleanup_success') for g in guards),'guardian cleanup incomplete'
    sensor=thermal();assert sensor['thermal_state'] is not None and sensor['thermal_state']<2
    assert shutil.disk_usage(ROOT).free>=512*1024**2
    from all_methods_dataset_loader import load_all_methods_equations
    from all_methods_bounds import certify_all_methods_forcing
    from mc_solver import auto_T
    data=ROOT/plan['data_dir'];eqs={e.equation_id:e for e in load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl')};assert len(eqs)==60
    for c in plan['cases']:
        e=eqs[c['equation_id']];p=e.remizov_problem;assert e.metadata['split']=='test' and e.metadata['expression_hash']==c['expression_hash']
        G=certify_all_methods_forcing(p,e.metadata).forcing_bound.abs_upper_bound;T=auto_T(p.lambda_value,p.c_abs_upper_bound,G,1e-4)
        assert G==c['G'] and T==c['T'] and math.ceil(128*T)==c['N']
    records,requests,persist,state=collect(out,plan)
    source=ROOT/plan['source_plan'];assert not list((source.parent/'records').glob('*.json')) and not list((source.parent/'events').glob('*.request.json')),'superseded720 test already started'
    for root in (ROOT/'results',ROOT/'reports'):
        for path in list(root.rglob('records/*.json'))+list(root.rglob('*.request.json')):
            if out in path.parents:continue
            r=read_json(path);j=r.get('job',{});assert j.get('config',r.get('config',{})).get('seed')!=plan['MC_seed'],'MC entropy already reserved elsewhere'
    approved=(out/'execution_authorization.json').exists()
    if approved:check_authorization(out)
    return dict(passed=True,execution_authorized=approved,families=60,MC_families=20,maximum_calls=140,cap_seconds=7464,
        state=state,thermal=sensor,new_solver_calls=0,worker_started=False,can_start_after_authorization=not state['expired'] and not state['critical'])


def reconcile_orphans(out,plan,records,requests):
    """Called only after cleanup and controller-lock acquisition; no repeat solve."""
    ph=digest(out/'plan.json');state=resume_state(plan,records,requests,None,ph,time.time(),time.monotonic());lookup={j['key']:j for j in plan['jobs']}
    for key in state['orphan_keys']:
        j=lookup[key];request=next(r for r in requests if r['job']['key']==key)
        events=list((out/'events').glob(f"*{key}.interrupted.json"));event=read_json(events[-1]) if events else {}
        record=dict(job=j,plan_sha256=ph,session=request['session'],equation_id=j['equation_id'],identity=j['identity'],profile=j['profile'],
            expression_hash=j['expression_hash'],family_id=j['family_id'],split='test',config=j['config'],seed=j['config'].get('seed'),status='interrupted_unknown',
            result=None,E_grid=None,accuracy_met=None,method_wall_seconds=None,external_elapsed_seconds=event.get('wall_seconds'),
            peak_rss_mib=event.get('peak_rss_mib'),solver_called=None,call_slot_reserved=True,reconciled_without_retry=True,
            reason='request persisted but no durable result; numerical accuracy/time unknown',finished_utc=utc())
        write_json(out/'records'/f'{key}.json',record,exclusive=True);records[key]=record
        write_json(out/'reconciliations'/f'{key}.json',dict(request=request,interruption_event=event,reason='reserved slot never redispatched'),exclusive=True)
    return state['orphan_keys']


def execute(out,authorization=None):
    preflight(out)
    if authorization is not None:
        a=read_json(authorization)
        if a.get('approved') is not True or a.get('plan_sha256')!=digest(out/'plan.json') or a.get('maximum_calls')!=140 or a.get('series_seconds')!=7464 or not a.get('source_instruction'):
            raise PermissionError('Explicit start approval for this140-call/7464s plan required')
        target=out/'execution_authorization.json'
        if target.exists():assert read_json(target)==a,'cannot replace existing approval'
        else:write_json(target,a,exclusive=True)
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
    with controller_locks(out) as fds:
        done,requests,budget,state=collect(out,plan)
        def finish_without_worker(reason,reconciled=()):
            value=dict(new_calls=0,worker_started=False,reason=reason,reconciled=list(reconciled),terminal=len(done),requested=len(requests))
            write_json(out/'resume_summary.json',value)
            write_json(out/'pending.json',dict(keys=[j['key'] for j in jobs if j['key'] not in done],reason=reason))
            print(json.dumps(value),flush=True)
        if state['critical']:
            finish_without_worker('critical_saved_result_requires_review');return
        reconciled=reconcile_orphans(out,plan,done,requests)
        state=resume_state(plan,done,requests,budget,ph,time.time(),time.monotonic())
        if len(done)==140:
            finish_without_worker('all_slots_accounted',reconciled);return
        if state['expired']:
            finish_without_worker('original_budget_exhausted',reconciled);return
        dispatched={r['job']['key'] for r in requests}
        pending=[j for j in jobs if j['key'] in state['pending_keys']]
        if not pending:
            finish_without_worker('no_unrequested_jobs');return
        if not (out/'budget.json').exists():
            start=time.time();startmono=time.monotonic();write_json(out/'budget.json',dict(plan_sha256=ph,started_epoch=start,deadline_epoch=start+limits['series_seconds'],started_monotonic=startmono,deadline_monotonic=startmono+limits['series_seconds'],started_utc=utc(),approved_cap_seconds=limits['series_seconds'],maximum_physical_calls=140),exclusive=True)
        budget=read_json(out/'budget.json');deadline=budget['deadline_epoch'];deadline_mono=budget['deadline_monotonic'];session='final140_'+uuid.uuid4().hex[:10]
        active=new_control(out,session,lease_seconds=300.);control=Path(active['control_dir'])
        began=time.perf_counter();child=None;log=None;state_path=None;reason=None;new=0;last_thermal=0.;sensor={}
        def checkpoint():
            write_json(out/'progress.json',dict(session=session,terminal=len(done),pending=140-len(done),
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
