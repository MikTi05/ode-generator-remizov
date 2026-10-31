"""Contract section206 restart of one guardian-interrupted R0 key.
Original plan/results and numerical configs remain immutable; deadline is unchanged.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from diagnostics.remizov_r0 import *
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
    pending=[j for j in jobs if j['key'] not in done]
    amendment=read_json(out/'resume_amendment.json')
    assert amendment['original_plan_sha256']==ph and amendment['maximum_physical_attempts']==39
    repeats=dispatched-set(done)
    assert repeats<=set(amendment['permitted_retry_keys'])
    assert all(not (out/'records'/f'{key}.json').exists() for key in repeats)
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
                assert len(list((out/'events').glob('*.request.json')))<=39
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
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('run','worker'));ap.add_argument('--out',type=Path,required=True);ap.add_argument('--state',type=Path);ap.add_argument('--parent-pid',type=int)
    a=ap.parse_args();out=a.out.resolve()
    if a.action=='run':run(out)
    else:worker(out,a.state,a.parent_pid)
