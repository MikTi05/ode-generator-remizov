"""Frozen train-only grid100 experiment, one worker, absolute 4h budget."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
from datetime import datetime,timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import threading
import time
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')
for n in THREADS:os.environ[n]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from diagnostics.run_chernov_batch import write_json,read_json,digest
from diagnostics.chernov_batch_control import controller_locks,new_control,finish_control,GuardedProcess
from diagnostics.mc_production_x0 import wait,stop_child,rss


def utc():return datetime.now(timezone.utc).isoformat()


def initialize(out):
    import numpy as np
    from all_methods_dataset_loader import load_all_methods_equations
    from mc_optimized import prepare_all_methods_mc
    from mc_solver import MCConfig,auto_T
    from chernov_solver import ChernovConfig
    from diagnostics.plan_joint_benchmark import protected_snapshot
    d=ROOT/'data/all_methods_v1_1000_2026-10-18'
    # Selection uses metadata, and historical TRAIN data only.
    meta=[json.loads(s) for s in (d/'all_methods_equations_v1_metadata.jsonl').read_text().splitlines()]
    train={m['equation_id']:m for m in meta if m['split']=='train'}
    cells=defaultdict(lambda:defaultdict(list))
    for pid,m in train.items():cells[(m['complexity_level'],m['regime'])][m['family_id']].append(pid)
    rng=random.Random(202610250333);selection=[];chosen=set()
    for cell,families in sorted(cells.items()):
        for family in rng.sample(sorted(families),4):
            pid=rng.choice(sorted(families[family]));selection.append(dict(problem_id=pid,cohort='representative',reason='4 of7 families per cell,1 of5 samples',cell=list(cell)));chosen.add(pid)
    prior=ROOT/'results/chernov_solve1000_2026-10-18/solutions'
    old={pid:read_json(prior/f'{pid}.json') for pid in train}
    x0={pid:[read_json(ROOT/'results/mc1000_x0_2026-10-25/attempts'/f'{pid}_seed{i}.json') for i in range(4)] for pid in train}
    costs={pid:float(np.quantile([r['result']['runtime_seconds'] for r in rows],.95))*33 for pid,rows in x0.items()}
    def add_ranked(ranked,n,reason):
        for pid in ranked:
            if pid in chosen:continue
            chosen.add(pid);m=train[pid];selection.append(dict(problem_id=pid,cohort='stress',reason=reason,cell=[m['complexity_level'],m['regime']]))
            n-=1
            if n==0:break
        if n:raise ValueError('selection underfilled')
    add_ranked(sorted(train,key=lambda p:(-float(old[p].get('max_abs_error') or -1),p)),10,'largest saved TRAIN Chernov33 error, historical taper')
    add_ranked(sorted(train,key=lambda p:(-costs[p],p)),5,'highest TRAIN x0-based projected grid runtime')
    def nodes(node):return 1+sum(nodes(c) for c in node['children'])
    complexity={pid:sum(nodes(train[pid]['serialized_ast'][k]) for k in ('a_raw','b_raw','c_raw','y_shape')) for pid in train}
    add_ranked(sorted(train,key=lambda p:(-complexity[p],p)),5,'largest input AST node count')
    assert len(selection)==100 and len(chosen)==100 and all(p in train for p in chosen)
    # Safe loading verifies dataset identity; does not select numerical settings.
    expected=read_json(ROOT/'results/dataset1000_2026-10-18/artifacts_sha256.json')
    if any(digest(d/n)!=h for n,h in expected.items()):raise ValueError('dataset changed')
    eqs={e.equation_id:e for e in load_all_methods_equations(d/'all_methods_equations_v1.csv',d/'all_methods_equations_v1_metadata.jsonl')}
    base=read_json(ROOT/'results/chernov_solve1000_2026-10-18/plan.json')['config']
    rc=dict(base,boundary='zero',T=None,tail_eps=1e-4)
    ChernovConfig(**rc).validate()
    mc=asdict(MCConfig(M=1024,chunk=1024,dt=1/128,T=None,tail_eps=1e-4,seed=None,
        max_steps=100000,max_path_steps=600000000,max_batch_paths=1024,max_runtime_seconds=None))
    seeds=[202610250330+i for i in range(4)];tasks=[];jobs=[]
    controls=['AMV1_F0141_S00','AMV1_F0184_S01']
    for pid in controls:
        assert pid in train
        for boundary in ('taper','zero'):
            reference=ROOT/'results/chernov_boundary_diagnosis_2026-10-24/runs'/f'{pid}_cubic_{boundary}.json'
            jobs.append(dict(key=f'control_{pid}_{boundary}',problem_id=pid,method='R',stage='control',
                config=dict(base,boundary=boundary),watchdog_seconds=180.,reference=str(reference.relative_to(ROOT)),reference_sha256=digest(reference)))
    for idx,s in enumerate(selection):
        pid=s['problem_id'];p=prepare_all_methods_mc(eqs[pid].metadata)
        T=auto_T(p.problem.lambda_value,p.problem.c_abs_upper_bound,p.forcing_bound.abs_upper_bound,1e-4);N=math.ceil(T*128)
        task=dict(s,family_id=train[pid]['family_id'],split='train',expression_hash=train[pid]['expression_hash'],
            T=T,G=p.forcing_bound.abs_upper_bound,N=N,mc_path_steps=33*1024*N,
            mc_projected_seconds=costs[pid],mc_watchdog_seconds=max(180.,6*costs[pid]+15),
            r_watchdog_seconds=max(120.,6*float(old[pid].get('task_wall_seconds') or 20.)))
        tasks.append(task)
        rj=dict(key=f'{pid}_R',problem_id=pid,method='R',stage='primary',config=dict(rc,T=T),watchdog_seconds=task['r_watchdog_seconds'])
        mj=dict(key=f'{pid}_MC0',problem_id=pid,method='MC',seed_index=0,stage='primary',
            config=dict(mc,seed=seeds[0],max_runtime_seconds=task['mc_watchdog_seconds']),watchdog_seconds=task['mc_watchdog_seconds'])
        jobs.extend([rj,mj] if idx%2==0 else [mj,rj])
    sources=['diagnostics/joint_grid100.py','diagnostics/grid100_statistics.py','diagnostics/mc_production_x0.py',
        'diagnostics/run_chernov_batch.py','diagnostics/chernov_batch_control.py','chernov_solver.py','remizov_solver.py',
        'mc_optimized.py','mc_ast_kernels.py','mc_solver.py','all_methods_bounds.py','all_methods_dataset_loader.py','src/all_methods_expressions.py','src/all_methods_dataset_v1.py']
    hashes={n:digest(ROOT/n) for n in sources};hashes.update({str((d/n).relative_to(ROOT)):h for n,h in expected.items()})
    plan=dict(schema='joint_grid100_train_v1',created_utc=utc(),selection_seed=202610250333,selection=tasks,jobs=jobs,
        data_dir=str(d.relative_to(ROOT)),target_x=np.linspace(-1,1,33).tolist(),threshold=.01,
        r_profile='cubic_zero_n128_Q513_dx.0125_padding12_common_certifiedT',r_base_config=rc,mc_base_config=mc,seeds=seeds,
        resources=dict(series_seconds=14400.,rss_mib=512.,startup_seconds=90.,poll_seconds=.05,monitor_seconds=.25,owner_lease_seconds=300.,compute_workers=1),
        inference=dict(alpha=.05,maximum_looks=2,normal='Bonferroni33x2 asymptotic',hoeffding='analytic bounded discrete integrals, real arithmetic',time_bias='not estimated',winner_policy='no reliable fastest-method labels without accuracy/reliability evidence'),
        adaptive=dict(maximum_equations=12,new_seeds_each=[1,2,3],selection='six R-not-pass uncertain cases then lowest upper-error-band unresolved cases; fixed h/M/T',first_seed_cost_always_included=True),
        bias='80 representative rows from80 families,4/cell;20 outcome/cost/AST-enriched stress rows; adaptive repeat subset enriched; no validation/test tuning',
        cost_forecast=dict(primary_mc_seconds=sum(t['mc_projected_seconds'] for t in tasks),R_seconds_scenario=1200.,
            max_additional_mc_seconds_scenario=3*sum(sorted((t['mc_projected_seconds'] for t in tasks),reverse=True)[:12]),
            primary_path_steps=sum(t['mc_path_steps'] for t in tasks)),protected_sha256=hashes)
    out.mkdir(parents=True,exist_ok=False)
    for name in ('records','events','workers','sessions','sources'):(out/name).mkdir()
    shutil.copy2(ROOT/'PROJECT_STATUS.md',out/'PROJECT_STATUS.before.md')
    write_json(out/'protected_before.json',protected_snapshot(out),exclusive=True)
    write_json(out/'plan.json',plan,exclusive=True)
    for n in sources:
        dst=out/'sources'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dst)
    write_json(out/'environment.json',dict(python=sys.version,versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','numba','llvmlite')},threads={k:os.environ[k] for k in THREADS}),exclusive=True)
    print(json.dumps(dict(selection=100,cohorts=dict(Counter(t['cohort'] for t in tasks)),families=len({t['family_id'] for t in tasks}),cost=plan['cost_forecast'])),flush=True)


def worker(out,state_path,parent_pid):
    plan=read_json(out/'plan.json');stop=threading.Event();state=dict(pid=os.getpid(),phase='loading',key=None)
    def monitor():
        with state_path.with_suffix('.rss.jsonl').open('x') as f:
            while not stop.is_set():
                if os.getppid()!=parent_pid:os._exit(75)
                sample=dict(state,monotonic=time.monotonic(),peak_rss_mib=rss());write_json(state_path,sample,durable=False)
                f.write(json.dumps(sample)+'\n');f.flush();stop.wait(.25)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        import numpy as np
        from all_methods_dataset_loader import load_all_methods_equations
        from mc_optimized import prepare_all_methods_mc,compile_backend,solve_monte_carlo_optimized
        from mc_solver import MCConfig
        from chernov_solver import ChernovConfig,ChernovSolver
        d=ROOT/plan['data_dir'];t=time.perf_counter()
        eqs={e.equation_id:e for e in load_all_methods_equations(d/'all_methods_equations_v1.csv',d/'all_methods_equations_v1_metadata.jsonl')}
        loading=time.perf_counter()-t;jit=compile_backend();state.update(phase='ready',load_seconds=loading,compilation=jit)
        write_json(state_path,dict(state,monotonic=time.monotonic(),peak_rss_mib=rss()),durable=False)
        for line in sys.stdin:
            req=read_json(Path(json.loads(line)['request_path']));job=req['job'];pid=job['problem_id'];state.update(phase='solving',key=job['key'])
            began=time.perf_counter();r=dict(job=job,plan_sha256=req['plan_sha256'],session=req['session'],started_utc=utc(),worker_pid=os.getpid(),load_seconds_once=loading,compilation_once=jit)
            try:
                e=eqs[pid]
                if job['method']=='R':result=ChernovSolver(e.remizov_problem,ChernovConfig(**job['config'])).solve(plan['target_x'])
                else:
                    p=prepare_all_methods_mc(e.metadata)
                    result=solve_monte_carlo_optimized(p,plan['target_x'],MCConfig(**job['config']),engine='numba',block_steps=64)
                # Exact is used for metrics only after the numerical return.
                exact=e.y_exact(np.asarray(plan['target_x']));error=float(np.max(abs(result.f-exact))) if result.success else None
                status='completed' if result.success else ('external_time_limit' if result.failure_reason=='time_limit_exceeded' else 'solver_failure')
                r.update(status=status,result=asdict(result),y_exact=exact,E_inf=error,accuracy_met=error<=.01 if error is not None else None,
                    expression_hash=e.metadata['expression_hash'],family_id=e.metadata['family_id'],split=e.metadata['split'])
            except Exception:r.update(status='worker_exception',result=None,E_inf=None,traceback=traceback.format_exc())
            elapsed=time.perf_counter()-began
            r.update(finished_utc=utc(),task_wall_seconds=elapsed,peak_rss_mib=rss())
            if elapsed>job['watchdog_seconds']:r['status']='external_time_limit'
            if rss()>512:r['status']='external_memory_limit'
            write_json(out/'records'/f"{job['key']}.json",r,exclusive=True);state.update(phase='ready',key=None)
    finally:stop.set();thread.join()


def controls_pass(out,plan):
    import numpy as np
    for j in plan['jobs'][:4]:
        path=out/'records'/f"{j['key']}.json"
        if not path.exists():return False
        r=read_json(path)
        if r['status']!='completed':return False
        ref=ROOT/j['reference'];assert digest(ref)==j['reference_sha256']
        old=read_json(ref)
        expected=np.asarray(old['result']['f']);got=np.asarray(r['result']['f'])
        if np.max(abs(got-expected))>1e-10*max(1.,float(np.max(abs(expected)))):return False
    return True


def primary_rows(out,plan):
    from diagnostics.grid100_statistics import aggregate_mc
    rows=[]
    for task in plan['selection']:
        pid=task['problem_id'];rp=out/'records'/f'{pid}_R.json';mp=out/'records'/f'{pid}_MC0.json'
        if not rp.exists() or not mp.exists():continue
        r,m=read_json(rp),read_json(mp);ok=r['status']==m['status']=='completed'
        row=dict(problem_id=pid,both_completed=ok,R_pass=r.get('accuracy_met'))
        if ok:row['mc']=aggregate_mc([m])
        rows.append(row)
    return rows


def run(out):
    plan=read_json(out/'plan.json');ph=digest(out/'plan.json');limits=plan['resources']
    for n,h in plan['protected_sha256'].items():
        if digest(ROOT/n)!=h:raise ValueError(f'changed source/data: {n}')
    records={p.stem:read_json(p) for p in (out/'records').glob('*.json')}
    if any(r['plan_sha256']!=ph for r in records.values()):raise ValueError('record plan mismatch')
    with controller_locks(out) as fds:
        if not (out/'budget.json').exists():
            now=time.time();write_json(out/'budget.json',dict(started_epoch=now,deadline_epoch=now+14400.,started_utc=utc()),exclusive=True)
        deadline=read_json(out/'budget.json')['deadline_epoch'];session='grid_'+uuid.uuid4().hex[:10]
        active=new_control(out,session,lease_seconds=300.);control=Path(active['control_dir']);began=time.perf_counter();child=None;log=None;reason=None
        jobs=list(plan['jobs']);new=0
        old_handlers={s:signal.signal(s,lambda *_:write_json(control/'stop.json',dict(utc=utc()))) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
        def checkpoint():
            pairs=sum(all(f'{t["problem_id"]}_{m}' in records for m in ('R','MC0')) for t in plan['selection'])
            write_json(out/'progress.json',dict(session=session,terminal=len(records),primary_pairs_terminal=pairs,
                planned_so_far=len(jobs),remaining_seconds=max(0.,deadline-time.time()),session_seconds=time.perf_counter()-began,
                outcomes=dict(Counter(r['status'] for r in records.values())),updated_utc=utc()))
        try:
            pos=0;checkpoint()
            while True:
                if pos==4:
                    passed=controls_pass(out,plan)
                    if not (out/'profile_check.json').exists():write_json(out/'profile_check.json',dict(passed=passed,controls=4,used_train_only=True),exclusive=True)
                    if not passed:reason='profile_control_failed';break
                if pos==len(jobs):
                    if (out/'adaptive_selection.json').exists():extra=read_json(out/'adaptive_selection.json')['jobs']
                    else:
                        from diagnostics.grid100_statistics import adaptive_ids
                        ids=adaptive_ids(primary_rows(out,plan),12);extra=[]
                        for pid in ids:
                            base=next(j for j in plan['jobs'] if j['key']==pid+'_MC0')
                            for i in (1,2,3):extra.append(dict(base,key=f'{pid}_MC{i}',seed_index=i,stage='repeat',config=dict(base['config'],seed=plan['seeds'][i])))
                        write_json(out/'adaptive_selection.json',dict(selected=ids,jobs=extra,created_utc=utc(),rule=plan['adaptive'],primary_sha256={k:digest(out/'records'/f'{k}.json') for k in records if not k.startswith('control_')}),exclusive=True)
                    if len(jobs)>len(plan['jobs']) or not extra:break
                    jobs.extend(extra);checkpoint()
                j=jobs[pos];pos+=1
                if j['key'] in records:continue
                if time.time()>=deadline:reason='series_budget_exhausted';break
                if (control/'stop.json').exists():reason='interrupted';break
                if child is None or child.poll() is not None:
                    if log:log.close()
                    token=session+'_'+uuid.uuid4().hex[:6];state=out/'workers'/f'{token}.json';log=(out/'workers'/f'{token}.log').open('x')
                    child=GuardedProcess([sys.executable,'-B',str(Path(__file__).resolve()),'worker','--out',str(out),'--state',str(state),'--parent-pid','{GUARD_PID}'],log,control/f'{token}.guardian.json',active,fds)
                    startup,peak=wait(child,None,state,time.monotonic()+90.,limits,deadline,control/'stop.json')
                    if startup:reason='startup_'+startup;break
                rp=out/'records'/f"{j['key']}.json";req=out/'events'/f"{session}_{j['key']}.request.json"
                write_json(req,dict(job=j,session=session,plan_sha256=ph),exclusive=True)
                t=time.perf_counter();child.stdin.write(json.dumps(dict(request_path=str(req.resolve())))+'\n');child.stdin.flush()
                outcome,peak=wait(child,rp,state,time.monotonic()+j['watchdog_seconds'],limits,deadline,control/'stop.json')
                if outcome:
                    stop_child(child)
                    if rp.exists():outcome=None
                if outcome in ('interrupted','series_budget_exhausted','telemetry_stale'):
                    reason=outcome;write_json(out/'events'/f"{session}_{j['key']}.interrupted.json",dict(reason=outcome,wall_seconds=time.perf_counter()-t,peak_rss_mib=peak),exclusive=True);break
                if outcome:write_json(rp,dict(job=j,plan_sha256=ph,session=session,status=outcome,result=None,E_inf=None,accuracy_met=None,task_wall_seconds=time.perf_counter()-t,peak_rss_mib=peak,finished_utc=utc()),exclusive=True)
                records[j['key']]=read_json(rp);new+=1;checkpoint()
                print(json.dumps(dict(new=new,key=j['key'],status=records[j['key']]['status'],seconds=records[j['key']]['task_wall_seconds'])),flush=True)
        finally:
            stop_child(child)
            if log:log.close()
            for sig,handler in old_handlers.items():signal.signal(sig,handler)
            write_json(out/'sessions'/f'{session}.json',dict(session=session,started_previously=len(records)-new,new_terminal=new,reason=reason,wall_seconds=time.perf_counter()-began,finished_utc=utc()),exclusive=True)
            checkpoint();finish_control(out,active)
            write_json(out/'pending.json',dict(keys=[j['key'] for j in jobs if j['key'] not in records],adaptive_decision_pending=not (out/'adaptive_selection.json').exists(),reason=reason))
        print(json.dumps(dict(new_terminal=new,reason=reason)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=('init','run','worker'));ap.add_argument('--out',type=Path,required=True);ap.add_argument('--state',type=Path);ap.add_argument('--parent-pid',type=int)
    a=ap.parse_args();out=a.out.resolve()
    if a.action=='init':initialize(out)
    elif a.action=='run':run(out)
    else:worker(out,a.state,a.parent_pid)
