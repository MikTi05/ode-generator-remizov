"""Only the five frozen train tasks: R n256 and four fresh MC4096 seeds.

Reuse the already tested guarded worker, without its grid100 controller,
controls, adaptive selection or any other equation solves.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime,timezone
import importlib.metadata
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')
for n in THREADS:os.environ[n]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from diagnostics.run_chernov_batch import write_json,read_json,digest
from diagnostics.chernov_batch_control import controller_locks,new_control,finish_control,GuardedProcess
from diagnostics.mc_production_x0 import wait,stop_child

IDS=('AMV1_F0047_S00','AMV1_F0005_S02','AMV1_F0042_S00','AMV1_F0059_S04','AMV1_F0125_S04')
PRIOR=ROOT/'results/joint_grid100_2026-10-25'


def utc():return datetime.now(timezone.utc).isoformat()


def initialize(out):
    import numpy as np
    from all_methods_dataset_loader import load_all_methods_equations
    from mc_optimized import prepare_all_methods_mc
    from mc_solver import MCConfig,auto_T
    from chernov_solver import ChernovConfig
    from diagnostics.plan_joint_benchmark import protected_snapshot
    proposal=read_json(PRIOR/'next_experiment_proposal.json');previous=read_json(PRIOR/'plan.json')
    if tuple(r['problem_id'] for r in proposal['equations'])!=IDS:raise ValueError('proposal IDs changed')
    if proposal['MC_profile']['M']!=4096 or proposal['MC_profile']['h']!=1/128 or proposal['R_profile']['n']!=256:raise ValueError('proposal profiles changed')
    if proposal['resources']!={**proposal['resources'],'total_seconds':1200,'rss_mib':512,'workers':1}:raise ValueError('proposal limits changed')
    for n,h in previous['protected_sha256'].items():
        if digest(ROOT/n)!=h:raise ValueError(f'prior core/data changed: {n}')
    d=ROOT/proposal['dataset'];eqs={e.equation_id:e for e in load_all_methods_equations(d/'all_methods_equations_v1.csv',d/'all_methods_equations_v1_metadata.jsonl')}
    tasks=[];jobs=[];baselines={};seeds=proposal['MC_profile']['new_independent_seeds']
    if len(set(seeds))!=4 or set(seeds)&set(previous['seeds']):raise ValueError('fresh independent seeds required')
    for spec in proposal['equations']:
        pid=spec['problem_id'];e=eqs[pid];p=prepare_all_methods_mc(e.metadata)
        T=auto_T(p.problem.lambda_value,p.problem.c_abs_upper_bound,p.forcing_bound.abs_upper_bound,1e-4);N=math.ceil(T*128)
        if e.metadata['split']!='train' or e.metadata['expression_hash']!=spec['expression_hash'] or T!=spec['T']:raise ValueError('problem identity/horizon differs')
        path=PRIOR/'records'/f'{pid}_R.json';baseline=read_json(path);c=baseline['job']['config']
        if baseline['status']!='completed' or c['n']!=128 or c['num_t']!=513 or c['boundary']!='zero' or c['interp']!='cubic' or c['T']!=T or baseline['result']['x']!=proposal['target_x']:raise ValueError('invalid baseline profile')
        if baseline['expression_hash']!=spec['expression_hash']:raise ValueError('baseline identity mismatch')
        rc=dict(c,n=256);ChernovConfig(**rc).validate()
        base_mc=previous['mc_base_config']
        work=4096*N*33
        if work!=spec['mc_max_path_steps_per_trial']:raise ValueError('preflight work differs')
        mc=dict(base_mc,M=4096,chunk=1024,T=None,dt=1/128,tail_eps=1e-4,max_path_steps=work,max_batch_paths=1024,
            max_runtime_seconds=spec['mc_watchdog_seconds'])
        MCConfig(**dict(mc,seed=seeds[0])).validate()
        task=dict(spec,N=N,G=p.forcing_bound.abs_upper_bound,mc_path_steps_per_trial=work,
            baseline_source=str(path.relative_to(ROOT)),baseline_sha256=digest(path),R_config=rc,MC_base_config=mc)
        tasks.append(task);baselines[pid]=path
        # Fixed order, exactly five calls per equation; no accuracy-dependent stop.
        jobs.append(dict(key=pid+'_R256',problem_id=pid,method='R',stage='final_diagnostic',config=rc,watchdog_seconds=180.))
        for i,seed in enumerate(seeds):jobs.append(dict(key=pid+f'_MC{i}',problem_id=pid,method='MC',stage='final_diagnostic',seed_index=i,
            config=dict(mc,seed=seed),watchdog_seconds=spec['mc_watchdog_seconds']))
    source_names=['diagnostics/final_grid5.py','diagnostics/final_grid5_stats.py','diagnostics/joint_grid100.py',
        'diagnostics/grid100_statistics.py','diagnostics/chernov_batch_control.py','diagnostics/mc_production_x0.py',
        'diagnostics/run_chernov_batch.py','chernov_solver.py','remizov_solver.py','mc_optimized.py','mc_ast_kernels.py',
        'mc_solver.py','all_methods_bounds.py','all_methods_dataset_loader.py','src/all_methods_expressions.py','src/all_methods_dataset_v1.py']
    hashes={n:digest(ROOT/n) for n in source_names};hashes.update({n:h for n,h in previous['protected_sha256'].items() if n.startswith('data/')})
    plan=dict(schema='final_train_grid5_v1',created_utc=utc(),proposal_source=str((PRIOR/'next_experiment_proposal.json').relative_to(ROOT)),
        proposal_sha256=digest(PRIOR/'next_experiment_proposal.json'),data_dir=proposal['dataset'],target_x=proposal['target_x'],threshold=.01,
        tasks=tasks,jobs=jobs,seeds=seeds,exact_policy='metrics only after solve; no settings or seed-count adaptation',
        resources=dict(series_seconds=1200.,rss_mib=512.,compute_workers=1,startup_seconds=90.,poll_seconds=.05,monitor_seconds=.25,owner_lease_seconds=300.),
        inference=dict(family_alpha=.05,equations=5,estimators_each=5,coordinates=33,
            normal_and_hoeffding='Bonferroni5 equations x5 estimators x33 coordinates; raw point-CI separate',
            probability_interval='exact Clopper-Pearson,5 equations two-sided family95,tail=.005',
            reliability_test='H0 p(success)>=.95; reject with exact binomial p<=.01 per task;4/4 cannot confirm',
            temporal_bias='uncontrolled; no total-error certification',old_seeds_excluded=True),
        estimate_seconds=proposal['resources']['estimate_seconds'],total_new_calls=25,
        total_MC_path_steps=sum(4*t['mc_path_steps_per_trial'] for t in tasks),protected_sha256=hashes,
        no_ML=True,no_additional_equations=True,selection_scope='the5 prior targeted train cases,not representative population')
    out.mkdir(parents=True,exist_ok=False)
    for folder in ('records','events','workers','sessions','sources','baseline'):(out/folder).mkdir()
    shutil.copy2(ROOT/'PROJECT_STATUS.md',out/'PROJECT_STATUS.before.md')
    write_json(out/'protected_before.json',protected_snapshot(out),exclusive=True)
    for n in source_names:
        dst=out/'sources'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dst)
    for pid,path in baselines.items():
        shutil.copy2(path,out/'baseline'/f'{pid}_R128.json')
        for i in range(4):shutil.copy2(PRIOR/'records'/f'{pid}_MC{i}.json',out/'baseline'/f'{pid}_old_MC{i}.json')
    write_json(out/'plan.json',plan,exclusive=True)
    write_json(out/'environment.json',dict(python=sys.version,versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','numba','llvmlite')},threads={n:os.environ[n] for n in THREADS}),exclusive=True)
    print(dict(tasks=5,calls=25,expected_seconds=plan['estimate_seconds'],MC_path_steps=plan['total_MC_path_steps']),flush=True)


def run(out):
    plan=read_json(out/'plan.json');ph=digest(out/'plan.json');jobs=plan['jobs'];by_key={j['key']:j for j in jobs};limits=plan['resources']
    for n,h in plan['protected_sha256'].items():
        if digest(ROOT/n)!=h:raise ValueError(f'protected source/data changed: {n}')
    done={p.stem:read_json(p) for p in (out/'records').glob('*.json')}
    for k,r in done.items():
        if k not in by_key or r['plan_sha256']!=ph or r['job']!=by_key[k]:raise ValueError('record/profile mismatch')
    with controller_locks(out) as fds:
        if not (out/'budget.json').exists():
            now=time.time();write_json(out/'budget.json',dict(started_epoch=now,deadline_epoch=now+1200.,started_utc=utc()),exclusive=True)
        deadline=read_json(out/'budget.json')['deadline_epoch'];session='final5_'+uuid.uuid4().hex[:10]
        active=new_control(out,session,lease_seconds=300.);control=Path(active['control_dir']);child=None;log=None;new=0;reason=None;began=time.perf_counter();startups=[]
        def checkpoint():
            write_json(out/'progress.json',dict(session=session,terminal=len(done),pending=25-len(done),
                outcomes=dict(Counter(r['status'] for r in done.values())),remaining_seconds=max(0.,deadline-time.time()),updated_utc=utc()))
        old={s:signal.signal(s,lambda *_:write_json(control/'stop.json',dict(utc=utc()))) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
        try:
            checkpoint()
            for j in jobs:
                if j['key'] in done:continue
                if time.time()>=deadline:reason='series_budget_exhausted';break
                if (control/'stop.json').exists():reason='interrupted';break
                if child is None or child.poll() is not None:
                    if log:log.close()
                    token=session+'_'+uuid.uuid4().hex[:5];state=out/'workers'/f'{token}.json';log=(out/'workers'/f'{token}.log').open('x')
                    child=GuardedProcess([sys.executable,'-B',str(Path(__file__).resolve()),'worker','--out',str(out),'--state',str(state),'--parent-pid','{GUARD_PID}'],log,control/f'{token}.guardian.json',active,fds)
                    startup,peak=wait(child,None,state,time.monotonic()+90.,limits,deadline,control/'stop.json');startups.append(dict(state=str(state.relative_to(out)),status=startup or 'ready',peak_rss_mib=peak))
                    if startup:reason='startup_'+startup;break
                path=out/'records'/f"{j['key']}.json";req=out/'events'/f"{session}_{j['key']}.request.json"
                write_json(req,dict(job=j,session=session,plan_sha256=ph),exclusive=True)
                t=time.perf_counter();child.stdin.write(__import__('json').dumps(dict(request_path=str(req.resolve())))+'\n');child.stdin.flush()
                outcome,peak=wait(child,path,state,time.monotonic()+j['watchdog_seconds'],limits,deadline,control/'stop.json')
                if outcome:
                    stop_child(child)
                    if path.exists():outcome=None
                if outcome in ('interrupted','series_budget_exhausted','telemetry_stale'):
                    reason=outcome;write_json(out/'events'/f"{session}_{j['key']}.interrupted.json",dict(reason=outcome,wall_seconds=time.perf_counter()-t,peak_rss_mib=peak),exclusive=True);break
                if outcome:write_json(path,dict(job=j,plan_sha256=ph,session=session,status=outcome,result=None,E_inf=None,accuracy_met=None,task_wall_seconds=time.perf_counter()-t,peak_rss_mib=peak,finished_utc=utc()),exclusive=True)
                done[j['key']]=read_json(path);new+=1;checkpoint()
                print(dict(key=j['key'],new_terminal=new,status=done[j['key']]['status'],seconds=done[j['key']]['task_wall_seconds']),flush=True)
        finally:
            stop_child(child)
            if log:log.close()
            for sig,handler in old.items():signal.signal(sig,handler)
            write_json(out/'sessions'/f'{session}.json',dict(session=session,new_terminal=new,previously_terminal=len(done)-new,startups=startups,
                reason=reason,wall_seconds=time.perf_counter()-began,finished_utc=utc()),exclusive=True)
            checkpoint();finish_control(out,active);write_json(out/'pending.json',dict(keys=[j['key'] for j in jobs if j['key'] not in done],reason=reason))
        print(dict(new_terminal=new,reason=reason),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=('init','run','worker'));ap.add_argument('--out',type=Path,required=True);ap.add_argument('--state',type=Path);ap.add_argument('--parent-pid',type=int)
    a=ap.parse_args();out=a.out.resolve()
    if a.action=='init':initialize(out)
    elif a.action=='run':run(out)
    else:
        from diagnostics.joint_grid100 import worker
        worker(out,a.state,a.parent_pid)
