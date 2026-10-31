"""Separate coupled-path diagnostic: unchanged production EM/trapezoid kernel.

Coarse normals are normalized sums of disjoint blocks of finest normals.
Replaying the same PCG64 stream at each level costs extra normal generation
but needs O(M) memory. Pilot and confirmation streams are independent.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from diagnostics.compare_chernov import THREAD_VARS, load_task, mc_forcing_certificate, write_json
from diagnostics.mc_convergence import digest, now, protected_hashes

for name in THREAD_VARS:
    os.environ[name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
PROBLEM_ID='AMV1_F0002_S00'
MASTER_SEED=2026102408
SOURCES=('diagnostics/mc_f0002_time_refinement.py',
         'diagnostics/report_mc_f0002_time_refinement.py',
         'tests/test_mc_coupled_refinement.py')


class CoupledNormals:
    """Each request returns sum(q finest N(0,1))/sqrt(q), independently.

    At a common seed and fixed M, q=1,2,4 traverse the identical finest
    Gaussian array by time then path. Thus sqrt(h)*normal is the sum of
    the corresponding fine Brownian increments, up to float rounding.
    """
    def __init__(self, rng, factor):
        if isinstance(factor,bool) or not isinstance(factor,int) or factor<1 or factor&(factor-1):
            raise ValueError('factor must be a positive power of two')
        self.rng=rng
        self.factor=factor
        self.normal_values_generated=0

    def standard_normal(self, n):
        z=self.rng.standard_normal(n)
        for _ in range(1,self.factor):
            z=z+self.rng.standard_normal(n)
        self.normal_values_generated+=n*self.factor
        return z if self.factor==1 else z/math.sqrt(self.factor)


def costs(M,T,hs):
    Ns=[round(T/h) for h in hs]
    if any(N*h!=T for N,h in zip(Ns,hs)):
        raise ValueError('this diagnostic requires exactly aligned time grids')
    return dict(time_steps=Ns, path_steps=M*sum(Ns),
        unique_finest_normal_values=M*max(Ns),
        normal_values_generated_with_replay=M*max(Ns)*len(Ns),
        coefficient_calls=sum(2+4*N for N in Ns),
        coefficient_values=M*sum(2+4*N for N in Ns))


def coupled_paths(problem, x, configs, bound, *, runtime_seconds=20.):
    """Return paired integrals, diagnostics and partial counters on failure.

    Single batch per level. Finest-level paths agree with the production
    solver at identical MCConfig/point; coarser levels retain its marginal law.
    No exact solution is used here.
    """
    import numpy as np
    from mc_solver import _Run, _path_contributions, _spawn_key
    from remizov_solver import _problem_conditions_valid, _validate_target_x
    if not configs:
        raise ValueError('need at least one level')
    for c in configs:
        c.validate()
    c0=configs[0]
    for c in configs:
        if (c.M,c.T,c.seed,c.stream_id,c.chunk)!=(c0.M,c0.T,c0.seed,c0.stream_id,c0.chunk):
            raise ValueError('only dt may vary between coupled levels')
        if c.chunk!=c.M or c.M>c.max_batch_paths:
            raise ValueError('one full batch per replicate is required')
    hs=[c.dt for c in configs]
    if hs!=sorted(set(hs),reverse=True):
        raise ValueError('levels must have distinct decreasing steps')
    if not _problem_conditions_valid(problem):
        raise ValueError('invalid global coefficient conditions')
    _validate_target_x(problem,np.array([x]))
    bound.validate()
    work=costs(c0.M,c0.T,hs)
    finest=min(hs)
    factors=[round(h/finest) for h in hs]
    if any(q*finest!=h or q&(q-1) for q,h in zip(factors,hs)):
        raise ValueError('levels must be nested powers of two')
    for c,N in zip(configs,work['time_steps']):
        if N>c.max_steps or c.M*N>c.max_path_steps:
            raise ValueError('preflight budget exceeded')
    began=time.perf_counter()
    deadline=began+runtime_seconds
    values,levels=[],[]
    for cfg,N,factor in zip(configs,work['time_steps'],factors):
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(cfg.seed,
            spawn_key=_spawn_key(problem.problem_id,x,cfg.stream_id))))
        draws=CoupledNormals(rng,factor)
        run=_Run(deadline=deadline,n_batches=1)
        start=time.perf_counter()
        failure=None
        try:
            z=_path_contributions(problem,x,cfg.T,N,cfg.M,draws,bound,run)
            values.append(z)
        except Exception as exc:
            failure=dict(reason=getattr(exc,'reason','worker_exception'),
                         message=f'{type(exc).__name__}: {exc}')
        levels.append(dict(config=asdict(cfg),factor=factor,n_steps=N,
            runtime_seconds=time.perf_counter()-start,kernel_statistics=asdict(run),
            normal_values_generated_with_replay=draws.normal_values_generated,
            success=failure is None,failure=failure))
        if failure:
            return None,dict(success=False,failure=failure,levels=levels,
                runtime_seconds=time.perf_counter()-began,preflight=work)
    return np.column_stack(values),dict(success=True,failure=None,levels=levels,
        runtime_seconds=time.perf_counter()-began,preflight=work,
        coupling='PCG64 production point stream; normalized disjoint finest blocks v1',
        production_scheme_unchanged=True,total_error_certified=False)


def config_dict(M,h,seed):
    from mc_solver import MCConfig
    return asdict(MCConfig(M=M,dt=h,T=63.625,tail_eps=1e-4,seed=seed,
        chunk=M,max_steps=8192,max_path_steps=32_000_000,max_batch_paths=4096,
        max_runtime_seconds=20.))


def prepare(out):
    import numpy as np
    from mc_solver import _gap_lower, _tail_bound, auto_T
    out.mkdir(parents=True,exist_ok=False)
    (out/'before').mkdir()
    shutil.copy2(ROOT/'PROJECT_STATUS.md',out/'before/PROJECT_STATUS.md')
    write_json(out/'protected_before.json',protected_hashes(out))
    p,_,_,metadata=load_task(PROBLEM_ID)
    cert,proof=mc_forcing_certificate({},p,metadata)
    Tstar=auto_T(p.lambda_value,p.c_abs_upper_bound,cert.abs_upper_bound,1e-4)
    assert 63.625>=Tstar
    attempts=[]
    for stage,R,M,hs in [('pilot',4,1024,[1/32,1/64]),
                          ('confirmation',8,2048,[1/32,1/64,1/128])]:
        for rep in range(R):
            stage_id=0 if stage=='pilot' else 1
            seed=int(np.random.SeedSequence(MASTER_SEED,spawn_key=(stage_id,rep)).generate_state(1,dtype=np.uint64)[0])
            attempts.append(dict(attempt_id=f'{stage}_{rep:02d}',stage=stage,repeat=rep,
                seed=seed,M=M,configs=[config_dict(M,h,seed) for h in hs],
                maximum_cost=costs(M,63.625,hs)))
    previous=json.loads((ROOT/'results/mc_convergence_2026-10-24/plan.json').read_text())
    assert not {a['seed'] for a in attempts}&{a['config']['seed'] for a in previous['attempts']}
    total=sum(a['maximum_cost']['path_steps'] for a in attempts)
    assert total<=260_000_000
    plan=dict(version='F0002_coupled_h_v1',frozen_utc=now(),problem_id=PROBLEM_ID,
        x=0.,T=63.625,tail_eps=1e-4,certified_T=Tstar,forcing_bound=asdict(cert),
        forcing_proof=proof,tail_error_bound=_tail_bound(cert.abs_upper_bound,
            _gap_lower(p.lambda_value,p.c_abs_upper_bound),63.625),
        master_seed=MASTER_SEED,attempts=attempts,
        decision_rule=dict(minimum_pilot_repeats=4,maximum_relative_ci95_half=.5,
            minimum_variance_reduction=2.,
            proceed='All 4 pilots succeed, paired t95 CI excludes zero, its half-width '
                '<=0.5*abs(mean difference), and pooled path variance reduction>=2. '
                'Otherwise stop without expensive confirmation.',
            include_128='Include 1/128 unless pilot t95 interval for mu_64-y_exact '
                'expanded by certified tail lies strictly inside [-0.01,0.01]. '
                'If it does, confirm only 1/32 and 1/64.',
            inference='Pilot only selects levels; statistical conclusions use independent confirmation. '
                'No additional M, levels or retries in this study.'),
        analysis=dict(mean_intervals='t across independent seed means; pointwise and '
                'Bonferroni simultaneous over declared confirmation levels',
            paired_intervals='t across paired run means; Bonferroni over adjacent contrasts',
            bootstrap_seed=2026102409,bootstrap_replicates=10000,
            bootstrap='Resample independent seed vectors jointly for contrast ratio; '
                'diagnostic interval, not proof of asymptotic order',
            no_exact_in_sampling=True,richardson='Model diagnostic only; do not replace EM values'),
        resources=dict(workers=1,blas_threads=1,rss_limit_mib=384.,
            total_wall_seconds=300.,stage_wall_seconds=180.,attempt_cooperative_seconds=20.,
            max_path_steps=260_000_000,planned_maximum_path_steps=total,
            planned_normal_values_generated_with_replay=sum(a['maximum_cost']['normal_values_generated_with_replay'] for a in attempts),
            estimate='Prior certified h=1/32,M=2048 median 1.369s. Maximum update ratio '
                'gives ~85s; Gaussian replay overhead additional; allow 300s watchdog. '
                'O(M) working memory; save only paired path integrals, not trajectories.',
            stop='Stop on any failure or resource limit, keep partial counters and not_run statuses'),
        hypotheses=['mu_32-mu_64 is distinguishable with coupling at small M',
            'Further refinement reduces positive mean error; otherwise no practical h selected',
            'Contrast ratio may support first-order behavior, but three levels cannot prove asymptotics'])
    write_json(out/'plan.json',plan)
    write_json(out/'environment.json',dict(python=sys.version,executable=sys.executable,
        platform=platform.platform(),versions={n:importlib.metadata.version(n) for n in
        ('numpy','scipy','sympy','matplotlib','pytest')},
        thread_environment={k:os.environ[k] for k in THREAD_VARS}))
    (out/'sources').mkdir()
    for name in SOURCES:
        shutil.copy2(ROOT/name,out/'sources'/Path(name).name)
    print(json.dumps(dict(attempts=len(attempts),maximum_path_steps=total,
        normal_values_generated_with_replay=plan['resources']['planned_normal_values_generated_with_replay'],
        certified_T=Tstar,T=63.625,tail_bound=plan['tail_error_bound'])),flush=True)


def worker(out,stage):
    import numpy as np
    from mc_solver import MCConfig, MCForcingBound
    plan=json.loads((out/'plan.json').read_text())
    selection=([1/32,1/64] if stage=='pilot' else
        json.loads((out/'decision.json').read_text())['confirmation_hs'])
    began=time.perf_counter()
    p,exact,_,_=load_task(PROBLEM_ID)
    bound=MCForcingBound(**plan['forcing_bound'])
    stop=threading.Event()
    def monitor():
        with (out/f'{stage}.rss.jsonl').open('x') as f:
            while not stop.is_set():
                rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)
                f.write(json.dumps(dict(seconds=time.perf_counter()-began,peak_rss_mib=rss))+'\n'); f.flush()
                stop.wait(.1)
    thread=threading.Thread(target=monitor,daemon=True); thread.start()
    try:
        for spec in [a for a in plan['attempts'] if a['stage']==stage]:
            aid=spec['attempt_id']
            write_json(out/'records'/f'{aid}.started.json',dict(spec=spec,started_utc=now(),hs=selection))
            configs=[MCConfig(**c) for c in spec['configs'] if c['dt'] in selection]
            samples,result=coupled_paths(p,plan['x'],configs,bound,
                runtime_seconds=plan['resources']['attempt_cooperative_seconds'])
            y=float(exact(np.array([plan['x']]))[0])  # only after sampling
            if samples is not None:
                np.savez_compressed(out/'records'/f'{aid}.samples.npz',Z=samples,hs=selection)
            write_json(out/'records'/f'{aid}.result.json',dict(spec=spec,hs=selection,
                result=result,y_exact=y,
                process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/
                    (1024**2 if sys.platform=='darwin' else 1024)))
            print(json.dumps(dict(attempt=aid,success=result['success'],seconds=result['runtime_seconds'])),flush=True)
            if not result['success']:
                break
    finally:
        stop.set(); thread.join()


def bounded_stage(out,stage,overall_began,limits):
    start=time.perf_counter(); reason=None; peak=0.
    with (out/f'{stage}.log').open('x') as log:
        child=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'worker',
            '--out',str(out),'--stage',stage],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
        while child.poll() is None:
            sample=out/f'{stage}.rss.jsonl'
            if sample.exists():
                lines=sample.read_text().splitlines()
                try:
                    if lines:
                        peak=max(peak,json.loads(lines[-1])['peak_rss_mib'])
                except json.JSONDecodeError:
                    pass
            if time.perf_counter()-overall_began>limits['total_wall_seconds']:
                reason='external_total_time_limit'
            elif time.perf_counter()-start>limits['stage_wall_seconds']:
                reason='external_stage_time_limit'
            elif peak>limits['rss_limit_mib']:
                reason='external_memory_limit'
            if reason:
                child.terminate()
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
                break
            time.sleep(.1)
    if child.returncode and not reason:
        reason='worker_exception'
    files=list((out/'records').glob(f'{stage}_*.result.json'))
    if any(not json.loads(p.read_text())['result']['success'] for p in files):
        reason='solver_failure'
    return dict(stage=stage,reason=reason,returncode=child.returncode,
        wall_seconds=time.perf_counter()-start,sampled_peak_rss_mib=peak,completed=len(files))


def run(out):
    from diagnostics.report_mc_f0002_time_refinement import pilot_decision
    plan=json.loads((out/'plan.json').read_text()); frozen=digest(out/'plan.json')
    (out/'records').mkdir(exist_ok=False)
    (out/'run.lock').write_text(now())
    began=time.perf_counter()
    event=bounded_stage(out,'pilot',began,plan['resources']); events=[event]
    if event['reason']:
        decision=dict(proceed=False,confirmation_hs=[],reason=event['reason'])
    else:
        decision=pilot_decision(out,plan)
    write_json(out/'decision.json',decision)
    print(json.dumps(dict(stage='pilot',event=event,decision=decision)),flush=True)
    if decision['proceed']:
        events.append(bounded_stage(out,'confirmation',began,plan['resources']))
        print(json.dumps(events[-1]),flush=True)
    statuses=[]
    for spec in plan['attempts']:
        aid=spec['attempt_id']; path=out/'records'/f'{aid}.result.json'
        status='not_run'
        if path.exists():
            r=json.loads(path.read_text())['result']
            status='success' if r['success'] else r['failure']['reason']
        elif (out/'records'/f'{aid}.started.json').exists():
            status=events[-1]['reason'] or 'incomplete'
        statuses.append(dict(attempt_id=aid,status=status))
    before=json.loads((out/'protected_before.json').read_text()); after=protected_hashes(out)
    changed=[p for p,h in before.items() if after.get(p)!=h]
    write_json(out/'verification.json',dict(protected_count=len(before),changed_or_removed=changed,
        after_sha256=after,plan_sha256_before=frozen,plan_sha256_after=digest(out/'plan.json')))
    write_json(out/'run_summary.json',dict(finished_utc=now(),wall_seconds=time.perf_counter()-began,
        events=events,attempts=statuses))
    assert not changed and frozen==digest(out/'plan.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','worker'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stage',choices=('pilot','confirmation'))
    args=parser.parse_args(); out=args.out.resolve()
    if not out.is_relative_to(ROOT/'results') or out==ROOT/'results':
        parser.error('use a new directory under results/')
    if args.action=='worker':
        worker(out,args.stage)
    else:
        (prepare if args.action=='prepare' else run)(out)


if __name__=='__main__':
    main()
