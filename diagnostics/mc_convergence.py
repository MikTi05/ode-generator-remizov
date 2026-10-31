"""Frozen one-point M/h/T study; existing MC solver is used without changes.

prepare writes a complete plan without sampling; run consumes it once. Each
problem has one isolated worker, independent seeds, and per-attempt persistence.
The separate report script only reads saved records.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diagnostics.compare_chernov import (THREAD_VARS, load_task,
                                         mc_forcing_certificate, write_json)

for key in THREAD_VARS:
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
PROBLEMS = ('MC_constant', 'MC_gaussian', 'AMV1_F0000_S00',
            'AMV1_F0001_S00', 'AMV1_F0002_S00')
MASTER_SEED = 2026102407


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_hashes(out):
    """All existing results and inputs, and numerical/generator sources."""
    paths = []
    for folder in ('src', 'data', 'comparison_inputs', 'results'):
        paths.extend((ROOT/folder).rglob('*'))
    paths.extend(ROOT.glob('*.py'))
    paths.append(ROOT/'MIGRATION_MANIFEST.json')
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))
            if p.is_file() and not p.is_relative_to(out) and p.name != '.DS_Store'}


def step_cost(M, h, T):
    N = max(1, math.ceil(T/h)) if T else 0
    if N and T/N > h:
        N += 1
    return dict(time_steps=N, actual_dt=T/N if N else 0., path_steps=M*N,
                normal_draws=M*N, coefficient_calls=2+4*N if N else 0)


def configurations(certified_T, deterministic=False):
    """Merge shared baselines so a single set of samples has several views."""
    long_T = math.ceil(certified_T*8)/8
    rows = {}
    def add(M, h, T, factor, level):
        key = (M, h, T)
        row = rows.setdefault(key, dict(M=M, dt=h, T=T, views=[]))
        row['views'].append(dict(factor=factor, level=level))
    if deterministic:
        for h in (1/8, 1/16, 1/32):
            add(2, h, 8., 'h', h)
        for T in (.5, 1., 2., long_T, 8.):
            add(2, 1/32, T, 'T', T)
    else:
        for M in (512, 2048, 8192):
            add(M, 1/32, 8., 'M', M)
        for h in (1/8, 1/16, 1/32):
            add(8192, h, 8., 'h', h)
        for T in (1., 2., 4., 8., long_T):
            add(2048, 1/32, T, 'T', T)
    return list(rows.values())


def prepare(out):
    import numpy as np
    from mc_solver import MCConfig, _gap_lower, _tail_bound, auto_T
    out.mkdir(parents=True, exist_ok=False)
    (out/'before').mkdir()
    shutil.copy2(ROOT/'PROJECT_STATUS.md', out/'before/PROJECT_STATUS.md')
    before = protected_hashes(out)
    write_json(out/'protected_before.json', before)
    tasks, attempts = [], []
    index = 0
    for pi, problem_id in enumerate(PROBLEMS):
        p, _, _, metadata = load_task(problem_id)
        bound, proof = mc_forcing_certificate({}, p, metadata)
        Tstar = auto_T(p.lambda_value, p.c_abs_upper_bound, bound.abs_upper_bound, 1e-4)
        gap = _gap_lower(p.lambda_value, p.c_abs_upper_bound)
        task = dict(problem_id=problem_id, target_x=[0.], G=bound.abs_upper_bound,
                    gap=gap, certified_T=Tstar, rounded_certified_T=math.ceil(Tstar*8)/8,
                    T_tail_1e3=auto_T(p.lambda_value, p.c_abs_upper_bound, bound.abs_upper_bound, 1e-3),
                    forcing_bound=asdict(bound), forcing_proof=proof)
        tasks.append(task)
        for ci, row in enumerate(configurations(Tstar, pi == 0)):
            for rep in range(1 if pi == 0 else 8):
                seed = int(np.random.SeedSequence(MASTER_SEED,
                    spawn_key=(pi, ci, rep)).generate_state(1, dtype=np.uint64)[0])
                cfg = MCConfig(M=row['M'], dt=row['dt'], T=row['T'], seed=seed,
                    tail_eps=1e-4, chunk=8192, max_steps=4096,
                    max_path_steps=5_000_000, max_batch_paths=8192,
                    max_runtime_seconds=10.)
                cfg.validate()
                cost = step_cost(cfg.M, cfg.dt, cfg.T)
                assert cost['actual_dt'] == cfg.dt
                assert cost['path_steps'] <= cfg.max_path_steps
                attempts.append(dict(attempt_id=f'{index:03d}', problem_id=problem_id,
                    configuration_id=f'{pi}_{ci}', repeat=rep, views=row['views'],
                    target_x=[0.], config=asdict(cfg), preflight=cost,
                    tail_bound=_tail_bound(bound.abs_upper_bound, gap, cfg.T),
                    tail_tolerance_met=cfg.T >= Tstar))
                index += 1
    assert len({a['config']['seed'] for a in attempts}) == len(attempts)
    total = sum(a['preflight']['path_steps'] for a in attempts)
    assert total <= 250_000_000
    # Randomize levels/repeats within each task, independently of the MC streams.
    order = []
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(MASTER_SEED,
        spawn_key=(999,))))
    for problem_id in PROBLEMS:
        ids = [a['attempt_id'] for a in attempts if a['problem_id'] == problem_id]
        order.extend(rng.permutation(ids).tolist())
    plan = dict(version='mc_one_point_convergence_v1', frozen_utc=now(),
        master_seed=MASTER_SEED, tasks=tasks, attempts=attempts, execution_order=order,
        purpose='Separate M, h, T at x=0; no benchmark or spatial accuracy claim',
        statistical_protocol=dict(repeats=8, independent_seeds_between_all_attempts=True,
            shared_baselines_counted_once=True, brownian_coupling=False,
            interval='Solver: pointwise normal CI for discrete truncated expectation; '
                     'aggregate: Student t across independent replicates; no total-error CI',
            order='At least 3 h levels; report stochastic ambiguity. Analytic oracles '
                  'may establish trapezoid order only for the two analytic examples.',
            tuning='No removal of outliers or adjustment of M/h/T using exact errors'),
        resources=dict(workers=1, blas_threads=1, total_path_step_cap=250_000_000,
            planned_path_steps=total, max_attempt_path_steps=max(a['preflight']['path_steps'] for a in attempts),
            total_wall_seconds=360., task_wall_seconds=110., rss_limit_mib=384.,
            cooperative_solver_seconds=10., rss_sampling_seconds=.1,
            rough_prior_estimate_seconds=[60, 180],
            estimate_basis='Prior measured F0000 13.2M updates/1.815s, F0001 '
                '/4.170s, F0002 /4.688s; one-point time-loop overhead makes extrapolation approximate',
            stop='Stop entire study on external limit or solver failure; retain all '
                 'completed/partial/not_run attempts; no automatic retries'),
        hypotheses=['SE and replicate spread scale as M^-1/2; RMSE can plateau at bias',
            'Gaussian/constant exact discrete means exhibit trapezoid O(h^2)',
            'Dataset h order may be unresolved with this statistical budget',
            'Certified T controls continuous tail only; shorter T may work empirically'])
    write_json(out/'plan.json', plan)
    write_json(out/'environment.json', dict(python=sys.version, executable=sys.executable,
        platform=platform.platform(), versions={n:importlib.metadata.version(n)
        for n in ('numpy','scipy','sympy','matplotlib','pytest')},
        thread_environment={n:os.environ[n] for n in THREAD_VARS}))
    (out/'sources').mkdir()
    for name in ('diagnostics/mc_convergence.py','diagnostics/report_mc_convergence.py',
                 'tests/test_mc_convergence_protocol.py'):
        shutil.copy2(ROOT/name, out/'sources'/Path(name).name)
    print(json.dumps(dict(attempts=len(attempts), path_steps=total,
        max_attempt_path_steps=plan['resources']['max_attempt_path_steps'],
        tasks=[{k:t[k] for k in ('problem_id','G','certified_T','rounded_certified_T')}
               for t in tasks])), flush=True)


def worker(out, problem_id):
    import numpy as np
    from mc_solver import MCConfig, MCForcingBound, solve_monte_carlo
    plan = json.loads((out/'plan.json').read_text())
    specs = {a['attempt_id']:a for a in plan['attempts']}
    task = next(t for t in plan['tasks'] if t['problem_id'] == problem_id)
    began = time.perf_counter()
    p, exact, _, _ = load_task(problem_id)
    load_seconds = time.perf_counter()-began
    bound = MCForcingBound(**task['forcing_bound'])
    stop = threading.Event()
    def sample():
        with (out/f'{problem_id}.rss.jsonl').open('x') as handle:
            while not stop.is_set():
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                handle.write(json.dumps(dict(worker_wall_seconds=time.perf_counter()-began,
                    peak_rss_mib=rss/(1024**2 if sys.platform=='darwin' else 1024)))+'\n')
                handle.flush()
                stop.wait(.1)
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    try:
        for aid in plan['execution_order']:
            spec = specs[aid]
            if spec['problem_id'] != problem_id:
                continue
            write_json(out/'records'/f'{aid}.started.json', dict(started_utc=now(), spec=spec))
            cfg = MCConfig(**spec['config'])
            start = time.perf_counter()
            result = solve_monte_carlo(p, spec['target_x'], cfg, forcing_bound=bound)
            elapsed = time.perf_counter()-start
            # Exact is evaluated only after the solver has returned.
            y = np.asarray(exact(np.asarray(spec['target_x'])), dtype=float)
            record = dict(spec=spec, config=asdict(cfg), result=asdict(result), y_exact=y,
                max_abs_error=float(np.max(np.abs(result.f-y))) if result.success else None,
                signed_error=(result.f-y) if result.success else None,
                call_wall_seconds=elapsed, load_seconds=load_seconds,
                process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/
                    (1024**2 if sys.platform=='darwin' else 1024))
            write_json(out/'records'/f'{aid}.result.json', record)
            print(json.dumps(dict(attempt=aid, problem_id=problem_id, success=result.success,
                seconds=result.runtime_seconds)), flush=True)
            if not result.success:
                break
    finally:
        stop.set()
        monitor.join()


def run(out):
    plan_path = out/'plan.json'
    plan_digest = digest(plan_path)
    plan = json.loads(plan_path.read_text())
    (out/'records').mkdir(exist_ok=False)
    began = time.perf_counter()
    events, stop_reason = [], None
    limits = plan['resources']
    with (out/'run.lock').open('x') as lock:
        lock.write(now())
    for task in plan['tasks']:
        problem_id = task['problem_id']
        if stop_reason:
            break
        start = time.perf_counter()
        with (out/f'{problem_id}.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                'worker', '--out', str(out), '--problem-id', problem_id], cwd=ROOT,
                env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
            peak = 0.
            while child.poll() is None:
                sample = out/f'{problem_id}.rss.jsonl'
                if sample.exists():
                    lines = sample.read_text().splitlines()
                    try:
                        if lines:
                            peak = max(peak, json.loads(lines[-1])['peak_rss_mib'])
                    except json.JSONDecodeError:
                        pass
                if time.perf_counter()-began > limits['total_wall_seconds']:
                    stop_reason = 'external_total_time_limit'
                elif time.perf_counter()-start > limits['task_wall_seconds']:
                    stop_reason = 'external_task_time_limit'
                elif peak > limits['rss_limit_mib']:
                    stop_reason = 'external_memory_limit'
                if stop_reason:
                    child.terminate()
                    try:
                        child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        child.kill(); child.wait()
                    break
                time.sleep(.1)
        if child.returncode and not stop_reason:
            stop_reason = 'worker_exception'
        task_records = [json.loads(p.read_text()) for p in (out/'records').glob('*.result.json')
                        if json.loads(p.read_text())['spec']['problem_id'] == problem_id]
        if any(not r['result']['success'] for r in task_records):
            stop_reason = 'solver_failure'
        events.append(dict(problem_id=problem_id, wall_seconds=time.perf_counter()-start,
            returncode=child.returncode, sampled_peak_rss_mib=peak,
            completed_attempts=len(task_records), stop_reason=stop_reason))
        write_json(out/'worker_events.json', events)
        print(json.dumps(events[-1]), flush=True)
    attempts = []
    for a in plan['attempts']:
        p = out/'records'/f"{a['attempt_id']}.result.json"
        started = (out/'records'/f"{a['attempt_id']}.started.json").exists()
        status = (stop_reason or 'incomplete') if started else 'not_run'
        if p.exists():
            r = json.loads(p.read_text())['result']
            status = 'success' if r['success'] else r['failure_reason']
        attempts.append(dict(attempt_id=a['attempt_id'], status=status))
    after = protected_hashes(out)
    before = json.loads((out/'protected_before.json').read_text())
    changed = [p for p,h in before.items() if after.get(p) != h]
    write_json(out/'verification.json', dict(before_file='protected_before.json',
        after_sha256=after, changed_or_removed=changed, protected_count=len(before),
        plan_sha256_before_run=plan_digest, plan_sha256_after_run=digest(plan_path)))
    write_json(out/'run_summary.json', dict(finished_utc=now(), attempts=attempts,
        wall_seconds=time.perf_counter()-began, stop_reason=stop_reason, workers=events))
    assert not changed and digest(plan_path) == plan_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','run','worker'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--problem-id', choices=PROBLEMS)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT/'results') or out == ROOT/'results':
        parser.error('use a new directory under results/')
    if args.action == 'worker':
        worker(out, args.problem_id)
    else:
        (prepare if args.action == 'prepare' else run)(out)


if __name__ == '__main__':
    main()
