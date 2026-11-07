"""Parallel execution of frozen MC33 jobs; no numerical scheme changes.

One existing serial MC call per job. Guardians fence orphan workers; immutable
checksummed results are the checkpoint. A CLI controller renews its own lease
while alive (there is deliberately no dependence on a chat/session heartbeat).
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import threading
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS',
           'BLIS_NUM_THREADS', 'OMP_THREAD_LIMIT')
for name in THREADS:
    os.environ[name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

from diagnostics.run_chernov_batch import clean, digest, read_json, write_json
from diagnostics.chernov_batch_control import (
    GuardedProcess, controller_locks, new_control, finish_control, atomic,
    lock_is_held)

NUMERICAL = ('mc_solver.py', 'mc_optimized.py', 'mc_ast_kernels.py',
             'remizov_solver.py', 'all_methods_bounds.py',
             'all_methods_dataset_loader.py', 'src/all_methods_expressions.py',
             'src/all_methods_dataset_v1.py', 'src/problem_types.py')
RUNTIME = ('diagnostics/mc33_parallel.py', 'diagnostics/run_chernov_batch.py',
           'diagnostics/chernov_batch_control.py')
PACKAGES = ('numpy', 'scipy', 'sympy', 'numba', 'llvmlite', 'threadpoolctl')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_object(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def environment():
    return dict(python=platform.python_version(), platform=platform.platform(),
                machine=platform.machine(), executable=sys.executable,
                packages={p: importlib.metadata.version(p) for p in PACKAGES},
                threads={p: os.environ[p] for p in THREADS})


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        1024**2 if sys.platform == 'darwin' else 1024)


def publish(path, value, exclusive=True):
    """Reuse atomic no-clobber publication, additionally sync the directory."""
    write_json(path, value, exclusive=exclusive)
    fd = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def freeze(out, jobs, data_dir, pins, **extra):
    require(len({j['key'] for j in jobs}) == len(jobs), 'duplicate jobs')
    out.mkdir(parents=True, exist_ok=False)
    for name in ('records', 'attempts', 'workers', 'sessions', 'sources'):
        (out / name).mkdir()
    source_hashes = {p: digest(ROOT / p) for p in NUMERICAL + RUNTIME}
    for name in source_hashes:
        dest = out / 'sources' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / name).read_bytes())
    plan = dict(schema='mc33_parallel_v1', data_dir=str(data_dir), jobs=jobs,
                protected_sha256={**pins, **source_hashes}, environment=environment(),
                algorithm='unchanged mc_optimized; numba; block_steps=64',
                checkpoint_unit='one full equation/seed on all 33 nodes',
                aggregation='none; E_grid for each seed; episode0 primary',
                **extra)
    publish(out / 'plan.json', plan)
    publish(out / 'plan.sha256.json', dict(sha256=digest(out / 'plan.json')))
    return plan


def load_plan(out):
    require(digest(out / 'plan.json') == read_json(out / 'plan.sha256.json')['sha256'],
            'plan changed')
    plan = read_json(out / 'plan.json')
    for name, expected in plan['protected_sha256'].items():
        require(digest(ROOT / name) == expected, 'source/input changed: ' + name)
    current = environment()
    for name in ('python', 'packages', 'machine', 'threads'):
        require(current[name] == plan['environment'][name], 'environment changed: ' + name)
    require(len({j['key'] for j in plan['jobs']}) == len(plan['jobs']), 'duplicate jobs')
    return plan


def validate_success(job, record, exact=None):
    """Validate raw numbers, work, RNG identity and the original error metric."""
    import numpy as np
    from mc_solver import _spawn_key
    z = record['result']; c = job['config']
    require(record['status'] == 'completed' and z['success'] and not z['failure_reason'],
            'not a successful result')
    require(z['problem_id'] == job['equation_id'], 'wrong problem')
    require(record['expression_hash'] == job['expression_hash'], 'wrong expression')
    require(record['config'] == c, 'wrong config')
    require(z['backend'] == 'numba' and z['block_steps'] == 64, 'wrong backend')
    require(z['rng_scheme'] == 'PCG64/SeedSequence/point_address_v1', 'wrong RNG')
    require(z['seed_entropy'] == c['seed'] and z['stream_id'] == c['stream_id'], 'wrong seed')
    require(z['point_spawn_keys'] == [_spawn_key(job['equation_id'], x, c['stream_id'])
                                    for x in job['target_x']], 'wrong point streams')
    require(z['x'] == job['target_x'] and len(z['x']) == 33, 'wrong grid')
    require(z['M'] == c['M'] and z['completed_paths'] == [c['M']] * 33, 'incomplete paths')
    require(z['T'] == job['expected_T'] and z['n_steps'] == job['expected_steps'], 'wrong T/N')
    require(z['dt_requested'] == c['dt'] and z['dt_actual'] == z['T']/z['n_steps'], 'wrong h')
    require(z['forcing_abs_upper_bound'] == job['forcing_G'], 'wrong G')
    require(z['n_path_steps'] == z['n_normal_draws'] == job['numerical_work'], 'wrong work')
    require(z['tail_eps'] == c['tail_eps'] and z['tail_tolerance_met'], 'wrong tail')
    for name in ('f', 'stderr', 'ci95', 'ci_lower', 'ci_upper'):
        a = np.asarray(z[name]); require(a.shape == (33,) and np.isfinite(a).all(), name)
    target = np.asarray(record['y_exact'] if exact is None else exact)
    require(target.shape == (33,) and np.isfinite(target).all(), 'bad exact')
    require(np.array_equal(target, record['y_exact']), 'exact differs')
    error = float(np.max(np.abs(np.asarray(z['f']) - target)))
    require(record['E_grid'] == error, 'wrong error metric')
    require(record['accuracy_met'] == (error <= .01), 'wrong threshold')


def payload(path, job, plan_sha):
    value = read_json(path)
    require(value['sha256'] == sha_object(value['record']), 'corrupt checkpoint: ' + str(path))
    r = value['record']
    require(r['job'] == job and r['plan_sha256'] == plan_sha, 'wrong checkpoint identity')
    require(r['status'] in ('completed', 'solver_failure', 'worker_exception', 'invalid_result'),
            'unknown terminal status')
    if r['status'] == 'completed':
        validate_success(job, r)
    return r


def scan(out, plan):
    ph = digest(out / 'plan.json'); lookup = {j['key']: j for j in plan['jobs']}
    known = {p.stem for p in (out / 'records').glob('*.json')}
    require(known <= lookup.keys(), 'unrecognized record')
    done = {key: payload(out / 'records' / (key + '.json'), lookup[key], ph) for key in known}
    # A crash between durable attempt and record publication is recoverable
    # without another solver call. Different timing replays are never pooled.
    for path in sorted((out / 'attempts').glob('*.result.json')):
        raw = read_json(path); key = raw['record']['job']['key']
        require(key in lookup, 'unknown attempt')
        r = payload(path, lookup[key], ph)
        if key not in done:
            publish(out / 'records' / (key + '.json'), raw)
            done[key] = r
    return done


def worker(out, state_path, parent_pid):
    import numpy as np
    from all_methods_dataset_loader import load_all_methods_equations
    from mc_optimized import compile_backend, prepare_all_methods_mc, solve_monte_carlo_optimized
    from mc_solver import MCConfig
    from threadpoolctl import threadpool_info, threadpool_limits
    plan = load_plan(out); ph = digest(out / 'plan.json')
    state = dict(pid=os.getpid(), phase='loading', key=None)
    stop = threading.Event()
    def monitor():
        while not stop.is_set():
            if os.getppid() != parent_pid:
                os._exit(75)
            write_json(state_path, dict(state, peak_rss_mib=rss()), durable=False)
            stop.wait(.25)
    thread = threading.Thread(target=monitor, daemon=True); thread.start()
    try:
        began = time.perf_counter(); data = ROOT / plan['data_dir']
        eqs = {e.equation_id: e for e in load_all_methods_equations(
            data/'all_methods_equations_v1.csv', data/'all_methods_equations_v1_metadata.jsonl')}
        loading = time.perf_counter() - began
        with threadpool_limits(limits=1):
            started = time.perf_counter(); jit = compile_backend()
            import numba
            publish(state_path.with_suffix('.warmup.json'), dict(
                loading_seconds=loading, jit=jit, warmup_seconds=time.perf_counter()-started,
                numba_threads=numba.get_num_threads(), threadpools=threadpool_info(),
                environment=environment(), peak_rss_mib=rss()))
            lookup = {j['key']: j for j in plan['jobs']}
            state['phase'] = 'ready'
            for line in sys.stdin:
                request = read_json(Path(json.loads(line)['request_path'])); j = request['job']
                require(j == lookup[j['key']] and request['plan_sha256'] == ph, 'wrong request')
                e = eqs[j['equation_id']]
                require(e.metadata['expression_hash'] == j['expression_hash'], 'wrong dataset')
                state.update(phase='solving', key=j['key'])
                r = dict(job=j, config=j['config'], expression_hash=j['expression_hash'],
                         equation_id=j['equation_id'], plan_sha256=ph,
                         attempt=request['attempt'], session=request['session'], worker_pid=os.getpid(),
                         started_epoch=time.time(), solver_called=False)
                t0 = time.perf_counter()
                try:
                    prepared = prepare_all_methods_mc(e.metadata)
                    require(prepared.forcing_bound.abs_upper_bound == j['forcing_G'], 'G changed')
                    config = MCConfig(**j['config']); r['solver_called'] = True
                    z = solve_monte_carlo_optimized(prepared, j['target_x'], config,
                                                   engine='numba', block_steps=64)
                    r.update(method_wall_seconds=time.perf_counter()-t0, result=clean(asdict(z)),
                             status='completed' if z.success else 'solver_failure')
                    exact = np.asarray(e.y_exact(np.asarray(j['target_x'])))
                    error = float(np.max(abs(z.f-exact))) if z.success else None
                    r.update(y_exact=exact.tolist(), E_grid=error,
                             accuracy_met=None if error is None else error <= .01)
                    if z.success:
                        validate_success(j, r, exact)
                except Exception:
                    r.update(status='invalid_result' if 'result' in r else 'worker_exception',
                             traceback=traceback.format_exc(), E_grid=None, accuracy_met=None)
                r.update(finished_epoch=time.time(), peak_rss_mib=rss())
                wrapped = dict(record=r, sha256=sha_object(r))
                publish(out/'attempts'/(request['attempt']+'.result.json'), wrapped)
                publish(out/'records'/(j['key']+'.json'), wrapped)
                state.update(phase='ready', key=None)
    finally:
        stop.set(); thread.join()


def run(out, workers, resume=False, max_jobs=None):
    require(workers >= 1, 'workers must be positive')
    plan = load_plan(out); ph = digest(out/'plan.json')
    session = uuid.uuid4().hex
    with controller_locks(out) as fds:
        require(not lock_is_held(ROOT/'results/.chernov_batch_compute.lock'), 'old compute is active')
        done = scan(out, plan)
        require(resume or not done and not list((out/'attempts').glob('*.request.json')),
                'existing execution: use --resume')
        pending = [j for j in plan['jobs'] if j['key'] not in done]
        if not pending:
            result = dict(new_calls=0, worker_startups=0, completed=len(done), pending=0)
            print(json.dumps(result), flush=True); return result
        if max_jobs is not None:
            require(max_jobs >= 1, 'max_jobs must be positive'); pending = pending[:max_jobs]
        # Longest estimated tasks first; job RNG and reductions stay unchanged.
        pending.sort(key=lambda j: (-j.get('scenario_seconds', j['numerical_work']), j['key']))
        active = new_control(out, session, lease_seconds=60.)
        control = Path(active['control_dir']); states = []; new = 0
        began = time.perf_counter(); compute_start = None; stop_reason = None
        peak = 0.; peak_workers = 0.; last_lease = 0.; old = {}
        def interrupted(signum, frame):
            atomic(control/'stop.json', dict(signal=signum))
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            old[sig] = signal.signal(sig, interrupted)
        def checkpoint():
            write_json(out/'progress.json', dict(session=session, terminal=len(done),
                       statuses=dict(Counter(r['status'] for r in done.values())),
                       pending=len(plan['jobs'])-len(done), in_flight=[s['job']['key']
                       for s in states if s['job']], new_terminal=new))
        try:
            for i in range(min(workers, len(pending))):
                token = f'{session}_{i}'; path = out/'workers'/(token+'.json')
                log = (out/'workers'/(token+'.log')).open('x')
                child = GuardedProcess([sys.executable, '-B', str(Path(__file__).resolve()),
                    'worker', '--out', str(out), '--state', str(path), '--parent-pid', '{GUARD_PID}'],
                    log, control/(token+'.guardian.json'), active, fds,
                    compute_lock_path=out/'slots'/f'{i}.lock')
                states.append(dict(child=child, log=log, path=path, job=None, request=None))
            with (out/'sessions'/(session+'.telemetry.jsonl')).open('x') as telemetry:
                while True:
                    now = time.monotonic()
                    if now-last_lease > 1.:
                        atomic(control/'lease.json', dict(token=active['token'], renewed_monotonic=now))
                        last_lease = now
                    snapshots = []
                    for s in states:
                        child = s['child']; code = child.poll()
                        if code is not None:
                            stop_reason = 'worker_exit'; break
                        snapshot = read_json(s['path']) if s['path'].exists() else {}
                        snapshots.append(snapshot)
                        j = s['job']
                        if j and (out/'records'/(j['key']+'.json')).exists():
                            r = payload(out/'records'/(j['key']+'.json'), j, ph)
                            done[j['key']] = r; new += 1; s['job'] = None
                            print(json.dumps(dict(completed=len(done), total=len(plan['jobs']),
                                             key=j['key'], status=r['status'])), flush=True)
                            checkpoint()
                            if r['status'] != 'completed':
                                stop_reason = r['status']; break
                    worker_rss = sum(s.get('peak_rss_mib', 0) for s in snapshots)
                    peak_workers = max(peak_workers, worker_rss)
                    peak = max(peak, worker_rss + rss())
                    telemetry.write(json.dumps(dict(elapsed=time.perf_counter()-began,
                        worker_rss_highwater_sum_mib=worker_rss, controller_rss_highwater_mib=rss(),
                        workers=snapshots))+'\n'); telemetry.flush()
                    if stop_reason or (control/'stop.json').exists():
                        stop_reason = stop_reason or 'interrupted'; break
                    if len(snapshots) == len(states) and all(s.get('phase') == 'ready' for s in snapshots):
                        if compute_start is None:
                            compute_start = time.perf_counter()
                    if compute_start is not None:
                        for s in states:
                            if s['job'] is None and pending:
                                j = pending.pop(0); attempt = uuid.uuid4().hex
                                request = dict(attempt=attempt, session=session, job=j,
                                               plan_sha256=ph, requested_epoch=time.time())
                                path = out/'attempts'/(attempt+'.request.json'); publish(path, request)
                                s.update(job=j, request=request)
                                s['child'].stdin.write(json.dumps(dict(request_path=str(path)))+'\n')
                                s['child'].stdin.flush()
                                checkpoint()
                        if not pending and not any(s['job'] for s in states):
                            break
                    time.sleep(.1)
        finally:
            compute_end = time.perf_counter()
            for s in states:
                s['child'].terminate()
            for s in states:
                s['child'].wait(timeout=15); s['log'].close()
            # Capture records published between the last poll and shutdown.
            done = scan(out, plan); checkpoint()
            incomplete = []
            for s in states:
                if s['job'] and s['job']['key'] not in done:
                    incomplete.append(s['request']['attempt'])
                    publish(out/'attempts'/(s['request']['attempt']+'.interrupted.json'), dict(
                        request=s['request'], reason=stop_reason or 'controller_exception',
                        finished_epoch=time.time(), partial_solver_counters='unknown'))
            finish_control(out, active)
            for sig, handler in old.items():
                signal.signal(sig, handler)
            summary = dict(session=session, workers=workers, worker_startups=len(states),
                wall_seconds=time.perf_counter()-began,
                compute_seconds=None if compute_start is None else compute_end-compute_start,
                startup_seconds=None if compute_start is None else compute_start-began,
                peak_controller_plus_worker_highwater_mib=peak,
                peak_worker_highwater_sum_mib=peak_workers,
                memory_measurement='sum of sampled process lifetime RSS maxima; conservative; guardians excluded',
                completed=len(done), pending=len(plan['jobs'])-len(done), new_terminal=new,
                statuses=dict(Counter(r['status'] for r in done.values())),
                interrupted_attempts=incomplete, stop_reason=stop_reason)
            publish(out/'sessions'/(session+'.json'), summary)
        print(json.dumps(summary), flush=True)
        return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('run', 'status', 'worker'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--max-jobs', type=int)
    p.add_argument('--state', type=Path)
    p.add_argument('--parent-pid', type=int)
    a = p.parse_args(); out = a.out.resolve()
    if a.command == 'worker':
        worker(out, a.state, a.parent_pid)
    elif a.command == 'status':
        with controller_locks(out):
            plan = load_plan(out); done = scan(out, plan)
            print(json.dumps(dict(terminal=len(done), pending=len(plan['jobs'])-len(done),
                                  statuses=dict(Counter(r['status'] for r in done.values())))))
    else:
        run(out, a.workers, a.resume, a.max_jobs)


if __name__ == '__main__':
    main()
