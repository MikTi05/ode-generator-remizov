"""Saved-only MC33 inventory, frozen manifests and independent replay checks."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diagnostics.mc33_parallel import (NUMERICAL, digest, read_json, publish, freeze,
    require, validate_success, payload, load_plan)

CONTRACT = Path('reports/ml_selector_contract/plan.json')
X0 = Path('results/mc1000_x0_2026-10-25')
# 4 complexity levels, all 5 coefficient regimes, two historical seeds each.
# Selected for computational geometry/cost, never for observed accuracy.
BENCH_IDS = ('AMV1_F0004_S04', 'AMV1_F0028_S03', 'AMV1_F0067_S01',
             'AMV1_F0089_S04', 'AMV1_F0108_S03', 'AMV1_F0130_S02',
             'AMV1_F0163_S02', 'AMV1_F0199_S01')


def write_csv(path, rows):
    with path.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def make_job(pid, cfg, metadata, preflight, episode, key):
    p = preflight[pid]; m = metadata[pid]
    n = max(1, math.ceil(p['T']/cfg['dt']))
    if p['T']/n > cfg['dt']:
        n += 1
    return dict(key=key, equation_id=pid, expression_hash=m['expression_hash'],
                family_id=m['family_id'], split=m['split'], config=cfg, episode=episode,
                profile='MC1' if cfg['M'] == 4096 else 'MC0',
                target_x=[-1+i/16 for i in range(33)], expected_T=p['T'],
                forcing_G=p['G'], expected_steps=n, numerical_work=33*cfg['M']*n)


def canonical_record(d):
    cfg = d.get('config', d.get('job', {}).get('config'))
    error = d.get('E_grid', d.get('E_inf'))
    return dict(d, config=cfg, E_grid=error)


def prepare(out):
    import numpy as np
    from all_methods_dataset_loader import load_all_methods_equations
    from mc_optimized import prepare_all_methods_mc
    from mc_solver import auto_T
    contract = read_json(ROOT/CONTRACT); data = Path(contract['data_dir'])
    pins = {str(data/name): h for name, h in contract['dataset_sha256'].items()}
    pins[str(CONTRACT)] = digest(ROOT/CONTRACT)
    for name, h in pins.items():
        require(digest(ROOT/name) == h, name)
    eqs = {e.equation_id: e for e in load_all_methods_equations(
        ROOT/data/'all_methods_equations_v1.csv', ROOT/data/'all_methods_equations_v1_metadata.jsonl')}
    metadata = {pid: e.metadata for pid, e in eqs.items()}
    preflight = {p['problem_id']: p for p in read_json(ROOT/X0/'plan.json')['preflight']}
    # Re-derive analytical G/T/N for all 1000, without any stochastic solve.
    for pid, e in eqs.items():
        z = prepare_all_methods_mc(e.metadata); p = z.problem
        G = z.forcing_bound.abs_upper_bound
        T = auto_T(p.lambda_value, p.c_abs_upper_bound, G, 1e-4)
        require(T == preflight[pid]['T'] and G == preflight[pid]['G'], 'G/T changed: '+pid)
    out.mkdir(parents=True, exist_ok=False)
    (out/'PROJECT_STATUS.before.md').write_bytes((ROOT/'PROJECT_STATUS.md').read_bytes())
    protected = {}
    for folder in ('data', 'results', 'reports', 'ml'):
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and out not in path.parents and not any(
                    x in path.parts for x in ('__pycache__', '.DS_Store')) and not path.name.endswith('.lock'):
                protected[str(path.relative_to(ROOT))] = digest(path)
    protected.update({name: digest(ROOT/name) for name in NUMERICAL})
    publish(out/'protected_before.json', protected)
    inventory = []; seen = defaultdict(list); groups = defaultdict(list); rejected = []
    final_seeds = contract['final_seed_schedule']['MC1']
    for folder in ('results', 'reports'):
        for path in sorted((ROOT/folder).rglob('*.json')):
            if out in path.parents or path.parent.name not in ('records', 'attempts', 'runs', 'solutions'):
                continue
            if any(x in path.parts for x in ('sources', 'SOURCES_SNAPSHOT', 'snapshot', 'snapshots')):
                continue
            d = read_json(path); z = d.get('result', {}) if isinstance(d, dict) else {}
            if not isinstance(z, dict) or 'rng_scheme' not in z:
                continue
            pid = z.get('problem_id')
            if pid not in metadata or d.get('expression_hash') != metadata[pid]['expression_hash']:
                continue
            if len(z.get('x', [])) != 33 or z.get('M') not in (1024, 4096) or z.get('dt_requested') != 1/128:
                continue
            try:
                r = canonical_record(d); cfg = r['config']
                episode = d.get('job', {}).get('episode', d.get('job', {}).get('seed_index', 0))
                j = make_job(pid, cfg, metadata, preflight, episode, path.stem)
                exact = eqs[pid].y_exact(np.asarray(j['target_x']))
                validate_success(j, r, exact)
                plan_path = path.parent.parent/'plan.json'; old_plan = read_json(plan_path)
                require(d.get('plan_sha256') == digest(plan_path), 'old plan hash mismatch')
                old_pins = old_plan['protected_sha256']
                for name in NUMERICAL:
                    if name == 'src/problem_types.py':
                        continue  # older plans did not pin this data-only type module
                    require(old_pins.get(name) == digest(ROOT/name), 'old source mismatch: '+name)
            except (ValueError, KeyError) as exc:
                rejected.append(dict(path=str(path.relative_to(ROOT)), reason=str(exc))); continue
            # Ignore resource caps only for descriptive profile coverage; exact
            # replay manifests below keep every original config field unchanged.
            signature = (pid, cfg['M'], cfg['seed'], cfg['dt'], cfg['chunk'], cfg['stream_id'])
            for earlier in seen[signature]:
                require(earlier['result']['f'] == z['f'], 'conflicting duplicate outputs')
            seen[signature].append(r)
            group = str(path.parent.parent.relative_to(ROOT)); groups[group].append(j)
            inventory.append(dict(path=str(path.relative_to(ROOT)), sha256=digest(path),
                group=group, equation_id=pid, M=cfg['M'], seed=cfg['seed'], episode=episode,
                profile=j['profile'], E_grid=r['E_grid'], work=j['numerical_work'],
                seconds=d.get('method_wall_seconds', z['runtime_seconds']),
                complexity=metadata[pid]['complexity_level'], regime=metadata[pid]['regime'],
                final_seed_matches=cfg['M'] == 4096 and cfg['seed'] in final_seeds))
    require(not rejected, 'invalid historical results: '+str(rejected[:4]))
    summary_groups = []
    for group, jobs in groups.items():
        summary_groups.append(dict(group=group, records=len(jobs),
            equations=len({j['equation_id'] for j in jobs}), seeds=sorted({j['config']['seed'] for j in jobs})))
    mc1 = defaultdict(set); mc0 = defaultdict(set)
    for row in inventory:
        (mc1 if row['M'] == 4096 else mc0)[row['equation_id']].add(row['seed'])
    summary = dict(groups=summary_groups, validated_records=len(inventory), rejected=rejected,
        any_MC33_equations=len(set(mc1)|set(mc0)), MC1024_grid33_equations=len(mc0),
        MC1_grid33_equations=len(mc1), MC1_at_least_four_unique_seeds=sum(len(s)>=4 for s in mc1.values()),
        MC1_unique_equation_seed_calls=sum(map(len, mc1.values())),
        final_seeds=final_seeds, final_matching_calls=sum(len(s.intersection(final_seeds)) for s in mc1.values()),
        final_complete_equations=sum(set(final_seeds)<=s for s in mc1.values()),
        input_equations=len(eqs), final_total_calls=4*len(eqs),
        historical_numeric_coverage_is_not_final_seed_coverage=True)
    publish(out/'audit.json', summary); write_csv(out/'historical_mc33.csv', inventory)
    # All-1000 time proxy is measured MCx0 x 33 x 4 (paths), per equation.
    x0_times = defaultdict(list)
    for path in (ROOT/X0/'attempts').glob('*.json'):
        d = read_json(path); z = d.get('result', {})
        if z.get('success'):
            x0_times[z['problem_id']].append(z['runtime_seconds'])
    require(len(x0_times) == 1000 and all(len(t) == 4 for t in x0_times.values()), 'x0 coverage')
    final_jobs = []; coverage = []
    for pid in sorted(metadata, key=lambda p: ({'train':0, 'validation':1, 'test':2}[metadata[p]['split']], p)):
        cfg0 = dict(contract['profiles']['MC1']['config'])
        n = preflight[pid]['N']; work = 33*4096*n
        scenario = float(np.mean(x0_times[pid]))*33*4
        for episode, seed in enumerate(final_seeds):
            # Existing contract's job expansion rule, not the generic 1e8 cap.
            cfg = dict(cfg0, seed=seed, max_path_steps=work)
            j = make_job(pid, cfg, metadata, preflight, episode, f'{pid}_MC1_episode{episode}')
            j['scenario_seconds'] = scenario; final_jobs.append(j)
        coverage.append(dict(equation_id=pid, split=metadata[pid]['split'],
            historical_MC1_seeds=len(mc1[pid]), final_matching_seeds=len(mc1[pid].intersection(final_seeds)),
            pending_calls=4-len(mc1[pid].intersection(final_seeds)), work_per_call=work,
            scenario_seconds_per_call=scenario))
    # There are currently no matching final seeds. Do not silently discard any
    # future compatible record: a subsequent preparation must implement reuse.
    require(summary['final_matching_calls'] == 0, 'matching final records exist: reuse them')
    freeze(out/'production', final_jobs, data, pins, purpose='MC1 full1000; prepared, NOT started',
           final_seed_schedule=final_seeds, full_run_authorized=False,
           expansion_source='diagnostics/plan_ml_selector_contract.py:job; max_path_steps=33*M*N',
           priority='throughput, not isolated per-equation timing or ML labels')
    write_csv(out/'coverage1000.csv', coverage)
    publish(out/'remaining.json', dict(equations=1000, calls=len(final_jobs),
        node_seed_estimates=33*len(final_jobs), paths=33*4096*len(final_jobs),
        path_updates=sum(j['numerical_work'] for j in final_jobs),
        historical_x0_scaled_serial_seconds=sum(j['scenario_seconds'] for j in final_jobs)))
    bench_jobs = []; reference_pins = dict(pins)
    for pid in BENCH_IDS:
        for episode in (0, 1):
            row = next(r for r in inventory if r['equation_id'] == pid and r['episode'] == episode
                       and 'pilot_p2_' in r['group'] and r['M'] == 4096)
            old = read_json(ROOT/row['path']); j = dict(old['job'])
            j['reference'] = row['path']; j['reference_sha256'] = row['sha256']
            j['scenario_seconds'] = row['seconds']; bench_jobs.append(j)
            reference_pins[row['path']] = row['sha256']
    for workers in (1, 2, 4, 8):
        freeze(out/f'benchmark_w{workers}', bench_jobs, data, reference_pins,
               purpose='isolated exact historical replay; never merged into production',
               selection=dict(ids=BENCH_IDS, episodes=[0, 1],
                   reason='all four L levels/all five regimes; typical costs; not an accuracy sample',
                   limitation='does not include rare 100-315s historical heavy tails'),
               requested_workers=workers)
    publish(out/'benchmark_protocol.json', dict(workers=[1,2,4,8], calls_each=len(bench_jobs),
        reference_seconds=sum(j['scenario_seconds'] for j in bench_jobs),
        numerical_changes=False, seed_changes=False, baseline='same jobs, one worker',
        gate='verify old numerical fields after w1 before any other worker count',
        memory='sampled RSS lifetime maxima, controller plus workers; guardians separately small',
        no_full_run=True))
    print(json.dumps(summary, indent=2))


def verify_replay(out):
    import numpy as np
    plan = load_plan(out); ph = digest(out/'plan.json'); rows = []
    ignored = {'runtime_seconds','preparation_seconds','ast_preparation_seconds',
               'numba_import_seconds','compilation_seconds','numerical_seconds','rng_seconds'}
    for j in plan['jobs']:
        r = payload(out/'records'/(j['key']+'.json'), j, ph)
        old = canonical_record(read_json(ROOT/j['reference']))
        require(r['status'] == 'completed', j['key'])
        fields = set(old['result']) | set(r['result'])
        differences = [k for k in sorted(fields-ignored) if old['result'].get(k) != r['result'].get(k)]
        for name in ('f', 'stderr', 'ci95', 'ci_lower', 'ci_upper', 'x'):
            require(np.asarray(old['result'][name], dtype=np.float64).tobytes() ==
                    np.asarray(r['result'][name], dtype=np.float64).tobytes(),
                    'bitwise array mismatch: '+j['key']+' '+name)
        require(not differences and old['E_grid'] == r['E_grid'] and old['y_exact'] == r['y_exact'],
                'replay mismatch: '+j['key']+' '+str(differences))
        rows.append(dict(key=j['key'], reference=j['reference'], bitwise_numeric_equal=True,
                         exact_equal=True, error_equal=True, fields_checked=len(fields-ignored)))
    report = dict(passed=True, completed=len(rows), checks=rows)
    publish(out/'equivalence.json', report)
    print(json.dumps(dict(passed=True, completed=len(rows))))


def integrity(out):
    before = read_json(out/'protected_before.json')
    changed = [name for name, h in before.items() if not (ROOT/name).is_file() or digest(ROOT/name) != h]
    report = dict(passed=not changed, checked=len(before), changed=changed)
    publish(out/'integrity.json', report)
    require(not changed, str(changed)); print(json.dumps(report))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare','verify-replay','integrity'))
    p.add_argument('--out', required=True, type=Path); a = p.parse_args()
    {'prepare':prepare, 'verify-replay':verify_replay, 'integrity':integrity}[a.command](a.out.resolve())


if __name__ == '__main__':
    main()
