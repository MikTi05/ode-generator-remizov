"""Read saved grid33 values only; never run a numerical method or fit a model.

The main comparison uses historical mass R128 (taper), without replacing it
with later R128 zero-boundary runs. MC1024 uses the complete joint100 primary
seed cohort. MC4096 is a separate, descriptive fixed-profile comparison.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

EPSILON = 0.05
GRID = [-1.0 + i / 16 for i in range(33)]
ORIGINAL = 'data/all_methods_v1_1000_2026-10-18'
EXTERNAL = 'results/ml_selector_final140_protocol_2026-10-27'
MC4096_SOURCES = (
    'results/final_grid5_2026-10-25',
    'reports/ml_selector_contract/sentinel_p1_2026-10-26',
    'reports/ml_selector_contract/pilot_p2_2026-10-26',
    'results/f0047_observed_timing_2026-10-26',
    'results/f0047_three_profile_2026-10-26',
    'results/ml_probability_validation_2026-10-26',
)
R256_SOURCES = (
    'results/final_grid5_2026-10-25',
    'results/remizov_r0_2026-10-26',
    'results/ml_probability_validation_2026-10-26',
)


def build_grid_evidence(root, original_metadata=None, external_metadata=None):
    """Return ``rows, seed_rows, source_sha256, audit`` in a dictionary.

    Equation identity includes its dataset SHA and expression hash. Metadata
    arguments, if provided, are dictionaries keyed by equation_id. All raw
    values and configs are read independently from immutable result files.
    """
    root = Path(root).resolve()
    pins = {}

    def sha(path):
        path = Path(path)
        name = str(path.relative_to(root))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if name in pins and pins[name] != digest:
            raise ValueError('input changed during grid audit: ' + name)
        pins[name] = digest
        return digest

    def read(path):
        sha(path)
        return json.loads(Path(path).read_text())

    def metadata(folder):
        path = folder / 'all_methods_equations_v1_metadata.jsonl'
        sha(path)
        return {d['equation_id']: d for line in path.read_text().splitlines()
                for d in [json.loads(line)]}

    original_dir = root / ORIGINAL
    original_sha = sha(original_dir / 'all_methods_equations_v1.csv')
    original_metadata = original_metadata or metadata(original_dir)
    final_dir = root / EXTERNAL
    final_plan = read(final_dir / 'plan.json')
    external_dir = root / final_plan['data_dir']
    sha(external_dir / 'all_methods_equations_v1.csv')
    # test_inputs is a saved subset; jobs identify the parent generated pool.
    external_sha = sha(external_dir.parent / 'input_pool' / 'all_methods_equations_v1.csv')
    external_metadata = external_metadata or metadata(external_dir)
    if original_sha == external_sha:
        raise ValueError('external dataset identity must differ from original')

    def verify_plan(path, expected_csv, expected_sha):
        plan = read(path)
        name = str(expected_csv.relative_to(root))
        if plan.get('protected_sha256', {}).get(name) != expected_sha:
            raise ValueError('plan dataset SHA mismatch: ' + str(path))
        return plan

    verify_plan(final_dir / 'plan.json',
                external_dir.parent / 'input_pool' / 'all_methods_equations_v1.csv',
                external_sha)

    def checked(path, meta, dataset_sha, method, M=None):
        d = read(path)
        j = d.get('job', {})
        z = d.get('result') or {}
        cfg = d.get('config', j.get('config', {}))
        pid = d.get('equation_id', d.get('problem_id',
                    j.get('equation_id', j.get('problem_id', z.get('problem_id')))))
        m = meta.get(pid)
        if m is None:
            raise ValueError('unknown grid equation: ' + str(pid))
        if not z.get('success') or z.get('failure_reason'):
            return None
        identity = d.get('metadata', d)
        if identity.get('expression_hash') != m['expression_hash']:
            raise ValueError('grid expression identity mismatch: ' + str(path))
        if identity.get('family_id') != m['family_id'] or identity.get('split') != m['split']:
            raise ValueError('grid split identity mismatch: ' + str(path))
        if z.get('x') != GRID or d.get('target_x', j.get('target_x', GRID)) != GRID:
            raise ValueError('grid33 required: ' + str(path))
        f, exact = z['f'], d['y_exact']
        if len(f) != 33 or len(exact) != 33 or not all(math.isfinite(v) for v in [*f, *exact]):
            raise ValueError('finite saved grid arrays required')
        errors = [abs(a - b) for a, b in zip(f, exact)]
        row = dict(equation_id=pid, family_id=m['family_id'], split=m['split'],
                   expression_hash=m['expression_hash'], dataset_sha256=dataset_sha,
                   method=method, config=cfg, f=f, y_exact=exact,
                   E_grid33=max(errors), grid_abs_errors=errors,
                   source=str(path.relative_to(root)), source_sha256=sha(path),
                   T=z['T'], n_steps=z.get('n_steps'), seed=None)
        if method == 'monte_carlo':
            required = dict(M=M, dt=1 / 128, chunk=1024, max_batch_paths=1024,
                            tail_eps=1e-4, stream_id=0, T=None)
            if any(cfg.get(k) != v for k, v in required.items()):
                raise ValueError('different MC physical profile: ' + str(path))
            if (z.get('M') != M or z.get('completed_paths') != [M] * 33
                    or z.get('dt_requested') != 1 / 128
                    or z.get('backend') != 'numba' or z.get('block_steps') != 64
                    or z.get('rng_scheme') != 'PCG64/SeedSequence/point_address_v1'
                    or z.get('seed_entropy') != cfg['seed']
                    or z.get('stream_id') != cfg['stream_id']):
                raise ValueError('inconsistent MC result/config')
            row.update(seed=z['seed_entropy'], point_spawn_keys=z['point_spawn_keys'],
                       rng_scheme=z['rng_scheme'], mc_M=M,
                       dt_actual=z['dt_actual'], stream_id=z['stream_id'])
            expected_keys = [[*struct.unpack('<8I', hashlib.sha256(
                b'mc_point_v1\0' + pid.encode() + struct.pack('<d', 0. if x == 0 else x)
            ).digest()), cfg['stream_id']] for x in GRID]
            if z['point_spawn_keys'] != expected_keys:
                raise ValueError('33 correct semantic point streams required')
            if (not 0 < z['dt_actual'] <= cfg['dt']
                    or z['n_steps'] != math.ceil(z['T'] / cfg['dt'])
                    or z['n_path_steps'] != 33 * M * z['n_steps']
                    or z['n_normal_draws'] != z['n_path_steps']):
                raise ValueError('MC completed work/discretization mismatch')
        else:
            if cfg['n'] != z['n'] or z.get('scheme') != 'solve':
                raise ValueError('inconsistent Chernoff result/config')
            if cfg['n'] == 256 and cfg['T'] != z['T']:
                raise ValueError('R256 frozen horizon mismatch')
            row['remizov_n'] = cfg['n']
        return row

    mass_plan_path = root / 'results/chernov_solve1000_2026-10-18/plan.json'
    original_csv = original_dir / 'all_methods_equations_v1.csv'
    mass_plan = verify_plan(mass_plan_path, original_csv, original_sha)
    mass_r = {}
    for path in sorted((mass_plan_path.parent / 'solutions').glob('*.json')):
        d = read(path)
        if d['config'] != mass_plan['config'] or d['config']['n'] != 128:
            raise ValueError('mass R128 profile mismatch')
        # This historical driver hashed normalized plan JSON, rather than bytes.
        normalized_plan_sha = hashlib.sha256(json.dumps(mass_plan, sort_keys=True).encode()).hexdigest()
        if d['plan_sha256'] != normalized_plan_sha:
            raise ValueError('mass R128 result/plan mismatch')
        row = checked(path, original_metadata, original_sha, 'remizov')
        if row is not None:
            mass_r[row['equation_id']] = row

    r256 = {}
    for folder in R256_SOURCES:
        plan_path = root / folder / 'plan.json'
        verify_plan(plan_path, original_csv, original_sha)
        for path in sorted((root / folder / 'records').glob('*.json')):
            d = read(path)
            cfg = d.get('config', d.get('job', {}).get('config', {}))
            if cfg.get('n') != 256:
                continue
            if d['plan_sha256'] != sha(plan_path):
                raise ValueError('R256 result/plan mismatch')
            if any(cfg.get(k) != v for k, v in dict(boundary='zero', interp='cubic',
                    dx=.0125, padding=12., num_t=513, t_grading='quadratic', tail_eps=1e-4).items()):
                raise ValueError('different secondary R256 physical profile')
            row = checked(path, original_metadata, original_sha, 'remizov')
            if row is None:
                continue
            old = r256.get(row['equation_id'])
            if old and (old['f'] != row['f'] or old['config'] != row['config']):
                raise ValueError('conflicting deterministic R256 repeats')
            r256.setdefault(row['equation_id'], row)

    joint_dir = root / 'results/joint_grid100_2026-10-25'
    joint_plan = verify_plan(joint_dir / 'plan.json', original_csv, original_sha)
    primary_mc = defaultdict(list)
    for path in sorted((joint_dir / 'records').glob('*.json')):
        d = read(path)
        j = d.get('job', {})
        if j.get('method') != 'MC' or j.get('stage') != 'primary':
            continue
        if j['config']['seed'] != joint_plan['seeds'][0] or j['seed_index'] != 0:
            raise ValueError('primary joint seed mismatch')
        if d['plan_sha256'] != sha(joint_dir / 'plan.json'):
            raise ValueError('joint plan mismatch')
        row = checked(path, original_metadata, original_sha, 'monte_carlo', 1024)
        if row:
            primary_mc[row['equation_id']].append(row)
    if any(len(rs) != 1 for rs in primary_mc.values()):
        raise ValueError('one primary MC1024 seed required')

    mc4096 = defaultdict(list)
    duplicates = []
    seen = {}
    for folder in MC4096_SOURCES:
        plan_path = root / folder / 'plan.json'
        verify_plan(plan_path, original_csv, original_sha)
        for path in sorted((root / folder / 'records').glob('*.json')):
            d = read(path)
            cfg = d.get('config', d.get('job', {}).get('config', {}))
            if cfg.get('M') != 4096 or cfg.get('dt') != 1 / 128:
                continue
            if d['plan_sha256'] != sha(plan_path):
                raise ValueError('MC4096 plan mismatch')
            row = checked(path, original_metadata, original_sha, 'monte_carlo', 4096)
            if row is None:
                continue
            physical_cfg = {k: v for k, v in cfg.items()
                            if k not in {'max_path_steps', 'max_runtime_seconds', 'max_steps'}}
            key = json.dumps(dict(dataset_sha256=original_sha,
                    expression_hash=row['expression_hash'], equation_id=row['equation_id'],
                    config=physical_cfg, T=row['T'], n_steps=row['n_steps'],
                    point_spawn_keys=row['point_spawn_keys'], rng_scheme=row['rng_scheme']), sort_keys=True)
            if key in seen:
                old = seen[key]
                if old['f'] != row['f'] or old['y_exact'] != row['y_exact']:
                    raise ValueError('conflicting semantic MC replay')
                duplicates.append(dict(kept=old['source'], duplicate=row['source']))
            else:
                seen[key] = row
                mc4096[row['equation_id']].append(row)

    ext_r256, ext_mc = {}, defaultdict(list)
    for path in sorted((final_dir / 'records').glob('*.json')):
        d = read(path)
        if d['plan_sha256'] != sha(final_dir / 'plan.json'):
            raise ValueError('external result/plan mismatch')
        if d['job']['identity'].split('|')[0] != external_sha:
            raise ValueError('external parent dataset identity mismatch')
        cfg = d['config']
        if cfg.get('n') == 256:
            row = checked(path, external_metadata, external_sha, 'remizov')
            if row:
                ext_r256[row['equation_id']] = row
        elif cfg.get('M') == 4096:
            if cfg['seed'] != final_plan['MC_seed']:
                raise ValueError('external MC seed mismatch')
            row = checked(path, external_metadata, external_sha, 'monte_carlo', 4096)
            if row:
                ext_mc[row['equation_id']].append(row)

    rows, seed_rows, groups = [], [], []

    def group(name, cohort, context, universe, rr, mm):
        pairs = sorted(set(rr) & set(mm))
        selected_rows = []
        for pid in pairs:
            r, ms = rr[pid], mm[pid]
            if any(r['y_exact'] != m['y_exact'] for m in ms):
                raise ValueError('R/MC exact grid mismatch: ' + pid)
            if r['remizov_n'] == 256 and any(r['T'] != m['T'] for m in ms):
                raise ValueError('R256/MC common frozen horizon mismatch: ' + pid)
            errors = [m['E_grid33'] for m in ms]
            er, em = r['E_grid33'], statistics.median(errors)
            best = 'remizov' if er <= em else 'monte_carlo'
            row = {k: r[k] for k in ('equation_id', 'family_id', 'split', 'expression_hash', 'dataset_sha256')}
            row.update(group=name, cohort=cohort, context=context,
                       metric='max_abs_error_on_fixed_grid33[-1,1]', epsilon=EPSILON,
                       E_R=er, E_MC=em, best_method=best,
                       remizov_success=er <= EPSILON, mc_success=em <= EPSILON,
                       chosen_success=min(er, em) <= EPSILON,
                       remizov_n=r['remizov_n'], mc_M=ms[0]['mc_M'],
                       mc_seed_count=len(ms), mc_seed_errors=errors,
                       mc_error_min=min(errors), mc_error_max=max(errors),
                       mc_error_std=statistics.pstdev(errors),
                       mc_seeds=[m['seed'] for m in ms],
                       remizov_config=r['config'], mc_configs=[m['config'] for m in ms],
                       remizov_source=r['source'], mc_sources=[m['source'] for m in ms])
            rows.append(row)
            selected_rows.append(row)
            for m in ms:
                seed_rows.append(dict(m, group=name, cohort=cohort, context=context,
                                      epsilon=EPSILON, mc_success=m['E_grid33'] <= EPSILON))
        by_split = {}
        for split in ('train', 'validation', 'test'):
            n = sum(m['split'] == split for m in universe.values())
            p = sum(universe[pid]['split'] == split for pid in pairs)
            by_split[split] = dict(universe_n=n, pairs_n=p, missing_pairs_n=n-p)
        groups.append(dict(name=name, cohort=cohort, context=context,
                           universe_n=len(universe), pairs_n=len(pairs),
                           missing_pairs_n=len(universe)-len(pairs), by_split=by_split,
                           winners={m: sum(r['best_method'] == m for r in selected_rows)
                                    for m in ('remizov', 'monte_carlo')},
                           success_counts={m: sum(r[k] for r in selected_rows)
                                           for m, k in [('remizov', 'remizov_success'),
                                                        ('monte_carlo', 'mc_success'),
                                                        ('oracle', 'chosen_success')]},
                           seed_count_distribution=dict(sorted(Counter(len(mm[pid]) for pid in pairs).items())),
                           mc_aggregation='median of per-seed max absolute grid33 errors',
                           pairing_scope='saved grid33 nodes only; no continuous interval certificate'))

    group('mass_R128_MC1024_primary_grid33', 'original_1000',
          'historical mass R128 taper vs joint100 primary MC1024; one MC seed, separate from x0 median4',
          original_metadata, mass_r, primary_mc)
    group('secondary_R256_MC1024_primary_grid33', 'original_1000',
          'fixed R256 zero/commonT vs joint100 primary MC1024; one MC seed',
          original_metadata, r256, primary_mc)
    group('secondary_R256_MC4096_grid33', 'original_1000',
          'fixed R256 zero/commonT vs MC4096; all saved independent semantic seeds, unequal seed budgets',
          original_metadata, r256, mc4096)
    group('external_R256_MC4096_grid33', 'independent_60',
          'separate external dataset; R256 zero/commonT vs MC4096 episode0; one MC seed on twenty of sixty',
          external_metadata, ext_r256, ext_mc)
    return dict(rows=rows, seed_rows=seed_rows, source_sha256=pins,
                audit=dict(grid_groups=groups, mc4096_replays_excluded=duplicates,
                           numerical_solver_calls=0, model_fit_calls=0))
