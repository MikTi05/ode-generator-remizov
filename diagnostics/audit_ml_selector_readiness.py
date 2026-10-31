"""Read-only audit of saved solver/ML evidence; never solve, fit or regenerate.

Run from the project root with .venv/bin/python -B. Outputs must be new.
The main CSV retains the MASS Remizov taper profile and records MC grid33
separately from MC x0. Comparable pairs have their own explicit joint columns;
pair_comparable refers to those columns, never to historical taper timing.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT/'data/all_methods_v1_1000_2026-10-18'
R = ROOT/'results/chernov_solve1000_2026-10-18'
C = ROOT/'results/mc1000_x0_2026-10-25'
J = ROOT/'results/joint_grid100_2026-10-25'
F = ROOT/'results/final_grid5_2026-10-25'
INPUTS = {}


def sha(path):
    path = Path(path)
    name = str(path.relative_to(ROOT))
    if name in INPUTS:
        return INPUTS[name]
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    INPUTS[name] = value
    return value


def read(path):
    sha(path)
    return json.loads(Path(path).read_text())


def csvread(path):
    sha(path)
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write(path, data):
    payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True,
                         allow_nan=False, default=lambda value: value.item())+'\n'
    with Path(path).open('x') as stream:
        stream.write(payload)


def csvwrite(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def audit(out):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
    from all_methods_dataset_loader import load_all_methods_equations
    from ml.remizov_features import load_inputs, coefficient_features, CERT_FIELDS

    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ('audit_by_equation.csv', 'audit_summary.json', 'run_evidence.csv',
                 'input_sha256.json', 'special_equations.csv', 'ml_checks.json', 'artifact_inventory.json'):
        if (out/name).exists():
            raise FileExistsError(f'refusing to overwrite {out/name}')
    report = read(DATA/'all_methods_equations_v1_report.json')
    for name, digest in report['output_sha256'].items():
        assert sha(DATA/name) == digest, name
    meta_path = DATA/'all_methods_equations_v1_metadata.jsonl'
    sha(meta_path)
    metadata = [json.loads(s) for s in meta_path.read_text().splitlines()]
    meta = {m['equation_id']: m for m in metadata}
    source = csvread(DATA/'all_methods_equations_v1.csv')
    ids = [r['equation_id'] for r in source]
    canonical = set(ids)
    assert len(ids) == len(canonical) == len(meta) == len(metadata) == 1000
    assert canonical == set(meta)
    assert {m['master_seed'] for m in metadata} == {2488}
    assert {m['generation_version'] for m in metadata} == {'all_methods_v1'}
    families = defaultdict(list)
    topologies = defaultdict(set)
    for m in metadata:
        families[m['family_id']].append(m)
        topologies[m['structure_signature']].add(m['split'])
        assert m['equation_id'] == f"AMV1_{m['family_id']}_S{m['sample_index']:02d}"
    assert len(families) == len(topologies) == 200
    assert all(len(v) == 5 and len({m['split'] for m in v}) == 1 for v in families.values())
    assert all(len(v) == 1 for v in topologies.values())
    assert len({m['expression_hash'] for m in metadata}) == 1000
    assert len({m['feature_hash'] for m in metadata}) == 1000
    equations = load_all_methods_equations(DATA/'all_methods_equations_v1.csv', meta_path)
    by_id = {q.equation_id: q for q in equations}
    grid = np.linspace(-1., 1., 33)
    reference = {q.equation_id: np.asarray(q.y_exact(grid)) for q in equations}
    maxima = Counter()
    with np.load(DATA/'all_methods_equations_v1_ml.npz', allow_pickle=False) as z:
        z = {key: z[key] for key in z.files}
        assert list(z['equation_id']) == ids
        assert list(z['family_id']) == [meta[i]['family_id'] for i in ids]
        assert list(z['split']) == [meta[i]['split'] for i in ids]
        assert list(z['channel_order']) == ['a', 'b', 'c', 'g']
        assert z['X'].shape == (1000, 4, 256)
        x = z['x_grid']; original_values = z['X'].copy()
        for i, q in enumerate(equations):
            p, ivp, m = q.remizov_problem, q.cauchy_problem, q.metadata
            assert p.bounds_certification == 'analytic' and p.bounds_domain == 'all_real_x'
            assert 0 < m['a_lower_cert'] <= m['a_upper_cert'] and m['remizov_gap_cert'] > 0
            assert m['support_left'] < -1 and m['support_right'] > 1
            arrays = [p.a(x), p.b(x), p.c_remizov(x)-p.lambda_value, -p.g_remizov(x)]
            np.testing.assert_allclose(arrays, z['X'][i], rtol=2e-13, atol=2e-13)
            np.testing.assert_allclose(q.y_exact(x), z['y_exact'][i], rtol=2e-13, atol=2e-13)
            np.testing.assert_allclose(q.dy_exact(x), z['dy_exact'][i], rtol=2e-13, atol=2e-13)
            assert abs(q.y_exact(-1) - z['y0'][i]) < 2e-13
            assert abs(q.dy_exact(-1) - z['v0'][i]) < 2e-13
            np.testing.assert_allclose(p.c_remizov(x)-p.lambda_value, arrays[2], atol=2e-13)
            np.testing.assert_allclose(p.g_remizov(x), -arrays[3], atol=2e-13)
            residual = arrays[0]*q.d2y_exact(x)+arrays[1]*q.dy_exact(x)+arrays[2]*q.y_exact(x)-arrays[3]
            maxima['residual_max'] = max(maxima['residual_max'], float(np.max(abs(residual))))
            assert np.all(np.isfinite(reference[q.equation_id]))
        assert all(np.all(np.isfinite(z[k])) for k in ('X', 'y_exact', 'dy_exact', 'y0', 'v0', 'x_grid'))
    # Explicitly identify the deceptive ID collision with the old 20-row dataset.
    small = [json.loads(s) for s in (ROOT/'data/generated/all_methods_equations_v1_metadata.jsonl').read_text().splitlines()]
    sha(ROOT/'data/generated/all_methods_equations_v1_metadata.jsonl')
    collisions = [m['equation_id'] for m in small if m['equation_id'] in canonical]
    identical = [m['equation_id'] for m in small if m['equation_id'] in canonical and
                 m['expression_hash'] == meta[m['equation_id']]['expression_hash']]
    plans = {name: read(folder/'plan.json') for name, folder in [('R', R), ('C', C), ('J', J), ('F', F)]}
    source_versions = {}
    for plan in plans.values():
        for name, digest in plan['protected_sha256'].items():
            source_versions.setdefault(name, {})[digest] = sha(ROOT/name) == digest
            if name.startswith('data/'):
                assert sha(ROOT/name) == digest, name
    evidence = []
    def record(path, value, method, profile, config, stored_error, status):
        pid = value.get('problem_id', (value.get('result') or {}).get('problem_id')); assert pid in canonical
        m = meta[pid]
        identity = value.get('expression_hash', value.get('metadata', {}).get('expression_hash'))
        if identity is not None: assert identity == m['expression_hash'], str(path)
        result = value.get('result') or {}
        success = bool(result.get('success'))
        err = None
        if success:
            xx = np.asarray(result['x']); ff = np.asarray(result['f'])
            assert xx.shape == ff.shape and np.isfinite(ff).all()
            assert np.array_equal(xx, grid) or np.array_equal(xx, [0.])
            yy = reference[pid] if len(xx) == 33 else reference[pid][16:17]
            np.testing.assert_allclose(np.asarray(value['y_exact']).reshape(-1), yy, rtol=0, atol=2e-13)
            err = float(np.max(abs(ff-yy)))
            assert math.isclose(err, stored_error, rel_tol=0, abs_tol=3e-13), str(path)
            assert result['runtime_seconds'] > 0 and math.isfinite(result['runtime_seconds'])
            if method == 'MC':
                assert result['completed_paths'] == [config['M']]*len(xx)
                assert result['backend'] == 'numba' and result['rng_scheme'] == 'PCG64/SeedSequence/point_address_v1'
                assert result['seed_entropy'] == config['seed'] and result['tail_tolerance_met']
                assert result['n_path_steps'] == result['n_normal_draws'] == config['M']*result['n_steps']*len(xx)
                assert np.isfinite(result['stderr']).all() and min(result['stderr']) >= 0
                assert result['discretization_error_bound'] is None and not result['total_error_certified']
            else:
                assert result['n'] == config['n'] and result['time_nodes'] == config['num_t']
                assert result['n_operator_applications'] == config['n']*(config['num_t']-1)
        row = dict(equation_id=pid, family_id=m['family_id'], split=m['split'], method=method,
                   profile=profile, source=str(path.relative_to(ROOT)), source_sha256=sha(path),
                   status=status, success=success, grid_points=len(result.get('x', [])), error=err,
                   accuracy_pass=None if err is None else err <= .01,
                   runtime=result.get('runtime_seconds'), T=result.get('T'),
                   config_json=json.dumps(config, sort_keys=True), identity_verified=identity is not None)
        evidence.append(row)
        return row
    history = {}
    for path in sorted((R/'solutions').glob('*.json')):
        v = read(path); pid = v['problem_id']; assert path.stem == pid and pid not in history
        assert v['config'] == plans['R']['config']
        assert v['plan_sha256'] == hashlib.sha256(json.dumps(plans['R'], sort_keys=True).encode()).hexdigest()
        assert v['implementation'] == 'chernov_solver.ChernovSolver.solve'
        request = R/'attempts'/v['attempt_id']/'request.json'
        assert read(request)['problem_id'] == pid
        assert read(request.parent/'result.json') == v
        row = record(path, v, 'Remizov', 'mass_taper_n128_legacyT_grid33', v['config'],
                     v['max_abs_error'], v['computation_status'])
        assert v['accuracy_reached'] == bool(row['accuracy_pass'])
        history[pid] = row
    assert set(history) == canonical
    # Terminal solutions are not additional attempts. Count request and result stores separately.
    requests = [read(p) for p in sorted((R/'attempts').glob('*/request.json'))]
    terminal_attempts = [read(p) for p in sorted((R/'attempts').glob('*/result.json'))]
    x0 = defaultdict(list)
    for path in sorted((C/'attempts').glob('*.json')):
        v = read(path); pid = v['problem_id']
        assert v['plan_sha256'] == sha(C/'plan.json') and v['status'] == 'completed'
        assert v['target_x'] == [0.] and v['metadata_identity']['family_id'] == meta[pid]['family_id']
        assert v['metadata_identity']['split'] == meta[pid]['split']
        assert v['config'] == dict(plans['C']['config'], seed=plans['C']['seeds'][v['seed_index']])
        row = record(path, v, 'MC', 'mass_x0_M1024_single_seed', v['config'], v['abs_error'], v['status'])
        assert row['accuracy_pass'] == v['accuracy_met']
        x0[pid].append((v, row))
    assert set(x0) == canonical
    assert all(sorted(v['seed_index'] for v, _ in rows) == [0, 1, 2, 3] for rows in x0.values())
    joint = defaultdict(dict); fresh = defaultdict(list); boundary = []
    for folder, name in [(J, 'J'), (F, 'F')]:
        for path in sorted((folder/'records').glob('*.json')):
            v = read(path); job = v['job']; pid = job['problem_id']; method = job['method']
            assert v['plan_sha256'] == sha(folder/'plan.json') and v['status'] == 'completed'
            assert v['family_id'] == meta[pid]['family_id'] and v['split'] == meta[pid]['split'] == 'train'
            profile = ('joint100' if name == 'J' else 'final5')+'_'+method+'_'+job['stage']
            value = dict(v, problem_id=pid)
            row = record(path, value, 'MC' if method == 'MC' else 'Remizov', profile,
                         job['config'], v['E_inf'], v['status'])
            assert row['accuracy_pass'] == v['accuracy_met']
            if name == 'J' and job['stage'] == 'primary':
                assert (method, job.get('seed_index', 0)) not in joint[pid]
                joint[pid][method] = (v, row)
            if name == 'F': fresh[pid].append((v, row))
    assert len(joint) == 100 and all(set(v) == {'R', 'MC'} for v in joint.values())
    for pid, pair in joint.items():
        rv, rr = pair['R']; cv, cr = pair['MC']
        assert rv['result']['x'] == cv['result']['x'] == list(grid)
        assert rv['result']['T'] == cv['result']['T']
        assert rv['y_exact'] == cv['y_exact']
        assert rv['job']['config']['boundary'] == 'zero'
    for path in sorted((ROOT/'results/chernov_boundary_diagnosis_2026-10-24/runs').glob('*.json')):
        v = read(path); pid = v['problem_id']
        assert v['production_result_sha256'] == sha(R/'solutions'/f'{pid}.json')
        row = record(path, v, 'Remizov', 'boundary_'+v['mode'], v['config'], v['max_abs_error'],
                     'completed' if v['result']['success'] else 'failed')
        boundary.append(row)
    for folder in ('chernov_time_quadrature1025_2026-10-24', 'chernov_window_padding24_2026-10-24'):
        path = ROOT/'results'/folder/'result.json'; v = read(path)
        record(path, v, 'Remizov', folder, v['config'], v['max_abs_error'],
               'completed' if v['result']['success'] else 'failed')
    table = []
    for pid in ids:
        m = meta[pid]; hr = history[pid]; mcrows = x0[pid]
        mean0 = float(np.mean([v['result']['f'][0] for v, _ in mcrows]))
        error0 = float(abs(mean0-reference[pid][16]))
        pair = joint.get(pid)
        mcgrid = pair['MC'][1] if pair else None
        rpair = pair['R'][1] if pair else None
        reasons = []
        if not hr['success']: reasons.append('remizov_worker_exception_no_error')
        if pair is None: reasons.append('mc_grid33_not_run;only_x0_available')
        else: reasons.append('mc_reliability_and_time_bias_unconfirmed')
        row = dict(equation_id=pid, family_id=m['family_id'], split=m['split'], reference_valid=True,
            remizov_run_found=True, remizov_status=hr['status'], remizov_error=hr['error'],
            remizov_runtime=hr['runtime'], remizov_profile=hr['profile'], remizov_source=hr['source'],
            remizov_accuracy_pass=hr['accuracy_pass'], mc_run_found=True,
            mc_status='completed_grid33' if pair else 'completed_x0_only',
            mc_error=mcgrid['error'] if pair else None, mc_runtime=mcgrid['runtime'] if pair else None,
            mc_profile=mcgrid['profile'] if pair else 'mass_x0_M1024_single_seed',
            mc_accuracy_pass=mcgrid['accuracy_pass'] if pair else None,
            mc_grid33_run_found=bool(pair), mc_x0_runs=4, mc_x0_mean_error=error0,
            mc_x0_mean_runtime=sum(r['runtime'] for _, r in mcrows),
            mc_x0_observed_successes=sum(r['accuracy_pass'] for _, r in mcrows),
            pair_comparable=bool(pair), pair_scope='joint100_primary_commonT_grid33' if pair else None,
            pair_remizov_error=rpair['error'] if pair else None,
            pair_remizov_runtime=rpair['runtime'] if pair else None,
            pair_remizov_profile=rpair['profile'] if pair else None,
            pair_remizov_source=rpair['source'] if pair else None,
            pair_mc_source=mcgrid['source'] if pair else None,
            pair_timing_scope='warm_solver_wall_excludes_external_AST_G_preparation' if pair else None,
            label='unknown', label_confidence_or_status='stochastic_uncertain' if pair else 'missing_grid33',
            blocking_reason=';'.join(reasons))
        table.append(row)
    # Cross-check published derived tables, never use their labels as truth.
    published = {r['equation_id']: r for r in csvread(C/'analysis/equations.csv')}
    for row in table:
        p = published[row['equation_id']]
        assert abs(float(p['mc_aggregate_error'])-row['mc_x0_mean_error']) < 3e-13
        assert int(p['mc_success_count']) == row['mc_x0_observed_successes']
    jpub = {r['problem_id']: r for r in csvread(J/'analysis/comparison.csv')}
    for pid, pair in joint.items():
        for method, col in [('R', 'R_error'), ('MC', 'MC_primary_error')]:
            assert abs(float(jpub[pid][col])-pair[method][1]['error']) < 3e-13
    adaptive_pass = 0
    for pid in joint:
        rows = [r for r in evidence if r['equation_id']==pid and r['method']=='MC' and r['profile'].startswith('joint100_')]
        values = [read(ROOT/r['source'])['result']['f'] for r in rows]
        error = float(np.max(abs(np.mean(values, axis=0)-reference[pid])))
        assert abs(error-float(jpub[pid]['MC_final_error'])) < 3e-13
        adaptive_pass += error <= .01
    final_pub = {r['problem_id']:r for r in csvread(F/'analysis/comparison.csv')}
    from scipy.stats import beta, binom
    final_checks = {}
    for pid, rows in fresh.items():
        ms = [(v,r) for v,r in rows if r['method']=='MC']
        successes = sum(r['accuracy_pass'] for _,r in ms)
        average = np.mean([v['result']['f'] for v,_ in ms], axis=0)
        error = float(np.max(abs(average-reference[pid])))
        assert successes == int(final_pub[pid]['MC_successes'])
        assert abs(error-float(final_pub[pid]['MC_mean16384_error'])) < 3e-13
        low = float(beta.ppf(.005, successes, 5-successes)) if successes else 0.
        high = float(beta.ppf(.995, successes+1, 4-successes)) if successes<4 else 1.
        assert abs(low-float(final_pub[pid]['MC_probability_family95_lower'])) < 1e-12
        assert abs(high-float(final_pub[pid]['MC_probability_family95_upper'])) < 1e-12
        final_checks[pid] = dict(successes=successes, repeats=4, aggregate_error=error,
                                p95_test_pvalue=float(binom.cdf(successes,4,.95)),
                                family95_interval=[low,high])
    # Rebuild only input features and recompute metrics from frozen predictions.
    inputs = load_inputs(DATA); matrix = inputs['X']
    for folder in (ROOT/'results/remizov_ml_v1_2026-10-25', C/'ml'):
        sha(folder/'features.npz')
        with np.load(folder/'features.npz', allow_pickle=False) as z:
            np.testing.assert_array_equal(z['X'], matrix)
            np.testing.assert_array_equal(z['equation_id'], ids)
            np.testing.assert_array_equal(z['split'], inputs['splits'])
        assert read(folder/'feature_schema.json')['feature_names'] == inputs['feature_names']
    clean = {k: metadata[0][k] for k in CERT_FIELDS}
    clean['serialized_ast'] = {k: metadata[0]['serialized_ast'][k] for k in ('a_raw','b_raw','c_raw')}
    assert coefficient_features(original_values[0], inputs['x_grid'], clean) == coefficient_features(original_values[0], inputs['x_grid'], metadata[0])
    rp = pd.DataFrame(csvread(ROOT/'results/remizov_ml_v1_2026-10-25/predictions.csv'))
    cp = pd.DataFrame(csvread(C/'ml/predictions.csv'))
    assert rp.equation_id.tolist() == cp.equation_id.tolist() == ids
    rm = read(ROOT/'results/remizov_ml_v1_2026-10-25/metrics.json'); cm = read(C/'ml/metrics.json')
    checks = {}
    def metrics(name, y, p, expected):
        expected = expected.get('default_threshold', expected)
        auc = float(roc_auc_score(y, p)); ll = float(log_loss(y, p, labels=[0,1])); br = float(brier_score_loss(y,p))
        assert abs(auc-expected['roc_auc']) < 1e-12
        assert abs(ll-expected['logloss']) < 1e-12 and abs(br-expected['brier']) < 1e-12
        checks[name] = dict(n=len(y), positives=int(sum(y)), negatives=int(len(y)-sum(y)), auc=auc, logloss=ll, brier=br)
    for split in ('train', 'validation', 'test'):
        ix = [i for i,pid in enumerate(ids) if meta[pid]['split']==split and history[pid]['success']]
        metrics('R_grid33_'+split, [int(history[ids[i]]['accuracy_pass']) for i in ix],
                rp.iloc[ix].p_success_catboost.to_numpy(dtype=float), rm['classification']['catboost'][split])
        metrics('R_x0_'+split, [int(abs(read(R/'solutions'/f'{ids[i]}.json')['result']['f'][16]-reference[ids[i]][16])<=.01) for i in ix],
                cp.iloc[ix].p_remizov_point_accuracy_catboost.to_numpy(dtype=float), cm['classification']['remizov_point_accuracy']['catboost'][split])
        ys, ps = [], []
        for i, pid in enumerate(ids):
            if meta[pid]['split'] != split: continue
            for _, row in x0[pid]:
                ys.append(int(row['accuracy_pass'])); ps.append(float(cp.iloc[i].p_mc_single_seed_accuracy_catboost))
        metrics('MC_x0_single_'+split, ys, ps, cm['classification']['mc_single_seed_accuracy']['catboost'][split])
    scaler_checks = {}
    import joblib
    targets = pd.DataFrame(csvread(ROOT/'results/remizov_ml_v1_2026-10-25/targets.csv'))
    for folder in (ROOT/'results/remizov_ml_v1_2026-10-25/models', C/'ml/models'):
        for path in sorted(folder.glob('*.joblib')):
            sha(path); model = joblib.load(path)
            if not hasattr(model, 'steps'): continue
            scaler = model.steps[0][1]
            if not hasattr(scaler, 'mean_'): continue
            mc = path.name.startswith('mc_')
            mask = np.array([meta[pid]['split']=='train' and (mc or history[pid]['success']) for pid in ids])
            xx = matrix
            if len(scaler.mean_) == 254:
                xx = np.column_stack([xx, targets.planned_external_workers.to_numpy(dtype=float)])
            mean, var = np.mean(xx[mask], axis=0), np.var(xx[mask], axis=0)
            np.testing.assert_allclose(scaler.mean_, mean, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(scaler.var_, var, rtol=1e-11, atol=1e-11)
            expected_count = int(sum(mask))*(4 if path.name.startswith('mc_single_seed') else 1)
            assert scaler.n_samples_seen_ == expected_count
            scaler_checks[str(path.relative_to(ROOT))] = dict(train_count=expected_count, columns=len(mean))
    registry = read(ROOT/'results/ml_research_closure_2026-10-25/registry.json')
    for spec in registry['artifacts'].values(): assert sha(ROOT/spec['path']) == spec['sha256']
    assert sha(ROOT/'ml/remizov_features.py') == registry['feature_code_sha256']
    # Search for overlooked saved MC records across all result directories.
    # Copies are excluded explicitly; small-set runs remain inventoried but are
    # never admitted to production coverage without matching expression hashes.
    inventory = defaultdict(Counter)
    for path in sorted((ROOT/'results').rglob('*.json')):
        if any(p in path.parts for p in ('sources','source','sources_final','sources_executed',
                                        'SOURCES','RAW','SUPPORT','reference','baseline','baseline128','snapshots')):
            continue
        try:
            value = json.loads(path.read_text())
        except (ValueError, UnicodeError):
            continue
        if not isinstance(value,dict) or not isinstance(value.get('result'),dict): continue
        result=value['result']
        if 'M' not in result: continue
        sha(path)
        identity=value.get('expression_hash',value.get('metadata_identity',{}).get('expression_hash'))
        pid=value.get('problem_id',result.get('problem_id'))
        valid=bool(pid in canonical and identity==meta[pid]['expression_hash'])
        scope=f"grid{len(result.get('x',[]))}_"+('production_hash_verified' if valid else 'not_admitted_by_identity')
        inventory[path.relative_to(ROOT).parts[1]][scope]+=1
        if valid and len(result.get('x',[]))==33:
            assert pid in joint, f'unexpected production grid33 MC record: {path}'
    optimization=ROOT/'results/mc_optimized_backend_2026-10-25'
    replay_checks={}
    for path in sorted((optimization/'replay').glob('*.npz')):
        sha(path)
        with np.load(path,allow_pickle=False) as z:
            keys=list(z.files)
            # These are saved path contributions, never newly simulated paths.
            ref=z['reference']; deltas={k:float(np.max(abs(z[k]-ref))) for k in keys if k!='reference'}
            assert max(deltas.values()) < 1e-12
            replay_checks[path.stem]=deltas
    timing=csvread(optimization/'timing_summary.csv')
    speedups=[float(r['reference_seconds'])/float(r['numba_seconds']) for r in timing]
    write(out/'artifact_inventory.json',dict(mc_saved_records={k:dict(v) for k,v in inventory.items()},
        protected_source_versions=source_versions, optimization_saved_path_replay=replay_checks,
        optimization_speedup_range=[min(speedups),max(speedups)],
        caveat='Inventory counts records, including legacy wrappers, not unique attempts or production equations'))
    csvwrite(out/'audit_by_equation.csv', table)
    csvwrite(out/'run_evidence.csv', evidence)
    special = [r for r in evidence if r['equation_id'] in {'AMV1_F0141_S00','AMV1_F0180_S04','AMV1_F0184_S01','AMV1_F0180_S03'}]
    csvwrite(out/'special_equations.csv', special)
    def stats(rows):
        rows = list(rows)
        return dict(records=len(rows), unique_equations=len({r['equation_id'] for r in rows}),
                    successful=sum(r['success'] for r in rows), unsuccessful=sum(not r['success'] for r in rows),
                    observed_accuracy_pass=sum(r['accuracy_pass'] is True for r in rows),
                    observed_accuracy_fail=sum(r['accuracy_pass'] is False for r in rows),
                    missing_error=sum(r['error'] is None for r in rows))
    jrs = [pair['R'][1] for pair in joint.values()]; jcs = [pair['MC'][1] for pair in joint.values()]
    errids = [pid for pid,r in history.items() if not r['success']]
    missing_grid = sorted(canonical-set(joint))
    # Recover physical interruptions from immutable events instead of treating resumed records as duplicate equations.
    events = {}
    for folder in (C, J):
        ev = [dict(source=str(p.relative_to(ROOT)), **read(p))
              for p in sorted((folder/'events').glob('*.interrupted.json'))]
        for path in sorted(folder.rglob('events.jsonl')):
            sha(path)
            for line in path.read_text().splitlines():
                value=json.loads(line)
                if 'interrupt' in str(value).lower():ev.append(value)
        events[folder.name] = ev
    summary = dict(
        schema='ml_selector_readiness_audit_v1', date='2026-10-25', local_only=True,
        new_solver_calls=0, new_model_fits=0,
        canonical=dict(path=str(DATA.relative_to(ROOT)), equations=1000, families=200, samples_per_family=5,
            seed=2488, splits=dict(Counter(m['split'] for m in metadata)),
            split_families=dict(Counter(v[0]['split'] for v in families.values())),
            csv_jsonl_npz_identity_valid=True, reference_checks_valid=True,
            residual_grid256_max=float(maxima['residual_max']), family_leakage=False, topology_leakage=False,
            duplicate_ids=[], duplicate_expression_hashes=0, duplicate_feature_hashes=0),
        small_dataset_id_collisions=dict(count=len(collisions), ids=sorted(collisions), identical_expression_count=len(identical)),
        historical_remizov=stats(history.values()), mc_x0=stats(r for rows in x0.values() for _,r in rows),
        joint_grid33_remizov=stats(jrs), joint_grid33_mc=stats(jcs),
        final5=stats(r for rows in fresh.values() for _,r in rows),
        final5_remizov=stats(r for rows in fresh.values() for _,r in rows if r['method']=='Remizov'),
        final5_mc=stats(r for rows in fresh.values() for _,r in rows if r['method']=='MC'),
        final5_seed_statistics=final_checks,
        joint_grid33_adaptive_observed_mc_pass=adaptive_pass,
        mc_x0_aggregate=dict(equations=1000, pass_count=sum(r['mc_x0_mean_error']<=.01 for r in table),
                            all_four_success=sum(r['mc_x0_observed_successes']==4 for r in table)),
        coverage=dict(any_scope_launch_intersection=1000, same_grid33_pairs=100, comparable_pair_fraction=.1,
                      grid33_missing_mc=missing_grid, remizov_failed=errids,
                      remizov_missing=[], mc_x0_missing=[], extra_canonical_result_ids=[],
                      duplicate_terminal_remizov_ids=[], duplicate_mc_equation_seed_keys=[],
                      remizov_request_count=len(requests), remizov_terminal_attempt_count=len(terminal_attempts),
                      remizov_nonterminal_request_ids=sorted(r['attempt_id'] for r in requests if r['attempt_id'] not in {v['attempt_id'] for v in terminal_attempts}),
                      mc_grid33_validation=0, mc_grid33_test=0),
        labels=dict(remizov=0, monte_carlo=0, none=0, unknown=1000),
        label_statuses=dict(Counter(r['label_confidence_or_status'] for r in table)),
        observed_primary_grid33_categories=dict(Counter('remizov_only' if p['R'][1]['accuracy_pass'] else 'neither' for p in joint.values())),
        observed_primary_labels_not_reliable_training_targets=dict(remizov=73,monte_carlo=0,none=27,unknown=900),
        timing=dict(joint_R_median=float(np.median([r['runtime'] for r in jrs])),
                    joint_MC_median=float(np.median([r['runtime'] for r in jcs])),
                    historical_R_sum=float(sum(r['runtime'] or 0 for r in history.values())),
                    x0_MC_sum=float(sum(r['runtime'] for rows in x0.values() for _,r in rows))),
        fixed_profiles=dict(saved_numeric_comparison='PARTIAL:100 train pairs; warm solver timing only',
                            reliable_fastest_labels='NO', minimum_measured_time_selection='NO'),
        readiness=dict(DATA_COVERAGE='INCOMPLETE', ANY_SPATIAL_SCOPE_LAUNCH_COVERAGE='COMPLETE',
                       READY_FOR_IMPLEMENTATION='PARTIAL', READY_FOR_TRAINING_AND_EVALUATION='NO'),
        blockers=['900 production equations lack MC grid33 runs', 'MC total accuracy/reliability unconfirmed',
                  'historical taper has boundary artifacts; zero profile only100 train cases',
                  'no grid33 validation/test pairs', 'timing excludes external MC AST/G prep and JIT warmup',
                  'no sufficiently explored comparable configuration sets for time-to-accuracy'],
        event_interruption_evidence=events,
        environment=dict(python=sys.version, platform=platform.platform(), numpy=np.__version__),
        source_semantics=dict(main_remizov='historical mass taper only', main_mc_error='joint100 primary grid33 or missing',
            pair_comparable='joint pair_* Remizov + main MC; never historical Remizov timing',
            mc_run_found='any spatial scope; x0 does not establish grid33 accuracy',
            label='conservative fastest-reliable grid33; all unknown',
            success='finite computational answer, not full error certification'))
    write(out/'ml_checks.json', dict(features=253, feature_matrix_rebuilt_exactly=True,
        input_only_metadata_invariance=True, scaler_checks=scaler_checks, metrics=checks,
        registry_hash_checks=len(registry['artifacts']), existing_comparative_model=cm['comparative']))
    write(out/'audit_summary.json', summary)
    sha(Path(__file__))
    write(out/'input_sha256.json', INPUTS)
    print(json.dumps({k:summary[k] for k in ('historical_remizov','mc_x0','joint_grid33_remizov','joint_grid33_mc','labels','readiness')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    audit(parser.parse_args().out)
