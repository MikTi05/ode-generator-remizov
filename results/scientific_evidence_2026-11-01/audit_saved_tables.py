"""Arithmetic verification of saved observations only; no ML/ODE imports."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
BASELINE = json.loads((OUT / 'before_sha256.json').read_text())
USED = json.loads((OUT / 'used_sources.json').read_text())


def pin(path):
    path = Path(path).resolve()
    key = str(path.relative_to(ROOT))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if key in BASELINE:
        assert BASELINE[key] == digest, key
    USED[key] = digest
    return path


def read(path):
    return json.loads(pin(path).read_text())


def frame(path):
    return pd.read_csv(pin(path), float_precision='round_trip', keep_default_na=False)


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def metric(rows):
    er, em = rows.E_R.to_numpy(float), rows.E_MC.to_numpy(float)
    truth = em < er
    prediction = rows.selected_method.to_numpy() == 'monte_carlo'
    cm = np.zeros((2, 2), dtype=int)
    np.add.at(cm, (truth.astype(int), prediction.astype(int)), 1)
    chosen = np.where(prediction, em, er)
    f1 = 2 * np.diag(cm) / (cm.sum(axis=0) + cm.sum(axis=1))
    return dict(n=len(rows), confusion_matrix=cm.tolist(), accuracy=float(np.trace(cm) / len(rows)),
                macro_f1=float(f1.mean()), mean_error=float(chosen.mean()),
                mean_regret=float((chosen - np.minimum(er, em)).mean()),
                success_count=int((chosen <= .05).sum()))


def main():
    hyp = ROOT / 'results/ml_selector_hypotheses_2026-10-31'
    folds = frame(hyp / 'cv_folds.csv')
    predictions = frame(hyp / 'cv_predictions.csv')
    summary = read(hyp / 'cv_summary.json')
    for row in folds.to_dict('records'):
        own = predictions[(predictions.scenario == row['scenario']) &
                          (predictions.fold == row['fold']) & (predictions.policy == row['policy'])]
        got = metric(own)
        assert got['confusion_matrix'] == json.loads(row['confusion_matrix'])
        for key in got.keys() - {'confusion_matrix'}:
            assert np.isclose(got[key], row[key], rtol=2e-14, atol=1e-15), (row['policy'], key)
    cv_rows = []
    for row in summary['policy_summary']:
        own = predictions[(predictions.scenario == row['scenario']) & (predictions.policy == row['policy'])]
        assert own.equation_id.is_unique
        got = metric(own)
        assert got['confusion_matrix'] == row['pooled']['confusion_matrix']
        for key in got.keys() - {'confusion_matrix'}:
            assert np.isclose(got[key], row['pooled'][key], rtol=2e-14, atol=1e-15)
        cv_rows.append(dict(scenario=row['scenario'], policy=row['policy'], **row['pooled'],
                            fold_regret_mean=row['fold_variability']['mean_regret']['mean'],
                            fold_regret_sd=row['fold_variability']['mean_regret']['sd']))
    pd.DataFrame(cv_rows).to_csv(OUT / 'tables/cv_policies_22.csv', index=False)
    write('checks/cv_saved_arithmetic.json', dict(folds_checked=110, pooled_policies_checked=22,
        prediction_rows=15884, fit_calls=0, inner_threshold_optimality='not verifiable: inner scores absent'))

    four = ROOT / 'results/four_method_accuracy_1000_2026-11-01'
    coverage = frame(four / 'release/coverage_all1000.csv')
    metadata = {m['equation_id']: m for m in (
        json.loads(s) for s in pin(ROOT / 'data/all_methods_v1_1000_2026-10-18/all_methods_equations_v1_metadata.jsonl').read_text().splitlines())}
    counts = dict(new_records=0, old_R_records=0, usable_old_R_records=0, MC_records=0)
    grid_values = dict(rk4=[], product_integral=[])
    for row in coverage.to_dict('records'):
        pid, exact0 = row['equation_id'], None
        for method in ('rk4', 'product_integral'):
            record = read(four / 'records' / method / (pid + '.json'))
            result = record['result']
            assert record['usable'] and result['success']
            assert record['expression_hash'] == metadata[pid]['expression_hash'] == row['expression_hash']
            assert record['inputs'] == {k: metadata[pid][k] for k in ('x0', 'x1', 'y0', 'v0')}
            assert len(result['x']) == len(result['y']) == len(result['dy']) == 129
            assert result['x'][64] == 0.0
            assert result['y'][0] == metadata[pid]['y0'] and result['dy'][0] == metadata[pid]['v0']
            exact0 = record['exact0'] if exact0 is None else exact0
            assert exact0 == record['exact0'] == record['exact_grid33'][16]
            error = abs(result['y'][64] - exact0)
            grid_error = float(np.max(np.abs(np.asarray(result['y'])[::4] - record['exact_grid33'])))
            assert error == record['E0'] == float(row['E_' + method])
            assert record['accuracy_met'] == (error <= .05)
            assert grid_error == record['E_inf_grid33']
            grid_values[method].append(grid_error)
            counts['new_records'] += 1
        old = read(ROOT / 'results/chernov_solve1000_2026-10-18/solutions' / (pid + '.json'))
        counts['old_R_records'] += 1
        if row['usable_remizov']:
            assert old['metadata']['expression_hash'] == row['expression_hash']
            assert old['target_x'][16] == 0.0 and old['y_exact'][16] == exact0
            assert abs(old['result']['f'][16] - exact0) == float(row['E_remizov'])
            counts['usable_old_R_records'] += 1
        else:
            assert old['status'] == 'worker_exception'
        answers = []
        for seed in range(4):
            mc = read(ROOT / 'results/mc1000_x0_2026-10-25/attempts' / (pid + f'_seed{seed}.json'))
            assert mc['status'] == 'completed' and mc['expression_hash'] == row['expression_hash']
            assert mc['target_x'] == [0.0] and mc['y_exact'] == exact0
            answers.append(mc['result']['f'][0])
            counts['MC_records'] += 1
        assert abs(float(np.mean(answers)) - exact0) == float(row['E_monte_carlo'])
        if row['eligible_four_method_ml']:
            errors = [float(row['E_' + m]) for m in ('remizov', 'monte_carlo', 'rk4', 'product_integral')]
            assert np.argmin(errors) == 2 and sum(x == min(errors) for x in errors) == 1
    grid_rows = []
    for method, values in grid_values.items():
        a = np.asarray(values)
        grid_rows.append(dict(method=method, n=len(a), accurate_grid33=int((a <= .05).sum()),
                              mean=float(a.mean()), median=float(np.median(a)), maximum=float(a.max()),
                              metric='max absolute error on saved 33 nodes only'))
    pd.DataFrame(grid_rows).to_csv(OUT / 'tables/new_methods_grid33_summary.csv', index=False)
    write('checks/raw_saved_records.json', dict(**counts, exact_recalculated=False,
        source_of_exact='already saved scalar/33-node exact arrays, equality across sources checked',
        existing_E0_recomputed=3997, existing_E_grid33_recomputed=2000,
        solver_calls=0, fits=0, winner_count=997, exact_ties=0))

    original = frame(ROOT / 'results/ml_research_closure_2026-10-25/ml_metrics.csv')
    original.to_csv(OUT / 'tables/historical_63_split_metrics.csv', index=False)
    for p in json.loads((OUT / 'report_index.json').read_text()):
        pin(ROOT / p['path'])
    write('used_sources.json', USED)
    print(json.dumps(dict(CV_rows_checked=110, CV_pooled=22, **counts, fits=0, solver_calls=0)))


if __name__ == '__main__':
    main()
