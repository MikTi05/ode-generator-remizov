"""Recompute report evidence from saved answers only; no solver/ML imports."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math
import re
import shutil

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = Path('/Users/mikti/.codex/attachments/61bbc5b1-4b6f-4055-96d8-278a70f3a628/Pasted text.txt')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def table(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def summary(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                q95=float(np.quantile(a, .95)), maximum=float(a.max()),
                accurate=int(np.count_nonzero(a <= .05)))


def assert_close(left, right):
    assert math.isclose(left, right, rel_tol=2e-13, abs_tol=1e-15), (left, right)


def verify():
    rdir = ROOT / 'results/chernov_solve1000_2026-10-18/solutions'
    ivp = ROOT / 'results/four_method_accuracy_1000_2026-11-01'
    mcdir = ROOT / 'results/mc1000_x0_2026-10-25/attempts'
    coverage = table(ivp / 'release/coverage_all1000.csv')
    metadata = {m['equation_id']: m for m in map(json.loads, (
        ROOT / 'data/all_methods_v1_1000_2026-10-18/all_methods_equations_v1_metadata.jsonl').read_text().splitlines())}
    assert len(coverage) == len(metadata) == 1000
    grid = np.linspace(-1, 1, 33)
    rows = []
    old_failures = []
    mc_configs = []
    counts = Counter()
    input_paths = set()
    for record in coverage:
        pid = record['equation_id']
        row = dict(equation_id=pid, R_available=False)
        exact = None
        for method, label in (('rk4', 'RK4'), ('product_integral', 'PI')):
            path = ivp / 'records' / method / (pid + '.json')
            raw = read(path); input_paths.add(path)
            q = raw['result']
            assert raw['usable'] and q['success'] and raw['status'] == 'success'
            assert raw['expression_hash'] == metadata[pid]['expression_hash']
            body = {k: v for k, v in raw.items() if k != 'record_sha256'}
            encoded = json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
            assert hashlib.sha256(encoded).hexdigest() == raw['record_sha256']
            assert q['x'] == np.linspace(-1, 1, 129).tolist()
            assert len(q['y']) == len(q['dy']) == 129
            assert np.isfinite(q['y']).all() and np.isfinite(q['dy']).all()
            assert q['y'][0] == metadata[pid]['y0'] and q['dy'][0] == metadata[pid]['v0']
            local_exact = np.asarray(raw['exact_grid33'])
            if exact is None:
                exact = local_exact
            else:
                assert np.array_equal(exact, local_exact)
            error = float(np.max(np.abs(np.asarray(q['y'])[::4] - exact)))
            e0 = abs(q['y'][64] - exact[16])
            assert error == raw['E_inf_grid33'] and e0 == raw['E0']
            assert e0 == float(record['E_' + method])
            row[label + '_E33'] = error; row[label + '_E0'] = e0
            counts[method] += 1
        path = rdir / (pid + '.json'); r = read(path); input_paths.add(path)
        assert r['problem_id'] == pid
        if r['computation_status'] == 'success':
            q = r['result']; assert q['success'] and q['failure_reason'] == ''
            assert r['metadata']['expression_hash'] == metadata[pid]['expression_hash']
            assert q['x'] == r['target_x'] == grid.tolist()
            assert np.array_equal(exact, r['y_exact']) and len(q['f']) == 33
            assert np.isfinite(q['f']).all()
            assert r['config']['n'] == 128 and r['config']['boundary'] == 'taper'
            assert r['config']['num_t'] == 513 and r['config']['dx'] == .0125
            error = float(np.max(np.abs(np.asarray(q['f']) - exact)))
            assert error == r['max_abs_error']
            row.update(R_available=True, R_E33=error, R_E0=abs(q['f'][16] - exact[16]))
            assert row['R_E0'] == float(record['E_remizov'])
            counts['remizov_success'] += 1
        else:
            assert r['result'] is None and r['computation_status'] == 'worker_exception'
            old_failures.append(pid)
        values = []
        for seed_index in range(4):
            path = mcdir / (pid + f'_seed{seed_index}.json'); raw = read(path); input_paths.add(path)
            q = raw['result']; c = raw['config']
            assert raw['status'] == 'completed' and q['success'] and q['x'] == [0.0]
            assert raw['expression_hash'] == metadata[pid]['expression_hash']
            assert raw['y_exact'] == exact[16]
            assert c['seed'] == 202610250101 + seed_index
            assert c['M'] == c['chunk'] == 1024 and c['dt'] == 1 / 128
            assert c['T'] is None and c['tail_eps'] == 1e-4 and c['stream_id'] == 0
            assert math.isfinite(q['f'][0]) and len(q['f']) == 1
            values.append(q['f'][0]); mc_configs.append(c); counts['mc_seed_records'] += 1
        e0_mc = abs(float(np.mean(values, dtype=np.float64)) - exact[16])
        assert e0_mc == float(record['E_monte_carlo'])
        row.update(MC_E0_lower_bound=e0_mc, MC_full_E33=None,
                   margin_MC=e0_mc - row['RK4_E33'],
                   margin_PI=row['PI_E33'] - row['RK4_E33'])
        assert row['margin_MC'] > 0 and row['margin_PI'] > 0
        if row['R_available']:
            row['margin_R'] = row['R_E33'] - row['RK4_E33']
            assert row['margin_R'] > 0
            row['inferred_unique_grid33_winner'] = 'RK4'
        else:
            row['inferred_unique_grid33_winner'] = None
        rows.append(row)
    assert counts == dict(rk4=1000, product_integral=1000, remizov_success=997, mc_seed_records=4000)
    common = [r for r in rows if r['R_available']]
    stats = {
        'common997': {m: summary([r[m + '_E33'] for r in common]) for m in ('R', 'RK4', 'PI')},
        'full_available': {m: summary([r[m + '_E33'] for r in rows if m + '_E33' in r]) for m in ('R', 'RK4', 'PI')},
        'point0_common997': {m: summary([r[m + '_E0'] for r in common]) for m in ('R', 'RK4', 'PI')},
    }
    stats['point0_common997']['MC'] = summary([r['MC_E0_lower_bound'] for r in common])
    assert [stats['common997'][m]['accurate'] for m in ('R', 'RK4', 'PI')] == [970, 997, 402]
    assert [stats['full_available'][m]['accurate'] for m in ('R', 'RK4', 'PI')] == [970, 1000, 404]
    assert [stats['point0_common997'][m]['accurate'] for m in ('R', 'MC', 'RK4', 'PI')] == [982, 935, 997, 758]
    margins = {m: min((r['margin_' + m], r['equation_id']) for r in common) for m in ('R', 'MC', 'PI')}
    assert_close(margins['MC'][0], 2.6073122595920983e-5)
    fields = ['equation_id', 'R_available', 'R_E0', 'R_E33', 'RK4_E0', 'RK4_E33',
              'PI_E0', 'PI_E33', 'MC_E0_lower_bound', 'MC_full_E33',
              'margin_R', 'margin_PI', 'margin_MC', 'inferred_unique_grid33_winner']
    with (OUT / 'grid33_per_equation.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    write(OUT / 'grid33_verification.json', dict(
        metric='max absolute error at linspace(-1,1,33); not a continuous norm',
        target_x=grid.tolist(), epsilon=.05, raw_records_checked=dict(counts),
        exact_source='saved arrays only, equality across methods checked',
        summaries=stats, common_equations=997, inferred_unique_RK4_winners=997,
        minimum_positive_margins={m: dict(margin=value, equation_id=pid) for m, (value, pid) in margins.items()},
        inference_condition='extend the same saved mean4 MC value at zero; not a new independent MC run',
        MC_full_grid33_calculated=False, solver_calls=0, fit_calls=0, inference_calls=0,
        missing_remizov_ids=old_failures))
    models = table(ROOT / 'results/scientific_evidence_2026-11-01/tables/main_model_comparisons.csv')
    selected = [r for r in models if r['label'] == 'selected253 new']
    diagnostic = [r for r in models if r['label'] == 'diagnostic258 new']
    assert len(selected) == len(diagnostic) == 3
    lock = read(ROOT / 'results/ml_selector_hypotheses_2026-10-31/selection_lock.json')['selected']['new']
    assert lock['candidate_id'] == 'new__X__classifier__w1__baseline' and lock['threshold'] == .25
    sv = next(r for r in selected if r['split'] == 'validation')
    dv = next(r for r in diagnostic if r['split'] == 'validation')
    assert float(sv['mean_regret']) == lock['mean_regret'] < float(dv['mean_regret'])
    st = next(r for r in selected if r['split'] == 'test')
    dt = next(r for r in diagnostic if r['split'] == 'test')
    assert [int(st[k]) for k in ('mc_correct', 'mc_false', 'mc_missed', 'success_count')] == [1, 5, 8, 146]
    assert [int(dt[k]) for k in ('mc_correct', 'mc_false', 'mc_missed', 'success_count')] == [6, 14, 3, 145]
    assert_close(float(st['accuracy']), 137 / 150)
    assert_close(float(st['macro_f1']), .543859649122807)
    assert_close(float(dt['mean_error']), .005604052376948204)
    three = read(ROOT / 'results/ml_three_method_selector_2026-11-03/test_metrics.json')['policies']
    chosen = [p for p in three if p['scenario'] == 'new' and p.get('selected_on_validation')]
    assert len(chosen) == 1
    tm = chosen[0]
    assert tm['confusion_matrix'] == [[133, 3, 0], [7, 0, 0], [6, 1, 0]]
    assert tm['success_count'] == 147 and tm['selection_counts'] == dict(remizov=146, monte_carlo=4, product_integral=0)
    assert_close(tm['mean_error'], 1.65885046045115)
    tex = (OUT / 'report.tex').read_text()
    assert sha(SOURCE) == sha(OUT / 'original_report.tex')
    figures = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', tex)
    assert len(figures) == 13 and r'\graphicspath{{figures/}}' in tex
    figure_sources = read(OUT / 'figure_sources.json')
    for name in figures:
        entry = figure_sources[name]
        assert sha(OUT / 'figures' / name) == sha(ROOT / entry['source']) == entry['sha256']
    # TeX escaping depends on parity: a brace after \\ ends a row and is active.
    def active(line, index):
        backslashes = 0
        index -= 1
        while index >= 0 and line[index] == '\\':
            backslashes += 1
            index -= 1
        return backslashes % 2 == 0
    uncommented = []
    for line in tex.splitlines():
        cuts = [i for i, char in enumerate(line) if char == '%' and active(line, i)]
        uncommented.append(line[:cuts[0]] if cuts else line)
    content = '\n'.join(uncommented)
    braces = 0
    math_dollars = 0
    for i, char in enumerate(content):
        if active(content, i):
            if char in '{}':
                braces += 1 if char == '{' else -1
                assert braces >= 0
            if char == '$': math_dollars += 1
    assert braces == 0
    assert math_dollars % 2 == 0
    assert len(re.findall(r'(?<!\\)\\\[', content)) == len(re.findall(r'(?<!\\)\\\]', content))
    environments = []
    for kind, name in re.findall(r'\\(begin|end)\{([^}]+)\}', content):
        if kind == 'begin': environments.append(name)
        else: assert environments.pop() == name
    assert not environments
    labels = re.findall(r'\\label\{([^}]+)\}', content)
    assert len(labels) == len(set(labels))
    refs = re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', content)
    assert not set(refs) - set(labels)
    bibs = set(re.findall(r'\\bibitem\{([^}]+)\}', content))
    cites = {key for group in re.findall(r'\\cite\{([^}]+)\}', content) for key in group.split(',')}
    assert not cites - bibs
    assert 'Матричная экспонента вычисляется точно' not in tex
    assert 'П.7, признак 3) и величина' not in tex
    assert 'полная сеточная ошибка не вычислена' in tex
    for token in ('39.4119', '0.273965', '970/997', '402/997', '404', '2.6073122596',
                  '0.0007103513', '0.0008760301', '1.65980', '6/14/3', '1/5/8'):
        assert token in tex, token
    write(OUT / 'report_verification.json', dict(
        original_sha256=sha(SOURCE), original_backup_identical=True,
        report_sha256=sha(OUT / 'report.tex'), latex_static=dict(
            braces_balanced=True, environments_balanced=True, labels_unique=len(labels),
            references_resolved=len(refs), citations_resolved=sorted(cites),
            figures_present_and_byte_identical=len(figures)),
        binary_selected253=dict(validation=sv, test=st),
        binary_diagnostic258=dict(validation=dv, test=dt), three_selected_test=tm,
        grid_evidence='grid33_verification.json and grid33_per_equation.csv',
        new_solver_calls=0, new_fit_calls=0, new_model_inference_calls=0,
        latex_compiler_available={name: shutil.which(name) for name in ('pdflatex', 'xelatex', 'lualatex', 'tectonic', 'latexmk')},
        pdf_compiled=False, visual_page_layout_verified=False,
        limitation='No local TeX engine; static validation only. Original PNG figures preserved.'))
    write(OUT / 'raw_input_sha256.json', {str(p.relative_to(ROOT)): sha(p) for p in sorted(input_paths)})
    print(json.dumps(dict(raw_records=dict(counts), common=997, RK4_winners=997,
                         figures=13, labels=len(labels), references=len(refs),
                         selected_binary=lock['candidate_id'], solver_calls=0, fit_calls=0), ensure_ascii=False))


if __name__ == '__main__':
    verify()
