"""Audit the report against frozen evidence. Never solve, train or predict."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import math
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/ml_research_report_part3_2026-10-27'
MAT = ROOT / 'results/ml_report_illustrations_2026-10-27'
FINAL = ROOT / 'results/ml_selector_final140_protocol_2026-10-27'


def read(path):
    return json.loads(path.read_text())


def stated_rule(estimates):
    """Independent transcription of equations (14)--(15), on saved forecasts."""
    p = {k: v['probability_accuracy'] for k, v in estimates.items()}
    t = {k: v['predicted_full_warm_seconds'] for k, v in estimates.items()}
    eligible = {k for k in p if p[k] >= max(p.values()) - .02}
    rs = eligible & {'R128', 'R256'}
    if not rs:
        return 'MC1'
    best = min(rs, key=lambda k: (t[k], k == 'R256'))
    if 'MC1' in eligible and t['MC1'] <= .9 * t[best]:
        return 'MC1'
    return best


def audit():
    checks = {}
    before = read(OUT / 'internal/source_hashes.before.json')
    after = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in before}
    checks['protected_sources_unchanged'] = before == after
    assert checks['protected_sources_unchanged']
    analysis = read(FINAL / 'analysis.json')
    cases = analysis['cases']
    assert len(cases) == 60 and len({r['case']['structure_signature'] for r in cases}) == 60
    totals = {}
    for name, expected in [('constant_R128', 53), ('constant_R256', 57), ('selector', 57)]:
        accurate = sum(r['policies'][name]['correct'] for r in cases)
        seconds = sum(r['policies'][name]['cost'] for r in cases)
        assert accurate == expected
        assert math.isclose(seconds, analysis['primary'][name]['total_warm_cost'], abs_tol=1e-8)
        totals[name] = {'accurate': accurate, 'seconds': seconds}
    checks['primary_metrics_recomputed_from_60_saved_cases'] = True
    saving = 1 - totals['selector']['seconds'] / totals['constant_R256']['seconds']
    assert round(100 * saving, 2) == 17.49
    measured_mc = [r['profiles']['MC1'] for r in cases if r['profiles'].get('MC1', {}).get('status') == 'completed']
    assert len(measured_mc) == 20 and sum(r['correct'] for r in measured_mc) == 0
    assert sum(r['policies']['selector']['profile'] == 'MC1' for r in cases) == 0
    feasible = [r for r in cases if any(r['profiles'][p]['correct'] for p in ('R128', 'R256'))]
    fastest = sum(r['profiles'][r['policies']['selector']['profile']]['seconds'] == min(
        r['profiles'][p]['seconds'] for p in ('R128', 'R256') if r['profiles'][p]['correct']) for r in feasible)
    assert len(feasible) == 57 and fastest == 23
    checks['MC20_and_23_of_57_selection_limits_confirmed'] = True
    forecasts = read(FINAL / 'frozen_predictions.json')['rows']
    assert len(forecasts) == 60
    for row in forecasts:
        pred = row['prediction']
        assert stated_rule(pred['estimates']) == pred['selected_profile']
    checks['piecewise_rule_matches_all_60_frozen_decisions'] = True

    files = [OUT / 'sections' / n for n in ('main.tex', 'appendix_a.tex', 'appendix_b.tex', 'appendix_c.tex')]
    text = '\n'.join(p.read_text() for p in files)
    assert not re.search(r'AMV1_|F\d{4}_S\d+|\.json\b|\.py\b|\.csv\b|\bMC1\b|\bR128\b|\bR256\b|\bAPI\b|serialized_ast', text)
    assert not any(ord(c) < 32 and c not in '\n\t\r' for c in text)
    labels = re.findall(r'\\label\{([^}]+)\}', text)
    labels += re.findall(r'\\fig\{[^}]+\}\{([^}]+)\}', text)
    assert len(labels) == len(set(labels))
    assert set(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', text)) <= set(labels)
    checks['scientific_text_has_no_internal_ids_and_all_references_resolve'] = True
    app_a = files[1].read_text()
    for i in range(1, 8):
        section = app_a.split(r'\subsection*{А.' + str(i) + '. Пример ' + str(i) + '}')[1].split(r'\subsection*')[0]
        old = (MAT / 'equations' / f'example_{i:02d}_factored.tex').read_text()
        # Long numeric parameters from the original math cards must survive factorization.
        assert set(re.findall(r'\d+\.\d{4,}', old)) <= set(re.findall(r'\d+\.\d{4,}', section))
        for suffix in ('ode', 'forcing', 'solution'):
            assert f'eq:ex{i}-{suffix}' in section
        for expression in ('\\Gamma_2', '\\Gamma_1', '\\Gamma_0', 'g_R', 'c_R', 'y(-1)', r'y^\prime(-1)'):
            assert expression in section, (i, expression)
    checks['all_seven_equations_retain_numeric_parameters_and_full_definitions'] = True
    figures = re.findall(r'\\includegraphics\[[^]]*\]\{figures/([^}]+)\.pdf\}', text)
    figures += re.findall(r'\\fig\{([^}]+)\}', text)
    prepared = {r['name'] for r in read(MAT / 'internal/figures_manifest.json')}
    assert len(figures) == len(set(figures)) == 34 and set(figures) == prepared
    assert all((OUT / 'figures' / (n + '.pdf')).is_file() for n in figures)
    assert len(re.findall(r'\\fig\{', files[0].read_text())) == 4
    checks['all_34_prepared_figures_used_once_including_4_main_figures'] = True
    log = (OUT / 'report.log').read_text()
    assert 'MAIN-END-PAGE=8' in log
    assert not any(s in log for s in ('Overfull', 'Undefined', 'Missing character', 'Warning', 'Error'))
    pdf = read(OUT / 'internal/pdf_qa.json')
    starts = {p['first_lines'][0].split('.')[0]: p['page'] for p in pdf['page_checks'] if p['first_lines'] and p['first_lines'][0].startswith('Приложение')}
    assert starts['Приложение А'] == 9
    assert pdf['pages'] >= 40 and all(abs(p['width'] - 595.276) < .1 and abs(p['height'] - 841.89) < .1 for p in pdf['page_checks'])
    checks['compiled_A4_pdf_main_8_pages_no_TeX_warnings'] = True
    # These drivers contain no numerical solver, model fit or inference calls.
    for name in ('build_part3_report.py', 'check_part3_report.py'):
        tree = ast.parse((ROOT / 'diagnostics' / name).read_text())
        forbidden = {'solve', 'fit', 'predict', 'solve_monte_carlo_optimized', 'compile_backend'}
        assert all(getattr(n.func, 'attr', getattr(n.func, 'id', '')) not in forbidden for n in ast.walk(tree) if isinstance(n, ast.Call))
    checks['no_solver_training_or_model_inference_calls'] = True
    return dict(passed=all(checks.values()), checks=checks, protected_files=len(before),
                main_pages=8, total_pages=pdf['pages'], primary=totals, saving_fraction=saving,
                solver_calls=0, new_training=0, new_model_inference=0,
                visual_qa='All pages rendered; contact sheets and representative full-size pages inspected.',
                formulas='Inherited independent analytic verification: 7 examples × 515 points, max discrepancy 8.53e-14; no new ODE solutions.')


if __name__ == '__main__':
    result = audit()
    (OUT / 'internal/source_hashes.after.json').write_text(json.dumps({p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in read(OUT / 'internal/source_hashes.before.json')}, indent=2))
    (OUT / 'internal/verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
