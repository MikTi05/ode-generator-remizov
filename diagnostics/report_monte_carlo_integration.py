"""Recheck saved MC arrays and integrity; never call a numerical solver."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(out: Path):
    summary = json.loads((out/'runs/summary.json').read_text())
    plan = json.loads((out/'plan.json').read_text())
    if len(summary['runs']) != len(plan['attempts']) or len(summary['runs']) != 5:
        raise ValueError('expected all five predeclared attempts, including failures')
    rows = []
    for index, run in enumerate(summary['runs']):
        if run['spec'] != plan['attempts'][index]:
            raise ValueError('saved attempt differs from the predeclared plan')
        row = dict(problem_id=run['spec']['problem_id'], status=run['status'],
                   E=None, max_stderr=None, T=None, dt=None, tail=None,
                   seconds=run['runtime_seconds'], rss=run['observed_peak_rss_mib'])
        if run['result_file'] is not None:
            data = json.loads((out/'runs'/run['result_file']).read_text())
            r = data['result']
            if r['success']:
                x, f, y, se = (np.asarray(v) for v in (r['x'], r['f'], data['y_exact'], r['stderr']))
                assert x.shape == f.shape == y.shape == se.shape == (33,)
                assert np.all(np.isfinite(f)) and np.all(np.isfinite(se)) and np.all(se >= 0)
                error = float(np.max(abs(f-y)))
                assert error == data['max_abs_error'] == run['max_abs_error']
                np.testing.assert_allclose(r['ci95'], 1.96*se, rtol=0, atol=0)
                assert r['n_normal_draws'] == r['n_path_steps'] == summary['preflight'][index]['path_steps']
                assert all(m == r['M'] for m in r['completed_paths'])
                assert r['dt_actual'] <= r['dt_requested']
                assert not r['total_error_certified'] and r['discretization_error_bound'] is None
                if row['problem_id'].startswith('AMV1_'):
                    assert r['tail_error_bound'] is None and r['tail_tolerance_met'] is None
                else:
                    assert r['tail_error_bound'] <= r['tail_eps'] and r['tail_tolerance_met']
                row.update(E=error, max_stderr=float(np.max(se)), T=r['T'],
                           dt=r['dt_actual'], tail=r['tail_error_bound'])
        threshold = run['spec']['observational_threshold']
        row['threshold'] = threshold
        row['observed_target_met'] = None if threshold is None else row['E'] is not None and row['E'] <= threshold
        rows.append(row)

    before = json.loads((out/'before_sha256.json').read_text())
    allowed = {'README.md', 'PROJECT_STATUS.md', 'diagnostics/compare_chernov.py'}
    changed = [p for p,h in before.items() if not (ROOT/p).is_file() or digest(ROOT/p) != h]
    unexpected = sorted(set(changed)-allowed)
    assert not unexpected, unexpected
    manifest = json.loads((ROOT/'MIGRATION_MANIFEST.json').read_text())
    core = {p:digest(ROOT/p) == manifest['source_sha256'][p] for p in manifest['unchanged_mathematical_core']}
    assert all(core.values())
    backups = {p: digest(out/snapshot) == before[p] for p,snapshot in [
        ('README.md', 'README.md.before'), ('PROJECT_STATUS.md', 'PROJECT_STATUS.md.before'),
        ('diagnostics/compare_chernov.py', 'compare_chernov.py.before')]}
    assert all(backups.values())
    experiment_sources = {
        name: digest(out/snapshot) == summary['before_sha256'][name]
        for name,snapshot in [('mc_solver.py', 'mc_solver.experiment.py'),
                              ('diagnostics/compare_chernov.py', 'compare_chernov.experiment.py')]}
    assert all(experiment_sources.values())
    provenance = json.loads((ROOT/'comparison_inputs/monte_carlo/provenance.json').read_text())
    originals = {name: digest(ROOT/'comparison_inputs/monte_carlo'/name) == item['sha256']
                 and digest(Path(item['source'])) == item['sha256'] for name,item in provenance.items()}
    assert all(originals.values())
    verification = dict(old_files=len(before), unchanged_old_files=len(before)-len(changed),
        permitted_changes=sorted(changed), unexpected_changes=unexpected,
        mathematical_core=core, recoverable_previous_versions=backups,
        experiment_source_snapshots=experiment_sources,
        original_materials=originals, recalculated_rows=rows,
        current_source_sha256={p:digest(ROOT/p) for p in ['mc_solver.py',
            'diagnostics/compare_chernov.py','examples/solve_mc_equation.py',
            'tests/test_mc_solver.py','tests/test_all_methods_mc_pipeline.py']})
    (out/'verification.json').write_text(json.dumps(verification,indent=2,sort_keys=True)+'\n')

    lines = ['# Интеграция Монте-Карло — 2026-10-24', '',
        'Пять заранее записанных запусков, один worker/BLAS=1, seed=2488, без подбора и повторов.',
        'Протокол: [plan.json](plan.json), полные ответы и ресурсы: [runs/summary.json](runs/summary.json).',
        'Все ошибки пересчитаны по сохранённым f и exact на 33 точках; решатели при подготовке отчёта не вызываются.', '',
        '| Задача | Статус | M | h фактический | T | E∞ на сетке | max stderr | Граница хвоста | Время MC, с |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    fmt = lambda v: 'не определено' if v is None else f'{v:.9g}'
    for row,run in zip(rows,summary['runs']):
        lines.append('| '+' | '.join([row['problem_id'],row['status'],str(run['spec']['config']['M']),
            *[fmt(row[k]) for k in ('dt','T','E','max_stderr','tail','seconds')]])+' |')
    lines += ['', '## Интерпретация', '',
        'MC_constant: a=1, b=c_R=0, lambda=2, g_R=1, exact=1/2, G=1.',
        'MC_gaussian: exact=exp(-x²), g_R=(4-4x²)exp(-x²), G=4: для u=x²>=0 максимум |(1-u)e^-u| равен 1.',
        'Обе аналитические задачи достигли заранее заданных наблюдаемых порогов 2e-5 и .03: '+str([r['observed_target_met'] for r in rows[:2]])+'.',
        'Это одиночные результаты с фиксированным seed, не утверждение о покрытии или вероятности достижения допуска.',
        'Интервалы — точечные асимптотические для среднего усечённой дискретной схемы; временное смещение в них не входит.',
        'На датасете выбраны L1/constant, L1/slow_varying, L1/moderate_varying; M=2000, dt=.02, T=4.',
        'Численного сертификата G нет: хвост не подтверждён, общий допуск не заявлен. E∞ — наблюдаемая ошибка на 33 точках [-1,1].',
        'Аналитическая сетка — 33 точки [-2,2]. Результаты между точками и на всей прямой этим измерением не сертифицируются.',
        'Граница хвоста следует из доверенных аналитических сертификатов и вычисляется в float64; произвольные callable и регулярность автоматически не доказываются.',
        'Массовый benchmark, сравнение победителей методов и ML не выполнялись.', '',
        '## Проверки и сохранность', '',
        'Полная регрессия без длительного R1: [pytest_final.txt](pytest_final.txt); первая проверка — [pytest.txt](pytest.txt).',
        'Пересчёт метрик и SHA: [verification.json](verification.json). '+str(len(before)-len(changed))+' прежних файлов сохранены побайтово.',
        'Разрешённые изменения прежних файлов: '+', '.join(sorted(changed))+'. Численные методы, генератор, данные и все прежние результаты сохранены.',
        'Предоставленные три MC-материала скопированы без изменений, исходные внешние файлы также совпадают с хешами.',
        'Версии MC и сравнительного драйвера, использованные в пяти запусках, сохранены как mc_solver.experiment.py и compare_chernov.experiment.py; их хеши совпадают с summary.',
        'После запусков добавлены только защита положительного T при underflow T/dt и отказ при непредставимом полушаге, с отдельным тестом. Обычные режимы этих пяти запусков защиту не задействуют; повторных расчётов не было.',
        'Дефолты и пределы MC записаны в каждой полной конфигурации. Внешний watchdog: 60 с / 512 МиБ на попытку, внутренние лимиты 30/45 с.',
        'Исходная версия сравнительного драйвера и прежние README/PROJECT_STATUS сохранены рядом с отчётом.', '',
        '## Ограничения и следующий шаг', '',
        'Автоматического аналитического вывода G из AST пока нет. Для датасета нужен отдельный вывод сертификата либо явный T без утверждения о хвосте.',
        'Не установлены общий временной порядок, полная ошибка, номинальное покрытие CI и вероятность достижения допуска.',
        'Следующая отдельная задача: сертификат G и изолированные уточнения h/T/M на малой подвыборке перед согласованием сравнительного benchmark.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return rows, verification


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    rows, verification = build_report(args.output_dir.resolve())
    print(json.dumps(dict(rows=rows, unchanged_old_files=verification['unchanged_old_files'],
                          unexpected_changes=verification['unexpected_changes']),ensure_ascii=False))
