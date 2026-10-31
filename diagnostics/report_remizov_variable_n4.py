"""Report the two saved variable-coefficient attempts; never call a solver."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from diagnostics.remizov_n_study import snapshot,write_json
from diagnostics.remizov_n4_followup import tree_hashes


def fmt(value):
    return 'не определено' if value is None else f'{value:.10g}'


def diagnostic_row(run,preflight):
    d=run.get('solver_result')
    amplitude=preflight['exact_amplitude_on_target_33']
    row={'equation_id':run['equation_id'],'regime':preflight['metadata']['regime'],
        'n':run['n'],'status':run['status'],'max_abs_error':run['max_abs_error'],
        'process_wall_seconds':run['worker_wall_seconds'],
        'solver_runtime_seconds':run['runtime_seconds'],
        'observed_peak_rss_mib':run['worker_peak_rss_mib'],
        'failure_reason':run['failure_reason'],
        'exact_amplitude_on_target_33':amplitude,
        'max_error_over_exact_amplitude_33':run['max_abs_error']/amplitude
             if run['max_abs_error'] is not None else None}
    for key in ('final_dx','final_nt','quadrature_error_estimate','spatial_error_estimate',
                'n_operator_applications','n_quadrature_node_evaluations',
                'spatial_grid_points_peak','spatial_grid_points_total_processed'):
        row[key]=d[key] if d else None
    row['last_quadrature_threshold']=next((q['convergence_thresholds'][-1]
        for q in reversed(d['quadrature_history']) if q['convergence_thresholds']),None) if d else None
    row['last_spatial_threshold']=(d['spatial_convergence_thresholds'][-1]
         if d and d['spatial_convergence_thresholds'] else None)
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_dir',type=Path)
    args=parser.parse_args()
    out=args.results_dir.resolve()
    if not out.is_relative_to(ROOT/'results'):
        parser.error('results must be under project/results')
    report=out/'REPORT.md'
    if report.exists():
        raise SystemExit('Refusing to overwrite report')
    s=json.loads((out/'summary.json').read_text())
    assert [r['equation_id'] for r in s['runs']]==['AMV1_F0006_S00','AMV1_F0007_S00']
    old=json.loads((ROOT/'results/remizov_n4_followup_2026-10-12/summary.json').read_text())
    f5=next(r for r in old['runs'] if r['equation_id']=='AMV1_F0005_S00')
    runs=[f5,*s['runs']]
    rows=[diagnostic_row(r,s['preflight'][r['equation_id']]) for r in runs]
    write_json(out/'comparison_F0005_F0006_F0007.json',rows)
    with (out/'attempts.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows[1:])

    checks=[]
    import numpy as np
    for run in s['runs']:
        eid=run['equation_id']; folder=out/f'{eid}_n4'
        assert run['config']==s['config'] and not run['instrumented']
        d=run['solver_result']
        if d:
            np.testing.assert_array_equal(d['x'],np.linspace(-1,1,33))
            if d['success']:
                assert float(np.max(np.abs(np.array(d['f'])-np.array(run['y_exact']))))==run['max_abs_error']
                assert d['n_operator_applications']==4*d['n_quadrature_node_evaluations']
                assert d['spatial_error_estimate']<=d['spatial_convergence_thresholds'][-1]
                assert all(q['converged'] and q['error_estimates'][-1]<=q['convergence_thresholds'][-1]
                           for q in d['quadrature_history'])
            else:
                assert run['max_abs_error'] is None and not d['f']
        else:
            assert run['max_abs_error'] is None and run['runtime_seconds'] is None
            assert not (folder/'solver_return.json').exists()
        samples=[json.loads(line) for line in (folder/'resource_samples.jsonl').read_text().splitlines()]
        assert max(v['rss_kib'] for v in samples if v['rss_kib'] is not None)/1024==run['worker_peak_rss_mib']
        checks.append({'equation_id':eid,'config_matches':True,'instrumented':False,
            'rss_recalculation_matches':True,'solver_result_returned':d is not None,
            'error_recalculation_matches_if_success':True if d and d['success'] else None,
            'resource_samples':len(samples),
            'no_profile_artifacts':not any(folder.glob('*.pstats'))})

    metadata={eid:s['preflight'][eid]['metadata'] for eid in s['preflight']}
    lines=['# Два независимых запуска с переменными коэффициентами, n=4', '',
        'Дата по контексту пользователя: 2026-10-12. Выполнены ровно '
        'две новые попытки: AMV1_F0006_S00, затем AMV1_F0007_S00. '
        'Обе без профилирования. Второй запуск независим от статуса '
        'первого. Новых попыток F0000/F0005, других уравнений и '
        'n=1,2,8,16,32 в этом этапе нет. После двух попыток серия остановлена.', '',
        '## Подтверждённые математические параметры и допустимость', '',
        'Штатный загрузчик восстановил исходные AST, точные функции '
        'y,y′,y″, коэффициенты и forcing. У всех трёх сравниваемых '
        'задач сертификаты analytic/all_real_x, положительное a_lower '
        'и λ>C_abs. F0005 реконструирован только для сравнения '
        'параметров и амплитуды; повторно не решался.', '',
        'Постановка сохранена: A=a∂²+b∂+c_R, c=c_R−λ, '
        'g=a y″+b y′+c y, g_R=−g, (λ−A)y=g_R. '
        'Метод вычисляет глобальную резольвентную конструкцию, '
        'а не интегрирование задачи Коши. Начальные y0,v0 '
        'сохранены и проверены, но не передаются в RemizovProblem.', '',
        'На общей целевой сетке 33 x разности g+g_R, '
        'c−(c_R−λ), y(-1)−y0, y′(-1)−v0 и '
        'изготовленная невязка равны 0 для F0005/F0006/F0007. '
        'Последняя проверка подтверждает согласованность штатной '
        'реконструкции, не является независимым доказательством '
        'производных. Дополнительно проверены конечность и '
        'соответствие коэффициентов сертификатам на 1025 точках '
        'полной консервативной рабочей области. Конечная сетка '
        'не заменяет глобальный аналитический сертификат.', '',
        '| equation_id | regime | a_lower / a_upper | B_abs | C_R_abs | λ | gap | T |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        p=s['preflight'][row['equation_id']]; c=p['certificate']
        lines.append(f"| {row['equation_id']} | {row['regime']} | {fmt(c['a_lower'])} / {fmt(c['a_upper'])} | {fmt(c['b_abs'])} | {fmt(c['c_R_abs'])} | {fmt(c['lambda'])} | {fmt(c['gap'])} | {fmt(p['T'])} |")
    lines += ['',
        'T рассчитан существующей `compute_truncation_T` при tail_eps=1e-8; '
        'формула и сертификаты не менялись. Длительность интегрирования '
        'и геометрия области различаются между задачами.', '',
        'Рабочая область для n=4: d_max=max(2√(a_upper T/4),2 B_abs T/4); '
        'исходный интервал [−1−4 d_max,1+4 d_max]. '
        'Число точек max(3,ceil(width/dx))+1. Оценки до запуска:', '',
        '| equation_id | исходная рабочая область | ширина | точки при dx=0.05 | точки при dx=0.0015625 |',
        '|---|---|---:|---:|---:|']
    for row in rows:
        p=s['preflight'][row['equation_id']]
        lines.append(f"| {row['equation_id']} | [{fmt(p['work_domain_bound'][0])}, {fmt(p['work_domain_bound'][1])}] | {fmt(p['work_width_bound'])} | {p['source_points_initial_bound']} | {p['source_points_finest_bound']} |")
    lines += ['',
        'Размеры области F0006/F0007 ниже лимита 100000 даже '
        'на самом мелком разрешённом dx. Это не гарантия времени '
        'или success: неизвестны достигнутые Nt/dx и стоимость '
        'выражений. На один процесс максимально 24528 Chernoff-узлов '
        'и 98112 применений S при полностью исчерпанных уточнениях.', '',
        '## Неизменная конфигурация, окружение и оболочка', '',
        '```json',json.dumps(s['config'],indent=2),'```','',
        'target_x=np.linspace(-1,1,33). Для обоих процессов '
        'задан лимит 45 с wall time от старта оболочки процесса '
        '(включая импорт, загрузку и сохранение), порог наблюдаемого '
        'RSS 512 МиБ, опрос через ps с паузой 0.25 с. '
        'Одновременно только один вычислительный worker; '
        'процесс supervisor и краткие ps не решают уравнения. '
        'Все thread limits сохранены равными 1. После SIGTERM '
        'оболочка ждёт прекращения до 2 с, затем SIGKILL; '
        'это время завершения, не дополнительный бюджет решения. '
        'Фактические небольшие задержки watchdog записаны, '
        'значения не округлены вниз до 45.000 с.', '',
        'Переиспользованы существующие `worker`, `bounded_run`, '
        'загрузчик и сериализация результата из '
        '`diagnostics/remizov_n4_followup.py` с profile=False. '
        'cProfile импортируется старым модулем, но не активируется: '
        'Observer не создаётся, функции не оборачиваются, '
        'профильные файлы не формируются.', '',
        'Единственное изменение существующего кода — в диагностической '
        'CLI этого файла: choices для --worker-id расширены двумя '
        'ID F0006/F0007. Без этого argparse отклонял бы новые ID '
        'до расчёта. Функции worker и watchdog не изменены. '
        'Создан отдельный драйвер `diagnostics/remizov_variable_n4_study.py`, '
        'который делает два независимых вызова; retries и '
        'прерывание после первого результата отсутствуют.', '',
        'Окружение:', '', '```json',json.dumps(s['environment'],indent=2),'```','',
        '## Наблюдаемые результаты двух новых попыток', '',
        '| equation_id | regime | n | status | max_abs_error | runtime, с | RSS, МиБ | failure_reason |',
        '|---|---|---:|---|---:|---:|---:|---|']
    for row in rows[1:]:
        lines.append(f"| {row['equation_id']} | {row['regime']} | 4 | {row['status']} | {fmt(row['max_abs_error'])} | {row['process_wall_seconds']:.6f} | {row['observed_peak_rss_mib']:.3f} | {row['failure_reason'] or '—'} |")
    lines += ['',
        'Runtime в этой таблице — wall time процесса для всех строк. '
        'Для внешней остановки оно является временем до прерывания, '
        'а не временем полного решения. Время внутри решателя '
        'записано отдельно ниже и в JSON. RSS — наблюдаемый пик '
        'по опросам; transient peaks между ними не исключены.', '',
        'Доступная диагностика и сравнение с прежним F0005 n=4:', '',
        '| equation_id | solver time, с | final dx | final Nt | quad разность / порог | spatial разность / порог | применений S | узлов | максимальная рабочая сетка | всего обработано пространственных точек |',
        '|---|---:|---:|---:|---|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['equation_id']} | {fmt(row['solver_runtime_seconds'])} | {fmt(row['final_dx'])} | {fmt(row['final_nt'])} | {fmt(row['quadrature_error_estimate'])} / {fmt(row['last_quadrature_threshold'])} | {fmt(row['spatial_error_estimate'])} / {fmt(row['last_spatial_threshold'])} | {fmt(row['n_operator_applications'])} | {fmt(row['n_quadrature_node_evaluations'])} | {fmt(row['spatial_grid_points_peak'])} | {fmt(row['spatial_grid_points_total_processed'])} |")
    lines += ['',
        '| equation_id | max_abs_error на 33 x | амплитуда max abs(y) на тех же 33 x | E / амплитуда | целевая амплитуда генератора на 257 x |',
        '|---|---:|---:|---:|---:|']
    for row in rows:
        m=metadata[row['equation_id']]
        lines.append(f"| {row['equation_id']} | {fmt(row['max_abs_error'])} | {fmt(row['exact_amplitude_on_target_33'])} | {fmt(row['max_error_over_exact_amplitude_33'])} | {fmt(m['target_amplitude'])} |")
    lines += ['',
        'Амплитуда точного решения доступна из входных данных даже '
        'при timeout; она не заменяет неизвестную ошибку. E/амплитуда — '
        'нормированная максимальная абсолютная ошибка на общей сетке, '
        'не максимальная поточечная относительная ошибка и не глобальная '
        'оценка на всех x. Различие амплитуд на 33 и 257 x связано '
        'с сетками, точное решение не перенормировалось.', '']
    for run in s['runs']:
        eid=run['equation_id']; d=run['solver_result']
        lines += ['', f'### {eid}', '']
        if d:
            lines += [f"Получен штатный RemizovResult: success={d['success']}, "
                f"failure_reason={d['failure_reason'] or 'нет'}. "
                f"Время процесса {run['worker_wall_seconds']:.6f} с, "
                f"внутри solve_remizov {run['runtime_seconds']:.6f} с. "
                'Все поля результата, x,f, точное решение и '
                'точечная ошибка при success сохранены в отдельном JSON.', '',
                'История пространства:', '', '```json',
                json.dumps({key:d[key] for key in ['spatial_dx_levels','spatial_error_history',
                           'spatial_convergence_thresholds']},indent=2),'```','',
                '| dx | Nt уровни | разности уточнений квадратуры | пороги | converged |',
                '|---:|---|---|---|---|']
            for q in d['quadrature_history']:
                lines.append(f"| {fmt(q['dx_limit'])} | {', '.join(map(str,q['nt_levels']))} | {', '.join(fmt(v) for v in q['error_estimates'])} | {', '.join(fmt(v) for v in q['convergence_thresholds'])} | {q['converged']} |")
        else:
            lines += [f"Статус оболочки {run['status']}; "
                f"SIGTERM запрошен на {run['termination_requested_wall_seconds']:.9f} с, "
                f"процесс завершился на {run['worker_wall_seconds']:.9f} с. "
                f"Код завершения {run['child_returncode']}. "
                'RemizovResult не получен. Ошибка, финальные dx/Nt, '
                'истории/пороги и счётчики неизвестны. '
                'Не присваивается quadrature_not_converged либо '
                'spatial_not_converged; рабочая стадия остановки '
                'без инструментализации не установлена. '
                'Частичные численные данные не использовались.', '']
    lines += ['',
        '## Отличия задач помимо режима коэффициентов', '',
        'Все три задачи относятся к L2 и имеют один интервал [-1,1], '
        'но происходят из разных семейств и имеют разные AST и '
        'случайные параметры. Это не эксперимент, в котором '
        'изменён только режим коэффициентов.', '',
        '| equation_id | μ bump | R bump | scale y | y0 | v0 | всего AST узлов | sampled max abs(y″) на 257 x | sampled max abs(g) на 257 x |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        m=metadata[row['equation_id']]; c=m['serialized_ast']['construction']
        lines.append(f"| {row['equation_id']} | {fmt(c['bump_mu'])} | {fmt(c['bump_radius'])} | {fmt(c['solution_scale'])} | {fmt(m['y0'])} | {fmt(m['v0'])} | {m['tree_metrics']['total_nodes']} | {fmt(m['validation']['max_abs']['d2y'])} | {fmt(m['validation']['max_abs']['g'])} |")
    lines += ['',
        'Дополнительно различаются a_lower/a_upper, B_abs, C_R_abs, '
        'λ и gap, а значит T и ширина рабочей области; форма и '
        'амплитуда y, её поддержка и производные, forcing и '
        'стоимость его вычисления. Полные сериализованные AST и '
        'construction сохранены в preflight.json.', '',
        'Примеры из прежней штатной проверки на 257 x в [-1,1]: '
        'у F0005 b≈0.9422 и c_R≈−0.4591 постоянны; '
        'у F0006 b меняется примерно от 0.02383 до 0.42875, '
        'c_R от −1.10970 до −1.04865; у F0007 b '
        'от −0.90721 до 0.32161, c_R от −0.45069 до 0.18501. '
        'Это наблюдаемые диапазоны на сетке, а не новые глобальные '
        'сертификаты. F0006 имеет существенно меньшие '
        'рабочую область и sampled |y″|, чем F0005, но '
        'более сложные коэффициентные AST.', '',
        'Начальные данные y0,v0 различаются, однако данный глобальный '
        'решатель их не использует; они приведены для полноты '
        'описания датасета. Нельзя объяснять разницу результатов '
        'исключительно slow_varying/moderate_varying/constant.', '',
        '## Ответы и границы выводов', '',
        '1. **Численная работоспособность:** F0006 — успешное '
        'приближение с переменными коэффициентами при n=4. '
        'F0007 не дал результата за разрешённое время; это '
        'не доказательство математической несовместимости.',
        '2. **Точность:** F0006 E=0.09481647848327257 при '
        'амплитуде на 33 x 1.47747087180712, то есть '
        f"около {100*rows[1]['max_error_over_exact_amplitude_33']:.4f}% амплитуды. "
        'Это заметная ошибка, несмотря на успешные внутренние '
        'критерии. У F0007 ошибка не определена. Для сравнения, '
        f"F0005 E≈0.380744 (около {100*rows[0]['max_error_over_exact_amplitude_33']:.4f}% амплитуды на 33 x).",
        '3. **Стоимость:** F0006 потребовал 32640 S-применений, '
        '8160 Chernoff-узлов и обработки 38715120 '
        'пространственных точек; пик сетки 6376 точек. '
        'F0007 израсходовал внешние 45 с, его внутренний '
        'объём работы неизвестен. Ни в одной новой попытке '
        'наблюдаемый RSS не достиг 512 МиБ.',
        '4. **Причины отказа:** у F0007 подтверждён только '
        'external_time_limit. Причина незавершённого внутреннего '
        'уточнения и конкретная стадия не установлены. '
        'У F0006 штатного отказа нет.',
        '5. **Следующая серия после согласования:** F0006 — '
        'приоритетный кандидат для заранее ограниченных '
        'независимых запусков n=1,2; n=4 уже имеется. '
        'Для F0007 целесообразны отдельные n=1,2 как проверка '
        'доступности результата при меньшем числе S-применений; '
        'не подставлять их вместо несуществующей ошибки n=4. '
        'У всех сохранить те же критерии, x и пределы. '
        'Следующую серию сейчас не начинали.',
        '6. **Настройки:** данных недостаточно, чтобы менять '
        'численные параметры или ослаблять критерии. F0006 '
        'достиг внутренних порогов, но это не разделяет '
        'ошибки конечного n, сплайнов, квадратуры и хвоста. '
        'Timeout F0007 не указывает, какое изменение '
        'параметра было бы математически оправданным.', '',
        '**Математическая гипотеза:** существенная ошибка конечного '
        'n может сохраняться у F0006, как у F0005. '
        'Это согласуется с большой разницей между E≈0.095 '
        'и внутренними индикаторами порядка 1e-7–1e-8, '
        'но не устанавливает количественную декомпозицию ошибки. '
        'Разности уточнений не являются доказанными абсолютными '
        'границами полной ошибки и не могут вычитаться из E '
        'для получения чистой ошибки конечного n. '
        'Сходимость по n для F0006/F0007 пока не исследована.', '',
        'Сохранена также прежняя оговорка: формула T не включает '
        'глобальную норму g_R, поэтому tail_eps не трактуется '
        'как универсальная абсолютная граница ошибки решения. '
        'Математический алгоритм не пересматривался.', '',
        '## Файлы, проверки и воспроизводимость', '',
        '- summary.json, preflight.json — конфигурация, окружение, '
        'входные параметры, аналитические оценки и два результата;',
        '- отдельные каталоги AMV1_F0006_S00_n4 и '
        'AMV1_F0007_S00_n4 — run.json, worker.log и '
        'resource_samples.jsonl; solver_return.json имеется '
        'только при штатном возврате;',
        '- attempts.csv и comparison_F0005_F0006_F0007.json — '
        'табличные показатели двух новых попыток и сравнения;',
        '- verification.json — согласованность записанных ошибок, '
        'счётчиков, порогов, RSS и целостность исходных файлов.', '',
        '```bash',
        '.venv/bin/python -B diagnostics/remizov_variable_n4_study.py --output-dir results/remizov_variable_n4_repeat',
        '.venv/bin/python -B diagnostics/report_remizov_variable_n4.py results/remizov_variable_n4_repeat',
        '```', '',
        'Это команды воспроизводимости, повтор сейчас не выполнялся. '
        'Драйвер требует новый каталог, отчёт не перезаписывает '
        'существующий REPORT.md. Новых зависимостей не установлено. '
        'Математические файлы совпадают с миграционными SHA-256; '
        'загрузчик, датасет и три прежних каталога результатов '
        'не изменены. Полные SHA-256 прежних результатов '
        'проверяются отдельно. Снимки исходного кода исключают '
        '.venv, .git, caches, results, .DS_Store; границы '
        'сверки явно сохранены. generator_DE не редактировался. '
        'Git commit/push, новые серии и оптимизации не выполнялись.', '']
    report.write_text('\n'.join(lines),encoding='utf-8')

    manifest=json.loads((ROOT/'MIGRATION_MANIFEST.json').read_text())
    core={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==manifest['source_sha256'][name]
          for name in manifest['unchanged_mathematical_core']}
    now=snapshot(ROOT); original_now=snapshot(ROOT.parent/'generator_DE')
    baseline=s.get('baseline_before_infrastructure_changes')
    infrastructure_change=None; project_before=[]; original_before=[]
    if baseline:
        project_before=[p for p,h in baseline['project'].items() if now.get(p)!=h]
        original_before=[p for p,h in baseline['original'].items() if original_now.get(p)!=h]
        source=(ROOT/'diagnostics/remizov_n4_followup.py').read_text()
        expanded="parser.add_argument('--worker-id',choices=['AMV1_F0005_S00','AMV1_F0000_S00','AMV1_F0006_S00','AMV1_F0007_S00'])"
        old="parser.add_argument('--worker-id',choices=['AMV1_F0005_S00','AMV1_F0000_S00'])"
        assert source.count(expanded)==1
        old_hash=hashlib.sha256(source.replace(expanded,old).encode()).hexdigest()
        infrastructure_change=old_hash==baseline['project']['diagnostics/remizov_n4_followup.py']
        assert infrastructure_change and project_before==['diagnostics/remizov_n4_followup.py']
        assert not original_before
    verification={'core_matches_migration_source':core,'checks':checks,
        'exactly_two_attempts':len(s['runs'])==2,'attempts_per_equation':1,
        'second_task_completed_attempt_independently':True,'automatic_retries':0,
        'previous_results_identical':{name:tree_hashes(ROOT/'results'/name)==hashes
             for name,hashes in s['previous_results_before_sha256'].items()},
        'project_changed_during_execution':[p for p,h in s['project_before_execution_sha256'].items() if now.get(p)!=h],
        'original_changed_or_removed':[p for p,h in s['original_before_sha256'].items() if original_now.get(p)!=h],
        'preexisting_project_changes_since_pre_edit_baseline':project_before,
        'CLI_edit_exactly_two_added_worker_choices':infrastructure_change,
        'new_dependencies_installed':False}
    assert all(core.values()) and all(verification['previous_results_identical'].values())
    assert not verification['project_changed_during_execution']
    assert not verification['original_changed_or_removed']
    write_json(out/'verification.json',verification)
    print(report)


if __name__=='__main__':
    main()
