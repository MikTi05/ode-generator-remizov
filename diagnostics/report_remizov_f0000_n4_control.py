"""Report the saved single control attempt without invoking any solver."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from diagnostics.remizov_n_study import snapshot, write_json
from diagnostics.remizov_n4_followup import tree_hashes


def fmt(value):
    return 'не определено' if value is None else f'{value:.10g}'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_dir',type=Path)
    args=parser.parse_args()
    out=args.results_dir.resolve()
    if not out.is_relative_to(ROOT/'results'):
        parser.error('results directory must be under project/results')
    report=out/'REPORT.md'
    if report.exists():
        raise SystemExit('Refusing to overwrite an existing report')
    s=json.loads((out/'summary.json').read_text())
    assert len(s['runs'])==1
    run=s['runs'][0]
    assert run['equation_id']=='AMV1_F0000_S00' and run['n']==4
    assert run['config']==s['config'] and not run['instrumented']
    prior_names=['remizov_n_study_2026-10-11','remizov_n4_followup_2026-10-12']
    prior=[json.loads((ROOT/'results'/name/'summary.json').read_text()) for name in prior_names]
    old_plain=next(r for r in prior[0]['runs'] if r['equation_id']==run['equation_id'] and r['n']==4)
    old_profile=next(r for r in prior[1]['runs'] if r['equation_id']==run['equation_id'])
    records=[]
    for label,r,source in [('2026-10-11 без профиля',old_plain,prior_names[0]),
                           ('2026-10-12 cProfile',old_profile,prior_names[1]),
                           ('2026-10-12 контроль без профиля',run,out.name)]:
        records.append({'attempt':label,'equation_id':r['equation_id'],'n':r['n'],
            'status':r['status'],'max_abs_error':r['max_abs_error'],
            'solver_runtime_seconds':r['runtime_seconds'],
            'worker_wall_seconds':r['worker_wall_seconds'],
            'worker_peak_rss_mib':r['worker_peak_rss_mib'],'source':source})
    write_json(out/'comparison.json',records)
    with (out/'comparison.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    sample_path=out/'F0000_n4_unprofiled/resource_samples.jsonl'
    samples=[json.loads(line) for line in sample_path.read_text().splitlines()]
    rss=[sample for sample in samples if sample['rss_kib'] is not None]
    measured_peak=max(sample['rss_kib'] for sample in rss)/1024
    assert measured_peak==run['worker_peak_rss_mib']
    d=run['solver_result']
    if d is None:
        assert run['max_abs_error'] is None and run['runtime_seconds'] is None
        assert not (out/'F0000_n4_unprofiled/solver_return.json').exists()
    elif d['success']:
        import numpy as np
        np.testing.assert_array_equal(d['x'],np.linspace(-1,1,33))
        assert float(np.max(np.abs(np.array(d['f'])-np.array(run['y_exact']))))==run['max_abs_error']
    else:
        assert run['max_abs_error'] is None and not d['f']

    # Only inspect metadata for recommendations; do not load or solve these tasks.
    wanted={'AMV1_F0006_S00','AMV1_F0007_S00','AMV1_F0003_S00'}
    candidates=[]
    for line in (ROOT/'data/generated/all_methods_equations_v1_metadata.jsonl').read_text().splitlines():
        m=json.loads(line)
        if m['equation_id'] not in wanted:
            continue
        gap=m['remizov_lambda']-m['remizov_c_abs_cert']
        T=max(0.0,(math.log(2)-math.log(gap)-math.log(s['config']['tail_eps']))/gap)
        shift=max(2*math.sqrt(m['a_upper_cert']*T/4),2*m['b_abs_cert']*T/4)
        width=m['x1']-m['x0']+8*shift
        candidates.append({'equation_id':m['equation_id'],'complexity':m['complexity_level'],
            'regime':m['regime'],'gap':gap,'T':T,'width_bound_n4':width,
            'source_points_initial_bound':max(3,math.ceil(width/.05))+1,
            'source_points_finest_bound':max(3,math.ceil(width/(.05/32)))+1,
            'analytical_preflight_only_not_executed':True,
            'cost_and_success_not_guaranteed':True})
    candidates.sort(key=lambda r:r['width_bound_n4'])
    write_json(out/'next_task_preflight.json',candidates)

    lines=['# Контроль F0000 при n=4 без профилирования', '',
        'Дата по контексту пользователя: 2026-10-12. Выполнена ровно '
        'одна новая попытка решения `AMV1_F0000_S00`, n=4. '
        'Другие уравнения и значения n не запускались; повторов не было. '
        'Данные и прежние результаты не перезаписывались.', '',
        '## Состояние проекта и возможность повторного использования', '',
        'Изучены предыдущие отчёты и JSON из двух каталогов, а также '
        '`diagnostics/remizov_n_study.py` и `diagnostics/remizov_n4_followup.py`. '
        'Существующая функция `bounded_run(equation_id, profile, folder)` '
        'подходит для контрольной попытки без изменений. При `profile=False` '
        'она не передаёт флаг --profile дочернему процессу; worker '
        'устанавливает observer=None и вызывает оригинальный solve_remizov '
        'без инструментальных обёрток. Модуль cProfile импортируется '
        'существующим файлом, но Profile не создаётся, enable не вызывается '
        'и профилирование не выполняется. Сплайны и функции решателя не заменяются.', '',
        'Основная CLI прежнего скрипта запускает две задачи, поэтому '
        'она не использовалась. Новый небольшой драйвер '
        '`diagnostics/remizov_f0000_n4_control.py` делает ровно один вызов:', '',
        '```python',
        "bounded_run('AMV1_F0000_S00', False, new_results_dir / 'F0000_n4_unprofiled')",
        '```', '',
        'Исходный решатель, генератор, загрузчик и прежняя оболочка '
        'не редактировались. До старта сверены математические SHA-256 '
        'миграционного манифеста, точная конфигурация и ограничения. '
        'Новый каталог создаётся с exist_ok=False, вложенная папка '
        'запуска тоже должна быть новой.', '',
        '## Неизменная конфигурация и ограничения', '', '```json',
        json.dumps(s['config'],indent=2,ensure_ascii=False),'```','',
        '`target_x=np.linspace(-1,1,33)`. При штатном возврате сравнение '
        'выполняется с восстановленным `equation.y_exact(result.x)`, '
        'не с интерполированными значениями NPZ и не с данными задачи Коши.', '',
        'Ровно один процесс решателя; отдельный процесс оболочки следит '
        'за временем и RSS. Параллельных задач, потоков профилирования '
        'и сэмплирования не было. Монитор запускает краткие команды ps; '
        'они не выполняют математические расчёты. Временной предел '
        '45 с включает импорт, загрузку, решение и запись результата. '
        'RSS наблюдается примерно раз в 0.25 с, порог 512 МиБ. '
        'Наблюдаемый пик не является строгим мгновенным максимумом. '
        'При достижении лимита SIGTERM прекращает непрофилируемый процесс '
        'с обычным обработчиком; существующая оболочка допускает до '
        '2 с ожидания завершения, а затем SIGKILL. Это ожидание не '
        'дополнительный бюджет расчёта. Ограничения и бюджеты не повышались.', '',
        'Срабатывание wall-clock watchdog имеет небольшую задержку '
        'опроса и команды ps; фактические моменты сигнала и завершения '
        'записаны отдельно. Они не округлены вниз до 45.000 с.', '',
        'В предыдущей попытке без профиля был внешний timeout, '
        'а профилируемая попытка достигла dx=0.003125/Nt=2048 к '
        'моменту timeout. Поэтому завершение этой контрольной попытки '
        'в пределах лимита заранее не гарантировалось. Возможная '
        'максимальная рабочая сетка F0000 n=4 — 26 331 точка, '
        'максимальный бюджет применений S — 98 112. Эти оценки '
        'не ограничивают время. Повторная попытка при отказе не предусмотрена.', '',
        'Окружение:', '', '```json',json.dumps(s['environment'],indent=2,ensure_ascii=False),'```','',
        '## Подтверждённый результат текущей попытки', '',
        '| Показатель | Значение |','|---|---|',
        f"| equation_id / n | {run['equation_id']} / 4 |",
        f"| Итоговый статус оболочки | `{run['status']}` |",
        f"| Получен RemizovResult | {'да' if d is not None else 'нет'} |",
        f"| Успешное решение | {'да' if d is not None and d['success'] else 'не получено'} |",
        f"| Время процесса, с | {run['worker_wall_seconds']:.9f} |",
        f"| Время внутри решателя, с | {fmt(run['runtime_seconds'])} |",
        f"| Наблюдаемый пик RSS, МиБ | {run['worker_peak_rss_mib']:.6f} |",
        f"| max_abs_error | {fmt(run['max_abs_error'])} |",
        f"| Код завершения процесса | {run['child_returncode']} |", '']
    if d is None:
        lines += [f"SIGTERM запрошен на {run['termination_requested_wall_seconds']:.9f} с "
            'от старта измерения оболочки. Для текущей попытки '
            'достоверно установлена внешняя причина прекращения. '
            'Наблюдаемый RSS не достиг 512 МиБ; ограничение памяти '
            'не сработало. `solver_return.json` отсутствует, '
            'worker.log не содержит сообщения штатного отказа. '
            'Итоговый внутренний статус решателя неизвестен.', '',
            'Последние dx, Nt, квадратурные/пространственные сравнения '
            'и счётчики именно этой попытки недоступны: без '
            'инструментализации они возвращаются только вместе '
            'с RemizovResult. Рабочая стадия на момент SIGTERM '
            'не установлена. Частичных решений нет, ошибка не '
            'вычислялась. Прежние профилированные уровни не '
            'переносятся в текущую попытку.', '']
    else:
        lines += [f"Штатный результат: success={d['success']}, "
            f"failure_reason={d['failure_reason'] or 'нет'}. "
            f"Итоговые dx={fmt(d['final_dx'])}, Nt={d['final_nt']}, T={fmt(d['T'])}. "
            'Все поля RemizovResult, включая истории, пороги и '
            'счётчики, сохранены в `F0000_n4_unprofiled/solver_return.json`.', '',
            '```json',json.dumps({key:d[key] for key in ['spatial_dx_levels',
                'spatial_error_history','spatial_convergence_thresholds','quadrature_history']},indent=2),'```','']
    lines += [f"В журнале RSS {len(samples)} наблюдений; пик на первой "
        f"достигшей его записи при wall≈{max(rss,key=lambda v:v['rss_kib'])['wall_seconds']:.6f} с. "
        'Не наблюдалось достижения порога памяти. '
        'Отсутствие transient spikes между опросами не доказывается.', '',
        '## Сравнение с предыдущими попытками', '',
        '| Попытка | Статус | Время процесса, с | Наблюдаемый RSS, МиБ | Ошибка |',
        '|---|---|---:|---:|---|']
    for r in records:
        lines.append(f"| {r['attempt']} | {r['status']} | {r['worker_wall_seconds']:.6f} | {r['worker_peak_rss_mib']:.2f} | {fmt(r['max_abs_error'])} |")
    lines += ['',
        'У профилируемого запуска последний снимок на 44.6166 с '
        'наблюдаемого решателя показывал Nt=2048, dx=0.003125. '
        'Первые четыре квадратурные стадии приняты; последнее '
        'завершённое пространственное сравнение на dx=0.00625 '
        'равнялось 1.011460737e-6 при пороге 7.077089274e-8. '
        'Это сведения только о прежнем профилированном запуске.', '',
        'Сравнение одной профилированной и одной контрольной попытки '
        'не является точной оценкой накладных расходов. Время '
        'внешне остановленной попытки — момент обрыва, а не время '
        'до естественного окончания. По двум timeout нельзя '
        'получить разницу/коэффициент замедления полной задачи. '
        'Также не контролировались нагрузка компьютера, тепловой '
        'режим и состояние системных кэшей. Отличие RSS не '
        'устанавливает отличие достигнутой стадии.', '',
        '## Выводы: наблюдения и гипотезы', '']
    if d is None:
        lines += ['**Подтверждённое наблюдение.** Отключение профилирования '
            'не обеспечило штатный результат F0000 n=4 в пределах '
            '45 с именно в этой контрольной попытке. Следовательно, '
            'нынешний практический лимит сохраняется и без cProfile. '
            'Это не утверждение, что такой расчёт никогда не '
            'уложится в 45 с при иной загрузке или на другом компьютере.', '']
    else:
        lines += [f"**Подтверждённое наблюдение.** Без профилирования "
            f"штатный результат получен со статусом {run['status']}. "
            'Одна пара попыток не даёт точной оценки стоимости '
            'профилирования; результат не переносится автоматически '
            'на другие уравнения.', '']
    lines += ['**Вычислительное ограничение.** Прежний профиль указывает '
        'на многократные применения S, построение/вычисление сплайнов '
        'и полный пересчёт квадратуры при уточнении dx. Он объясняет '
        'структуру затрат реализации, но точная стадия текущего '
        'timeout по нему не восстанавливается. Суммарные рабочие '
        'объёмы могут быть велики даже при небольшом пике памяти.', '',
        '**Математические границы вывода.** Timeout не доказывает '
        'несовместимость задачи с методом, несходимость по n, '
        'невыполнимость критериев или ошибку знака. Ни '
        'quadrature_not_converged, ни spatial_not_converged текущей '
        'внешне остановленной попытке не присваиваются. '
        'Причина того, что прежняя профилируемая попытка требовала '
        'многих уточнений, остаётся предметом отдельного исследования; '
        'новый запуск не разделяет погрешности конечного n, '
        'интерполяции, квадратуры и усечения.', '',
        '**Гипотеза.** Повторные вычисления на мелких расширенных '
        'сетках могут быть основным источником длительности и '
        'без профиля. Эта гипотеза согласуется со старым профилем, '
        'но непосредственно не измерена в текущей попытке.', '',
        '## Следующие эксперименты на других задачах', '',
        'Не повторять автоматически F0000. Сначала предложены '
        'две отдельно назначенные попытки на других режимах: '
        '`AMV1_F0006_S00` (L2, slow_varying) и '
        '`AMV1_F0007_S00` (L2, moderate_varying). Сохранить n=4, '
        'ту же целевую сетку, конфигурацию и пределы 45 с/512 МиБ; '
        'по одной попытке на задачу без профилирования и без '
        'повторной настройки при отказе. Цель — проверить, '
        'насколько ограничение характерно для других коэффициентных '
        'режимов, а не только F0000.', '',
        'Предварительно прочитаны только метаданные выбранных '
        'кандидатов и рассчитана консервативная геометрия исходной '
        'рабочей сетки; никаких решений этих задач сейчас не '
        'выполнялось. Дополнительный кандидат для отдельного '
        'этапа — F0003 (L1, oscillatory).', '',
        '| equation_id | сложность / режим | T | максимальная ширина при n=4 | исходная сетка при dx=0.05 | при dx=0.0015625 |',
        '|---|---|---:|---:|---:|---:|']
    for c in candidates:
        lines.append(f"| {c['equation_id']} | {c['complexity']} / {c['regime']} | {c['T']:.6f} | {c['width_bound_n4']:.6f} | {c['source_points_initial_bound']} | {c['source_points_finest_bound']} |")
    lines += ['',
        'Геометрическая оценка не предсказывает количество принятых '
        'Nt/dx, стоимость AST, ошибку или вероятность success. '
        'Кандидаты F0006/F0007 имеют сравнительно небольшую '
        'сертифицированную область; у F0003 она шире. '
        'F0001/F0002 не предлагаются первыми: малый gap ведёт '
        'к T≈79 и существенно более широкой области; на самом '
        'мелком разрешённом dx она превысила бы штатный лимит '
        '100000 точек, если уточнение дойдёт до этого уровня. '
        'Это ресурсная особенность текущих сертификатов и '
        'конфигурации, не математическая недопустимость задач.', '',
        'Для каждого будущего штатного возврата сохранять полный '
        'RemizovResult и ошибку только при success. После такой '
        'малой проверки можно выбирать задачу для отдельного '
        'исследования зависимости ошибки от n=1,2,4, заранее '
        'ограничив число попыток и сохранив единую точность. '
        'Переходить к n=8,16,32 или массовой серии оснований '
        'пока нет. Ни эти эксперименты, ни оптимизация сейчас '
        'не выполнялись.', '',
        '## Сохранённые файлы и воспроизводимость', '',
        '- `summary.json` — конфигурация, окружение, одна попытка, '
        'снимки SHA-256 и сверка;',
        '- `F0000_n4_unprofiled/run.json` — статус и ресурсы '
        'контрольной попытки;',
        '- `F0000_n4_unprofiled/resource_samples.jsonl` — весь '
        'ряд наблюдений RSS и wall time;',
        '- `F0000_n4_unprofiled/worker.log` — вывод worker;',
        '- `comparison.json`, `comparison.csv` — три исторические '
        'попытки F0000 n=4;',
        '- `next_task_preflight.json` — только аналитические '
        'оценки для предложенных будущих задач;',
        '- `verification.json` — контроль целостности и '
        'согласованности сохранённых результатов.', '',
        'Новые модули: `diagnostics/remizov_f0000_n4_control.py` '
        'и `diagnostics/report_remizov_f0000_n4_control.py`. '
        'Второй модуль читает результаты и не вызывает решатель.', '',
        '```bash',
        '.venv/bin/python -B diagnostics/remizov_f0000_n4_control.py --output-dir results/remizov_f0000_n4_control_repeat',
        '.venv/bin/python -B diagnostics/report_remizov_f0000_n4_control.py results/remizov_f0000_n4_control_repeat',
        '```', '',
        'Команды приведены только для воспроизводимости; повторный '
        'расчёт сейчас не выполнен. Драйвер требует новый каталог, '
        'отчёт также не перезаписывает существующий REPORT.md.', '',
        'Математические файлы проверены по исходным SHA-256 '
        'миграционного манифеста, предыдущие два каталога '
        'результатов — по полным деревьям SHA-256. Предсуществующий '
        'код и датасет нового проекта и проверенные файлы '
        'generator_DE не изменены. Снимок кода исключает .venv, '
        '.git, caches, results, .DS_Store; это не утверждение '
        'о неизменности каждого байта окружения. Старые результаты '
        'проверяются отдельно без этого исключения. Новые '
        'зависимости не установлены; commit/push не выполнялись.', '']
    report.write_text('\n'.join(lines),encoding='utf-8')
    manifest=json.loads((ROOT/'MIGRATION_MANIFEST.json').read_text())
    core={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==manifest['source_sha256'][name]
          for name in manifest['unchanged_mathematical_core']}
    after=snapshot(ROOT); original_after=snapshot(ROOT.parent/'generator_DE')
    verification={'core_matches_migration_source':core,
        'preexisting_project_changed_or_removed':[p for p,h in s['project_before_sha256'].items() if after.get(p)!=h],
        'original_changed_or_removed':[p for p,h in s['original_before_sha256'].items() if original_after.get(p)!=h],
        'previous_results_identical':{name:tree_hashes(ROOT/'results'/name)==hashes for name,hashes in s['previous_results_before_sha256'].items()},
        'exactly_one_control_attempt':len(s['runs'])==1,
        'config_matches_requested':run['config']==s['config'],
        'rss_recalculation_matches':measured_peak==run['worker_peak_rss_mib'],
        'resource_samples':len(samples),'instrumented':False,
        'no_profile_artifacts':not any((out/'F0000_n4_unprofiled').glob('*.pstats')),
        'solver_result_returned':d is not None,'error_is_null_if_no_success':
            run['max_abs_error'] is None if d is None or not d['success'] else False,
        'automatic_retries':0,'other_tasks_solved':0,
        'new_dependencies_installed':False}
    assert all(core.values())
    assert all(verification['previous_results_identical'].values())
    assert not verification['preexisting_project_changed_or_removed']
    assert not verification['original_changed_or_removed']
    write_json(out/'verification.json',verification)
    print(report)


if __name__=='__main__':
    main()
