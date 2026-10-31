"""Build reports/figures from saved data only; no solver execution."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diagnostics.remizov_n_study import write_json, snapshot
from diagnostics.remizov_n4_followup import tree_hashes


def fmt(value):
    return 'не определено' if value is None else f'{value:.10g}'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_dir',type=Path)
    args=parser.parse_args()
    out=args.results_dir.resolve()
    report=out/'REPORT.md'
    if report.exists():
        raise SystemExit('Refusing to overwrite report')
    current=json.loads((out/'summary.json').read_text())
    previous_path=ROOT/'results/remizov_n_study_2026-10-11'
    prior=json.loads((previous_path/'summary.json').read_text())
    f5=current['runs'][0]
    f0=current['runs'][1]
    comparisons=[dict(r,source='../remizov_n_study_2026-10-11/summary.json')
        for r in prior['runs'] if r['equation_id']=='AMV1_F0005_S00' and r['n'] in (1,2)]
    comparisons.append(dict(f5,source='F0005_n4/run.json'))
    comparison=[{key:r.get(key) for key in ('equation_id','n','status','max_abs_error',
        'runtime_seconds','worker_wall_seconds','worker_peak_rss_mib','source')}
        for r in comparisons]
    write_json(out/'F0005_comparison.json',comparison)
    with (out/'F0005_comparison.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(comparison[0]))
        writer.writeheader(); writer.writerows(comparison)

    progress=f0['partial_diagnostics']
    diag=progress['partial_solver_diagnostics']
    profile=json.loads((out/'F0000_n4_profile/profile_summary.json').read_text())
    rows=profile['rows_by_self_seconds']
    def find(function,suffix,line=None):
        matches=[r for r in rows if r['function']==function and r['filename'].endswith(suffix)
                 and (line is None or r['line']==line)]
        assert len(matches)==1, (function,suffix,line)
        return matches[0]
    from scipy.interpolate import CubicSpline, PPoly
    metrics=[('Применение S, включая вложенные операции',find('_apply_operator_on_grid','remizov_solver.py')),
        ('Построение CubicSpline',find('__init__','_cubic.py',CubicSpline.__init__.__code__.co_firstlineno)),
        ('Вызовы сплайна',find('__call__','_interpolate.py',PPoly.__call__.__code__.co_firstlineno)),
        ('Коэффициенты и проверка сертификатов внутри S',find('_evaluate_coefficients','remizov_solver.py')),
        ('Проверка границ сдвигов внутри S',find('_coordinates_inside_source','remizov_solver.py')),
        ('Forcing g_R на исходной сетке',find('g_remizov','all_methods_dataset_loader.py')),
        ('Построение рабочих сеток',find('_uniform_grid','remizov_solver.py')),
        ('Построение узлов Gauss–Legendre',find('leggauss','legendre.py'))]
    events=[json.loads(line) for line in (out/'F0000_n4_profile/stage_events.jsonl').read_text().splitlines()]
    stage_start={}; stages=[]
    for e in events:
        dx=e['partial_solver_diagnostics']['final_dx']
        elapsed=e['elapsed_solver_observed_seconds']
        if e['event']=='enter:_converge_quadrature':
            stage_start[dx]=elapsed
        elif e['event']=='return:_converge_quadrature':
            stages.append({'dx':dx,'completed':True,
                           'stage_elapsed_seconds':elapsed-stage_start[dx]})
        elif e['event']=='external_SIGTERM':
            stages.append({'dx':dx,'completed':False,
                           'stage_elapsed_seconds':elapsed-stage_start[dx]})
    derived={'profile_metrics':[{'label':name,**row} for name,row in metrics],
        'quadrature_stages':stages,'last_active_calls':progress['active_calls'],
        'partial_diagnostics_are_not_a_solution':True,
        'profile_inclusive_metrics_overlap_do_not_sum':True,
        'coarse_completed_node_timings':progress['completed_inclusive_timings'],
        'SIGTERM_checkpoint_captured':events[-1]['event']=='external_SIGTERM'}
    write_json(out/'F0000_profile_analysis.json',derived)

    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/ode-remizov-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    valid=[r for r in comparison if r['status']=='success' and r['max_abs_error'] is not None]
    fig,ax=plt.subplots(figsize=(7.4,4.5),constrained_layout=True)
    ax.plot([r['n'] for r in valid],[r['max_abs_error'] for r in valid],marker='o')
    for r in valid:
        ax.annotate(f"{r['max_abs_error']:.6f}",(r['n'],r['max_abs_error']),
                    xytext=(7,7),textcoords='offset points')
    ax.set(xlabel='n (Chernoff operator applications per quadrature node)',
           ylabel='Max absolute error on 33 points in [-1, 1]',
           title='AMV1_F0005_S00: fixed criteria, adaptive working grids',
           xticks=[1,2,4],ylim=(0,3.5),xlim=(.8,4.55))
    ax.grid(alpha=.25)
    for extension in ('png','svg'):
        fig.savefig(out/f'F0005_error_vs_n.{extension}',dpi=160)
    plt.close(fig)

    config=current['config']
    lines=['# Продолжение исследования: независимые запуски n=4', '',
        'Дата исследования по контексту пользователя: 2026-10-12. '
        'Результаты сохранены отдельно от датасета и предыдущего исследования. '
        'Новые вызовы: F0005 без инструментализации; F0000 с cProfile и '
        'наблюдением стадий. Никаких повторов общей серии или изменений '
        'математической реализации не было.', '',
        '## Подтверждённые настройки и оценка стоимости до запуска', '',
        'Перед вызовом конфигурация `make_verification_config(4)` проверялась '
        'на полное равенство требуемым значениям:', '', '```json',
        json.dumps(config,indent=2,ensure_ascii=False), '```', '',
        'Целевая сетка `np.linspace(-1,1,33)`, точное решение '
        '`equation.y_exact(result.x)`. Материалы сравнения n=1,2 взяты '
        'из прежнего JSON без повторного вычисления. Сетка результатов '
        'проверена на точное равенство заданной.', '',
        'Прежнее время F0005: 8.89 с при n=1 и 21.14 с при n=2; '
        'n=2 обработал 64 624 136 пространственных точек. Простая '
        'экстраполяция пропорционально n даёт 42.28 с при n=4, '
        'но это не граница стоимости: при n=4 три промежуточные рабочие '
        'сетки вместо одной, а адаптивные Nt/dx могут остановиться на '
        'других уровнях. Верхняя оценка числа применений S — 98 112, '
        'размер исходной сетки на самом мелком разрешённом dx — 99 707 '
        'точек при лимите 100 000. Завершение за 45 с заранее не гарантировалось.', '',
        'Каждый процесс независимо ограничен 45 с wall time от старта '
        'процесса и 512 МиБ наблюдаемого RSS. Запуски последовательные. '
        'Память опрашивается через ps с паузой 0.25 с: пик наблюдаемый, '
        'не гарантированный мгновенный максимум. При превышении отправляется '
        'SIGTERM; до 2 с предусмотрено только на сохранение диагностики, '
        'после чего SIGKILL. Это не дополнительный бюджет решения. '
        'Все thread limits сохранены равными 1. После timeout F0005 '
        'независимая диагностическая задача F0000 всё равно была бы '
        'выполнена; при memory limit/неожиданном исключении оболочка '
        'останавливается. Повторов и автоматического увеличения бюджетов нет.', '',
        'Окружение:', '', '```json',json.dumps(current['environment'],indent=2,ensure_ascii=False),'```','',
        '## Наблюдения: F0005, n=1,2,4', '',
        '| n | status | max_abs_error | время решателя, с | время процесса, с | наблюдаемый RSS, МиБ |',
        '|---:|---|---:|---:|---:|---:|']
    for r in comparison:
        lines.append(f"| {r['n']} | {r['status']} | {fmt(r['max_abs_error'])} | {fmt(r['runtime_seconds'])} | {fmt(r['worker_wall_seconds'])} | {r['worker_peak_rss_mib']:.2f} |")
    lines += ['',
        'n=1,2 относятся к 2026-10-11; n=4 — к текущему запуску. '
        'Время зависит от загрузки компьютера, поэтому отношение времён '
        'не является чистым измерением сложности. Настройки, версии '
        'вычислительных библиотек и ограничения потоков совпадают.', '',
        '![Ошибка F0005](F0005_error_vs_n.png)', '',
        'Ошибка убывает на всех трёх исследованных n. Отношения E1/E2 '
        f"и E2/E4 равны {comparison[0]['max_abs_error']/comparison[1]['max_abs_error']:.6f} и "
        f"{comparison[1]['max_abs_error']/comparison[2]['max_abs_error']:.6f}. "
        'Это подтверждение эмпирического уменьшения ошибки на n=1,2,4 '
        'для одной задачи, а не доказательство предела n→∞, порядка '
        'сходимости или поведения остальных уравнений.', '',
        'Диагностика успешных расчётов:', '',
        '| n | final dx | final Nt | quad разность / порог | spatial разность / порог | узлов | применений S | обработано пространственных точек | пиковая рабочая сетка |',
        '|---:|---:|---:|---|---|---:|---:|---:|---:|']
    for r in comparisons:
        d=r.get('solver_result')
        if not d:
            continue
        q=d['quadrature_history'][-1]
        lines.append(f"| {r['n']} | {fmt(d['final_dx'])} | {d['final_nt']} | {fmt(d['quadrature_error_estimate'])} / {fmt(q['convergence_thresholds'][-1])} | {fmt(d['spatial_error_estimate'])} / {fmt(d['spatial_convergence_thresholds'][-1])} | {d['n_quadrature_node_evaluations']} | {d['n_operator_applications']} | {d['spatial_grid_points_total_processed']} | {d['spatial_grid_points_peak']} |")
    d5=f5['solver_result']
    lines += ['',
        f"При n=4 T={fmt(d5['T'])}, gap={fmt(d5['lambda_c_gap'])}, tail_eps=1e-8. "
        'Успешный статус означает прохождение внутренних сравнений, '
        'а не сертифицированную малую ошибку относительно точного y. '
        'Финальные разности квадратуры/пространства порядка 1e-9 '
        'намного меньше полной ошибки 0.3807, но не являются '
        'независимыми гарантированными составляющими этой ошибки.', '',
        'F0005 n=4 потребовал меньше квадратурных узлов, чем прежний n=2 '
        '(9176 против 11224), при том же final dx=0.003125; число '
        'применений S и обработанных пространственных точек выросло. '
        'Поэтому простая линейная оценка времени по n была неточной.', '',
        'Полная новая история пространственного уточнения:', '',
        '```json',json.dumps({k:d5[k] for k in ['spatial_dx_levels','spatial_error_history',
             'spatial_convergence_thresholds']},indent=2),'```', '',
        '| dx | Nt уровни | quad разности | quad пороги |', '|---:|---|---|---|']
    for q in d5['quadrature_history']:
        lines.append(f"| {fmt(q['dx_limit'])} | {', '.join(map(str,q['nt_levels']))} | {', '.join(fmt(v) for v in q['error_estimates'])} | {', '.join(fmt(v) for v in q['convergence_thresholds'])} |")
    lines += ['',
        '## Метод диагностики F0000: вычисления сохранены', '',
        'Отдельный модуль `diagnostics/remizov_n4_followup.py` временно '
        'оборачивает оригинальные функции `_converge_spatial`, '
        '`_converge_quadrature`, `_integrate_fixed_nt`, '
        '`_compute_chernoff_at_t`, `_apply_operator_on_grid` и '
        '`numpy.polynomial.legendre.leggauss`. Каждый исходный вызов '
        'выполняется один раз с теми же аргументами и возвращает '
        'свой исходный результат; исключения не перехватываются для '
        'подбора параметров. CubicSpline и его результаты не заменяются. '
        'cProfile наблюдает вызовы SciPy напрямую.', '',
        'Протокол сохраняется на границах квадратурных стадий и примерно '
        'раз в секунду внутри вычислений. SIGTERM инициирует отдельное '
        'сохранение частичной диагностики и профиля, затем процесс '
        'завершается. Последний сохранённый event — external_SIGTERM. '
        'В файл записываются существующие диагностические dataclass '
        'решателя и стек наблюдаемых стадий; промежуточный вектор '
        'не объявляется решением.', '',
        'Перед запуском инструментализация проверена на малом вычислении '
        'штатного интеграла R1: n=4, Nt=8, T=1, dx=0.05, три целевые '
        'точки. Векторы без/с наблюдателем побитово равны, все штатные '
        'счётчики равны. Это проверка прозрачности обёртки, не новая '
        'серия сходимости. Сравнение не включает математическую '
        'перепроверку всей реализации, уже выполненную в предыдущем этапе.', '',
        'Профилирование и обёртки изменяют время исполнения. Полная '
        'величина этого влияния не измерена. Суммарное явно измеренное '
        f"время завершённых операций сохранения наблюдателя — {progress['observer_io_seconds_completed']:.6f} с; "
        'оно не включает все расходы cProfile, вызовов обёрток и '
        'последнего сохранения. Отнимать только его и считать остаток '
        'временем обычного решателя нельзя.', '',
        'Для сохранения снимков cProfile временно отключается и '
        'включается вновь. Поэтому cumulative time долгоживущих '
        'внешних функций (`_converge_spatial`, `_converge_quadrature`, '
        '`_integrate_fixed_nt`) неполон и не используется для определения '
        'длительности стадий. Для стадий используются отдельные '
        'wall-clock метки событий; для внутренних коротких функций — '
        'профиль. У активных при остановке вызовов время также может '
        'быть неполным. Inclusive/cumulative строки вложены друг в '
        'друга, их нельзя складывать.', '',
        '## Подтверждённая стадия остановки F0000, n=4', '',
        f"Статус `{f0['status']}`. SIGTERM запрошен на {f0['termination_requested_wall_seconds']:.6f} с "
        f"от старта процесса; процесс завершился на {f0['worker_wall_seconds']:.6f} с. "
        f"Наблюдаемый RSS {f0['worker_peak_rss_mib']:.2f} МиБ. "
        'Математическая ошибка не определена: RemizovResult не возвращён. '
        'Это не штатный отказ quadrature_not_converged/spatial_not_converged.', '',
        f"Последний снимок относится к {progress['elapsed_solver_observed_seconds']:.6f} с "
        'наблюдаемого решателя; время процесса дополнительно содержит '
        'импорт и загрузку. На момент сигнала выполнялась квадратура '
        'Nt=2048 при dx=0.003125, внутри вычисления S(t/n)^4. '
        'Узел t≈11.9396459615, tau≈2.9849114904; текущий оператор '
        'использовал исходную сетку 13166 точек и сетку назначения '
        '10035 точек. Это место остановки в данном запуске, '
        'а не доказательство того же места в предыдущем запуске.', '',
        'Четыре квадратурных стадии завершились успешно; на каждой '
        'понадобились все Nt до 2048:', '',
        '| dx | состояние | время стадии, с | последний quad error / threshold |',
        '|---:|---|---:|---|']
    for stage,q in zip(stages,diag['quadrature_history']):
        detail=f"{fmt(q['error_estimates'][-1])} / {fmt(q['convergence_thresholds'][-1])}"
        if not stage['completed']:
            detail += ' (последнее завершённое сравнение 512→1024; Nt=2048 ещё выполняется)'
        lines.append(f"| {fmt(stage['dx'])} | {'принята' if stage['completed'] else 'прервана извне'} | {stage['stage_elapsed_seconds']:.6f} | {detail} |")
    lines += ['',
        'Сравнения пространства:', '',
        '| новый dx | spatial error | threshold | принят? |', '|---:|---:|---:|---|']
    for dx,error,threshold in zip(diag['spatial_dx_levels'][1:],diag['spatial_error_history'],
                                  diag['spatial_convergence_thresholds']):
        lines.append(f"| {fmt(dx)} | {fmt(error)} | {fmt(threshold)} | {'да' if error<=threshold else 'нет'} |")
    lines += ['',
        'Для dx=0.003125 пространственное сравнение ещё не было выполнено. '
        'Не следует трактовать `converged=false` у активной квадратурной '
        'истории как завершённый отказ: сравнение Nt=2048 отсутствует. '
        'После текущей стадии остаётся ещё допустимый dx=0.0015625, '
        'но необходимость его использования и итоговый статус неизвестны.', '',
        'Частичные штатные счётчики на момент SIGTERM:', '', '```json',
        json.dumps(diag['counters'],indent=2),'```','',
        'Счётчики отражают начатую работу; завершённых Chernoff-узлов '
        '20039, завершённых S-вызовов 80156. Нельзя считать текущий '
        'незавершённый узел/оператор завершённым. T=13.142092786567558, '
        'gap=1.4273241529715541, tail_eps=1e-8; параметры усечения '
        'не менялись.', '',
        '## Наблюдения профиля: где тратится время', '',
        '| операция | число вызовов в снимке | inclusive/cumulative, с |',
        '|---|---:|---:|']
    for label,row in metrics:
        lines.append(f"| {label} | {row['calls']} | {row['cumulative_seconds']:.6f} |")
    completed=progress['completed_inclusive_timings']
    lines += ['',
        'Строки таблицы вложены: применение S включает построение '
        'и вызовы сплайна, коэффициенты и проверку границ. Их суммы '
        'не являются полной длительностью. Независимые обёртки '
        f"зафиксировали {completed['_compute_chernoff_at_t']['inclusive_seconds']:.6f} с "
        'в завершённых Chernoff-узлах, из них '
        f"{completed['_apply_operator_on_grid']['inclusive_seconds']:.6f} с "
        'в завершённых применениях S. Построение квадратурных '
        f"узлов по этим меткам занимает {completed['leggauss']['inclusive_seconds']:.6f} с.", '',
        'Подтверждённый основной источник затрат в профилируемой попытке — '
        'последовательная обработка многих Chernoff-узлов и пространственных '
        'сеток. Построение/использование сплайнов и вычисление/проверка '
        'коэффициентов — существенные вложенные затраты. Построение '
        'квадратурных узлов занимает заметно меньше времени; одно его '
        'кэширование не устранит основную стоимость. Прямое сравнение '
        'пространственных векторов мало; цена уточнения — повторное '
        'выполнение полной квадратуры на более мелкой сетке.', '',
        '## Математические, численные и вычислительные препятствия', '',
        '**Подтверждённые наблюдения.** F0005 проходит внутренние критерии '
        'при всех n=1,2,4, но сохраняет заметную ошибку относительно точного '
        'решения. F0000 при n=4 удовлетворяет критерию квадратуры на первых '
        'четырёх dx, однако все три завершённых пространственных сравнения '
        'не приняты. Текущий запуск остановлен по времени; память и размер '
        'рабочей сетки не исчерпали свои ограничения.', '',
        '**Математическая гипотеза.** Конечное n существенно влияет на '
        'ошибку F0005; монотонное уменьшение согласуется с этой гипотезой. '
        'Вклад конечного n не измерен независимо от квадратуры, сплайнов '
        'и хвоста, поэтому ошибка 0.3807 не объявляется чистой ошибкой '
        'Chernoff-аппроксимации. Из новой диагностики не следует '
        'математическая несовместимость F0000 с методом.', '',
        '**Численные ограничения.** При текущих критериях пространственная '
        'разность F0000 ещё превышает порог; квадратура на каждой '
        'завершённой стадии требует максимальный Nt. Это необходимость '
        'уточнения и расход численного бюджета, а не доказательство '
        'невозможности сходимости. Прежний отказ F0000 n=1,2 на '
        'начальном dx не повторяется при n=4: причина более долгого '
        'расчёта здесь включает дальнейшее пространственное уточнение.', '',
        '**Вычислительное препятствие.** Повторные сериализованные '
        'операторы на расширенных сетках с реконструкцией сплайнов '
        'на каждом шаге и полным пересчётом квадратуры при каждом dx '
        'расходуют доступные 45 с. Инструментализация также замедляет '
        'запуск; её полная стоимость отдельно неизвестна. '
        'Нельзя утверждать, что обычный запуск имеет те же проценты затрат '
        'или завершился бы в то же время.', '',
        '**Ограничение математической интерпретации хвоста.** '
        'Сохранена прежняя формула T. Она не использует глобальную '
        'норму g_R, поэтому tail_eps не интерпретируется как '
        'универсальная абсолютная граница всей ошибки. Ни усечение, '
        'ни формулы сертификатов здесь не пересматривались.', '',
        '## Обоснованный следующий шаг', '',
        'Следующее исследование следует ограничить одним F0000 n=4 '
        'с теми же 45 с/512 МиБ и более лёгким внешним сэмплированием '
        '(без cProfile и частых Python-обёрток). Это позволит проверить '
        'штатное завершение, не увеличивая бюджет и не принимая '
        'профилируемое время за обычное. При повторной внешней остановке '
        'результат остаётся неопределённым; скрытых повторов быть не должно.', '',
        'Если затем потребуется работа над производительностью, она '
        'должна идти отдельным согласованным этапом с проверкой '
        'эквивалентности численных результатов и истории критериев. '
        'Профиль указывает на сплайны и повторную обработку рабочих '
        'сеток/коэффициентов; кэширование неизменной геометрии/значений '
        'можно сначала оценить как возможность, но в этом этапе '
        'ничего не кэшировалось и решатель не изменялся. Простое '
        'увеличение n или бюджетов сейчас увеличит стоимость и не '
        'решит установленное вычислительное препятствие.', '',
        'Для количественного разделения ошибок F0005 после этого нужен '
        'контролируемый эксперимент при фиксированном n и T с одним '
        'численным параметром либо независимый эталон конечного n. '
        'Новые n=8,16,32 не нужны для нынешнего вывода и не выполнялись. '
        'Рекомендованные следующие запуски сейчас не выполнялись.', '',
        '## Воспроизводимость и сохранённые файлы', '',
        'Диагностические модули:', '',
        '- `diagnostics/remizov_n4_followup.py` — два независимых '
        'ограниченных вызова, наблюдатель и ресурсный контроль;',
        '- `diagnostics/report_remizov_n4_followup.py` — читает '
        'сохранённые файлы, строит сравнение, график и этот отчёт; решатель не вызывает.', '',
        'Новый каталог содержит summary, JSON результатов, resource samples, '
        'лог событий стадий, частичный профиль pstats, табличный профиль, '
        'машинный анализ профиля, сравнение F0005 CSV/JSON и PNG/SVG графика. '
        'Полные истории F0005 находятся в `F0005_n4/solver_return.json`; '
        'неполные истории F0000 — в `F0000_n4_profile/progress.json`.', '',
        '```bash',
        '.venv/bin/python -B diagnostics/remizov_n4_followup.py --output-dir results/remizov_n4_followup_repeat',
        '.venv/bin/python -B diagnostics/report_remizov_n4_followup.py results/remizov_n4_followup_repeat',
        '```', '',
        'Команды только для воспроизводимости: повторный запуск требует '
        'нового каталога, существующие результаты не перезаписываются.', '',
        'Контроль неизменности: SHA-256 математических файлов совпадают '
        'с миграционным манифестом; старые результаты проверены отдельным '
        'полным набором SHA-256. Исходный датасет и предсуществующий код '
        'не изменены. Сверка исходного проекта исключает .venv, .git, '
        'caches, results и .DS_Store; границы сверки явно сохранены. '
        'Новых зависимостей не установлено. Commit/push не выполнялись.', '']
    report.write_text('\n'.join(lines),encoding='utf-8')
    manifest=json.loads((ROOT/'MIGRATION_MANIFEST.json').read_text())
    core={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==manifest['source_sha256'][name]
          for name in manifest['unchanged_mathematical_core']}
    after=snapshot(ROOT)
    original_after=snapshot(ROOT.parent/'generator_DE')
    persisted=f5['solver_result']
    import numpy as np
    reconstructed_error=float(np.max(np.abs(np.array(persisted['f'])-np.array(f5['y_exact']))))
    assert reconstructed_error==f5['max_abs_error']
    verification={'core_matches_migration_source':core,
        'preexisting_project_changed_or_removed':[p for p,h in current['previous_dataset_and_code_sha256'].items() if after.get(p)!=h],
        'original_changed_or_removed':[p for p,h in current['original_sha256'].items() if original_after.get(p)!=h],
        'previous_results_identical':tree_hashes(previous_path)==current['previous_results_sha256'],
        'F0005_persisted_error_recalculation_matches':True,
        'F0000_error_is_null':f0['max_abs_error'] is None,
        'observer_sanity_check':{'problem':'R1','n':4,'nt':8,'T':1.0,'dx':.05,
            'target_points':3,'bit_identical_values':True,'identical_work_counters':True},
        'automatic_retries':0,'executed_study_n_values':[4],
        'new_dependencies_installed':False}
    assert all(core.values()) and verification['previous_results_identical']
    assert not verification['preexisting_project_changed_or_removed']
    assert not verification['original_changed_or_removed']
    write_json(out/'verification.json',verification)
    print(report)


if __name__=='__main__':
    main()
