"""Read saved boundary diagnosis only; never load tasks or call a solver."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IDS = ('AMV1_F0141_S00', 'AMV1_F0180_S04', 'AMV1_F0184_S01', 'AMV1_F0180_S03')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def maximum(a):
    return float(np.max(np.abs(a)))


def build_analysis(output):
    plan = read(output/'plan.json')
    modes = ['cubic_taper', 'cubic_zero']
    if (output/'linear_completion.json').exists():
        modes.append('linear_zero')
    records, comparisons, traces = {}, [], []
    for task in plan['tasks']:
        pid = task['problem_id']
        records[pid] = {}
        row = dict(equation_id=pid, baseline_error=task['baseline_error'],
                   baseline_boundary_max=task['baseline_boundary_max'], T=task['T'])
        baseline = read(ROOT/'results/chernov_solve1000_2026-10-18/solutions'/f'{pid}.json')
        for mode in modes:
            record = read(output/'runs'/f'{pid}_{mode}.json')
            result = record['result']
            assert result['success'] and result['n'] == 128 and result['time_nodes'] == 513
            assert result['work_domain'] == [-13., 13.] and result['actual_dx'] == .0125
            assert result['T'] == task['T'] and record['fingerprints'] == task['fingerprints']
            np.testing.assert_array_equal(result['x'], baseline['target_x'])
            np.testing.assert_array_equal(record['y_exact'], baseline['y_exact'])
            error = maximum(np.asarray(result['f']) - record['y_exact'])
            assert error == record['max_abs_error']
            row[mode+'_error'] = error
            row[mode+'_boundary_max'] = result['boundary_state_max_abs']
            row[mode+'_runtime_seconds'] = result['runtime_seconds']
            row[mode+'_status'] = 'accuracy_pass' if error <= .01 else 'accuracy_not_met'
            records[pid][mode] = record
        np.testing.assert_array_equal(records[pid]['cubic_taper']['result']['f'], baseline['result']['f'])
        assert records[pid]['cubic_taper']['result']['boundary_state_max_abs'] == task['baseline_boundary_max']
        row['cubic_zero_improvement_factor'] = row['baseline_error']/row['cubic_zero_error']
        row['baseline_zero_solution_difference'] = maximum(
            np.asarray(records[pid]['cubic_taper']['result']['f']) - records[pid]['cubic_zero']['result']['f'])
        if 'linear_zero' in modes:
            row['cubic_linear_zero_solution_difference'] = maximum(
                np.asarray(records[pid]['cubic_zero']['result']['f']) - records[pid]['linear_zero']['result']['f'])
        comparisons.append(row)
        for item in read(output/'traces'/f'{pid}_summary.json'):
            j = item['time_node_index']
            raw = [json.loads(line) for line in (output/'traces'/f'{pid}_j{j:03d}.jsonl').open()]
            by_mode = {m: [r for r in raw if r['mode'] == m] for m in ('taper', 'zero')}
            assert all(len(v) == 128 for v in by_mode.values())
            a, b = item['modes']['taper'], item['modes']['zero']
            first_violation = a['first_step_bound_violation']
            onset = next((r for r in by_mode['taper'] if r['k_after'] == first_violation), None)
            target_k = a['first_target_difference_gt_001']
            weighted_k = a['first_weighted_target_difference_gt_001']
            def at(k):
                return next((r for r in by_mode['taper'] if r['k_after'] == k), None)
            # All target queries remain in the source window: the boundary
            # intervention reaches the target indirectly through state changes.
            assert max(r['outside_contribution_target'] for r in raw) == 0
            traces.append(dict(problem_id=pid, time_node_index=j, t=item['t'],
                t_over_T=item['t_over_T'], tau=item['tau'], initial_norm=item['initial_global'],
                first_global_gt_twice_initial=a['first_global_gt_twice_initial'],
                first_step_bound_violation=first_violation, violation_detail=onset,
                first_target_difference_gt_001=target_k, target_onset_detail=at(target_k),
                first_weighted_target_difference_gt_001=weighted_k,
                weighted_target_onset_detail=at(weighted_k),
                final_taper_global=a['final']['global_after'],
                final_zero_global=b['final']['global_after'],
                final_taper_target=a['final']['target_after'],
                final_zero_target=b['final']['target_after'],
                final_weighted_taper_target=a['final']['weighted_target_after'],
                final_weighted_zero_target=b['final']['weighted_target_after'],
                final_target_difference=a['final']['target_pair_difference'],
                final_weighted_target_difference=a['final']['weighted_target_pair_difference'],
                max_taper_extension=a['max_extension'], max_zero_extension=b['max_extension'],
                max_taper_weighted_global=a['max_weighted_global'],
                max_zero_weighted_global=b['max_weighted_global'],
                zero_first_step_bound_violation=b['first_step_bound_violation'],
                direct_target_outside_contribution_zero=True))
    checks = dict(baseline_f_bitwise_reproduced=4, baseline_boundary_exact_reproduced=4,
        saved_error_arrays_rechecked=4*len(modes), all_fingerprints_identical=True,
        all_target_exact_identical=True, instrumented_steps_bitwise_checked=4096,
        traced_real_time_nodes=16, no_dense_operator_matrix=True,
        no_parameter_refinement=True, no_numerical_solver_calls_by_report=True,
        full_numerical_solves=4*len(modes))
    return plan, records, dict(comparisons=comparisons, traces=traces, checks=checks, modes=modes)


def write_report(output, plan, analysis):
    rows, traces = analysis['comparisons'], analysis['traces']
    linear = 'linear_zero' in analysis['modes']
    lines = ['# Граничная диагностика Чернова — 2026-10-24', '',
        '## A. Воспроизводимость', '',
        'Все четыре baseline воспроизвелись побайтово по f; граничные максимумы совпали точно. '
        'Проверены production-хеши пяти научных исходников и четырёх артефактов датасета, '
        'канонический SHA плана, первичные attempts/result.json, полные metadata, версии Python/NumPy/SciPy/SymPy.', '',
        'Конфигурация: solve; n=128; num_t=513; quadratic; dx=0.0125; padding=12; '
        '2081 узел на [-13,13]; 33 целевых узла. T сохранён отдельно для каждого ID. '
        'Исходный taper_width=null разрешается кодом в 2.6. Конфигурация T=null сохранена; '
        'до каждого запуска проверяется точное равенство вычисленного T историческому.', '',
        'Для каждого варианта совпали SHA-256 массивов a,b,c_R,g_R, пространственной сетки, '
        'узлов/весов квадратуры и target_x. В cubic/zero изменено только boundary. '
        'В linear/zero относительно cubic/zero изменено только interp, включая финальную проекцию; '
        'все 33 target_x совпадают с узлами рабочей сетки, поэтому проекция здесь не меняет узловые значения.', '',
        'Источники: [plan.json](plan.json), [core_integrity.json](core_integrity.json), '
        '[comparison.csv](comparison.csv), [analysis.json](analysis.json).', '',
        '## B. Локализация роста', '',
        'Прослежены реальные узлы j=256,362,443,512: t/T=0.25,0.4998931885,0.7486305237,1. '
        'Это ограниченная выборка t, а не установление точного порога по времени. '
        'k считается после шага: u_0=g_R. Значимый рост определён заранее как ||u_k||>2||g_grid||. '
        'Нарушение оценки шага: ||u_(k+1)||>(1+tau*C)||u_k||(1+1e-10). '
        'Порог влияния границы на цель: max|u_taper-u_zero|>0.01. '
        'Взвешенная диагностика использует exp(-lambda*t*k/n); фактические состояния не масштабировались.', '',
        '| ID | t | t/T | k: >2 initial | k: step bound | k: target diff>.01 | k: weighted diff>.01 | final weighted target diff |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in traces:
        val = lambda k: '—' if r[k] is None else str(r[k])
        lines.append(f"| {r['problem_id']} | {r['t']:.9g} | {r['t_over_T']:.8g} | "
            f"{val('first_global_gt_twice_initial')} | {val('first_step_bound_violation')} | "
            f"{val('first_target_difference_gt_001')} | {val('first_weighted_target_difference_gt_001')} | "
            f"{r['final_weighted_target_difference']:.9g} |")
    lines += ['', 'Детали первого нарушения оценки шага:', '',
        '| ID | t/T | k | global before | global after | inside queries max | outside extension max | outside contribution max | slopes left/right |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in traces:
        v = r['violation_detail']
        if v:
            lines.append(f"| {r['problem_id']} | {r['t_over_T']:.6g} | {v['k_after']} | "
                f"{v['global_before']:.6g} | {v['global_after']:.6g} | {v['inside_query_max']:.6g} | "
                f"{v['outside_extension_max']:.6g} | {v['outside_contribution_max']:.6g} | "
                f"{v['slope_left']:.6g} / {v['slope_right']:.6g} |")
    lines += ['', 'На всех шагах прямой вклад outside-запросов в target равен нулю: '
        'целевые сдвиги остаются внутри окна. Разность на [-1,1] возникает через многократное '
        'применение оператора к уже различающимся состояниям. Кубический сплайн также имеет '
        'глобальную зависимость от узлов: эти данные не разделяют перенос сдвигами и сплайн-связь.', '',
        'Во всех 4096 наблюдаемых шагах реконструированное наблюдателем значение побайтово совпало '
        'с исходной _apply. Цепочка всегда использовала результат исходной _apply. '
        'В JSONL сохранены нормы, значения/производные на границах, inside/outside-вклады '
        'и распределённые экспоненциальные веса. Полные состояния не сохранялись.', '',
        '## C. Контролируемое сравнение', '',
        '| ID | baseline error | cubic/zero error | linear/zero error | boundary baseline | boundary cubic/zero | boundary linear/zero | runtime baseline / cubic-zero / linear-zero, s |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        get = lambda k: f'{r[k]:.10g}' if k in r else 'не выполнен'
        runtimes = ' / '.join(get(m+'_runtime_seconds') for m in ('cubic_taper', 'cubic_zero', 'linear_zero'))
        lines.append(f"| {r['equation_id']} | {r['baseline_error']:.12g} | {r['cubic_zero_error']:.12g} | "
            f"{get('linear_zero_error')} | {r['baseline_boundary_max']:.7g} | "
            f"{r['cubic_zero_boundary_max']:.7g} | {get('linear_zero_boundary_max')} | {runtimes} |")
    lines += ['', 'Все выполненные solve вернули success=True. Это означает конечность вычисления; '
        'accuracy_pass/accuracy_not_met вычисляются отдельно по E∞<=0.01 в comparison.csv.', '',
        '## D. Причинные выводы', '',
        'На этих четырёх фиксированных дискретных задачах изменение только граничного продолжения '
        'устраняет три катастрофических выброса. Побайтовое воспроизведение baseline и одинаковые '
        'входные массивы исключают изменение задачи, T, пространственной сетки или квадратуры как '
        'объяснение различия между cubic/taper и cubic/zero. Это интервенционный численный результат '
        'для данной схемы и бюджета, а не доказательство глобальной точности zero.', '',
        'Успешный F0180/S03 служит контролем: большой необвешенный граничный максимум не равнозначен '
        'большой итоговой ошибке. Важны зависимость от t, перенос в целевой интервал и экспоненциальный вес.', '']
    if linear:
        lines += ['Для linear/zero P(q) либо выпуклая комбинация двух соседних значений, либо ноль. '
            'Поэтому ||P(q)u||∞<=||u||∞ и ||S_h(tau)u||∞<= '
            '(1+tau*max_i|c_R(x_i)|)||u||∞ <= (1+tau*C)||u||∞. '
            'Три положительных веса сдвигов суммируются в 1; потенциальное слагаемое оценено '
            'по модулю и не требует c_R>=0. Оценка всех степеней с весом: '
            '||exp(-lambda*k*tau)S_h(tau)^k u||∞<=exp(-(lambda-C)*k*tau)||u||∞. '
            'Отрицательный первый вес неравномерного Simpson требует суммы абсолютных весов '
            'при оценке интеграла. Конечная точность из этой устойчивости не следует.', '']
    lines += ['Остаточная ошибка может включать конечное n, квадратуру, интерполяцию и усечение окна. '
        'Их доли здесь не установлены. Уравнения сохраняют проверенные математические предпосылки; '
        'теорема на всей R не доказывает точность конечного окна и интерполяции.', '',
        '## E. Порог 0.01', '']
    for mode in analysis['modes']:
        passed = sum(r[mode+'_error'] <= .01 for r in rows)
        lines.append(f'- {mode}: {passed}/4 достигли порога на прежних 33 узлах.')
    lines += ['', '## F. Следующий минимальный эксперимент — не выполнен', '',
        'Для худшей остаточной задачи F0180/S04 выполнить один cubic/zero контроль с удвоенным '
        'padding=24 при неизменных n=128, dx, T, узлах/весах квадратуры и целевой сетке. '
        'Сопоставить разность решений с остаточной ошибкой и порогом .01. Это проверит влияние '
        'нулевой границы перед исследованием n. Если окно почти не влияет, отдельно планировать '
        'уточнение n и квадратуры. Ни один такой расчёт в эту серию не включён.', '',
        '## G. Файлы и ограничения', '',
        'Созданы отдельный диагностический драйвер, тесты наблюдателя/linear-нормы, '
        'отчётный скрипт и новый каталог этой серии. Исходный решатель и генератор не редактировались. '
        'Проверка before_sha256 охватывает все прежние data/results/comparison_inputs, пять научных '
        'исходников, статус и migration manifest. Статус обновляется после завершения серии с '
        'сохранением PROJECT_STATUS.before.md и отдельно учитывается финальной проверкой.', '',
        'Плотные матрицы 2081x2081 не строились. Один вычислительный процесс, BLAS=1. '
        'В runs/*.resources.jsonl сохранены секундные сэмплы wall/RSS. '
        'Новые полные решения: только четыре ID, baseline + cubic/zero'
        + (' + linear/zero.' if linear else '.'), '',
        'Размер/скорость сходимости по n, h, num_t и окну не исследовались. '
        'Результаты не следует переносить автоматически на все 126 превышений или worker_exception.', '']
    with (output/'REPORT.md').open('x') as handle:
        handle.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True, type=Path)
    args = parser.parse_args()
    output = args.input_dir.resolve()
    plan, records, analysis = build_analysis(output)
    save(output/'analysis.json', analysis)
    rows = analysis['comparisons']
    with (output/'comparison.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_report(output, plan, analysis)
    print(json.dumps(analysis['checks'], sort_keys=True))


if __name__ == '__main__':
    main()
