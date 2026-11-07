#!/usr/bin/env python3
"""Analyze saved timing pilot; never import solvers or load models."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from diagnostics.cost60_common import (METHODS,NS,EPS,POLICY,read,require,sha,atomic_json,choose,
                                      robust_faster,saving,summarize_profile)


def csv_write(path,rows,fields=None):
    with path.open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields or list(rows[0]));writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def analyze(records,ids):
    measured=defaultdict(list);raw_rows=[]
    for r in records:
        raw_rows.append({k:r[k] for k in ('sequence','equation_id','family_id','method','N','phase','episode','status','usable','E33','accuracy_met','solver_seconds','solver_nanoseconds','scoring_seconds')})
        if r['phase']=='measured':measured[(r['equation_id'],r['method'],r['N'])].append(r)
    require(len(records)==2880 and len(measured)==480,'unexpected run cardinality')
    profiles=[summarize_profile(sorted(group,key=lambda r:r['episode'])) for _,group in sorted(measured.items())]
    require(all(p['repeat_identical'] for p in profiles),'nondeterministic repeated trajectories')
    by_id={pid:[p for p in profiles if p['equation_id']==pid] for pid in ids}
    constants=[]
    for method in METHODS:
        for n in NS:
            group=[p for p in profiles if p['method']==method and p['N']==n]
            constants.append(dict(method=method,N=n,configuration=f'{method}_N{n}',
                successful=sum(p['usable'] for p in group),accurate=sum(p['accurate'] for p in group),
                total_solver_seconds=sum(p['solver_median_seconds'] for p in group),
                mean_solver_seconds=statistics.mean(p['solver_median_seconds'] for p in group),
                mean_E33=statistics.mean(p['E33'] for p in group if p['E33'] is not None) if any(p['E33'] is not None for p in group) else None,
                max_E33=max((p['E33'] for p in group if p['E33'] is not None),default=None),
                stable_profiles=sum(p['timing_stable'] for p in group)))
    admissible=[c for c in constants if c['accurate']==len(ids)]
    best_constant=min(admissible,key=lambda c:(c['total_solver_seconds'],METHODS.index(c['method']),c['N'])) if admissible else None
    fallback=min(constants,key=lambda c:(-c['accurate'],c['total_solver_seconds'],METHODS.index(c['method']),c['N']))
    winners=[];wins=Counter();robust_counts=Counter();no_solution=[];unknown=[]
    for pid in ids:
        group=by_id[pid];oracle=choose(group)
        rk=choose([p for p in group if p['method']=='rk4']);pi=choose([p for p in group if p['method']=='product_integral'])
        rk_minN=min((p for p in group if p['method']=='rk4' and p['accurate']),key=lambda p:p['N'],default=None)
        pi_robust=robust_faster(pi,rk) if rk else pi is not None
        rk_robust=robust_faster(rk,pi) if pi else rk is not None
        practical=pi if pi_robust else rk if rk else pi
        if pi_robust:robust_counts['product_integral']+=1
        elif rk_robust:robust_counts['rk4']+=1
        elif oracle:robust_counts['practical_tie_or_unstable']+=1
        if oracle:wins[oracle['method']]+=1
        elif all(p['usable'] for p in group):no_solution.append(pid)
        else:unknown.append(pid)
        const=next((p for p in group if best_constant and p['configuration']==best_constant['configuration']),None)
        item=dict(equation_id=pid,family_id=group[0]['family_id'],status='solved' if oracle else 'none_meets_tolerance' if pid in no_solution else 'unknown_due_to_failure',
            admissible_configurations=[p['configuration'] for p in group if p['accurate']],
            oracle_configuration=oracle['configuration'] if oracle else None,oracle_method=oracle['method'] if oracle else None,
            oracle_N=oracle['N'] if oracle else None,oracle_E33=oracle['E33'] if oracle else None,
            oracle_seconds=oracle['solver_median_seconds'] if oracle else None,
            rk4_only_configuration=rk['configuration'] if rk else None,rk4_only_seconds=rk['solver_median_seconds'] if rk else None,
            rk4_minN_configuration=rk_minN['configuration'] if rk_minN else None,rk4_minN_seconds=rk_minN['solver_median_seconds'] if rk_minN else None,
            PI_best_configuration=pi['configuration'] if pi else None,PI_best_seconds=pi['solver_median_seconds'] if pi else None,
            PI_robust_win=pi_robust,RK4_robust_win=rk_robust,
            conservative_configuration=practical['configuration'] if practical else None,
            conservative_seconds=practical['solver_median_seconds'] if practical else None,
            constant_configuration=const['configuration'] if const else None,constant_seconds=const['solver_median_seconds'] if const else None,
            gain_vs_constant=1-oracle['solver_median_seconds']/const['solver_median_seconds'] if oracle and const else None,
            gain_vs_rk4_only=1-oracle['solver_median_seconds']/rk['solver_median_seconds'] if oracle and rk else None,
            PI_gain_vs_rk4_only=1-pi['solver_median_seconds']/rk['solver_median_seconds'] if pi and rk else None)
        winners.append(item)
    gains={}
    for name,key in (('best_constant','constant_seconds'),('rk4_only_oracle','rk4_only_seconds'),('rk4_minN_rule','rk4_minN_seconds')):
        for variant,choice_key in (('raw_oracle','oracle_seconds'),('conservative_oracle','conservative_seconds')):
            paired=[r for r in winners if r[key] is not None and r[choice_key] is not None]
            gains[variant+'_vs_'+name]=saving([r[key] for r in paired],[r[choice_key] for r in paired]) if paired else None
    pi_raw=[r for r in winners if r['oracle_method']=='product_integral' and r['rk4_only_seconds'] is not None]
    pi_practical=[r for r in winners if r['PI_robust_win'] and r['rk4_only_seconds'] is not None]
    pi_savings={label:saving([r['rk4_only_seconds'] for r in group],[r['PI_best_seconds'] for r in group]) if group else None
                for label,group in (('raw_PI_winners',pi_raw),('robust_PI_winners',pi_practical))}
    stable=sum(p['timing_stable'] for p in profiles)/len(profiles)
    gates=dict(all60_solved=sum(wins.values())==60,no_unknown_minimum=not unknown,
        constant_meets_tolerance_on60=best_constant is not None,rk4_only_covers60=all(r['rk4_only_seconds'] is not None for r in winners),
        heterogeneous_raw_winners=all(wins[m]>=60*POLICY['minority_fraction'] for m in METHODS),
        robust_each_method=all(robust_counts[m]>=POLICY['minimum_robust_each_method'] for m in METHODS),
        stable_profile_fraction=stable>=POLICY['minimum_stable_profile_fraction'])
    for label in ('best_constant','rk4_only_oracle'):
        g=gains['conservative_oracle_vs_'+label]
        gates['practical_gain_vs_'+label]=bool(g and g['n']==60 and g['pooled_saving_fraction']>=POLICY['minimum_pooled_saving']
            and g['mean_relative_saving_fraction']>=POLICY['minimum_mean_relative_saving'])
    summary=dict(schema='cost60_analysis_v1',epsilon=EPS,equations=len(ids),profiles=480,solver_calls=2880,
        measured_calls=2400,warmup_calls=480,raw_method_winners={m:wins[m] for m in METHODS},
        raw_method_win_fraction={m:wins[m]/len(ids) for m in METHODS},robust_method_winners=dict(robust_counts),
        oracle_configuration_winners=dict(Counter(r['oracle_configuration'] for r in winners if r['oracle_configuration'])),
        no_configuration_meets_tolerance=no_solution,unknown_due_to_failure=unknown,best_constant=best_constant,
        max_coverage_constant=fallback,stable_profile_fraction=stable,gains=gains,PI_savings=pi_savings,
        GO_thresholds=POLICY,GO_gates=gates,recommendation='GO' if all(gates.values()) else 'NO-GO',
        failed_GO_gates=[k for k,v in gates.items() if not v],new_ML_fits=0,MC_calls=0,remizov_calls=0)
    return profiles,sorted(raw_rows,key=lambda r:r['sequence']),winners,constants,summary


def render_report(summary,out):
    s=summary;lines=[f"# Ограниченный пилот выбора стоимости · №2488 · {s['recommendation']}",'',
        f"**Рекомендация: {s['recommendation']} для дальнейшего ML в данной постановке.** "
        'Это решение о целесообразности отдельного будущего ML-эксперимента; обучаемость и переносимость модели здесь не проверялись.','',
        '## Методика и заранее зафиксированный план','',
        'Выбраны 60 исходных train-задач из 60 разных семейств и топологий: по три из каждой '
        'из 20 ячеек сложность×режим. Представитель семейства задан SHA(seed2488,ID); '
        'далее maximin по нормированным входным сертификатам и сохранённым диапазонам a,b,c_R,g. '
        'Отбор не использует ошибки, времена, ML-ответы или значения exact. Исторические Test '
        'использованы только как список исключений (включая топологии другого датасета). '
        'Из-за намеренного разнообразия эти 60 не являются случайной представительной выборкой production.','',
        'Неизменённые публичные RK4 и left-exponential product integral вызваны при N=32,64,128,256, '
        'a_min=.001, на [-1,1], с одними сохранёнными точными y(-1),y′(-1), обычными a,b,c,g. '
        'Оба решают одну задачу Коши. Новых MC, Ремизова и ML-вызовов нет.','',
        'Метрика E33=max|y_numeric−y_exact| на G33={−1+k/16:k=0,…,32}; '
        'срез траектории имеет шаг N/32 и точно совпадает с G33. Интерполяции нет. '
        'Допуск E33≤.01 включает равенство; непрерывная норма не сертифицируется.','',
        'На одной машине, один compute worker и один поток BLAS/OpenMP/Accelerate. '
        'Для всех 480 пар задача×конфигурация выполнены один прогрев и пять измеряемых повторов: '
        'всего 2880 новых вызовов, из них 2400 измеряемых. Порядки задач/конфигураций '
        'детерминированно перемешаны отдельно в каждом раунде. GC отключён на solver-фазе, '
        'сборка выполняется между группами вне таймера. Сырые ответы и времена сохранены для каждого вызова.','',
        'Основная стоимость — медиана пяти perf_counter_ns замеров вокруг публичной solver-обёртки, '
        'включая вычисление коэффициентов, шаги и формирование результата. Загрузка, подготовка '
        'входов/exact, asdict, scoring и JSON I/O исключены и сохранены отдельно в preparation.json. '
        'Системные/тепловые снимки по раундам и среда — environment.*.json/round_*.json. '
        'Температура кристалла напрямую не измерялась; сохранены только доступные системные сведения.','',
        'Oracle выбирает минимальную медиану среди точных конфигураций. Лучшая константа — '
        'одна конфигурация, точная на всех 60, с минимальной суммой медиан. Если такой нет, '
        'max-coverage fallback публикуется отдельно и не называется точным на всём наборе. '
        'RK4-only oracle выбирает самую быструю точную RK4-конфигурацию по каждому ID; '
        'дополнительно приведено правило минимального точного N. Оба используют hindsight/exact, '
        'не являются готовым адаптивным solver. Стоимость поиска/пробных решений в oracle не учтена; '
        'это оптимистическая верхняя граница потенциальной экономии селектора без его overhead.','',
        'До запуска зафиксирован практический запас10% и max(t_fast)<min(t_slow) при '
        'MAD/median≤.20 у обеих конфигураций. Консервативный oracle переключается на PI '
        'только при таком преимуществе над лучшим точным RK4; малые различия/перекрытия '
        'времён оставляют RK4. GO требует ≥12/60 побед каждого метода, ≥12 устойчивых '
        'побед каждого, ≥10% pooled и среднего относительного выигрыша консервативного '
        'oracle относительно константы и RK4-only oracle, полного покрытия и ≥90% стабильных '
        'профилей. Это заранее выбранный операционный критерий, а не тест значимости.','',
        '## Результаты','', '| Метод | Oracle-победы | Доля от60 |', '|---|---:|---:|']
    for m in METHODS:lines.append(f"| {m} | {s['raw_method_winners'][m]} | {100*s['raw_method_win_fraction'][m]:.2f}% |")
    lines.extend(['',f"Устойчивые методные победы: {json.dumps(s['robust_method_winners'],ensure_ascii=False)}. "
                  f"Стабильные профили: {s['stable_profile_fraction']*100:.2f}%.",
        f"Распределение конфигураций oracle: {json.dumps(s['oracle_configuration_winners'],ensure_ascii=False)}.",'',
        f"Лучшая константа: **{s['best_constant']['configuration'] if s['best_constant'] else 'нет точной на всех60'}**.",
        f"Ни один вариант не достиг допуска: **{len(s['no_configuration_meets_tolerance'])}/60**; ID: {s['no_configuration_meets_tolerance']}.",
        f"Неизвестный минимум из-за отказов: {s['unknown_due_to_failure']}.",'',
        '| Сравнение | Пар | Средняя экономия, % | Экономия суммарной стоимости, % | Средняя экономия, мс/задачу |',
        '|---|---:|---:|---:|---:|'])
    for label,g in s['gains'].items():
        if g:lines.append(f"| {label} | {g['n']} | {100*g['mean_relative_saving_fraction']:.4f} | {100*g['pooled_saving_fraction']:.4f} | {1000*g['mean_absolute_saving_seconds']:.4f} |")
    lines.extend(['','Средняя относительная экономия = mean(1−t_selected/t_reference); '
        'pooled = 1−sum(t_selected)/sum(t_reference). Все пары удовлетворяют одному допуску. '
        'Эти два показателя различаются из-за неодинаковой сложности задач.','', '### Где PI полезен',''])
    for label,g in s['PI_savings'].items():
        lines.append(f"- {label}: {g['n']} задач; mean saving {100*g['mean_relative_saving_fraction']:.4f}%, pooled {100*g['pooled_saving_fraction']:.4f}%, {1000*g['mean_absolute_saving_seconds']:.4f} мс/задачу." if g else f'- {label}: нет задач.')
    lines.extend(['','Конкретные ID, N, ошибки и все парные выигрыши — winners.csv; '
        'в profiles.csv есть времена пяти повторов, min/max/MAD и отметка устойчивости.','',
        '## Решение GO/NO-GO и ограничения','',
        f"**{s['recommendation']}**. Не пройдены заранее заданные условия: {', '.join(s['failed_GO_gates']) or 'нет'}.",
        'Все условия и их фактические значения сохранены в summary.json; параметры/пороги после результатов не менялись. '
        'Сравнение на development является исследованием возможности, не независимой оценкой ML. '
        'Константа и oracle выбраны по тому же пилоту; oracle оптимистичен и не доказывает '
        'предсказуемости лучшего метода по входам. Малые преимущества времени не считаются '
        'основанием для ML. Медианы относятся только к данной машине, реализации coefficient callbacks '
        'и протоколу; равный N не означает равную работу. Наблюдения не доказывают универсальное '
        'превосходство метода или порядок сходимости.','',
        'Никаких дополнительных повторов, задач, solver-серий или fit автоматически не выполняется. '
        'На этом ограниченный эксперимент остановлен.','',
        '## Файлы и проверки','',
        'plan.json записан до started.json; selection/candidate_pool фиксируют отбор. '
        'records.csv —2880 вызовов с фазой и повтором; profiles.csv —480 конфигураций; '
        'winners.csv —60 oracle-строк; configuration_summary.csv —8 постоянных правил; '
        'no_solution.csv сохраняет случаи без точного ответа и неизвестные. '
        'verify_cost60.py независимо проверяет сохранённые траектории/ошибки/времена/таблицы '
        'без новых решений; verification.json, checks.json и integrity.json — доказательства проверки.',''])
    return '\n'.join(lines)


def build(out):
    plan=read(out/'plan.json');session=read(out/'session.json');require(session['complete'],'incomplete experiment')
    paths=sorted((out/'records').glob('*.json'));records=[read(p) for p in paths]
    require(all(r['plan_sha256']==sha(out/'plan.json') for r in records),'record plan mismatch')
    profiles,raw,winners,constants,summary=analyze(records,plan['equation_ids'])
    csv_write(out/'records.csv',raw);csv_write(out/'profiles.csv',profiles)
    csv_write(out/'winners.csv',winners);csv_write(out/'configuration_summary.csv',constants)
    csv_write(out/'no_solution.csv',[r for r in winners if r['status']!='solved'],fields=list(winners[0]))
    atomic_json(out/'summary.json',summary)
    (out/'REPORT.md').write_text(render_report(summary,out))
    print(json.dumps({k:summary[k] for k in ('recommendation','raw_method_winners','robust_method_winners','best_constant','failed_GO_gates')},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    build(parser.parse_args().out.resolve())
