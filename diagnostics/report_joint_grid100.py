"""Read saved grid100 results; no numerical solver or ML fitting."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import json
import os
import re
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/ode-grid100-mpl')
from diagnostics.grid100_statistics import aggregate_mc


def read(p):return json.loads(p.read_text())


def table(rows):
    names=list(rows[0]);return '| '+' | '.join(names)+' |\n| '+' | '.join('---' for _ in names)+' |\n'+''.join('| '+' | '.join(str(r[n]) for n in names)+' |\n' for r in rows)


def report(out):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from diagnostics.train_remizov_ml import write
    plan=read(out/'plan.json');records={p.stem:read(p) for p in (out/'records').glob('*.json')}
    (out/'analysis').mkdir(exist_ok=True);(out/'plots').mkdir(exist_ok=True)
    rows=[];attempts=[];statistics={};controls=[]
    for key,r in records.items():
        j=r['job'];result=r.get('result') or {}
        attempts.append(dict(key=key,problem_id=j['problem_id'],method=j['method'],stage=j['stage'],status=r['status'],
            execution_session=r['session'],source='records/'+key+'.json',
            E_inf=r.get('E_inf'),accuracy_met=r.get('accuracy_met'),solver_seconds=result.get('runtime_seconds'),
            task_wall_seconds=r['task_wall_seconds'],peak_rss_mib=r['peak_rss_mib'],T=result.get('T'),
            path_steps=result.get('n_path_steps'),operator_applications=result.get('n_operator_applications'),
            config=json.dumps(j['config'],sort_keys=True)))
        if j['stage']=='control':
            old=read(ROOT/j['reference']);diff=float(np.max(abs(np.array(result['f'])-old['result']['f']))) if r['status']=='completed' else None
            controls.append(dict(ID=j['problem_id'],boundary=j['config']['boundary'],E_inf=r.get('E_inf'),max_difference_saved=diff,seconds=result.get('runtime_seconds')))
    for ip in (out/'events').glob('*.interrupted.json'):
        event=read(ip);request=read(ip.with_name(ip.name.replace('.interrupted.json','.request.json')));j=request['job']
        attempts.append(dict(key=j['key'],problem_id=j['problem_id'],method=j['method'],stage=j['stage'],status='interrupted',
            execution_session=request['session'],source=str(ip.relative_to(out)),E_inf=None,accuracy_met=None,solver_seconds=None,
            task_wall_seconds=event['wall_seconds'],peak_rss_mib=event['peak_rss_mib'],T=None,path_steps=None,operator_applications=None,
            config=json.dumps(j['config'],sort_keys=True)))
    for task in plan['selection']:
        pid=task['problem_id'];r=records.get(pid+'_R');m=records.get(pid+'_MC0')
        row=dict(task,representative_weight=8.75 if task['cohort']=='representative' else None,
            R_status=r['status'] if r else 'pending',MC_status=m['status'] if m else 'pending',
            R_error=r.get('E_inf') if r else None,MC_primary_error=m.get('E_inf') if m else None,
            R_seconds=r['result']['runtime_seconds'] if r and r.get('result') else None,
            MC_primary_seconds=m['result']['runtime_seconds'] if m and m.get('result') else None,
            observed_primary_category='unavailable',final_category='unavailable',statistical_accuracy='unavailable',
            reliable_fastest_label='unconfirmed',training_eligible_winner=False,MC_seed_count=0)
        good_r=r and r['status']=='completed';good_m=m and m['status']=='completed'
        if good_m:
            ms=[records[k] for i in range(4) if (k:=pid+f'_MC{i}') in records and records[k]['status']=='completed']
            primary=aggregate_mc([m]);final=aggregate_mc(ms if len(ms)==4 else [m])
            statistics[pid]=dict(primary=primary,final=final)
            all_mc=[rec for rec in records.values() if rec['job']['problem_id']==pid and rec['job']['method']=='MC']
            row.update(MC_seed_count=len(ms),MC_final_aggregate_seeds=4 if len(ms)==4 else 1,MC_final_error=final['E_inf'],
                MC_all_solve_seconds=sum(rec['result']['runtime_seconds'] for rec in all_mc if rec.get('result')),
                MC_all_task_wall_seconds=sum(rec['task_wall_seconds'] for rec in all_mc),
                MC_max_simultaneous_half=max(final['normal_simultaneous_half']),
                MC_error_lower=final['error_mean_interval'][0],MC_error_upper=final['error_mean_interval'][1],
                MC_hoeffding_half=final['hoeffding_half'],statistical_accuracy=final['statistical_accuracy'],
                finite_sample_accuracy=final['finite_sample_accuracy'],
                time_discretization_error_bound_available=False,MC_primary_simultaneous_half=max(primary['normal_simultaneous_half']))
        if good_m and good_r:
            def category(mcpass):
                return 'both_pass' if r['accuracy_met'] and mcpass else ('R_only' if r['accuracy_met'] else ('MC_only' if mcpass else 'neither_pass'))
            row['observed_primary_category']=category(m['accuracy_met']);row['final_category']=category(final['E_inf']<=.01)
            if r['accuracy_met'] and m['accuracy_met']:
                row['observed_lower_cost_primary']='R' if row['R_seconds']<row['MC_primary_seconds'] else 'MC'
            else:row['observed_lower_cost_primary']='not_jointly_accurate'
        rows.append(row)
    eq=pd.DataFrame(rows);att=pd.DataFrame(attempts);eq.to_csv(out/'analysis/comparison.csv',index=False);att.to_csv(out/'analysis/attempts.csv',index=False)
    write(out/'analysis/simultaneous_statistics.json',statistics)
    cohorts=[]
    for cohort,g in eq.groupby('cohort'):
        known=g.R_status.eq('completed')&g.MC_status.eq('completed')
        cohorts.append(dict(cohort=cohort,equations=len(g),families=g.family_id.nunique(),paired_completed=int(known.sum()),
            R_pass=int(g.R_error.le(.01).sum()),MC_primary_pass=int(g.MC_primary_error.le(.01).sum()),
            MC_final_pass=int(g.get('MC_final_error',pd.Series(dtype=float)).le(.01).sum()),
            primary_categories=g.observed_primary_category.value_counts().to_dict(),final_categories=g.final_category.value_counts().to_dict(),
            statistical_accuracy=g.statistical_accuracy.value_counts().to_dict(),
            R_solver_seconds=float(g.R_seconds.sum()),MC_primary_solver_seconds=float(g.MC_primary_seconds.sum()),
            MC_all_solver_seconds=float(g.get('MC_all_solve_seconds',pd.Series(dtype=float)).sum())))
    pd.DataFrame(cohorts).to_csv(out/'analysis/cohorts.csv',index=False)
    sessions=[read(p) for p in (out/'sessions').glob('*.json') if read(p)['new_terminal'] or read(p)['reason']]
    stopped={s['session'] for s in sessions if s['reason']}
    interruptions=[read(p).get('reason') for p in (out/'control').rglob('*.guardian.json') if p.parent.name in stopped]
    calendar=(max(datetime.fromisoformat(s['finished_utc']).timestamp() for s in sessions)-read(out/'budget.json')['started_epoch']) if sessions else 0
    summary=dict(equations=100,cohorts=cohorts,terminal_records=len(records),attempt_statuses=att.status.value_counts().to_dict(),
        primary_pairs_terminal=int(sum(r['R_status']!='pending' and r['MC_status']!='pending' for r in rows)),
        primary_pairs_completed=int(sum(r['R_status']==r['MC_status']=='completed' for r in rows)),
        repeated_equations=int(eq.MC_seed_count.gt(1).sum()),MC_solve_count=int((att.method.eq('MC')).sum()),
        MC_completed_solve_count=int((att.method.eq('MC')&att.status.eq('completed')).sum()),physical_attempts=len(att),
        R_solve_count=int(att.method.eq('R').sum()),active_session_seconds=sum(s['wall_seconds'] for s in sessions),calendar_seconds=calendar,
        peak_rss_mib=float(att.peak_rss_mib.max()),total_solver_seconds=float(att.solver_seconds.sum()),
        controls=controls,guardian_interruptions=interruptions,uncertain_fastest_labels=100,reliable_winner_labels=0,ML_training_performed=False,
        pending=read(out/'pending.json'),representative_selection='80 equal-probability train rows,80 families,4 per cell; stress20 separate',
        profile='new common certified T,cubic/zero; historical taper timings/errors not pooled',
        recommendation='Do not launch900 automatically: inspect accuracy-class balance and simultaneous uncertainty first')
    write(out/'analysis/summary.json',summary)
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for cohort,marker in [('representative','o'),('stress','x')]:
        g=eq[eq.cohort.eq(cohort)];axs[0].scatter(g.R_error,g.MC_primary_error,s=25,marker=marker,label=cohort,alpha=.65)
        axs[1].scatter(g.R_seconds,g.MC_primary_seconds,s=25,marker=marker,label=cohort,alpha=.65)
    axs[0].axhline(.01,color='black',ls='--');axs[0].axvline(.01,color='black',ls='--')
    axs[0].set(xscale='log',yscale='log',xlabel='R cubic/zero E_inf',ylabel='MC first-seed E_inf');axs[0].legend()
    lo=eq.R_seconds.min()*.95;hi=eq.R_seconds.max()*1.05
    axs[1].plot([lo,hi],[lo,hi],color='grey',lw=.8);axs[1].set_xlim(lo,hi)
    axs[1].set(yscale='log',xlabel='R direct33 solve, seconds',ylabel='MC first-seed direct33 solve, seconds');axs[1].legend()
    fig.savefig(out/'plots/accuracy_cost.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for cohort in ('representative','stress'):
        g=eq[eq.cohort.eq(cohort)]
        for method,col in [('R','R_error'),('MC','MC_primary_error')]:
            x=np.sort(g[col].dropna());axs[0].step(x,np.arange(1,len(x)+1)/len(x),label=cohort+' '+method)
    axs[0].axvline(.01,color='black',ls='--');axs[0].set(xscale='log',xlabel='Observed E_inf',ylabel='ECDF');axs[0].legend(fontsize=8)
    if 'MC_error_upper' in eq:
        axs[1].scatter(eq.MC_final_error,eq.MC_error_upper,s=18,alpha=.6);axs[1].axhline(.01,color='black',ls='--');axs[1].axvline(.01,color='black',ls='--')
    axs[1].set(xscale='log',yscale='log',xlabel='MC final observed E_inf',ylabel='Simultaneous upper E_inf\n(time bias excluded)')
    fig.savefig(out/'plots/distribution_uncertainty.png',dpi=170);plt.close(fig)
    repeated=eq[eq.MC_seed_count.gt(1)]
    if len(repeated):
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained');ids=np.arange(len(repeated))
        ax.plot(ids,repeated.MC_primary_error,'o',label='first seed,1024/point');ax.plot(ids,repeated.MC_final_error,'s',label='mean4x1024/point');ax.axhline(.01,color='black',ls='--')
        ax.set(yscale='log',xlabel='Adaptively selected equation index',ylabel='Observed E_inf',ylim=(.004,.11))
        ax.set_yticks([.005,.01,.02,.05,.1],labels=['.005','.01','.02','.05','.1'])
        ax.legend();fig.savefig(out/'plots/repeats.png',dpi=170);plt.close(fig)
    flat=[dict(cohort=r['cohort'],n=r['equations'],R_pass=r['R_pass'],MC_first_pass=r['MC_primary_pass'],MC_final_pass=r['MC_final_pass'],
        R_seconds=f"{r['R_solver_seconds']:.2f}",MC_first_seconds=f"{r['MC_primary_solver_seconds']:.2f}",MC_all_seconds=f"{r['MC_all_solver_seconds']:.2f}") for r in cohorts]
    text=f'''# Общий33-точечный pilot:100 train-уравнений

Выполнено **{summary['primary_pairs_completed']}/100** полных пар,
терминальных записей **{len(records)}**. Все исходы:
`{summary['attempt_statuses']}`. Pending: `{summary['pending']}`.

## Выбор и зафиксированные профили

80 representative: по4 из20 complexity×regime ячеек, случайно4 из7 train-семейств
и1 из5 образцов семейства; seed202610250333. Вероятность включения строки4/35,
вес8.75,80 разных семейств. 20 stress:10 наибольших исторических train R33
ошибок,5 дорогих по прежним train MCx0,5 сложных AST без повторения ID.
Стресс-часть намеренно обогащена отказами/стоимостью; общая доля100 не является
оценкой качества на700 train или1000 production. Исходный семейный split
сохранён, все выбранные строки train; validation/test не участвовали в отборе,
контролях или адаптации. [Выбор и план](plan.json).

Основная сетка linspace(-1,1,33), допуск E∞<=.01. R: именно ChernovSolver.solve,
n128,Q513,quadratic,dx.0125,padding12,cubic/zero. MC: существующий Numba backend,
M1024 на точку,h1/128,chunk1024. Общий T из аналитического G для tail1e-4;
MC вычисляет его через T=None, R получает то же явное T. Это новый профиль,
который не смешивается с историческим cubic/taper/legacyT. Численные схемы,
генератор и датасеты не менялись. Exact используется после solve для метрик.

Перед pilot проверены два известных train-выброса (F0141/S00,F0184/S01).
Контроли taper/zero воспроизводят сохранённые ответы при неизменном legacyT,
после чего основной pilot идёт с отдельным общим T. Изменён только способ
продолжения при контроле, не правые части или квадратура. Это исправляет
известный механизм роста краевой производной taper, но не доказывает .01
или устойчивость cubic/zero для всех функций.

{table(controls)}

## Ресурсы и прямое сравнение

Один worker/BLAS1, RSS<=512 МиБ, общий абсолютный дедлайн14400 с с сохранением
при resume. MC watchdog=max(180,6*33*прежний train p95 x0+15),
R watchdog=max(120,6*прежний task wall). Это оценки для33 узлов, не старые60 с.
Контрольные R-запуски имеют180 с. Один независимый guardian с heartbeat и
owner lease300 с, атомарные immutable records, interrupted остаётся pending.

Активный wall **{summary['active_session_seconds']:.2f} с / {summary['active_session_seconds']/60:.2f} мин**;
календарный **{calendar:.2f} с / {calendar/60:.2f} мин**;
solver wall суммарно **{summary['total_solver_seconds']:.2f} с**,
peak RSS **{summary['peak_rss_mib']:.2f} МиБ**. Полная стоимость всех дополнительных
seed включена в MC_all_seconds;4 контрольных R-расчёта учитываются отдельно.
Защитные прерывания guardian: `{interruptions}`. Во время паузы истёк owner
lease; worker безопасно остановлен. После команды продолжить восстановлены
только pending при прежнем абсолютном дедлайне, готовые записи не повторялись.
Это операторская остановка, не математический или ресурсный отказ solver.
Частичные solver-счётчики прерванного вызова недоступны; его известные wall/RSS
сохранены в events, время входит в session wall. Сумма solver wall относится
к возвращённым ответам. Начальная сборка выбора/плана выполнена до numerical
budget и не включена во время solve.
Повторы не трактуются как бесплатные. Первые seed имеют одинаковое M1024;
агрегат4 seed — другой статистический бюджет, не оптимальное время до .01.

{table(flat)}

[Полная сравнительная таблица](analysis/comparison.csv),
[все попытки и конфигурации](analysis/attempts.csv),
[сводка](analysis/summary.json). Время обоих методов теперь измерено прямо на33
узлах в одном окружении, с чередованием очередности. Это позволяет сравнить
стоимость фиксированных профилей; она не является минимумом времени достижения
допуска. Подготовка/общий JIT сохранены отдельно в worker/result, входят в wall.

![Точность и стоимость](plots/accuracy_cost.png)

## Исходы и статистическая неопределённость

Наблюдаемые категории primary и final приведены отдельно по когортам:
`{json.dumps([dict(cohort=r['cohort'],primary=r['primary_categories'],final=r['final_categories']) for r in cohorts],ensure_ascii=False)}`.
both_pass/R_only/MC_only/neither_pass означают только E∞ возвращённых ответов.
Final относится к адаптивному профилю: один seed на88 случаях и среднее4
на12 выбранных, а не к100 одинаковым4096-path расчётам.
unavailable — отсутствие корректной пары; ресурсный отказ не математическая
неточность. Статистическая неопределённость — отдельная колонка, она не
заменяется наблюдаемым успехом одного seed.

Одновременная нормальная полуширина z*stderr: Bonferroni на33 координаты и2
заранее разрешённых просмотра (first и final), alpha=.05 на уравнение.
Сырые pointwise CI95 из MCResult сохранены, но не используются как сеточный
интервал. Дополнительно сохранена консервативная Hoeffding-оценка для
дискретных вкладов |Z|<=G*(T+h), в идеальной вещественной арифметике;
её ширина может быть непрактичной. Нормальные bands асимптотические, не
точные finite-sample. Покрытие95% не заявляется одновременно для всех100
уравнений. Полную погрешность округления эти bands не сертифицируют.

Ни нормальная, ни Hoeffding-оценка не ограничивает EM/трапециевое временное
смещение. Bounds ошибки mean относительно exact диагностируют фиксированное
дискретное ожидание; добавленный continuous tail не заменяет оценку h-bias.
R n/space/window/quadrature ошибки также не сертифицированы. Итоговые
training_eligible_winner=false и reliable_fastest_label=unconfirmed сохранены.
Четыре seed не подтверждают p(success)>=.95: даже4/4 дают нижнюю95% границу.473.

После первых100 пар по замороженному правилу выбраны до12 неразрешённых
уравнений: сначала до6 с неточным R, затем с наименьшей верхней границей
MC-ошибки. Добавлены ровно3 независимых seed; M,h,T не менялись. Первый seed
сохраняется и входит в стоимость; repeat-часть намеренно смещена. Эта
адаптация и два просмотра учтены в Bonferroni. [Решение](adaptive_selection.json).
Повторены **{summary['repeated_equations']}** уравнений; MC вызовов
**{summary['MC_solve_count']}** (полных **{summary['MC_completed_solve_count']}**, остальное прерывание),
R вызовов с контролями **{summary['R_solve_count']}**.
Pooled дисперсия включает межseed-компонент; seed-t bands сохранены отдельно.
Если повторная тройка не завершена полностью, итоговым статистическим просмотром
остаётся первый seed; незапланированные n=2/3 в итоговый band не подставляются.
Стоимость незавершённых/неуспешных попыток при этом сохраняется отдельно.

![Распределения и статистические bounds](plots/distribution_uncertainty.png)
![Повторы](plots/repeats.png)

## Информативность для ML и решение

Это train-only диагностический набор, а не новый train/validation/test split.
80 representative дают начальную оценку частот по train при семейной
зависимости;20 stress нужны для обнаружения ограничений. Их смешивание без
признака когорты/весов создаёт смещение. Повторно отобранные случаи особенно
не репрезентативны. Историческая ML для taper не переносится на новый zero
профиль автоматически. Новые данные не добавлялись к прежним моделям.

**Дополнительные900 автоматически не запускать.** Сначала оценить баланс
наблюдаемых исходов и статистически разрешённых MC-результатов. Если MC
почти нигде не достигает .01, такой фиксированный профиль даёт вырожденную
задачу выбора; расширение объёма повторяет её, а не создаёт надёжные метки.
При широких bands увеличение M не назначается автоматически: для следующего
эксперимента сначала выделить небольшой train-набор потенциальных MC-преимуществ
с провалом zero и малым статистическим шумом, затем заранее зафиксировать
профиль/критерий успеха и выполнить независимую подтверждающую серию.
Конкретный следующий состав и бюджет уточняются в INTERPRETATION.md по
завершённым числам; новых расчётов для него здесь нет. ML не переобучалась.

Проверка источников/очистки и регрессии — verification.json, pytest.log.
Исторические результаты и защищённое математическое ядро сохранены;
PROJECT_STATUS.md обновлён после расчётов.
'''
    (out/'REPORT.md').write_text(text)
    log=(out/'pytest.log').read_text() if (out/'pytest.log').exists() else ''
    checks=re.findall(r'\d+ passed,.*',log)
    with (out/'REPORT.md').open('a') as f:
        f.write('\n## Проверки и воспроизведение\n\n')
        f.write(f"Регрессия: `{checks[-1] if checks else 'см. pytest.log'}`.\n\n")
        f.write('Новые файлы: diagnostics/joint_grid100.py, diagnostics/grid100_statistics.py, '
            'diagnostics/report_joint_grid100.py, diagnostics/verify_joint_grid100.py, '
            'tests/test_joint_grid100.py; обновлён PROJECT_STATUS.md с сохранением прежней версии. '
            'Проверены16379 прежних файлов, изменений нет. Ядро из MIGRATION_MANIFEST не редактировалось.\n\n'
            '[Интерпретация и решение](INTERPRETATION.md), [воспроизведение](REPRODUCE.md), '
            '[resume-проверка](resume_check.json), [манифест](artifact_manifest.json).\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);report(ap.parse_args().out.resolve())
