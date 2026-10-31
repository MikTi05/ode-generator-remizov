"""Read saved independent F0002 phases; no solver calls or pilot pooling."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'mc-convergence-mpl'))
import numpy as np
from scipy import stats
from diagnostics.report_mc_f0002_time_refinement import (
    order_diagnostic, save_csv, summarize_stage, table, write_json)


def gate_from_summary(summary,rule):
    if summary is None or summary['repeats']!=rule['required_repeats']:
        return dict(proceed=False,reason='incomplete_independent_control')
    d=summary['differences'][0]
    relative=d['ci95_half']/abs(d['mean']) if d['mean'] else None
    variance=d['variance_reduction_vs_independent']
    passed=(relative is not None and relative<=rule['maximum_relative_ci95_half'] and
            variance is not None and variance>=rule['minimum_variance_reduction'])
    return dict(proceed=bool(passed),reason='criteria_passed' if passed else 'criteria_not_met',
        repeats=summary['repeats'],paired_mean=d['mean'],ci95_half=d['ci95_half'],
        relative_ci95_half=relative,variance_reduction=variance,
        inference_data='Only 8 new control seeds; no prior pilot or refinement observations')


def statistical_cost(sigma,h,T,halfwidth,*,points=1,simultaneous=False,
                     reference_seconds=None,reference_M=1024):
    """Planning arithmetic only: independent EM paths at fixed h and T."""
    if sigma<=0 or h<=0 or T<=0 or halfwidth<=0 or points<1:
        raise ValueError('positive planning inputs required')
    z=float(stats.norm.isf(.05/(2*points))) if simultaneous else 1.96
    M=max(2,math.ceil((z*sigma/halfwidth)**2))
    N=math.ceil(T/h)
    if N*h!=T:
        raise ValueError('cost study uses aligned fixed grids')
    batches=math.ceil(M/reference_M)
    return dict(h=h,T=T,sigma_at_x0=sigma,statistical_halfwidth=halfwidth,points=points,
        simultaneous=simultaneous,z=z,M_per_point=M,time_steps=N,
        path_steps=points*M*N,production_batches_at_chunk_1024=points*batches,
        serial_seconds_at_chunk_1024=None if reference_seconds is None else
            points*batches*reference_seconds,
        assumption='Same variance as x=0 at every point; approximate normal CI. '
            'Time bias unbounded; these are statistical costs, not a total-accuracy guarantee.')


def analyze(out):
    policy=json.loads((out/'policy.json').read_text());run=json.loads((out/'run_summary.json').read_text())
    stages={};rows=[]
    for phase in ('control','refinement128'):
        directory=out/phase
        if (directory/'plan.json').exists():
            summary,attempts=summarize_stage(directory,phase)
            stages[phase]=summary;rows+=attempts
    save_csv(out/'attempt_levels.csv',rows)
    write_json(out/'aggregates.json',stages)
    all_records=[json.loads(p.read_text()) for phase in ('control','refinement128')
        for p in (out/phase/'records').glob('*.result.json')]
    all_levels=[l for r in all_records for l in r['result']['levels']]
    resource=dict(path_steps=sum(l['kernel_statistics']['n_path_steps'] for l in all_levels),
        normal_values_generated_with_replay=sum(l['normal_values_generated_with_replay'] for l in all_levels),
        kernel_seconds=sum(l['runtime_seconds'] for l in all_levels),
        max_rss_mib=max((r['process_peak_rss_mib'] for r in all_records),default=0.),
        statuses={s:sum(a['status']==s for a in run['attempts']) for s in {a['status'] for a in run['attempts']}},
        includes_partial_failure_counters=True,no_solver_called=True,no_old_pilot_data=True)
    assert resource['path_steps']<=policy['resources']['max_path_steps']
    write_json(out/'analysis_verification.json',resource)
    refinement=stages.get('refinement128')
    order=None
    if refinement and refinement['repeats']==8:
        differences=refinement['differences']
        if all(d['simultaneous_ci95'][0]>0 for d in differences):
            order=order_diagnostic(refinement,policy['analysis']['bootstrap_seed'],policy['analysis']['bootstrap_replicates'])
        else:
            order=dict(identified=False,reason='Not all adjacent differences positive in simultaneous 95% intervals')
    write_json(out/'order_diagnostic.json',order)
    # Cost scenarios use every successfully measured h of the latest phase,
    # not a selective best result or a mixture of independent phases.
    latest=refinement or stages.get('control')
    scenarios=[]
    if latest:
        for level in latest['levels']:
            for width in (.01,.005):
                for points,simultaneous in ((1,False),(33,False),(33,True)):
                    scenarios.append(statistical_cost(level['path_sigma'],level['h'],policy['T'],width,
                        points=points,simultaneous=simultaneous,
                        reference_seconds=level['median_runtime_seconds']))
    write_json(out/'cost_scenarios.json',scenarios)
    save_csv(out/'cost_scenarios.csv',scenarios)
    assessments=[]
    if latest:
        for level in latest['levels']:
            lo=level['signed_error']-level['simultaneous_ci95_half']
            hi=level['signed_error']+level['simultaneous_ci95_half']
            assessments.append(dict(h=level['h'],phase=latest['stage'],
                simultaneous_mean_bias_interval=[lo,hi],mean_bias_within_001=lo>=-.01 and hi<=.01,
                mean_bias_outside_001=lo>.01 or hi<-.01,
                combined_bias_and_future_stat_half_005=abs(level['signed_error'])+
                    level['simultaneous_ci95_half']+.005,
                caveat='Uncertainty of mean bias only; does not certify a future realization or grid'))
    write_json(out/'accuracy_assessment.json',assessments)
    return policy,run,stages,order,scenarios,assessments,resource


def plots(out,stages,scenarios):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'figure.dpi':130,'savefig.dpi':160})
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    def save(fig,name):
        fig.tight_layout();fig.savefig(figdir/f'{name}.png');fig.savefig(figdir/f'{name}.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for phase,color,marker in [('control','C0','o'),('refinement128','C1','s')]:
        a=stages.get(phase)
        if not a:continue
        x=[g['h'] for g in a['levels']]
        axes[0].errorbar(x,[g['signed_error'] for g in a['levels']],
            yerr=[g['simultaneous_ci95_half'] for g in a['levels']],fmt=marker+'-',color=color,
            capsize=4,label=phase+' (simultaneous 95%)')
        axes[1].errorbar([d['coarse_h'] for d in a['differences']],[d['mean'] for d in a['differences']],
            yerr=[d['simultaneous_ci95_half'] for d in a['differences']],fmt=marker+'-',color=color,
            capsize=4,label=phase+' (simultaneous 95%)')
    axes[0].axhspan(-.01,.01,color='gray',alpha=.15,label='Mean bias +/-0.01')
    axes[0].set_ylabel('Estimated mean bias to exact')
    axes[1].set_ylabel('Paired mean Z(h)-Z(h/2)')
    for ax in axes:ax.axhline(0,color='black',lw=.6);ax.set_xscale('log',base=2);ax.set_xlabel('h');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('F0002: fixed x=0, T=63.625; new independent phases, no old pilots')
    save(fig,'convergence')
    latest=stages.get('refinement128') or stages.get('control')
    if not latest:return
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    ds=latest['differences'];x=np.arange(len(ds))
    axes[0].bar(x-.17,[d['independent_stderr_at_equal_paths'] for d in ds],width=.34,label='Independent (estimated)')
    axes[0].bar(x+.17,[d['paired_stderr_from_paths'] for d in ds],width=.34,label='Coupled')
    axes[0].set_xticks(x,[f"1/{round(1/d['coarse_h'])} vs 1/{round(1/d['fine_h'])}" for d in ds])
    axes[0].set_ylabel('SE of level difference');axes[0].legend(fontsize=8)
    axes[1].plot([g['h'] for g in latest['levels']],[g['median_runtime_seconds'] for g in latest['levels']],'o-')
    axes[1].set(xlabel='h',ylabel='Median kernel seconds per 1024 paths');axes[1].set_xscale('log',base=2)
    for ax in axes:ax.grid(alpha=.2)
    save(fig,'variance_and_runtime')
    fig,ax=plt.subplots(figsize=(7,4))
    for pts,sim,style,label in [(1,False,'o-','1 point'),(33,False,'s-','33 pointwise intervals'),
                               (33,True,'^--','33 simultaneous intervals')]:
        a=sorted([s for s in scenarios if s['statistical_halfwidth']==.005 and
            s['points']==pts and s['simultaneous']==sim],key=lambda s:s['h'])
        ax.loglog([s['h'] for s in a],[s['path_steps'] for s in a],style,label=label)
    ax.set(xlabel='h',ylabel='Planned updates (statistical CI half-width 0.005)')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    ax.set_title('Cost scenarios assume variance at x=0; no total-error guarantee')
    save(fig,'cost_scaling')


def report(out,policy,run,stages,order,scenarios,assessments,resource):
    f=lambda x:f'{x:.7g}'
    gate=json.loads((out/'gate.json').read_text())
    lines=['# F0002: завершение ограниченного временного контроля — 2026-10-24','',
        'Только x=0, фиксированный сертифицированный T=63.625. '
        f"G={policy['forcing_bound']['abs_upper_bound']:.10g}, непрерывный хвост "
        f"≤{policy['tail_error_bound']:.9g}. Production EM и обе трапеции неизменны; "
        'использован прежний диагностический CoupledNormals и то же production-ядро.',
        'Новые independent control и refinement128 используют непересекающиеся seed. '
        'Прежний четырёхповторный пилот, старый M/h/T обзор и две новые фазы не объединяются '
        'в одну статистическую выборку. Для тройки заново считаются все три связанных уровня, '
        'поскольку поток зависит от самого мелкого h.', '',
        '## Протокол','',
        '[Политика до первого запуска](policy.json), [план контроля](control/plan.json), '
        '[решение о переходе](gate.json). '
        + ('[Отдельный план тройки, записанный после перехода и до её запуска](refinement128/plan.json).' if 'refinement128' in stages else 'Тройка не запускалась.'),
        'Контроль — ровно 8 новых seed×1024 пути на h=1/32,1/64. '
        'Порог перехода сохранён: относительная t95-полуширина парной разности ≤0.5 и '
        'оценённый выигрыш в дисперсии ≥2. При успехе — отдельная тройка '
        'h=1/32,1/64,1/128 на других 8 seed×1024. Никакого увеличения M/числа повторов, '
        'перезапуска отказов или h<1/128.',
        f"Решение: {gate['reason']}; относительная полуширина {gate.get('relative_ci95_half')}, "
        f"выигрыш {gate.get('variance_reduction')}.",
        'Общий предельный бюджет 170 млн обновлений / 240 с / 384 МиБ, '
        f"плановый максимум {policy['resources']['planned_maximum_path_steps']:,} обновлений. "
        'Один worker/BLAS=1; 20 с cooperative на связанный повтор, внешний watchdog.',
        f"Факт: статусы {resource['statuses']}, {resource['path_steps']:,} обновлений, "
        f"{resource['normal_values_generated_with_replay']:,} нормальных значений с replay; "
        f"ядра {resource['kernel_seconds']:.3f} с, серия {run['wall_seconds']:.3f} с, RSS "
        f"{resource['max_rss_mib']:.2f} МиБ. Replay повторяет поток и требует O(M) памяти.",
        'Источники: [окружение](environment.json), [все статусы и ресурсы](run_summary.json), '
        '[CI и ошибки каждого запуска CSV](attempt_levels.csv), [агрегаты/ковариации](aggregates.json), '
        '[планы/ответы контроля](control/), '
        + ('[планы/ответы тройки](refinement128/), ' if 'refinement128' in stages else '')+
        '[сохранность](verification.json). В каждой выполненной фазе сохранены NPZ с интегралами всех путей, '
        'полные config, stdout и rss.jsonl.','',
        '## Смещение и статистическая неопределённость','',
        '±t95 — точечный интервал для μ(h,T)−y_exact по независимым seed-средним. '
        'Он оценивает неопределённость среднего смещения, не полную ошибку нового запуска. '
        'Одновременные Bonferroni-интервалы по h и CI отдельных запусков сохранены отдельно. '
        'Чистое временное смещение μ(h,T)−y_T отличается не более чем на границу хвоста; '
        'в разности уровней общий непрерывный хвост сокращается. Разделения вклада EM и трапеций нет.','']
    lines+=table(['Фаза','h','f среднее','Смещение ±t95','Одновр.95 половина','RMSE 8 запусков','CI запуска, половина','Медиана, с'],[
        [phase,f(g['h']),f(g['mean']),f"{f(g['signed_error'])} ± {f(g['ci95_half'])}",
         f(g['simultaneous_ci95_half']),f(g['rmse_of_replicates']),f(g['mean_individual_ci95_half']),f(g['median_runtime_seconds'])]
        for phase,a in stages.items() if a for g in a['levels']])
    lines+=['','## Связанные разности','']
    lines+=table(['Фаза','h→h/2','Разность ±t95','Одновр.95 половина','Корреляция','Var reduction'],[
        [phase,f"1/{round(1/d['coarse_h'])}→1/{round(1/d['fine_h'])}",
         f"{f(d['mean'])} ± {f(d['ci95_half'])}",f(d['simultaneous_ci95_half']),
         f(d['correlation']),f(d['variance_reduction_vs_independent'])]
        for phase,a in stages.items() if a for d in a['differences']])
    lines+=['','![Временное уточнение](figures/convergence.png)','',
        '![Выигрыш в дисперсии и время](figures/variance_and_runtime.png)','',
        'Связывание уменьшает шум разностей, не абсолютную дисперсию каждого уровня. '
        'Интервалы приближённые: нормальность seed-средних и покрытие по 8 повторам не доказаны.', '',
        '## Порядок и точность 0.01','',
        '[Проверка точности среднего смещения](accuracy_assessment.json), '
        '[условная диагностика отношения разностей](order_diagnostic.json). '
        'Два уровня не дают порядка. Три успешных уровня допускают только наблюдаемое отношение; '
        'без различимых соседних разностей оно не вычисляется. Даже узкий bootstrap-интервал '
        'по трём уровням не устанавливает асимптотический порядок.']
    if order and order.get('identified'):
        lines += [f"Наблюдаемое p={order['observed_p']:.5g}, bootstrap95={order['bootstrap_p_ci95']}; "
            'это условный индикатор степенной модели, не утверждение о порядке метода. '
            'Экстраполированные модельные остатки не используются для подтверждения 0.01 или замены EM-ответов.']
    for a in assessments:
        lines += [f"h={a['h']}: одновременный 95% интервал смещения "
            f"{a['simultaneous_mean_bias_interval']}; весь внутри ±0.01: "
            f"{a['mean_bias_within_001']}; целиком вне ±0.01: {a['mean_bias_outside_001']}."]
    lines+=['','## Стоимость допуска: одна точка и сетка 33','',
        'Таблица выделяет статистике CI-полуширину 0.005, оставляя часть общего допуска '
        'для временного смещения и хвоста. Это **оценка статистической стоимости, не '
        'подтверждённая стоимость достижения полной ошибки 0.01**. В CSV/JSON также есть '
        'вариант 0.01 целиком на статистику; он не оставляет запаса на смещение.',
        'M=ceil((z·sigma/0.005)^2), sigma измерена только в x=0 на последней фазе. '
        'Столбцы сетки предполагают такую же дисперсию во всех точках. Одновременный '
        'вариант использует z=Phi^-1(1−0.05/(2·33)); pointwise×33 не контролирует максимум '
        'ошибки сетки. Регулярность ошибки между точками не исследовалась.', '']
    lines+=table(['h','Режим','M/точку','Обновлений','Оценка секунд, chunk=1024'],[
        [f(s['h']),'1 точка' if s['points']==1 else '33, совместный CI' if s['simultaneous'] else '33, точечные CI',
         s['M_per_point'],s['path_steps'],f(s['serial_seconds_at_chunk_1024'])]
        for s in scenarios if s['statistical_halfwidth']==.005])
    lines+=['','![Оценка масштабирования](figures/cost_scaling.png)','',
        'Полные сценарии: [cost_scenarios.csv](cost_scenarios.csv), [JSON](cost_scenarios.json). '
        'Время экстраполировано по медиане ядра на 1024 пути, последовательно теми же пакетами; '
        'округление числа пакетов учтено. Иной chunk, нагрев, загрузка, подготовка и дисперсия '
        'в других точках меняют время. Это не измерение работы на сетке.', '',
        '## Научный вывод и ограничения для будущего сравнения','',
        '[Краткая интерпретация](INTERPRETATION.md) отделяет установленные факты от гипотез. '
        'Если интервалы смещения не помещаются в допустимый бюджет или статистическая стоимость '
        'превышает исходный профиль, точность фиксируется как неподтверждённая. '
        'Дополнительные вычисления ради успешного результата не запускаются. '
        'Метку «решено с 0.01» для ML по этим данным не назначать без отдельной проверки полной ошибки. '
        'Сравнение методов должно учитывать стоимость, фиксированный бюджет и недостигнутый допуск; '
        'победитель по этой одной задаче не установлен.',
        'Проверки: [pytest.txt](pytest.txt), [финальная верификация](final_verification.json). '
        'Production MC, Ремизов, генератор, датасеты и прежние результаты сохранены; ML и массовых расчётов нет.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    data=analyze(args.out);plots(args.out,data[2],data[4]);report(args.out,*data)


if __name__=='__main__':main()
