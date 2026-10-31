"""Read-only analysis of saved one-point MC results; never invoke a solver."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'mc-convergence-mpl'))
import numpy as np
from scipy import stats


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def analytic_means(problem_id, h, T, x=0.):
    """Independent heat-kernel expectation, not a sampled Brownian reference.

    For the Gaussian, E[g(x+sqrt(2)W_t)] follows from the normal Gaussian
    integral; the continuous tail is exp(-2T) P_T exp(-x^2).
    """
    if problem_id not in ('MC_constant','MC_gaussian'):
        return None
    N = round(T/h)
    assert math.isclose(N*h, T, abs_tol=1e-13)
    t = np.arange(N+1)*h
    if problem_id == 'MC_constant':
        exact = .5
        tail = .5*math.exp(-2*T)
        integrand = np.exp(-2*t)
    else:
        exact = math.exp(-x*x)
        A = 1+4*t
        heat_y = np.exp(-x*x/A)/np.sqrt(A)
        integrand = np.exp(-2*t)*(2+2/A-4*x*x/(A*A))*heat_y
        tail = math.exp(-2*T-x*x/(1+4*T))/math.sqrt(1+4*T)
    continuous = exact-tail
    discrete = float(np.trapezoid(integrand, dx=h))
    return dict(exact=exact, continuous_finite_mean=continuous, discrete_mean=discrete,
                signed_truncation_bias=-tail, time_bias=discrete-continuous,
                discrete_total_bias=discrete-exact)


def ensemble(values, stderrs, exact):
    """t interval uses independent run means, not paths treated as repeats."""
    values, stderrs = np.asarray(values), np.asarray(stderrs)
    R = len(values)
    sd = float(np.std(values, ddof=1)) if R > 1 else 0.
    half = float(stats.t.ppf(.975, R-1)*sd/math.sqrt(R)) if R > 1 else 0.
    return dict(repeats=R, mean=float(np.mean(values)), signed_mean_error=float(np.mean(values)-exact),
        mean_ci95_half=half, replicate_sd=sd,
        replicate_sd_ci95=[math.sqrt((R-1)*sd*sd/stats.chi2.ppf(q, R-1))
                           for q in (.975,.025)] if R > 1 else [0.,0.],
        rmse=float(np.sqrt(np.mean((values-exact)**2))),
        rms_stderr=float(np.sqrt(np.mean(stderrs**2))),
        mean_individual_ci95_half=float(np.mean(1.96*stderrs)))


def difference(a, b):
    """Welch interval for independent configurations (unadjusted, pointwise)."""
    va = a['replicate_sd']**2/a['repeats']
    vb = b['replicate_sd']**2/b['repeats']
    if va+vb == 0:
        half, df = 0., None
    else:
        df = (va+vb)**2/(va**2/(a['repeats']-1)+vb**2/(b['repeats']-1))
        half = float(stats.t.ppf(.975,df)*math.sqrt(va+vb))
    d = a['mean']-b['mean']
    return dict(difference=d, ci95_half=half, lower=d-half, upper=d+half,
                excludes_zero=abs(d)>half, df=df)


def table(header, rows):
    return ['| '+' | '.join(header)+' |', '| '+' | '.join(['---']*len(header))+' |']+[
        '| '+' | '.join(str(x) for x in row)+' |' for row in rows]


def save_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer=csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def analyze(out):
    plan = json.loads((out/'plan.json').read_text())
    summary = json.loads((out/'run_summary.json').read_text())
    status = {a['attempt_id']:a['status'] for a in summary['attempts']}
    all_attempts, grouped, verified = [], {}, []
    for spec in plan['attempts']:
        aid=spec['attempt_id']
        row = dict(attempt_id=aid, problem_id=spec['problem_id'],
            configuration_id=spec['configuration_id'], repeat=spec['repeat'],
            status=status[aid], M=spec['config']['M'], h=spec['config']['dt'], T=spec['config']['T'],
            seed=spec['config']['seed'], x=0., f=None, exact=None, signed_error=None,
            abs_error=None, stderr=None, ci_lower=None, ci_upper=None,
            tail_bound=None, tail_tolerance_met=None, runtime_seconds=None,
            call_wall_seconds=None, path_steps=None, normal_draws=None, peak_rss_mib=None)
        path=out/'records'/f'{aid}.result.json'
        if path.exists():
            data=json.loads(path.read_text()); r=data['result']
            assert data['spec']==spec and data['config']==spec['config']
            if r['success']:
                assert r['x']==[0.] and r['completed_paths']==[r['M']]
                assert r['dt_actual']==spec['config']['dt']
                assert r['n_path_steps']==r['n_normal_draws']==spec['preflight']['path_steps']
                assert sum(r['coefficient_calls'].values())==spec['preflight']['coefficient_calls']
                assert r['n_steps']==spec['preflight']['time_steps']
                assert not r['total_error_certified'] and r['discretization_error_bound'] is None
                f,y,se=r['f'][0],data['y_exact'][0],r['stderr'][0]
                assert data['max_abs_error']==abs(f-y)
                assert r['ci95'][0]==1.96*se
                assert r['ci_lower'][0]==f-1.96*se and r['ci_upper'][0]==f+1.96*se
                assert r['tail_tolerance_met']==spec['tail_tolerance_met']
                row.update(f=f, exact=y, signed_error=f-y, abs_error=abs(f-y), stderr=se,
                    ci_lower=r['ci_lower'][0], ci_upper=r['ci_upper'][0],
                    tail_bound=r['tail_error_bound'], tail_tolerance_met=r['tail_tolerance_met'],
                    runtime_seconds=r['runtime_seconds'], call_wall_seconds=data['call_wall_seconds'],
                    path_steps=r['n_path_steps'], normal_draws=r['n_normal_draws'],
                    peak_rss_mib=data['process_peak_rss_mib'])
                grouped.setdefault(spec['configuration_id'], []).append((spec,row))
                verified.append(aid)
        all_attempts.append(row)
    save_csv(out/'attempts.csv', all_attempts)
    groups=[]
    for cid, entries in grouped.items():
        spec=entries[0][0]; rows=[e[1] for e in entries]
        a=ensemble([r['f'] for r in rows], [r['stderr'] for r in rows], rows[0]['exact'])
        a.update(configuration_id=cid, problem_id=spec['problem_id'], M=spec['config']['M'],
            h=spec['config']['dt'], T=spec['config']['T'], views=spec['views'], exact=rows[0]['exact'],
            min_abs_error=min(r['abs_error'] for r in rows), max_abs_error=max(r['abs_error'] for r in rows),
            count_abs_error_le_001=sum(r['abs_error']<=.01 for r in rows),
            exact_in_individual_ci=sum(r['ci_lower']<=r['exact']<=r['ci_upper'] for r in rows),
            tail_bound=rows[0]['tail_bound'], tail_tolerance_met=rows[0]['tail_tolerance_met'],
            median_seconds=float(np.median([r['runtime_seconds'] for r in rows])),
            sum_seconds=sum(r['runtime_seconds'] for r in rows), path_steps_per_run=rows[0]['path_steps'])
        oracle=analytic_means(a['problem_id'], a['h'], a['T'])
        if oracle:
            oracle['discrete_mean_in_individual_ci']=sum(
                r['ci_lower']-3e-14<=oracle['discrete_mean']<=r['ci_upper']+3e-14 for r in rows)
            oracle['sampling_errors']=[r['f']-oracle['discrete_mean'] for r in rows]
        a['analytic_oracle']=oracle
        groups.append(a)
    write_json(out/'aggregates.json', groups)
    save_csv(out/'aggregates.csv', [{k:v for k,v in a.items() if k not in
        ('views','analytic_oracle','replicate_sd_ci95')} for a in groups])
    comparisons=[]
    for task in plan['tasks']:
        pid=task['problem_id']
        for factor in ('h','T'):
            subset=select(groups,pid,factor)
            if len(subset)<2 or pid=='MC_constant':
                continue
            for a,b in zip(subset[:-1],subset[1:]):
                comparisons.append(dict(problem_id=pid, factor=factor,
                    first=a[factor], second=b[factor], **difference(a,b)))
    write_json(out/'differences.json',comparisons)
    recommendations=[]
    for task in plan['tasks'][1:]:
        base=next(g for g in groups if g['problem_id']==task['problem_id'] and
                  g['M']==8192 and g['h']==1/32 and g['T']==8)
        sigma=math.sqrt(base['M'])*base['rms_stderr']
        M=math.ceil((1.96*sigma/.005)**2)
        M=max(2,M)
        T=task['rounded_certified_T']
        long=next(g for g in groups if g['problem_id']==task['problem_id'] and
                  g['M']==2048 and g['h']==1/32 and g['T']==T)
        recommendations.append(dict(problem_id=task['problem_id'], pilot_path_sigma=sigma,
            M_for_statistical_halfwidth_005_pilot=M,
            M_for_statistical_halfwidth_001_pilot=math.ceil((1.96*sigma/.01)**2),
            T_tail_1e3=task['T_tail_1e3'], certified_T_tail_1e4=T, candidate_h=1/32,
            path_steps_at_candidate=M*round(T*32),
            seconds_at_candidate_linear_estimate=long['median_seconds']*M/2048,
            caveat='Pilot variance at T=8; not a guarantee, time bias unresolved on dataset. '
                   'Long-T variance and timing extrapolation require independent validation.'))
    write_json(out/'recommendations.json',recommendations)
    write_json(out/'analysis_verification.json',dict(verified_attempts=len(verified),
        total_declared=len(plan['attempts']), statuses={s:sum(a['status']==s for a in all_attempts)
        for s in sorted({a['status'] for a in all_attempts})},
        actual_path_steps=sum(r['path_steps'] or 0 for r in all_attempts),
        total_solver_seconds=sum(r['runtime_seconds'] or 0 for r in all_attempts),
        max_rss_mib=max(r['peak_rss_mib'] or 0 for r in all_attempts),
        source_snapshots={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'sources').glob('*')},
        no_solver_called=True))
    return plan,summary,groups,comparisons,recommendations


def select(groups,pid,factor):
    return sorted([g for g in groups if g['problem_id']==pid and
        any(v['factor']==factor for v in g['views'])], key=lambda a:a[factor])


def plots(out,plan,groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'figure.dpi':130,'savefig.dpi':160})
    pids=[t['problem_id'] for t in plan['tasks'][1:]]
    figdir=out/'figures'; figdir.mkdir(exist_ok=True)
    def save(fig,name):
        fig.tight_layout()
        fig.savefig(figdir/f'{name}.png')
        fig.savefig(figdir/f'{name}.pdf')
        plt.close(fig)
    for factor,name in [('M','M_convergence'),('h','h_convergence'),('T','T_convergence')]:
        fig,axes=plt.subplots(2,2,figsize=(10,7))
        for ax,pid in zip(axes.flat,pids):
            a=select(groups,pid,factor); x=[g[factor] for g in a]
            if factor=='M':
                ax.loglog(x,[g['rmse'] for g in a],'o-',label='RMSE to exact (8 seeds)')
                ax.loglog(x,[g['mean_individual_ci95_half'] for g in a],'s-',label='Mean 95% statistical half-width')
                ax.errorbar(x,[g['replicate_sd'] for g in a],
                    yerr=np.array([[g['replicate_sd']-g['replicate_sd_ci95'][0] for g in a],
                                   [g['replicate_sd_ci95'][1]-g['replicate_sd'] for g in a]]),
                    fmt='^',capsize=3,label='Across-seed SD + 95% interval*')
                ax.loglog(x,[a[0]['mean_individual_ci95_half']*math.sqrt(x[0]/v) for v in x],
                          ':',color='gray',label='M^-1/2 reference')
                ax.axhline(.01,color='gray',ls='--'); ax.set_ylabel('Error / statistical uncertainty')
            else:
                ax.errorbar(x,[g['signed_mean_error'] for g in a],
                    yerr=[g['mean_ci95_half'] for g in a],fmt='o-',capsize=4,label='Mean error + t 95% CI (8 seeds)')
                if pid=='MC_gaussian':
                    ax.plot(x,[g['analytic_oracle']['discrete_total_bias'] for g in a],
                            'x--',label='Exact discrete-expectation bias')
                ax.axhline(0,color='gray',lw=.7)
                ax.axhspan(-.01,.01,color='gray',alpha=.12,label='Pointwise +/-0.01')
                ax.set_ylabel('Signed error to exact')
                if factor=='h':
                    ax.set_xscale('log',base=2)
                else:
                    T=next(t['rounded_certified_T'] for t in plan['tasks'] if t['problem_id']==pid)
                    ax.axvline(T,color='purple',ls=':',label='Certified T (tail <= 1e-4)')
                    ax.set_xscale('log',base=2)
            ax.set_title(pid); ax.set_xlabel(factor); ax.grid(alpha=.25); ax.legend(fontsize=7)
        fig.suptitle({'M':'M: fixed h=1/32, T=8, x=0',
            'h':'h: fixed M=8192, T=8, x=0 (independent, uncoupled seeds)',
            'T':'T: fixed M=2048, h=1/32, x=0'}[factor])
        save(fig,name)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for pid in pids:
        a=select(groups,pid,'T')
        axes[0].semilogy([g['T'] for g in a],[g['tail_bound'] for g in a],'o-',label=pid)
        axes[1].plot([g['T'] for g in a],[g['path_steps_per_run']/1e6 for g in a],'o-',label=pid)
    axes[0].axhline(1e-4,color='gray',ls='--',label='tail_eps=1e-4')
    axes[0].axhline(.01,color='gray',ls=':',label='0.01 (tail alone)')
    axes[0].set_ylabel('Certified continuous tail bound'); axes[1].set_ylabel('Million path updates per run')
    for ax in axes:
        ax.set_xlabel('T'); ax.grid(alpha=.25); ax.legend(fontsize=7)
    fig.suptitle('T cost and certified tail: M=2048, h=1/32, x=0')
    save(fig,'tail_cost')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for pid in ('MC_constant','MC_gaussian'):
        a=select(groups,pid,'h')
        axes[0].loglog([g['h'] for g in a],[g['analytic_oracle']['time_bias'] for g in a],
                       'o-',label=pid)
        b=select(groups,pid,'T')
        axes[1].semilogy([g['T'] for g in b],[-g['analytic_oracle']['signed_truncation_bias'] for g in b],
                         'o-',label=pid+' actual tail')
        axes[1].semilogy([g['T'] for g in b],[g['tail_bound'] for g in b],':',label=pid+' bound')
    axes[0].set(xlabel='h',ylabel='Exact discrete minus continuous finite mean',title='Deterministic time bias at T=8')
    axes[1].set(xlabel='T',ylabel='Continuous tail',title='Independent analytic tail at x=0')
    for ax in axes:
        ax.grid(alpha=.25); ax.legend(fontsize=7)
    save(fig,'analytic_error_components')


def report(out,plan,summary,groups,comparisons,recs):
    v=json.loads((out/'analysis_verification.json').read_text())
    fmt=lambda x:f'{x:.6g}'
    lines=['# Контролируемое исследование Монте-Карло — 2026-10-24','',
        'Только x=0: абсолютные ошибки в этой точке, без утверждений о пространственном E∞.',
        'EM, обе трапеции, RNG и solver не менялись. Использован существующий малый датасет; exact вычислялся после возврата solver.',
        f"Выполнено {v['verified_attempts']}/{v['total_declared']} попыток; статусы: {v['statuses']}. "
        f"Обновлений {v['actual_path_steps']:,}; время solver {v['total_solver_seconds']:.3f} с, "
        f"всей серии {summary['wall_seconds']:.3f} с; максимальный RSS {v['max_rss_mib']:.1f} МиБ.",
        'Один worker, BLAS=1. До запуска зафиксированы бюджет 250 млн обновлений / 360 с / 384 МиБ, '
        'все конфигурации и seeds. Отказов/повторов/отбора по ошибке нет, если все статусы success.',
        'Источники: [план](plan.json), [окружение](environment.json), [все попытки CSV](attempts.csv), '
        '[агрегаты CSV](aggregates.csv), [полные агрегаты и оракулы](aggregates.json), '
        '[ответы MCResult](records/), [ресурсы worker](worker_events.json), [сохранность SHA](verification.json).','',
        '## Протокол и смысл интервалов','',
        'Для Gaussian и каждой F0000–F0002 — 8 независимых seed на конфигурацию. '
        'M=512/2048/8192 при h=1/32,T=8; h=1/8,1/16,1/32 при M=8192,T=8; '
        'T=1,2,4,8,Tсертификат при M=2048,h=1/32. Общие baseline решены один раз. '
        'Порядок запусков перемешан внутри задачи независимо от потоков MC; chunk=8192 фиксирован. '
        'Броуновского сопряжения между шагами нет. Постоянный пример: M=2 и один детерминированный запуск на уровень.',
        'Индивидуальный CI = f±1.96·stderr для μ(h,T). В таблицах ± — 95% Student-t интервал '
        'для среднего смещения по 8 независимым запускам; он также не включает неизвестное смещение схемы. '
        'SD-интервалы на графике M — chi-square при приближённой нормальности средних. '
        'Все интервалы точечные, без поправки за множественные сравнения. 8 повторов недостаточно '
        'для точной проверки покрытия 95% или вероятности достижения 0.01.',
        'Разложение: f−y=(f−μ(h,T))+(μ(h,T)−y_T)+(y_T−y). Первое — статистика, '
        'второе — EM/трапеции, третье — непрерывный хвост. G контролирует только третий член.','',
        '## Сертифицированные горизонты','']
    lines+=table(['Задача','G','λ−C','T* для 1e-4','T в опыте','Граница при T=8'],[
        [t['problem_id'],fmt(t['G']),fmt(t['gap']),fmt(t['certified_T']),fmt(t['rounded_certified_T']),
         fmt(next(g['tail_bound'] for g in groups if g['problem_id']==t['problem_id'] and g['T']==8))]
        for t in plan['tasks']])
    lines+=['','Горизонт округлён вверх до кратного 1/8; это сохраняет сертификат и один и тот же '
        'фактический h при изменении T. Более короткий T имеет аналитическую верхнюю границу, '
        'но **не сертифицирован к tail_eps=1e-4**, если tail_tolerance_met=False. '
        'Наблюдаемое малое отклонение короткого T не заменяет сертификат.','',
        '![Граница хвоста и стоимость](figures/tail_cost.png)','',
        '## Влияние M','',
        'RMSE считается по всем 8 ошибкам к exact; рост M уменьшает шум, но не устраняет смещение.', '']
    lines+=table(['Задача','M','RMSE','Средняя половина индивидуального CI','SD между seed','Среднее смещение ± t95','Медиана, с'],[
        [pid,g['M'],fmt(g['rmse']),fmt(g['mean_individual_ci95_half']),fmt(g['replicate_sd']),
         f"{fmt(g['signed_mean_error'])} ± {fmt(g['mean_ci95_half'])}",fmt(g['median_seconds'])]
        for pid in [t['problem_id'] for t in plan['tasks'][1:]] for g in select(groups,pid,'M')])
    lines+=['','![Серия M](figures/M_convergence.png)','',
        '## Влияние h','',
        'T=8, M=8192. Разности независимых средних и Welch-интервалы сохранены в [differences.json](differences.json).','']
    lines+=table(['Задача','h','Среднее смещение ± t95','RMSE','CI одного запуска, половина','Медиана, с'],[
        [pid,fmt(g['h']),f"{fmt(g['signed_mean_error'])} ± {fmt(g['mean_ci95_half'])}",
         fmt(g['rmse']),fmt(g['mean_individual_ci95_half']),fmt(g['median_seconds'])]
        for pid in [t['problem_id'] for t in plan['tasks'][1:]] for g in select(groups,pid,'h')])
    lines+=['','![Серия h](figures/h_convergence.png)','',
        '## Влияние T','',
        'M=2048, h=1/32. Граница хвоста глобальная; ошибки и интервалы только в x=0.','']
    lines+=table(['Задача','T','Среднее смещение ± t95','RMSE','Хвост ≤','Серт. 1e-4','Обновлений/запуск','Медиана, с'],[
        [pid,fmt(g['T']),f"{fmt(g['signed_mean_error'])} ± {fmt(g['mean_ci95_half'])}",fmt(g['rmse']),
         fmt(g['tail_bound']),'да' if g['tail_tolerance_met'] else 'нет',g['path_steps_per_run'],fmt(g['median_seconds'])]
        for pid in [t['problem_id'] for t in plan['tasks'][1:]] for g in select(groups,pid,'T')])
    lines+=['','![Серия T](figures/T_convergence.png)','',
        '## Независимая аналитическая проверка','',
        'Для y=1/2: y_T=(1−exp(−2T))/2; μ(h,T) — трапеции exp(−2t). '
        'Траектории не влияют на интеграл, поэтому нулевой статистический CI корректен для μ(h,T), '
        'но не для y. Для Gaussian: y_T(x)=exp(−x²)−exp(−2T−x²/(1+4T))/sqrt(1+4T). '
        'Heat-kernel интегранд равен exp(−2t)(2+2/A−4x²/A²)exp(−x²/A)/sqrt(A), A=1+4t. '
        'Его трапеции дают точное ожидание используемой схемы при постоянных коэффициентах.','']
    lines+=table(['Пример','h','Временное смещение μ−y_T','Хвост y−y_T','Полное смещение μ−y','CI покрывают μ / запуски'],[
        [pid,fmt(g['h']),fmt(g['analytic_oracle']['time_bias']),fmt(-g['analytic_oracle']['signed_truncation_bias']),
         fmt(g['analytic_oracle']['discrete_total_bias']),f"{g['analytic_oracle']['discrete_mean_in_individual_ci']}/{g['repeats']}"]
        for pid in ('MC_constant','MC_gaussian') for g in select(groups,pid,'h')])
    lines+=['','![Аналитическое разложение](figures/analytic_error_components.png)','',
        '## Ориентиры для точности 0.01','',
        'Планирование статистики: σ²≈M·mean(stderr²) по baseline M=8192,h=1/32,T=8. '
        'M ниже — предварительное число путей для индивидуальной половины CI 0.005; '
        'оставляет место для хвоста и временного смещения, но само по себе не гарантирует 0.01. '
        'При другом h/T дисперсию следует проверить заново. Время — грубая линейная экстраполяция '
        'от измеренного длинного T; мелкие M, AST и фиксированная стоимость цикла нарушают линейность.','']
    lines+=table(['Задача','σ pilot','M для CI±0.005','T для хвоста≤0.001','T для хвоста≤1e-4','Обновлений при h=1/32','Оценка секунд/точку'],[
        [r['problem_id'],fmt(r['pilot_path_sigma']),r['M_for_statistical_halfwidth_005_pilot'],
         fmt(r['T_tail_1e3']),fmt(r['certified_T_tail_1e4']),r['path_steps_at_candidate'],
         fmt(r['seconds_at_candidate_linear_estimate'])] for r in recs])
    lines+=['','h=1/32 — кандидат для следующей независимой проверки, не сертифицированный шаг датасета. '
        'Для контроля смещения уточнять h в фиксированной точке при том же длинном T и '
        'существенно меньшем шуме разности; общий порядок EM по этим трём задачам не переносится на весь датасет. '
        'Короткий T=4/8 допустим как явно экспериментальный pilot с сохранением tail_bound, '
        'но не как сертифицированная настройка, если граница больше выделенного допуска.',
        'Перед 33 точками сначала проверить несколько заранее выбранных точек, затем оценить бюджет K·M·ceil(T/h). '
        'Точечный 95% CI не даёт одновременного 95% покрытия сетки. Для статистического контроля '
        '33 точек Bonferroni требует большего квантиля, а временная ошибка остаётся отдельной.', '',
        '## Достоверность и ограничения','',
        'Численные выводы и различимость разностей см. [INTERPRETATION.md](INTERPRETATION.md). '
        'Все raw значения, seeds, конфигурации, ошибки, CI, время и стоимость сохранены. '
        'Нет пространственного исследования, строгой границы EM/трапеций на датасете, '
        'массового benchmark, сравнения с Ремизовым или обучения ML.',
        'Проверки протокола/регрессия: [pytest.txt](pytest.txt). '
        'Данные, генератор, solver и прежние результаты сверены побайтово; [verification.json](verification.json).']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    plan,summary,groups,comparisons,recs=analyze(args.out)
    plots(args.out,plan,groups)
    report(args.out,plan,summary,groups,comparisons,recs)


if __name__=='__main__':
    main()
