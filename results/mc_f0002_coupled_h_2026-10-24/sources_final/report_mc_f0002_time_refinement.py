"""Analysis of saved coupled integrals only; never call the numerical kernel."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'mc-convergence-mpl'))
import numpy as np
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]


def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n')


def centered_stats(Z):
    """Anchored means and centered covariance; no difference of squares."""
    Z=np.asarray(Z,dtype=float)
    if Z.ndim!=2 or Z.shape[0]<2 or not np.all(np.isfinite(Z)):
        raise ValueError('finite matrix with at least two independent paths required')
    mean=Z[0]+np.mean(Z-Z[0],axis=0)
    centered=Z-mean
    # Small number of columns. Explicit fsum is also an independent check of
    # the finite BLAS result: this host emitted matmul FP warnings despite
    # agreeing finite results. The original log and scalar recheck are saved.
    covariance=np.array([[math.fsum(float(x)*float(y) for x,y in
        zip(centered[:,i],centered[:,j]))/(Z.shape[0]-1)
        for j in range(Z.shape[1])] for i in range(Z.shape[1])])
    if not np.all(np.isfinite(covariance)):
        raise ValueError('nonfinite centered covariance')
    return mean,covariance


def stage_data(out,stage,verify=False):
    records=[]; samples=[]; means=[]; rows=[]
    plan=json.loads((out/'plan.json').read_text())
    specs={s['attempt_id']:s for s in plan['attempts']}
    for path in sorted((out/'records').glob(f'{stage}_*.result.json')):
        record=json.loads(path.read_text()); spec=record['spec']; r=record['result']
        assert spec==specs[spec['attempt_id']]
        records.append(record)
        if not r['success']:
            continue
        with np.load(path.with_name(path.name.replace('.result.json','.samples.npz')),allow_pickle=False) as saved:
            Z=saved['Z']; hs=saved['hs'].tolist()
        assert Z.shape==(spec['M'],len(hs)) and hs==record['hs']
        mean,cov=centered_stats(Z)
        if verify:
            assert r['preflight']['path_steps']==sum(l['kernel_statistics']['n_path_steps'] for l in r['levels'])
            assert r['preflight']['normal_values_generated_with_replay']==sum(l['normal_values_generated_with_replay'] for l in r['levels'])
            assert r['preflight']['coefficient_calls']==sum(sum(l['kernel_statistics']['calls'].values()) for l in r['levels'])
            assert all(l['config']['T']==plan['T'] and l['n_steps']*l['config']['dt']==plan['T'] for l in r['levels'])
        samples.append(Z); means.append(mean)
        for j,h in enumerate(hs):
            se=math.sqrt(cov[j,j]/spec['M'])
            f=float(mean[j]); y=record['y_exact']
            rows.append(dict(stage=stage,attempt_id=spec['attempt_id'],repeat=spec['repeat'],
                seed=spec['seed'],M=spec['M'],h=h,T=plan['T'],x=plan['x'],
                f=f,y_exact=y,signed_error=f-y,abs_error=abs(f-y),stderr=se,
                ci95_lower=f-1.96*se,ci95_upper=f+1.96*se,
                runtime_seconds=r['levels'][j]['runtime_seconds'],
                path_steps=r['levels'][j]['kernel_statistics']['n_path_steps'],
                normal_values_generated_with_replay=r['levels'][j]['normal_values_generated_with_replay'],
                tail_bound=plan['tail_error_bound'],peak_rss_mib=record['process_peak_rss_mib']))
    return records,samples,np.array(means),rows


def summarize_stage(out,stage):
    records,samples,means,rows=stage_data(out,stage,verify=True)
    successful=[r for r in records if r['result']['success']]
    if len(successful)<2:
        return None,rows
    R=len(successful); M=successful[0]['spec']['M']; hs=successful[0]['hs']; L=len(hs)
    y=successful[0]['y_exact']
    assert all(r['y_exact']==y and r['hs']==hs and r['spec']['M']==M for r in successful)
    mean,seed_cov=centered_stats(means)
    pooled=np.vstack(samples)
    _,path_cov=centered_stats(pooled)
    D=means[:,:-1]-means[:,1:]
    dmean,dcov=centered_stats(D)
    pathD=pooled[:,:-1]-pooled[:,1:]
    _,pathDcov=centered_stats(pathD)
    t95=float(stats.t.ppf(.975,R-1))
    level_q=float(stats.t.ppf(1-.05/(2*L),R-1))
    difference_q=float(stats.t.ppf(1-.05/(2*(L-1)),R-1))
    levels=[]; differences=[]
    for j,h in enumerate(hs):
        se=math.sqrt(seed_cov[j,j]/R)
        half=t95*se
        bias=float(mean[j]-y)
        levels.append(dict(h=h,mean=float(mean[j]),y_exact=y,signed_error=bias,
            abs_mean_error=abs(bias),seed_mean_stderr=se,ci95_half=half,
            ci95_error=[bias-half,bias+half],simultaneous_ci95_half=level_q*se,
            path_sigma=math.sqrt(path_cov[j,j]),
            pooled_path_mean_stderr=math.sqrt(path_cov[j,j]/(R*M)),
            rmse_of_replicates=float(np.sqrt(np.mean((means[:,j]-y)**2))),
            individual_runs_with_abs_error_le_001=int(np.sum(abs(means[:,j]-y)<=.01)),
            mean_individual_ci95_half=float(np.mean([1.96*r['stderr'] for r in rows if r['h']==h])),
            median_runtime_seconds=float(np.median([r['runtime_seconds'] for r in rows if r['h']==h])),
            sum_runtime_seconds=sum(r['runtime_seconds'] for r in rows if r['h']==h),
            path_steps=sum(r['path_steps'] for r in rows if r['h']==h),
            normal_values_generated_with_replay=sum(r['normal_values_generated_with_replay'] for r in rows if r['h']==h)))
    for j in range(L-1):
        se=math.sqrt(dcov[j,j]/R)
        half=t95*se
        independent_var=path_cov[j,j]+path_cov[j+1,j+1]
        paired_var=pathDcov[j,j]
        rho=path_cov[j,j+1]/math.sqrt(path_cov[j,j]*path_cov[j+1,j+1])
        differences.append(dict(coarse_h=hs[j],fine_h=hs[j+1],mean=float(dmean[j]),
            seed_mean_stderr=se,ci95_half=half,
            simultaneous_ci95_half=difference_q*se,
            ci95=[float(dmean[j]-half),float(dmean[j]+half)],
            simultaneous_ci95=[float(dmean[j]-difference_q*se),float(dmean[j]+difference_q*se)],
            pvalue=float(2*stats.t.sf(abs(dmean[j])/se,R-1)) if se else (0. if dmean[j] else 1.),
            path_sigma=math.sqrt(paired_var),correlation=float(rho),
            variance_reduction_vs_independent=float(independent_var/paired_var) if paired_var else None,
            independent_stderr_at_equal_paths=math.sqrt(independent_var/(R*M)),
            paired_stderr_from_paths=math.sqrt(paired_var/(R*M))))
    return dict(stage=stage,repeats=R,paths_per_repeat=M,total_independent_paths=R*M,
        hs=hs,levels=levels,differences=differences,means_by_seed=means.tolist(),
        covariance_seed_means=seed_cov.tolist(),covariance_path_integrals=path_cov.tolist(),
        covariance_paired_seed_differences=dcov.tolist(),
        sum_call_runtime_seconds=sum(r['result']['runtime_seconds'] for r in successful)),rows


def pilot_decision(out,plan):
    summary,_=summarize_stage(out,'pilot')
    rule=plan['decision_rule']
    if summary is None or summary['repeats']!=rule['minimum_pilot_repeats']:
        return dict(proceed=False,confirmation_hs=[],reason='incomplete_pilot')
    d=summary['differences'][0]
    relative=d['ci95_half']/abs(d['mean']) if d['mean'] else None
    proceed=(relative is not None and relative<=rule['maximum_relative_ci95_half'] and
             abs(d['mean'])>d['ci95_half'] and
             d['variance_reduction_vs_independent'] is not None and
             d['variance_reduction_vs_independent']>=rule['minimum_variance_reduction'])
    fine=summary['levels'][-1]
    certified_bias_indicator=abs(fine['signed_error'])+fine['ci95_half']+plan['tail_error_bound']
    need_128=certified_bias_indicator>=.01
    return dict(proceed=bool(proceed),confirmation_hs=([1/32,1/64,1/128] if need_128 else [1/32,1/64]) if proceed else [],
        reason='paired_signal_resolved' if proceed else 'insufficient_paired_resolution',
        paired_mean=d['mean'],paired_ci95_half=d['ci95_half'],relative_ci95_half=relative,
        variance_reduction=d['variance_reduction_vs_independent'],
        fine_bias_plus_ci_and_tail=certified_bias_indicator,need_128=bool(need_128),
        pilot_summary=summary)


def order_diagnostic(stage,seed,B):
    if stage is None or len(stage['levels'])!=3:
        return None
    means=np.asarray(stage['means_by_seed'])
    D=means[:,:-1]-means[:,1:]
    d=np.mean(D,axis=0)
    if np.any(d<=0) or d[0]<=d[1]:
        return dict(identified=False,reason='contrast ratio does not imply positive order')
    rng=np.random.Generator(np.random.PCG64(seed))
    indices=rng.integers(0,len(D),size=(B,len(D)))
    boot=D[indices].mean(axis=1)
    valid=(boot[:,0]>boot[:,1])&(boot[:,1]>0)
    ratios=boot[valid,0]/boot[valid,1]
    p=math.log2(d[0]/d[1])
    residual=d[1]/(2**p-1)
    residuals=boot[valid,1]/(ratios-1)
    h=stage['hs'][-1]
    target_h=h*(.0025/residual)**(1/p)
    suggested_h=2.**math.floor(math.log2(target_h))
    return dict(identified=True,observed_p=p,
        bootstrap_p_ci95=np.quantile(np.log2(ratios),[.025,.975]).tolist(),
        bootstrap_valid_fraction=float(np.mean(valid)),bootstrap_replicates=B,
        finest_bias_power_law_model=float(residual),
        finest_bias_model_ci95=np.quantile(residuals,[.025,.975]).tolist(),
        model_continuum_mean=stage['levels'][-1]['mean']-residual,
        model_h_for_time_bias_0025=suggested_h,
        caveat='Conditional three-level power-law model, seed-vector bootstrap with only 8 replicates. '
               'Neither asymptotic order, residual bound, nor recommended step is certified.')


def save_csv(path,rows):
    if not rows:
        return
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def analyze(out):
    plan=json.loads((out/'plan.json').read_text()); run=json.loads((out/'run_summary.json').read_text())
    pilot,rows=summarize_stage(out,'pilot')
    confirmation,other=summarize_stage(out,'confirmation');rows+=other
    save_csv(out/'attempt_levels.csv',rows)
    write_json(out/'aggregates.json',dict(pilot=pilot,confirmation=confirmation))
    order=order_diagnostic(confirmation,plan['analysis']['bootstrap_seed'],plan['analysis']['bootstrap_replicates'])
    write_json(out/'order_diagnostic.json',order)
    old=json.loads((ROOT/'results/mc_convergence_2026-10-24/aggregates.json').read_text())
    old=next(a for a in old if a['problem_id']==plan['problem_id'] and a['M']==2048 and
        a['h']==1/32 and a['T']==plan['T'])
    old_baseline=dict(mean=old['mean'],signed_error=old['signed_mean_error'],
        ci95_half=old['mean_ci95_half'],repeats=old['repeats'],M=old['M'],
        source='results/mc_convergence_2026-10-24/aggregates.json',pooled_with_new=False)
    rec=None
    planning=confirmation or pilot
    if planning:
        finest=planning['levels'][-1]
        M=math.ceil((1.96*finest['path_sigma']/.005)**2)
        rec=dict(source_stage=planning['stage'],finest_tested_h=finest['h'],pilot_M_for_stat_ci_half_005=M,
            path_steps_at_finest_tested=M*round(plan['T']/finest['h']),
            finest_bias_ci95=finest['ci95_error'],
            finest_simultaneous_bias_ci95=[finest['signed_error']-finest['simultaneous_ci95_half'],
                finest['signed_error']+finest['simultaneous_ci95_half']],
            tested_mean_bias_within_001=abs(finest['signed_error'])+finest['simultaneous_ci95_half']<.01,
            caveat='Only uncertainty of mean bias checked, not probability that a new run is within .01')
        if not confirmation:
            d=pilot['differences'][0]
            new_R,new_M=8,1024
            rec.update(next_small_pair_candidate=dict(repeats=new_R,M_per_seed=new_M,hs=pilot['hs'],
                projected_t95_difference_half=float(stats.t.ppf(.975,new_R-1)*d['path_sigma']/math.sqrt(new_R*new_M)),
                path_steps=new_R*new_M*sum(round(plan['T']/h) for h in pilot['hs']),
                executed=False,caveat='Planning estimate from four pilots, not a new experiment or a precision guarantee'))
        if order and order['identified']:
            rec.update(model_candidate_h=order['model_h_for_time_bias_0025'],
                model_candidate_path_steps=M*round(plan['T']/order['model_h_for_time_bias_0025']))
    write_json(out/'recommendations.json',rec)
    raw=[json.loads(p.read_text()) for p in (out/'records').glob('*.result.json')]
    raw_levels=[level for r in raw for level in r['result']['levels']]
    actual=sum(level['kernel_statistics']['n_path_steps'] for level in raw_levels)
    resource=dict(total_path_steps=actual,
        normal_values_generated_with_replay=sum(level['normal_values_generated_with_replay'] for level in raw_levels),
        level_runtime_seconds=sum(level['runtime_seconds'] for level in raw_levels),
        max_rss_mib=max((r['process_peak_rss_mib'] for r in raw),default=0.),
        includes_partial_failure_counters=True,
        successful_coupled_attempts=sum(a['status']=='success' for a in run['attempts']),
        statuses=run['attempts'],source_baseline=old_baseline,
        no_solver_called=True)
    assert actual<=plan['resources']['max_path_steps']
    write_json(out/'analysis_verification.json',resource)
    return plan,run,pilot,confirmation,order,rec,resource


def plots(out,pilot,confirmation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'figure.dpi':130,'savefig.dpi':160})
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    def save(fig,name):
        fig.tight_layout();fig.savefig(figdir/f'{name}.png');fig.savefig(figdir/f'{name}.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for a,color in ((pilot,'gray'),(confirmation,'C0')):
        if not a:
            continue
        x=[g['h'] for g in a['levels']]
        axes[0].errorbar(x,[g['signed_error'] for g in a['levels']],yerr=[g['ci95_half'] for g in a['levels']],
            fmt='o-',color=color,capsize=4,label=f"{a['stage']}: mean error, t95")
        x=[d['coarse_h'] for d in a['differences']]
        axes[1].errorbar(x,[d['mean'] for d in a['differences']],
            yerr=[d['simultaneous_ci95_half'] for d in a['differences']],fmt='o-',color=color,capsize=4,
            label=f"{a['stage']}: paired difference, simultaneous 95%")
    axes[0].axhspan(-.01,.01,color='gray',alpha=.12,label='Mean bias +/-0.01')
    axes[0].set_ylabel('Mean estimator minus exact y(0)')
    axes[1].set_ylabel('Mean Z(h)-Z(h/2)')
    for ax in axes:
        ax.axhline(0,color='black',lw=.7);ax.set_xscale('log',base=2);ax.set_xlabel('h');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('F0002 at x=0, fixed certified T=63.625; '+
                 ('independent confirmation' if confirmation else 'pilot only; confirmation not run'))
    save(fig,'bias_and_paired_differences')
    a=confirmation or pilot
    if not a:
        return
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    x=np.arange(len(a['differences']))
    axes[0].bar(x-.17,[d['independent_stderr_at_equal_paths'] for d in a['differences']],width=.34,label='Independent levels (estimated)')
    axes[0].bar(x+.17,[d['paired_stderr_from_paths'] for d in a['differences']],width=.34,label='Coupled levels')
    axes[0].set_xticks(x,[f"1/{round(1/d['coarse_h'])} vs 1/{round(1/d['fine_h'])}" for d in a['differences']])
    axes[0].set_ylabel('SE of level difference at equal path count');axes[0].legend(fontsize=8)
    axes[1].plot([g['h'] for g in a['levels']],[g['mean_individual_ci95_half'] for g in a['levels']],'o-')
    axes[1].set_xscale('log',base=2);axes[1].set(xlabel='h',ylabel='Mean individual 95% CI half-width')
    axes[1].set_title('Coupling reduces difference noise;\nmarginal uncertainty remains',fontsize=10)
    for ax in axes:ax.grid(alpha=.2)
    save(fig,'variance_reduction')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    hs=[g['h'] for g in a['levels']]
    axes[0].plot(hs,[g['path_steps']/1e6 for g in a['levels']],'o-',label='Path updates')
    axes[0].plot(hs,[g['normal_values_generated_with_replay']/1e6 for g in a['levels']],'s--',label='Normals generated incl. replay')
    axes[0].set_ylabel(f"Million operations across {a['repeats']} seeds");axes[0].legend(fontsize=8)
    axes[1].plot(hs,[g['median_runtime_seconds'] for g in a['levels']],'o-');axes[1].set_ylabel('Median kernel runtime per replicate, s')
    for ax in axes:ax.set_xscale('log',base=2);ax.set_xlabel('h');ax.grid(alpha=.2)
    save(fig,'cost')


def table(header,rows):
    return ['| '+' | '.join(header)+' |','| '+' | '.join(['---']*len(header))+' |']+[
        '| '+' | '.join(str(v) for v in row)+' |' for row in rows]


def report(out,plan,run,pilot,confirmation,order,rec,resource):
    fmt=lambda x:f'{x:.7g}'
    decision=json.loads((out/'decision.json').read_text())
    lines=['# F0002: связанное временное уточнение — 2026-10-24','',
        'Исследуется только AMV1_F0002_S00 в x=0 при фиксированном T=63.625. '
        f"Сертификат G={plan['forcing_bound']['abs_upper_bound']:.10g}, минимальный T*={plan['certified_T']:.10g}, "
        f"непрерывный хвост ≤{plan['tail_error_bound']:.10g}.",
        'Использован без изменений `_path_contributions` из mc_solver.py: старое положение '
        'в Эйлере–Маруяме, трапеции потенциала и forcing, c_R−lambda в весе. '
        'Диагностический CoupledNormals передаёт нормальные величины в то же ядро. '
        'Coarse normal = (z1+…+zq)/sqrt(q), q=1,2,4, где zj независимы N(0,1). '
        'Блоки не пересекаются, поэтому все маргинальные уровни имеют исходный закон EM, '
        'а sqrt(h)·normal совпадает с суммой мелких броуновских приращений до округления.',
        'Поток PCG64 воспроизводится на каждом уровне: добавочное генерирование нормальных '
        'чисел учтено в стоимости, память O(M), матрица всех временных приращений не хранится. '
        'Сохранены только интегралы всех отдельных путей, их парность не теряется.', '',
        '## План и выполнение','',
        'План заморожен до MC: [plan.json](plan.json). Пилот — 4 seed×1024 пути '
        'на паре 1/32,1/64. Подтверждение — независимые 8 seed×2048 путей; '
        '1/128 добавляется, если пилот не позволяет поместить смещение 1/64 в ±0.01. '
        'Переход разрешён только при относительной t95-полуширине парной разности ≤0.5 '
        'и оценённом снижении дисперсии ≥2. Пилот не объединяется с подтверждением.',
        f"Решение: {decision['reason']}; подтверждение {'запущено' if decision['proceed'] else 'не запускалось'}. "
        f"Выбранные h: {decision['confirmation_hs']}.",
        f"Максимальный предварительный бюджет: {plan['resources']['planned_maximum_path_steps']:,} обновлений, "
        f"{plan['resources']['planned_normal_values_generated_with_replay']:,} нормальных значений с replay; "
        '300 с / 384 МиБ / один worker / BLAS=1; внутренний предел 20 с на связанный повтор.',
        f"Реально: {resource['successful_coupled_attempts']} успешных связанных попыток; "
        f"not_run={sum(a['status']=='not_run' for a in run['attempts'])}, "
        f"отказов={sum(a['status'] not in ('success','not_run') for a in run['attempts'])}. "
        f"{resource['total_path_steps']:,} обновлений, {resource['normal_values_generated_with_replay']:,} "
        f"нормальных значений с replay; сумма времён уровней {resource['level_runtime_seconds']:.3f} с, "
        f"общее время {run['wall_seconds']:.3f} с, пик RSS {resource['max_rss_mib']:.2f} МиБ.",
        'Все статусы: [run_summary.json](run_summary.json); все ответы и O(M)-интегралы путей: '
        '[records/](records/); телеметрия каждого выполненного этапа — *.rss.jsonl. '
        '[Окружение](environment.json), [ошибки и CI каждого уровня CSV](attempt_levels.csv), '
        '[полные агрегаты и ковариации](aggregates.json).','',
        '## Смещение относительно exact','',
        'Каждая строка — среднее независимых повторов. ±t95 — неопределённость оценки '
        'μ(h,T)−y_exact по seed; она не является границей ошибки одного запуска. '
        'В aggregates.json дополнительно сохранены одновременные Bonferroni-интервалы '
        'по уровням каждой выполненной серии и интервалы отдельных запусков. Непрерывный хвост '
        'мал, но строгой границы временного смещения нет.','']
    levels=[]
    for s in (pilot,confirmation):
        if s:
            for g in s['levels']:
                levels.append([s['stage'],s['repeats'],s['paths_per_repeat'],fmt(g['h']),fmt(g['mean']),
                    f"{fmt(g['signed_error'])} ± {fmt(g['ci95_half'])}",fmt(g['rmse_of_replicates']),
                    fmt(g['mean_individual_ci95_half']),g['individual_runs_with_abs_error_le_001']])
    lines+=table(['Серия','seed','M/seed','h','f среднее','Смещение ±t95','RMSE повторов','Средняя CI-половина запуска','Ошибок≤0.01'],levels)
    old=resource['source_baseline']
    lines+=['',f"Прежний независимый baseline того же T,h=1/32,M=2048: "
        f"{old['signed_error']:.7g} ± {old['ci95_half']:.7g}, 8 seed. Он не объединён с новой серией.",
        '', '## Парные разности и уменьшение шума','',
        'Разность считается непосредственно по двум интегралам одной броуновской траектории. '
        'Индикатор смещения не вычитается из ошибки и не превращается в сертификат. '
        'Статистические выводы относятся только к выполненной серии; независимое '
        'подтверждение не подменяется пилотом. Для нескольких разностей предусмотрен '
        'Bonferroni95. Var reduction = (Var Z_h+Var Z_h/2)/Var(Z_h−Z_h/2), '
        'то есть оценка выигрыша относительно независимых уровней при одинаковом числе путей.','']
    diffs=[]
    for s in (pilot,confirmation):
        if s:
            for d in s['differences']:
                diffs.append([s['stage'],fmt(d['coarse_h']),fmt(d['fine_h']),
                    f"{fmt(d['mean'])} ± {fmt(d['ci95_half'])}",fmt(d['simultaneous_ci95_half']),
                    fmt(d['correlation']),fmt(d['variance_reduction_vs_independent'])])
    lines+=table(['Серия','h','h/2','Разность ±t95','Одновр.95 половина','Корреляция','Var reduction'],diffs)
    lines+=['','![Смещение и парные разности](figures/bias_and_paired_differences.png)','']
    if confirmation:
        lines+=['![Уменьшение дисперсии](figures/variance_reduction.png)','',
            'Связывание уменьшает неопределённость **разности уровней**, но не устраняет '
            'неопределённость абсолютного значения каждого уровня и его смещения к exact.', '',
            '## Вычислительная стоимость подтверждения','']
        lines+=table(['h','Обновлений, все seed','Нормальных значений с replay','Медиана секунд/seed'],[
            [fmt(g['h']),g['path_steps'],g['normal_values_generated_with_replay'],fmt(g['median_runtime_seconds'])]
            for g in confirmation['levels']])
        lines+=['','![Стоимость](figures/cost.png)','',
            '## Условная оценка порядка','',
            '[order_diagnostic.json](order_diagnostic.json) содержит отношение двух парных '
            'разностей и bootstrap независимых seed-векторов (10000 повторов). '
            'Совместное ресэмплирование сохраняет ковариацию общей средней точки. '
            'Три уровня не доказывают асимптотику или применимость степенной модели.']
        if order and order['identified']:
            lines += [f"Наблюдаемый p={order['observed_p']:.6g}; bootstrap95 "
                f"{order['bootstrap_p_ci95']}, допустимая доля {order['bootstrap_valid_fraction']:.4f}. "
                f"Модельное остаточное смещение на 1/128: {order['finest_bias_power_law_model']:.7g}, "
                f"bootstrap95 {order['finest_bias_model_ci95']}. Это гипотеза модели, не измеренная граница."]
        lines+=['','## Точность 0.01 и практический шаг','',
            f"На самом мелком испытанном h={rec['finest_tested_h']} одновременный 95% интервал "
            f"для среднего смещения: {rec['finest_simultaneous_bias_ci95']}. "
            f"Весь интервал внутри ±0.01: {rec['tested_mean_bias_within_001']}.",
            f"Оценка M для одной статистической CI-полуширины 0.005: "
            f"{rec['pilot_M_for_stat_ci_half_005']:,}, по дисперсии самого мелкого уровня. "
            f"Это {rec['path_steps_at_finest_tested']:,} обновлений на точку при том же T,h; "
            'этот большой расчёт не запускался и не решает вопрос временного смещения.']
        if order and order['identified']:
            lines += [f"Степенная экстраполяция к временному смещению 0.0025 предлагает "
                f"h={rec['model_candidate_h']} и {rec['model_candidate_path_steps']:,} "
                'обновлений при этом M. Это предварительный кандидат для отдельного контроля; '
                'он не испытан, производственная настройка по этой модели не выбирается.']
    elif pilot:
        g=pilot['levels'][-1]
        lines+=['![Выигрыш от связывания](figures/variance_reduction.png)','',
            '## Остановка, стоимость и точность 0.01','',
            f"Парный t95-интервал исключает ноль, однако относительная полуширина "
            f"{decision['relative_ci95_half']:.6g} превышает заранее зафиксированный порог "
            f"{plan['decision_rule']['maximum_relative_ci95_half']}. Это остановка по более строгому "
            'критерию точности разности, а не заявление об отсутствии обнаружимого эффекта. '
            'Порог после просмотра данных не менялся; 8 подтверждающих попыток отмечены not_run, '
            'h=1/128 не запускался. Программных или ресурсных отказов нет.', '',
            f"На h=1/64 среднее смещение {g['signed_error']:.8g}, t95 {g['ci95_error']}. "
            f"Одновременный интервал по двум уровням: {rec['finest_simultaneous_bias_ci95']}. "
            'Он включает как значения внутри ±0.01, так и значения выше 0.01. '
            'Практический шаг для достижения цели пока выбрать нельзя. '
            'Два уровня также не позволяют оценить временной порядок. '
            'Более мелкий шаг остаётся гипотезой для дальнейшей проверки, а не рекомендацией с гарантией.',
            f"Для одной статистической полуширины 0.005 на h=1/64 пилотная оценка "
            f"M={rec['pilot_M_for_stat_ci_half_005']:,}, стоимость "
            f"{rec['path_steps_at_finest_tested']:,} обновлений на точку. Большой расчёт не запускался. "
            'Он сам по себе не ограничивает смещение схемы.', '',
            'Более экономичный следующий контроль — сначала точнее измерить ту же пару на '
            'новых независимых seed, сохранив T: ориентир 8×1024 путей, '
            f"{rec['next_small_pair_candidate']['path_steps']:,} обновлений; "
            f"прогноз t95-полуширины разности {rec['next_small_pair_candidate']['projected_t95_difference_half']:.6g}. "
            'Это только оценка для нового плана, не выполненная серия и не гарантия. '
            'Переход к трём уровням должен сохранять парность и отдельный предварительный бюджет.','']
        lines+=table(['h','Обновлений, все seed','Нормальных значений с replay','Медиана секунд/seed'],[
            [fmt(g['h']),g['path_steps'],g['normal_values_generated_with_replay'],fmt(g['median_runtime_seconds'])]
            for g in pilot['levels']])
        lines+=['','![Стоимость пилота](figures/cost.png)']
    lines+=['','## Достоверность, ограничения и проверки','',
        'Содержательные выводы, сопоставление с прежней серией и следующий шаг: '
        '[INTERPRETATION.md](INTERPRETATION.md). Все попытки, включая не выполненные '
        'при остановке, остаются в протоколе. Пилот не служит подтверждающей выборкой. '
        'Точечные интервалы предполагают приблизительную нормальность средних; '
        'покрытие и вероятность достижения точности не оценены по столь малому числу seed.',
        'Проверки coupling и регрессия: [pytest.txt](pytest.txt). '
        'Независимый пересчёт статистики: [decision_recheck.json](decision_recheck.json); '
        'предупреждения первого матричного расчёта сохранены в driver.log. '
        'Сохранность: [verification.json](verification.json), [final_verification.json](final_verification.json). '
        'Старые данные, результаты, production MC, Ремизов и генератор не менялись. '
        'Нет пространственного benchmark и ML.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    plan,run,pilot,confirmation,order,rec,resource=analyze(args.out)
    plots(args.out,pilot,confirmation)
    report(args.out,plan,run,pilot,confirmation,order,rec,resource)


if __name__=='__main__':
    main()
