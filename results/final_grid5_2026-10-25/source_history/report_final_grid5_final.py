"""Saved-only final-five analysis: no numerical solves and no ML fitting."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/ode-final5-mpl')


def table(rows):
    names=list(rows[0]);return '| '+' | '.join(names)+' |\n| '+' | '.join('---' for _ in names)+' |\n'+''.join('| '+' | '.join(str(r[n]) for n in names)+' |\n' for r in rows)


def report(out):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from diagnostics.run_chernov_batch import read_json as read,write_json as write,digest
    from diagnostics.grid100_statistics import aggregate_mc
    from diagnostics.final_grid5_stats import reliability
    from diagnostics.chernov_batch_control import lock_is_held
    plan=read(out/'plan.json');ph=digest(out/'plan.json');bykey={j['key']:j for j in plan['jobs']}
    records={p.stem:read(p) for p in (out/'records').glob('*.json')}
    for k,r in records.items():
        if r['job']!=bykey[k] or r['plan_sha256']!=ph:raise ValueError('result identity/profile mismatch')
    (out/'analysis').mkdir(exist_ok=True);(out/'plots').mkdir(exist_ok=True)
    attempts=[];rows=[];stats={};identities=0
    for j in plan['jobs']:
        r=records.get(j['key']);z=(r.get('result') or {}) if r else {}
        attempts.append(dict(key=j['key'],problem_id=j['problem_id'],method=j['method'],seed=j['config'].get('seed'),
            status=r['status'] if r else 'pending',E_inf=r.get('E_inf') if r else None,
            accuracy_met=r.get('accuracy_met') if r else None,solver_seconds=z.get('runtime_seconds'),
            task_wall_seconds=r.get('task_wall_seconds') if r else None,peak_rss_mib=r.get('peak_rss_mib') if r else None,
            path_steps=z.get('n_path_steps'),operators=z.get('n_operator_applications'),T=z.get('T'),N=z.get('n_steps'),
            config=json.dumps(j['config'],sort_keys=True)))
    for task in plan['tasks']:
        pid=task['problem_id'];base=read(out/'baseline'/f'{pid}_R128.json');r=records.get(pid+'_R256')
        if digest(out/'baseline'/f'{pid}_R128.json')!=task['baseline_sha256'] or digest(ROOT/task['baseline_source'])!=task['baseline_sha256']:raise ValueError('baseline changed')
        oldcfg=base['job']['config'];newcfg=task['R_config']
        if {k for k in oldcfg if oldcfg[k]!=newcfg[k]}!={'n'}:raise ValueError('R changed beyond n')
        row=dict(problem_id=pid,family_id=task['family_id'],split='train',expression_hash=task['expression_hash'],T=task['T'],
            R128_error=base['E_inf'],R128_seconds=base['result']['runtime_seconds'],R128_pass=base['accuracy_met'],
            R256_error=None,R256_seconds=None,R256_pass=None,R256_status=r['status'] if r else 'pending',
            MC_completed=0,MC_successes=0,MC_errors=[],MC_seed_seconds=[],MC_reliability_status='insufficient_or_resource_censored',
            MC_only_observed_trials=0,MC_p95_confirmed=False,time_error_bound_available=False,ML_winner='unconfirmed')
        if r and r['status']=='completed':
            row.update(R256_error=r['E_inf'],R256_seconds=r['result']['runtime_seconds'],R256_pass=r['accuracy_met'],
                R_improved=r['E_inf']<base['E_inf'],R_error_ratio=base['E_inf']/r['E_inf'] if r['E_inf'] else None)
        ms=[];trial_bands=[]
        for i in range(4):
            rec=records.get(pid+f'_MC{i}')
            if rec and rec['status']=='completed':
                z=rec['result'];expected=task['MC_base_config']
                if z['x']!=plan['target_x'] or z['M']!=4096 or z['T']!=task['T'] or z['forcing_abs_upper_bound']!=task['G'] or not z['tail_tolerance_met']:raise ValueError('MC profile/certificate mismatch')
                if z['n_path_steps']!=task['mc_path_steps_per_trial'] or z['n_normal_draws']!=task['mc_path_steps_per_trial']:raise ValueError('work mismatch')
                np.testing.assert_array_equal(rec['y_exact'],base['y_exact'])
                if rec['expression_hash']!=task['expression_hash'] or rec['split']!='train':raise ValueError('equation identity mismatch')
                error=float(np.max(abs(np.asarray(z['f'])-np.asarray(rec['y_exact']))))
                if error!=rec['E_inf']:raise ValueError('stored error mismatch')
                ms.append(rec);trial_bands.append(aggregate_mc([rec],alpha=.01,looks=5));identities+=1
        row.update(MC_completed=len(ms),MC_successes=sum(x['accuracy_met'] for x in ms),
            MC_errors=[x['E_inf'] for x in ms],MC_seed_seconds=[x['result']['runtime_seconds'] for x in ms],
            MC_total_solver_seconds=sum(rec['result']['runtime_seconds'] for rec in records.values() if rec['job']['problem_id']==pid and rec['job']['method']=='MC' and rec.get('result')),
            MC_total_task_seconds=sum(rec['task_wall_seconds'] for rec in records.values() if rec['job']['problem_id']==pid and rec['job']['method']=='MC'))
        if ms:
            row.update(MC_error_min=min(row['MC_errors']),MC_error_max=max(row['MC_errors']),MC_median_seconds=float(np.median(row['MC_seed_seconds'])))
        if len(ms)==4:
            pooled=aggregate_mc(ms,alpha=.01,looks=5);p=reliability(row['MC_successes'])
            oldms=[read(out/'baseline'/f'{pid}_old_MC{i}.json') for i in range(4)]
            oldmean=aggregate_mc(oldms)
            row.update(MC_reliability_status='reject_p95' if p['reject_p95'] else 'p95_unconfirmed',
                MC_probability_family95_lower=p['exact_family95_probability_interval'][0],MC_probability_family95_upper=p['exact_family95_probability_interval'][1],
                MC_p95_pvalue=p['pvalue'],MC_p95_confirmed=p['confirm_p95'],
                MC_mean16384_error=pooled['E_inf'],MC_mean_error_lower=pooled['error_mean_interval'][0],MC_mean_error_upper=pooled['error_mean_interval'][1],
                MC_mean_statistical_accuracy=pooled['statistical_accuracy'],MC_mean_hoeffding_half=pooled['hoeffding_half'],
                old_MC_mean4096_error=oldmean['E_inf'],old_MC_mean4096_seconds=sum(x['result']['runtime_seconds'] for x in oldms),
                observed_stderr_reduction_1024_to4096=float(np.mean(oldms[0]['result']['stderr'])/np.mean([x['result']['stderr'] for x in ms])),
                MC_only_observed_trials=sum(x['accuracy_met'] for x in ms) if row['R256_pass'] is False else 0,
                both_pass_observed_trials=sum(x['accuracy_met'] for x in ms) if row['R256_pass'] is True else 0,
                MC_faster_success_trials=sum(x['accuracy_met'] and x['result']['runtime_seconds']<row['R256_seconds'] for x in ms) if row['R256_pass'] is True else 0)
            stats[pid]=dict(fresh_trials=trial_bands,fresh_pooled16384=pooled,success_probability=p,historical_mean4096=oldmean,
                historic_samples_in_confirmation=False)
        else:stats[pid]=dict(fresh_trials=trial_bands,pooled_unavailable='requires all4 predeclared fresh trials')
        if r and r['status']=='completed':
            if r['result']['T']!=task['T'] or r['result']['n']!=256 or r['result']['time_nodes']!=513:raise ValueError('R actual parameters differ')
            np.testing.assert_array_equal(r['y_exact'],base['y_exact'])
            np.testing.assert_array_equal(r['result']['x'],base['result']['x'])
            if r['expression_hash']!=task['expression_hash'] or r['result']['n_operator_applications']!=256*512:raise ValueError('R identity/work mismatch')
            if float(np.max(abs(np.asarray(r['result']['f'])-base['y_exact'])))!=r['E_inf']:raise ValueError('R metric mismatch')
            identities+=1
        rows.append(row)
    df=pd.DataFrame(rows);att=pd.DataFrame(attempts)
    df.to_csv(out/'analysis/comparison.csv',index=False);att.to_csv(out/'analysis/attempts.csv',index=False)
    write(out/'analysis/statistics.json',stats)
    sessions=[read(p) for p in (out/'sessions').glob('*.json') if read(p)['new_terminal'] or read(p)['reason']]
    wall=sum(s['wall_seconds'] for s in sessions)
    calendar=max(datetime.fromisoformat(s['finished_utc']).timestamp() for s in sessions)-read(out/'budget.json')['started_epoch'] if sessions else 0
    interruptions=[read(p).get('reason') for p in (out/'control').rglob('*.guardian.json') if p.parent.name in {s['session'] for s in sessions if s['reason']}]
    summary=dict(equations=5,terminal=len(records),planned=25,pending=25-len(records),statuses=att.status.value_counts().to_dict(),
        R128_pass=int(df.R128_pass.sum()),R256_pass=int(df.R256_pass.fillna(False).sum()),R_improved=int(df.get('R_improved',pd.Series(dtype=bool)).fillna(False).sum()),
        MC_fresh_successes=int(df.MC_successes.sum()),MC_fresh_completed=int(df.MC_completed.sum()),
        MC_only_observed_trials=int(df.MC_only_observed_trials.sum()),MC_p95_confirmed=int(df.MC_p95_confirmed.sum()),
        MC_faster_observed_success_trials=int(df.get('MC_faster_success_trials',pd.Series(dtype=float)).sum()),
        active_session_seconds=wall,calendar_seconds=calendar,solver_seconds=float(att.solver_seconds.sum()),
        R_solver_seconds=float(att.loc[att.method=='R','solver_seconds'].sum()),MC_solver_seconds=float(att.loc[att.method=='MC','solver_seconds'].sum()),
        peak_rss_mib=float(att.peak_rss_mib.max()),guardian_interruptions=interruptions,
        targeted_train_subset=True,time_discretization_bound=None,ML_training_runs=0,additional900_run=False,
        source_and_metric_checks=identities)
    write(out/'analysis/summary.json',summary)
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained');ix=np.arange(5);names=[r['problem_id'][5:] for r in rows]
    axs[0].plot(ix,df.R128_error,'o-',label='R n128 saved');axs[0].plot(ix,df.R256_error,'s-',label='R n256 new')
    for i,row in enumerate(rows):
        if row['MC_errors']:axs[0].scatter([i]*len(row['MC_errors']),row['MC_errors'],marker='x',color='C2',label='fresh MC4096' if i==0 else None)
    axs[0].axhline(.01,color='black',ls='--');axs[0].set(yscale='log',ylabel='Observed E_inf on33 nodes',xticks=ix,xticklabels=names);axs[0].tick_params(axis='x',rotation=25);axs[0].legend()
    width=.25;axs[1].bar(ix-width,df.R256_seconds,width,label='R n256 one solve');axs[1].bar(ix,df.get('MC_median_seconds'),width,label='MC4096 median one seed');axs[1].bar(ix+width,df.MC_total_solver_seconds,width,label='MC all4 seeds')
    axs[1].set(yscale='log',ylabel='Solver seconds',xticks=ix,xticklabels=names);axs[1].tick_params(axis='x',rotation=25);axs[1].legend()
    fig.savefig(out/'plots/accuracy_cost.png',dpi=170);plt.close(fig)
    tab=[dict(ID=r['problem_id'][5:],R128=f"{r['R128_error']:.6g}",R256='—' if r['R256_error'] is None else f"{r['R256_error']:.6g}",
        MC_success=f"{r['MC_successes']}/{r['MC_completed']}",MC_E_range='—' if not r['MC_errors'] else f"{min(r['MC_errors']):.6g}–{max(r['MC_errors']):.6g}",
        R256_s='—' if r['R256_seconds'] is None else f"{r['R256_seconds']:.2f}",MC_all_s=f"{r['MC_total_solver_seconds']:.2f}") for r in rows]
    text=f'''# Заключительная диагностика пяти train-уравнений

Выполнено **{len(records)}/25** терминальных вызовов, pending **{25-len(records)}**.
Статусы: `{summary['statuses']}`. Никаких дополнительных уравнений, ML-fit или
подстроек параметров по первым попыткам. Источники и конфигурации проверены
против неизменного proposal. Baseline n128 скопирован побайтово, не пересчитан.

## Точность и полная стоимость

{table(tab)}

R: ChernovSolver.solve,n256,Q513,quadratic,dx.0125,padding12,cubic/zero.
От baseline отличается только n128→256. MC: существующий Numba,M4096 на точку,
chunk1024,h1/128,четыре новых seed202610250440–443. Общая сетка33 узла[-1,1],
общий аналитический T по G,tail_eps1e-4; T одинаков у старого и нового R и MC.
Exact-метрики вычислены после возврата solver, параметры не менялись.

R: успехов {summary['R128_pass']}/5→{summary['R256_pass']}/5; E∞ уменьшилась
в {summary['R_improved']}/5 случаях. Два n не устанавливают порядок сходимости
или строгую границу остатка; другие ошибки схемы остались фиксированными.
MC: наблюдаемых успехов **{summary['MC_fresh_successes']}/{summary['MC_fresh_completed']}**.
MC-only наблюдаемых trial-успехов при неуспехе R256: **{summary['MC_only_observed_trials']}**.
Наблюдаемых точных MC-trial, быстрее также точного R256:
**{summary['MC_faster_observed_success_trials']}**. Это отдельно от подтверждения
надёжности; совместный успех с меньшим временем тоже мог бы быть полезен
для выбора, но четыре испытания не доказывают p(success)>=.95.

Активный wall **{wall:.2f} с /{wall/60:.2f} мин**, календарный **{calendar:.2f} с**,
известный solver wall **{summary['solver_seconds']:.2f} с**:
R **{summary['R_solver_seconds']:.2f} с**, все повторения MC **{summary['MC_solver_seconds']:.2f} с**.
RSS peak **{summary['peak_rss_mib']:.2f} МиБ**. Один worker,512 МиБ, абсолютный
бюджет1200 с, guardian/owner lease300 с; deadline при resume не продлевается.
Прерывания guardian: `{interruptions}`. Каждый результат атомарный immutable;
ресурсный или прерванный статус не является математической неточностью.

![Точность и стоимость](plots/accuracy_cost.png)

## Надёжность и временная ошибка

Для p=вероятность полного33-точечного успеха одного4096-path trial сохранены
точные биномиальные Clopper–Pearson интервалы с семейной поправкой на5 задач
(два хвоста .005). Прежние отобранные seed не включены в подтверждающие четыре.
Проверка H0:p>=.95 — exact binomial при alpha.01 на задачу. Три/четыре провала
из4 отвергают H0; даже4/4 не подтверждают p>=.95. Уверенность в p95 не
получена ни из наблюдаемого успеха, ни из близости одного среднего к exact.

Статистические bands дискретного усечённого среднего: Bonferroni на5 задач ×
5 оценок (4 trial и один pooled) ×33 точки при family alpha=.05.
Нормальные bands асимптотические, дополнительно сохранены консервативные
Hoeffding bands и seed-t диагностика. Raw MCResult CI95 остаются точечными и
не используются как совместный33-точечный CI. Pooled variance включает
межseed-компонент. Mean всех четырёх свежих trial — **16384 пути на точку**,
отдельный диагностический агрегат, а не один4096-path run и не бесплатный ответ.
Если все4 не завершены, такой агрегат не строится. [Все оценки](analysis/statistics.json).

Ни bands, ни аналитический хвост не ограничивают временное смещение EM и
трапеций. Оценка error дискретного среднего относительно exact позволяет
диагностировать фиксированное смещение, но не даёт строгой h-error bound.
У R отдельно не ограничены конечное n, пространство, окно и квадратура.
Здесь не исследовались h или интерполяция и схемы не корректировались.

## Область вывода

Пять train-случаев намеренно выбраны предыдущим исследованием: два удачных
MC-агрегата, два пограничных R-неуспеха и один контроль смещения. Это не
репрезентативная оценка production1000. Отсутствие преимущества MC на них
не является универсальным математическим утверждением о методе.
Новые seed независимы от прежних; baseline и исторические результаты сохранены.

Окончательные решения по900 задачам, увеличению M, R n256 и ML-меткам —
[INTERPRETATION.md](INTERPRETATION.md), на основании завершённых чисел.
Массовые расчёты900, дополнительные эксперименты и новое ML не запускались.

Новые файлы: diagnostics/final_grid5.py, diagnostics/final_grid5_stats.py,
diagnostics/report_final_grid5.py и tests/test_final_grid5.py. Численное ядро,
генератор, данные и старые результаты не редактировались; статус обновляется
в PROJECT_STATUS.md. [План](plan.json), [таблица](analysis/comparison.csv),
[попытки](analysis/attempts.csv), [проверки](verification.json).
'''
    (out/'REPORT.md').write_text(text)
    # Independent provenance and result integrity checks, without recomputation.
    before=read(out/'protected_before.json');changes=[n for n,h in before.items() if not (ROOT/n).is_file() or digest(ROOT/n)!=h]
    allowed=[]
    if (out/'analysis_source_history.json').exists():
        history=read(out/'analysis_source_history.json')
        if (history['path']=='diagnostics/report_final_grid5.py' and history['original_sha256']==before[history['path']]
            and digest(out/history['backup'])==history['original_sha256'] and digest(ROOT/history['path'])==history['final_sha256']):
            allowed=[history['path']]
    unexpected=[n for n in changes if n not in allowed]
    if unexpected:raise ValueError(f'protected files changed without preserved authorized analysis revision: {unexpected}')
    for n,h in plan['protected_sha256'].items():
        if digest(ROOT/n)!=h:raise ValueError('frozen source/data changed')
    for r in records.values():
        if r['status']=='completed' and r['task_wall_seconds']>r['job']['watchdog_seconds']:raise ValueError('time cap violated')
        if r['status']=='completed' and r['peak_rss_mib']>512:raise ValueError('memory cap violated')
    guardians=[read(p) for p in (out/'control').rglob('*.guardian.json')]
    if not all(g['finished'] and g.get('cleanup_success') for g in guardians):raise ValueError('worker cleanup incomplete')
    locks={n:lock_is_held(ROOT/'results'/n) for n in ('.chernov_batch_compute.lock','.chernov_batch_controller.lock')}
    if any(locks.values()) or not read(out/'active.json')['finished']:raise ValueError('live compute remains')
    write(out/'verification.json',dict(protected_files_checked=len(before),historical_or_numeric_files_changed=unexpected,
        authorized_new_analysis_source_revision=allowed,protected_files_changed=changes,
        terminal=len(records),pending=25-len(records),baseline_byte_copies=True,new_metric_identity_checks=identities,
        jobs_frozen_no_adaptation=True,fresh_seeds_only_in_confirmation=True,all_guardians_finished=True,locks_held=locks,
        ML_runs=0,additional900_equations=0,mathematical_schemes_changed=False))
    print(json.dumps(dict(summary=summary,table=tab),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);report(ap.parse_args().out.resolve())
