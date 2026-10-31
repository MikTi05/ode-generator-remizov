"""Report SAVED sentinel P1 records, fixed gates and operation-scaled costs.

This module never calls a solver or fits a model. Single-call outcomes remain
primary; mean4 is an explicitly separate diagnostic estimator.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvread(p):
    with Path(p).open(newline='') as stream:return list(csv.DictReader(stream))
def csvwrite(p,rows):
    with Path(p).open('x',newline='') as stream:
        w=csv.DictWriter(stream,list(rows[0]));w.writeheader();w.writerows(rows)
def write(p,value):
    with Path(p).open('x') as stream:stream.write(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')


def summary(values):
    import numpy as np
    a=np.asarray(values,dtype=float)
    if not len(a):return dict(n=0,min=None,median=None,max=None,mean=None,std=None,iqr=None,mad=None)
    med=float(np.median(a))
    return dict(n=len(a),min=float(np.min(a)),median=med,max=float(np.max(a)),mean=float(np.mean(a)),
        std=float(np.std(a,ddof=1)) if len(a)>1 else None,
        iqr=float(np.quantile(a,.75)-np.quantile(a,.25)),mad=float(np.median(abs(a-med))))


def choose_profile(profiles,gates,complete):
    """Exactly the predeclared P1 gates, no post-result threshold changes."""
    result={};base=profiles['MC0']
    for code,r in profiles.items():
        reasons=[];t=r['wall']['median'];e=r['error']['median']
        ratio=None if t is None or base['wall']['median'] is None else t/base['wall']['median']
        eratio=None if e is None or base['error']['median'] in (None,0.) else e/base['error']['median']
        if not complete:reasons.append('stage1_incomplete')
        if r['failures']>gates['MC_max_operational_failures_per20']:reasons.append('operational_failures_gt1')
        if t is None:reasons.append('missing_timing')
        if code!='MC0':
            if ratio is None:reasons.append('unknown_cost_ratio')
            elif ratio>gates['MC_expensive_cost_ratio_cap']:reasons.append('cost_ratio_gt8')
            if r['accuracy_pass']==0:
                if e is None or base['error']['median'] is None:reasons.append('unknown_futility_error')
                elif e>=gates['MC_expensive_zero_success_median_error_ratio_cutoff']*base['error']['median']:
                    reasons.append('zero_success_and_error_ratio_ge_0.8')
        result[code]=dict(eligible=not reasons,reasons=reasons,cost_ratio=ratio,median_error_ratio=eratio)
    if not complete:return dict(selected=None,status='incomplete_no_selection',profiles=result)
    if all(r['accuracy_pass']==0 for r in profiles.values()):
        return dict(selected=None,status='all_zero_stop_before_stage2',profiles=result)
    eligible=[c for c in profiles if result[c]['eligible']]
    if not eligible:return dict(selected=None,status='no_eligible_profile',profiles=result)
    successes=max(profiles[c]['accuracy_pass'] for c in eligible)
    candidates=[c for c in eligible if profiles[c]['accuracy_pass']==successes]
    fastest=min(candidates,key=lambda c:profiles[c]['wall']['median'])
    near=[c for c in candidates if profiles[c]['wall']['median']<=profiles[fastest]['wall']['median']/.9]
    selected=min(near,key=lambda c:int(c[-1]))
    return dict(selected=selected,status='candidate_for_separately_authorized_stage2',profiles=result,
        tie_priority_applied=len(near)>1,maximum_observed_successes=successes)


def report(out):
    import numpy as np
    from scipy.stats import beta
    plan=read(out/'plan.json');ph=sha(out/'plan.json');jobs={j['key']:j for j in plan['jobs']}
    records={p.stem:read(p) for p in (out/'records').glob('*.json')}
    assert set(records)<=set(jobs)
    attempts=[];groups={};raw_checks=0
    for key,j in jobs.items():
        r=records.get(key);z=(r or {}).get('result') or {};status=(r or {}).get('status','not_run')
        row=dict(key=key,equation_id=j['equation_id'],expression_hash=j['expression_hash'],family_id=j['family_id'],
            split='train',profile=j['profile'],episode=j['episode'],M=j['config']['M'],h_requested=j['config']['dt'],
            seed=j['config']['seed'],status=status,accuracy_met=(r or {}).get('accuracy_met'),
            E_grid=(r or {}).get('E_grid'),method_wall_seconds=(r or {}).get('method_wall_seconds'),
            solver_wall_seconds=z.get('runtime_seconds'),AST_G_preparation_seconds=z.get('ast_preparation_seconds'),
            JIT_in_call_seconds=z.get('compilation_seconds'),path_updates=z.get('n_path_steps'),
            peak_rss_mib=(r or {}).get('peak_rss_mib'),source=str((out/'records'/f'{key}.json').relative_to(ROOT)) if r else None)
        if r:
            assert r['job']==j and r['plan_sha256']==ph
            if status=='completed':
                assert r['solver_called'] and z['success'] and z['failure_reason']==''
                np.testing.assert_array_equal(z['x'],plan['target_x'])
                f=np.asarray(z['f']);exact=np.asarray(r['y_exact']);assert f.shape==exact.shape==(33,)
                assert np.isfinite(f).all() and np.isfinite(exact).all() and np.isfinite(z['stderr']).all()
                error=float(np.max(abs(f-exact)));assert error==r['E_grid']
                assert (error<=.01)==r['accuracy_met']
                assert z['M']==j['config']['M'] and z['completed_paths']==[z['M']]*33
                assert z['seed_entropy']==j['config']['seed'] and z['backend']=='numba'
                assert z['n_path_steps']==z['n_normal_draws']==j['numerical_work']
                assert z['T']==j['expected_T'] and z['n_steps']==j['expected_steps']
                assert z['forcing_abs_upper_bound']==j['forcing_G'] and z['tail_tolerance_met']
                assert z['dt_actual']<=j['config']['dt'] and z['compilation_seconds']==z['numba_import_seconds']==0.
                assert r['timer_stop_ns']>=r['timer_start_ns']
                assert math.isclose((r['timer_stop_ns']-r['timer_start_ns'])*1e-9,r['method_wall_seconds'],rel_tol=0,abs_tol=1e-12)
                assert r['method_wall_seconds']>=z['runtime_seconds']>0
                raw_checks+=1
        attempts.append(row);groups.setdefault((j['equation_id'],j['profile']),[]).append((j,r))
    cells=[];aggregates={}
    for (pid,code),items in groups.items():
        good=[r for _,r in items if r and r['status']=='completed']
        completed=len(good);terminal=sum(r is not None for _,r in items)
        successes=sum(r['accuracy_met'] for r in good);failures=terminal-completed
        errors=summary([r['E_grid'] for r in good]);times=summary([r['method_wall_seconds'] for r in good])
        trials=terminal
        lower=float(beta.ppf(.025,successes,trials-successes+1)) if successes else 0.
        upper=float(beta.ppf(.975,successes+1,trials-successes)) if successes<trials else 1.
        row=dict(equation_id=pid,profile=code,completed=completed,execution_failures=failures,not_run=4-terminal,
            accuracy_pass=successes,error_min=errors['min'],error_median=errors['median'],error_max=errors['max'],
            error_std=errors['std'],wall_median=times['median'],wall_min=times['min'],wall_max=times['max'],
            wall_IQR=times['iqr'],wall_MAD=times['mad'],wall_std=times['std'],
            probability_trials=trials,probability_lower95=lower,probability_upper95=upper,
            max_stderr_median=float(np.median([max(r['result']['stderr']) for r in good])) if good else None,
            all_four_accurate=successes==4,reliability95_confirmed=False)
        cells.append(row)
        if completed==4:
            fs=np.asarray([r['result']['f'] for r in good]);exact=np.asarray(good[0]['y_exact'])
            np.testing.assert_array_equal([r['y_exact'] for r in good],[good[0]['y_exact']]*4)
            mean=fs.mean(axis=0);error=float(np.max(abs(mean-exact)))
            aggregates[pid+'_'+code]=dict(kind='diagnostic_mean4_NOT_single_call_algorithm',
                f=mean.tolist(),y_exact=exact.tolist(),E_grid=error,accuracy_met=error<=.01,
                total_paths_per_point=4*good[0]['result']['M'],
                cost_seconds_sum_calls=sum(r['method_wall_seconds'] for r in good),
                independent_aggregate_replicates=1,reliability95_confirmed=False,
                point_seed_mean_sd=np.std(fs,axis=0,ddof=1).tolist(),
                seeds=[r['seed'] for r in good])
    profiles={}
    for code in ('MC0','MC1','MC2'):
        rs=[r for r in records.values() if r['profile']==code];good=[r for r in rs if r['status']=='completed']
        profiles[code]=dict(terminal=len(rs),completed=len(good),failures=len(rs)-len(good),
            accuracy_pass=sum(r['accuracy_met'] for r in good),
            equations_any_pass=sum(c['accuracy_pass']>0 for c in cells if c['profile']==code),
            error=summary([r['E_grid'] for r in good]),wall=summary([r['method_wall_seconds'] for r in good]),
            total_method_wall_seconds=sum(r.get('method_wall_seconds') or 0. for r in rs))
    valid_terminal={'completed','solver_failure','external_time_limit','external_memory_limit','worker_exception'}
    decision=choose_profile(profiles,plan['gates'],complete=len(records)==60 and all(r['status'] in valid_terminal for r in records.values()))
    # Update only a SCENARIO for the remaining35 using measured sentinel warm
    # time / original per-equation solver scenario. No further tasks are run.
    forecast=csvread(ROOT/'reports/ml_selector_contract/cost_forecast.csv')
    pilot=csvread(ROOT/'reports/ml_selector_contract/pilot_manifest.csv')
    remaining={r['equation_id'] for r in pilot if r['sentinel']=='False'}
    cost_rows=[]
    for code in ('MC0','MC1','MC2'):
        ratios=[]
        for pid in plan['sentinels']:
            own=next(c for c in cells if c['equation_id']==pid and c['profile']==code)
            baseline=next(r for r in forecast if r['equation_id']==pid and r['profile']==code)
            if own['wall_median'] is not None:ratios.append(own['wall_median']/float(baseline['scenario_seconds_one_call']))
        initial=sum(float(r['scenario_seconds_four_calls']) for r in forecast if r['profile']==code and r['equation_id'] in remaining)
        factor=statistics.median(ratios) if ratios else None
        central=initial*factor if factor is not None else None
        cost_rows.append(dict(profile=code,equations=35,calls=140,initial_operation_scaled_seconds=initial,
            median_measured_calibration_ratio=factor,scenario_end_to_end_seconds=central,
            sensitivity_seconds_low=initial*min(ratios) if ratios else None,
            sensitivity_seconds_high=initial*max(ratios) if ratios else None,
            confidence_interval=False,actual_remaining_calls=0,
            proposed_cap_seconds=math.ceil(2*central+300) if central is not None else None))
    sessions=[read(p) for p in (out/'sessions').glob('*.json')]
    warmups=[read(p) for p in (out/'workers').glob('*.warmup.json')]
    statuses=dict(Counter(r['status'] for r in records.values()))
    result=dict(schema='sentinel_MC_P1_summary_v1',planned=60,terminal=len(records),not_run=60-len(records),
        physical_requests=len(list((out/'events').glob('*.request.json'))),
        confirmed_solver_calls=sum(bool(r.get('solver_called')) for r in records.values()),
        statuses=statuses,completed=raw_checks,accuracy_pass=sum(r['accuracy_met'] is True for r in records.values()),
        equations_any_profile_pass=len({r['equation_id'] for r in records.values() if r['accuracy_met'] is True}),
        profiles=profiles,decision=decision,
        sum_method_wall_seconds=sum(r.get('method_wall_seconds') or 0. for r in records.values()),
        controller_wall_seconds=sum(s['wall_seconds'] for s in sessions),
        peak_rss_mib=max((r.get('peak_rss_mib',0.) for r in records.values()),default=None),
        warmups=warmups,remaining35_cost=cost_rows,numerical_metric_checks=raw_checks,
        new_R_calls=0,new_stage2_calls=0,new_model_fits=0,
        interpretation='five targeted train tasks; single calls primary; no95% reliability certificate; no temporal-order claim')
    dest=out/'analysis';dest.mkdir(exist_ok=False)
    csvwrite(dest/'attempts.csv',attempts);csvwrite(dest/'errors.csv',cells)
    csvwrite(dest/'timings.csv',[{k:v for k,v in c.items() if k in ('equation_id','profile','completed') or k.startswith('wall_')} for c in cells])
    csvwrite(dest/'remaining35_cost.csv',cost_rows);write(dest/'aggregates.json',aggregates);write(dest/'summary.json',result)
    lines=['# Sentinel MC P1 — фактический результат','',
        f"Выполнено {result['confirmed_solver_calls']}/60 MC-вызовов; completed {raw_checks}, pending {60-len(records)}.",
        f"Точных отдельных ответов: {result['accuracy_pass']}; задач хотя бы с одним успехом: {result['equations_any_profile_pass']}/5.",
        f"Warm end-to-end сумма {result['sum_method_wall_seconds']:.3f} с; controller wall {result['controller_wall_seconds']:.3f} с; peak RSS {result['peak_rss_mib']} МиБ.",
        '', '## Все задачи и профили','',
        '| ID | Profile | completed | E≤.01 | E min / median / max | warm wall median [min,max], с |',
        '|---|---|---:|---:|---|---|']
    def fmt(x):return 'missing' if x is None else f'{x:.6g}'
    for c in cells:
        lines.append(f"| {c['equation_id']} | {c['profile']} | {c['completed']}/4 | {c['accuracy_pass']}/4 | "+
            ' / '.join(fmt(c[k]) for k in ('error_min','error_median','error_max'))+' | '+fmt(c['wall_median'])+
            ' ['+fmt(c['wall_min'])+', '+fmt(c['wall_max'])+'] |')
    lines+=['','## Замороженные критерии P1','',
        '```json',json.dumps(decision,ensure_ascii=False,indent=2),'```','',
        'Все индивидуальные seed сохранены. Ошибки — max absolute на33 фиксированных точках; success не заменяет точность.',
        'Mean4 в analysis/aggregates.json — отдельная диагностика с4-кратной стоимостью, не4 повторения агрегированного алгоритма.',
        'Вероятностные CP-интервалы индивидуальные по4 испытаниям, без гарантии одновременного покрытия всех15 пар.',
        'Профили имеют независимые seeds: разности M/h не парные coupled оценки; по двум h порядок не установлен.',
        'Оценки стоимости оставшихся35 — operation-scaled сценарии, откалиброванные на5 train-sentinel; диапазон чувствительности не CI.',
        'Ни оставшиеся35, ни Remizov, ни benchmark1000, ни ML в этом запуске не выполнялись.',
        '', 'Источники: [summary](analysis/summary.json), [все вызовы](analysis/attempts.csv), [ошибки](analysis/errors.csv),',
        '[время](analysis/timings.csv), [стоимость35](analysis/remaining35_cost.csv), [план](plan.json), raw records/.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    report(ap.parse_args().out.resolve())
