"""Saved-results-only development96 diagnostics; never solve or fit."""
from __future__ import annotations
import csv,json,math,statistics,sys
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import beta
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnostics.prepare_development8 import read_json,write_json,digest,snapshot,SEEDS
from diagnostics.run_development8 import validate_scope


def interval(k,n):
    return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]


def timing(times):
    a=np.asarray(times);med=float(np.median(a));mad=float(np.median(abs(a-med)))
    return dict(times=times,median=med,min=float(min(a)),max=float(max(a)),IQR=float(np.quantile(a,.75)-np.quantile(a,.25)),MAD=mad,relative_MAD=mad/med)


def proper_scores(p,y):
    p=np.clip(np.asarray(p),1e-10,1-1e-10);y=np.asarray(y)
    return dict(brier=float(np.mean((p-y)**2)),logloss=float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))))


def cluster_score_uncertainty(probabilities,outcomes):
    """Family is the independent unit; SE is descriptive with only eight units."""
    brier=[];logloss=[]
    for p,ys in zip(probabilities,outcomes):
        s=proper_scores([p]*len(ys),ys);brier.append(s['brier']);logloss.append(s['logloss'])
    return {k:dict(family_values=v,mean=float(np.mean(v)),family_SE=float(np.std(v,ddof=1)/math.sqrt(len(v))),
        caveat='descriptive SE; eight families do not establish calibrated confidence coverage') for k,v in [('brier',brier),('logloss',logloss)]}


def verify(out):
    plan=read_json(out/'plan.json');validate_scope(plan);ph=digest(out/'plan.json')
    for name,h in plan['protected_sha256'].items():assert digest(ROOT/name)==h,name
    requests=[read_json(p) for p in (out/'events').glob('*.request.json')]
    assert len(requests)==96 and len({r['job']['key'] for r in requests})==96
    assert all(r['job'] in plan['jobs'] and r['plan_sha256']==ph for r in requests)
    assert len(list((out/'records').glob('*.json')))==96
    from all_methods_dataset_loader import load_all_methods_equations
    inputs=ROOT/plan['data_dir']
    eqs={e.equation_id:e for e in load_all_methods_equations(inputs/'all_methods_equations_v1.csv',inputs/'all_methods_equations_v1_metadata.jsonl')}
    assert len(eqs)==8 and all(e.metadata['split']=='train' for e in eqs.values())
    from all_methods_bounds import certify_all_methods_forcing
    from mc_solver import auto_T
    for c in plan['cases']:
        e=eqs[c['equation_id']];problem=e.remizov_problem
        G=certify_all_methods_forcing(problem,e.metadata).forcing_bound.abs_upper_bound
        T=auto_T(problem.lambda_value,problem.c_abs_upper_bound,G,1e-4)
        assert G==c['G'] and T==c['T'] and math.ceil(T*128)==c['N']
    from diagnostics.chernov_batch_control import lock_is_held
    active=read_json(out/'active.json');assert active['finished']
    guardians=[read_json(p) for p in Path(active['control_dir']).glob('*.guardian.json')]
    assert len(guardians)==1 and all(g['finished'] and g['cleanup_success'] for g in guardians)
    for n in ('.chernov_batch_controller.lock','.chernov_batch_compute.lock'):assert not lock_is_held(ROOT/'results'/n)
    records={};rng_addresses=[]
    for j in plan['jobs']:
        r=read_json(out/'records'/f"{j['key']}.json");z=r['result']
        assert r['job']==j and r['plan_sha256']==ph and r['status']=='completed' and r['solver_called']
        assert r['expression_hash']==j['expression_hash'] and r['split']=='train'
        assert z['success'] and not z['failure_reason'] and z['problem_id']==j['equation_id']
        assert z['T']==j['expected_T'] and z['x']==j['target_x'] and np.isfinite(z['f']).all()
        np.testing.assert_array_equal(eqs[j['equation_id']].y_exact(np.asarray(j['target_x'])),r['y_exact'])
        err=float(np.max(np.abs(np.asarray(z['f'])-np.asarray(r['y_exact']))))
        assert err==r['E_grid'] and r['accuracy_met']==(err<=.01)
        assert math.isclose(r['method_wall_seconds'],(r['timer_stop_ns']-r['timer_start_ns'])*1e-9,abs_tol=1e-15)
        assert r['method_wall_seconds']>0 and r['peak_rss_mib']<=512
        if j['profile']=='MC1':
            assert z['M']==4096 and z['seed_entropy']==j['config']['seed']==SEEDS[j['episode']]
            assert z['backend']=='numba' and z['rng_scheme']=='PCG64/SeedSequence/point_address_v1'
            assert z['n_path_steps']==z['n_normal_draws']==j['numerical_work']==33*4096*j['expected_steps']
            assert z['n_steps']==j['expected_steps'] and z['tail_tolerance_met'] and z['dt_requested']==1/128
            assert z['compilation_seconds']==z['numba_import_seconds']==0
            assert z['completed_paths']==[4096]*33
            for x,key in zip(z['x'],z['point_spawn_keys']):rng_addresses.append((z['seed_entropy'],tuple(key)))
        else:
            assert z['n']==j['config']['n'] and z['n_operator_applications']==j['numerical_work']
            assert z['n_spline_builds']==j['numerical_work']+1 and z['work_grid_points']==2081 and z['time_nodes']==513
            assert z['actual_dx']==.0125 and z['work_domain']==[-13.,13.]
        records[j['key']]=r
    assert len(rng_addresses)==len(set(rng_addresses))==32*33
    for c in plan['cases']:
        for profile in ('R128','R256'):
            rs=[r for r in records.values() if r['equation_id']==c['equation_id'] and r['profile']==profile]
            assert len(rs)==4 and all(r['result']['f']==rs[0]['result']['f'] for r in rs)
    for p in (out/'workers').glob('*.warmup.json'):
        warm=read_json(p);assert warm['extra_solver_calls']==0 and warm['numba_threads']==1 and warm['no_RNG_in_warmup']
    thermal=[json.loads(l) for l in (out/'thermal.jsonl').read_text().splitlines()];assert thermal and all(t['thermal_state'] is not None and t['thermal_state']<2 for t in thermal)
    rss_samples=[json.loads(l) for p in (out/'workers').glob('*.rss.jsonl') for l in p.read_text().splitlines()]
    assert rss_samples and max(r['peak_rss_mib'] for r in rss_samples)<=512
    sessions=[read_json(p) for p in (out/'sessions').glob('*.json')];assert len(sessions)==1 and sessions[0]['reason'] is None and sessions[0]['new_terminal']==96
    budget=read_json(out/'budget.json');end=max(r['timer_stop_ns'] for r in records.values())/1e9
    assert sessions[0]['wall_seconds']<5460 and end<=budget['deadline_monotonic']
    assert budget['deadline_epoch']-budget['started_epoch']==5460
    assert budget['deadline_monotonic']-budget['started_monotonic']==5460
    assert read_json(out/'pending.json')['keys']==[]
    result=dict(passed=True,requests=96,completed=96,unique_MC_point_streams=1056,no_retries=True,R_bitwise_repeats=True,E_grid_recomputed=96,
        model_source_and_input_pins_unchanged=True,wall_seconds=sessions[0]['wall_seconds'],cap_seconds=5460,
        peak_RSS_MiB=max(r['peak_rss_mib'] for r in rss_samples),RSS_samples=len(rss_samples),thermal_samples=len(thermal),thermal_states=sorted(set(t['thermal_state'] for t in thermal)),
        workers=len({r['worker_pid'] for r in records.values()}),threads=1,numerical_warmup_calls=0,no_validation_or_test_used=True)
    assert result['workers']==1
    write_json(out/'independent_verification.json',result)
    return plan,records,result


def analyze(out):
    plan,records,verification=verify(out);frozen=read_json(out/'frozen_predictions.json');pred={r['equation_id']:r for r in frozen['rows']}
    cases=[];attempts=[]
    for c in plan['cases']:
        pid=c['equation_id'];profiles={}
        for profile in ('R128','R256','MC1'):
            rs=sorted((r for r in records.values() if r['equation_id']==pid and r['profile']==profile),key=lambda r:r['job']['episode'])
            z=dict(accurate=sum(r['accuracy_met'] for r in rs),errors=[r['E_grid'] for r in rs],timing=timing([r['method_wall_seconds'] for r in rs]),primary_accurate=rs[0]['accuracy_met'],primary_seconds=rs[0]['method_wall_seconds'])
            if profile=='MC1':
                residual=np.array([np.array(r['result']['f'])-np.array(r['y_exact']) for r in rs]);stderr=np.array([r['result']['stderr'] for r in rs])
                z.update(seed_success_CP95=interval(z['accurate'],4),CP_note='four fixed seeds on this family only, not a population reliability certificate',
                    signed_mean_residual_max=float(np.max(abs(residual.mean(axis=0)))),between_seed_point_std_max=float(np.std(residual,axis=0,ddof=1).max()),
                    max_stderr_per_seed=[float(v.max()) for v in stderr],pointwise_CI_exact_coverage=[float(np.mean((np.asarray(r['result']['ci_lower'])<=r['y_exact'])&(np.asarray(r['y_exact'])<=r['result']['ci_upper']))) for r in rs],
                    discretization_bounds=[r['result']['discretization_error_bound'] for r in rs],total_error_certified=[r['result']['total_error_certified'] for r in rs],
                    diagnostic_mean_is_not_a_new_M16384_result=True)
            profiles[profile]=z
            for r in rs:attempts.append(dict(equation_id=pid,identity=c['identity'],profile=profile,episode=r['job']['episode'],seed=r['seed'],E_grid=r['E_grid'],accuracy_met=r['accuracy_met'],seconds=r['method_wall_seconds'],status=r['status'],peak_RSS_MiB=r['peak_rss_mib']))
        exact_R=[p for p in ('R128','R256') if profiles[p]['accurate']==4]
        fastest_R=min(exact_R,key=lambda p:profiles[p]['timing']['median']) if exact_R else None
        primary_feasible=[p for p in profiles if profiles[p]['primary_accurate']]
        primary_fastest=min(primary_feasible,key=lambda p:profiles[p]['primary_seconds']) if primary_feasible else 'none'
        if primary_fastest=='MC1' and fastest_R and profiles['MC1']['primary_seconds']>=.9*profiles[fastest_R]['primary_seconds']:primary_fastest=fastest_R
        mc=profiles['MC1'];r=profiles.get(fastest_R) if fastest_R else None
        speed=1-mc['timing']['median']/r['timing']['median'] if r else None
        stable=bool(r and speed>=.1 and mc['timing']['max']<r['timing']['min'] and mc['timing']['relative_MAD']<=.2 and r['timing']['relative_MAD']<=.2)
        cases.append(dict(**c,profiles=profiles,predictions=pred[pid],fastest_accurate_R=fastest_R,primary_observed_fastest_with_10percent_R_tie=primary_fastest,
            default_R256_below_q=pred[pid]['calibrated']['R256']['probability']['point']<.95,
            primary_accurate_profile_exists=bool(primary_feasible),
            R128_speed_advantage_stable=profiles['R128']['timing']['max']<.9*profiles['R256']['timing']['min'] and max(profiles[p]['timing']['relative_MAD'] for p in ('R128','R256'))<=.2,
            MC_saving_vs_fastest_accurate_R=speed,MC_timing_advantage_stable=stable,
            MC_accurate_calls_faster_than_R10percent=sum(a and t<.9*r['timing']['median'] for a,t in zip([e<=.01 for e in mc['errors']],mc['timing']['times'])) if r else None,
            MC_observed4of4_fast_candidate=bool(mc['accurate']==4 and stable),MC_new_seed_reliability_not_established=True))
    scores={}
    for p in ('R128','R256','MC1'):
        rows=[]
        for c in cases:
            forecast=c['predictions']['calibrated'][p];actual=c['profiles'][p];prob=forecast['probability'];cost=forecast['cost']
            rows.append(dict(equation_id=c['equation_id'],p=prob['point'],p_low=prob['low'],p_high=prob['high'],observed_fraction=actual['accurate']/4,
                primary_success=int(actual['primary_accurate']),observed_time=actual['timing']['median'],predicted_time=cost['point'],time_ratio=cost['point']/actual['timing']['median'],
                time_in_saved_band=cost['low']<=actual['timing']['median']<=cost['high'],cost_supported=cost['supported']))
        scores[p]=dict(rows=rows,primary=proper_scores([r['p'] for r in rows],[r['primary_success'] for r in rows]),
            all4_family_balanced=proper_scores([r['p'] for r in rows for _ in range(4)],[int(e<=.01) for c in cases for e in c['profiles'][p]['errors']]),
            family_uncertainty=cluster_score_uncertainty([r['p'] for r in rows],[[int(e<=.01) for e in c['profiles'][p]['errors']] for c in cases]),
            predicted_expected_successes=sum(r['p'] for r in rows),observed_primary_successes=sum(r['primary_success'] for r in rows),observed_successful_calls=sum(c['profiles'][p]['accurate'] for c in cases),
            time_MAPE=float(np.mean([abs(r['time_ratio']-1) for r in rows])),median_time_ratio=float(np.median([r['time_ratio'] for r in rows])),time_band_coverage=sum(r['time_in_saved_band'] for r in rows)/8,
            uncertainty_note='8 independent topology families; four replicates are clustered, model bands are empirical ensemble spread, not calibrated coverage guarantees')
    diagnoses=[]
    for c in cases:
        mc=c['profiles']['MC1'];rs=[r for r in records.values() if r['equation_id']==c['equation_id'] and r['profile']=='MC1']
        diagnoses.append(dict(equation_id=c['equation_id'],observed_accurate=mc['accurate'],max_stderr_range=[min(mc['max_stderr_per_seed']),max(mc['max_stderr_per_seed'])],
            max_signed_mean_residual=mc['signed_mean_residual_max'],max_between_seed_std=mc['between_seed_point_std_max'],
            tail_error_bounds=[r['result']['tail_error_bound'] for r in rs],discretization_error_certified=False,
            observation='sampling scatter is material at tolerance .01; maximum over33 points is stricter than a pointwise confidence interval',
            hypothesis='fixed-M variance and possible fixed-h bias; these repeats alone do not identify their separate contribution',
            no_resource_or_implementation_failure=True))
    write_json(out/'MC_error_diagnosis.json',dict(rows=diagnoses,new_MC_successes=0 if all(r['observed_accurate']==0 for r in diagnoses) else sum(r['observed_accurate'] for r in diagnoses),
        no_M_h_change=True,no_extra_seeds=True,no_aggregated_estimator_or_artificial_labels=True))
    def policy(kind,q=.95):
        actions=[('R256' if kind=='constant_R256' else c['predictions']['choices'][str(q)][kind]) for c in cases]
        exact=sum(c['profiles'][p]['primary_accurate'] for c,p in zip(cases,actions) if p!='abstain');n=sum(p!='abstain' for p in actions)
        cost=sum(c['profiles'][p]['timing']['median'] for c,p in zip(cases,actions) if p!='abstain')
        primary_cost=sum(c['profiles'][p]['primary_seconds'] for c,p in zip(cases,actions) if p!='abstain')
        overhead=0. if kind=='constant_R256' else sum(c['predictions']['feature_seconds']+c['predictions']['inference_seconds'] for c in cases)
        return dict(actions=Counter(actions),accurate=exact,delivered=n,violations=n-exact,abstentions=8-n,delivery_accuracy=exact/n if n else None,
            primary_method_seconds=primary_cost,median_method_seconds=cost,inference_overhead_seconds=overhead,total_warm_seconds=cost+overhead,
            cost_per_accurate=(cost+overhead)/exact if exact else None,accuracy_CP95=interval(exact,n) if n else None)
    policies=dict(primary=policy('baseline_default'),baseline=policy('constant_R256'),risk_q95=policy('risk'),
        already_preregistered_q_diagnostics={str(q):dict(baseline_default=policy('baseline_default',q),risk=policy('risk',q)) for q in (.5,.8)},
        production=dict(delivered=0,abstentions=8,reason='research-only guard remains in force; no production dispatch performed'))
    summary=dict(completed=96,cases=cases,probability_and_time=scores,policies=policies,resources=verification,
        interval_caveat='Binomial CP reference intervals assume iid trials within the stated unit; stratified eight-family pilot is not a final population coverage certificate. Seeds are not extra independent families.',
        accuracy_by_profile={p:sum(c['profiles'][p]['accurate'] for c in cases) for p in ('R128','R256','MC1')},
        MC_families_any_success=sum(c['profiles']['MC1']['accurate']>0 for c in cases),MC_families4of4=sum(c['profiles']['MC1']['accurate']==4 for c in cases),
        MC_observed4of4_fast_candidates=[c['equation_id'] for c in cases if c['MC_observed4of4_fast_candidate']],
        primary_oracle_counts=Counter(c['primary_observed_fastest_with_10percent_R_tie'] for c in cases),
        missed_R128_opportunities=[c['equation_id'] for c in cases if c['profiles']['R128']['accurate']==4 and c['profiles']['R128']['timing']['median']<.9*c['profiles']['R256']['timing']['median'] and c['predictions']['choices']['0.95']['baseline_default']=='R256'],
        MC_mechanism_limitations='M fixes sampling variance; h fixed, discretization bias unbounded. Grid point CI excludes time-discretization bias, not a simultaneous33-point E_grid certificate. Four repeats cannot uniquely identify bias versus random error.',
        development_only=True,model_or_q_tuned=False,test_or_validation_used=False,solver_calls_by_report=0)
    write_json(out/'analysis.json',summary)
    with (out/'attempts.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(attempts[0]));w.writeheader();w.writerows(attempts)
    return summary


def markdown_report(out,a):
    v=a['resources'];p=a['policies'];primary=p['primary'];baseline=p['baseline']
    lines=['# Fresh development8: R128 / R256 / MC1',
        '', 'Протокол/каталог: 2026-10-26; завершено2026-10-27 по Europe/Moscow (2026-10-26 UTC). Это development utility/futility pilot; данные теперь раскрыты и не могут служить финальным независимым тестом.',
        '', '## Протокол и сохранность', '',
        f"Выполнены96/96 физических вызовов:8 новых topology families, по4 R128/R256/MC1. Wall серии {v['wall_seconds']:.3f} с при cap5460 с;1 worker/поток, пик RSS {v['peak_RSS_MiB']:.3f} MiB≤512, thermal states {v['thermal_states']}. Guardian, внешний owner lease300 с, persistent epoch/monotonic deadlines и immutable records действовали. Повторов прерванных requests и дополнительных solver/JIT-warmup вызовов нет.",
        '', 'Пул seed2488000901/40семей/1sample создан неизменным all_methods_v1; в малом пуле20 train-role семей. Выбор8: по2 на L1..L4, least-represented regime + заранее заданный SHA256. Exclusion148 известных train/revealed topology signatures и аудит сохранённой train-истории выполнены до outcomes. Исходы/признаки прежних validation/test не использованы; held-out роли нового пула не оценивались. Все96 вычислений используют только отдельные8 train inputs.',
        '', 'Канонические ID повторяются между наборами. Идентичность здесь = SHA256 канонического CSV + topology signature + canonical ID. Нельзя соединять эти результаты со старым train по одному ID/family_id: будущему ingestion нужны qualified keys. Полные AST и четыре dataset-hash сохранены.',
        '', 'Профили и target33[-1,1] неизменны. Общие аналитические G/T и N=ceil(128T), max_path_steps=33×4096×N, seeds248800080100..103 заморожены до первого solve. Полный method wall включает method-specific preparation/config/solve; общие imports/load/Numba compilation вынесены в warmup, exact/scoring после таймера. Primary episode0 предопределён. Все4 MC-seed учитываются отдельно; успешные seeds не отбирались и не объединялись в M16384.',
        '', '## Точность и фактическое время', '',
        '| Новое canonical ID | R128 E_grid | R256 E_grid | MC1 точных/4; диапазон E_grid | median R128 / R256 / MC1, с |',
        '|---|---:|---:|---|---|']
    for c in a['cases']:
        z=c['profiles'];mc=z['MC1'];t=[z[x]['timing']['median'] for x in ('R128','R256','MC1')]
        lines.append(f"|{c['equation_id']}|{z['R128']['errors'][0]:.6g}|{z['R256']['errors'][0]:.6g}|{mc['accurate']}/4; {min(mc['errors']):.6g}–{max(mc['errors']):.6g}|{t[0]:.3f} / {t[1]:.3f} / {t[2]:.3f}|")
    lines += ['', f"Точных вызовов: {a['accuracy_by_profile']}. MC1 хоть раз точен на{a['MC_families_any_success']}/8семей,4/4 — на{a['MC_families4of4']}/8. Primary oracle с10% tie в пользу R: {a['primary_oracle_counts']}.",
        '', f"MC4/4 с устойчивой экономией≥10% относительно самого быстрого точного R: {a['MC_observed4of4_fast_candidates'] or 'не обнаружены'}. В analysis.json для всех семей приведены min/max/IQR/MAD, полные4 времени, отдельные успешные MC-вызовы быстрее R и признак disjoint ranges. Наличие случайного удачного seed не заменяет надёжность нового вызова.",
        '', '## Замороженная политика и baseline', '',
        f"Основная final nested sigmoid R256-default q=.95: actions {primary['actions']}; точных доставок{primary['accurate']}/8, нарушений{primary['violations']}/8, отказов{primary['abstentions']}/8. Constant_R256: точных{baseline['accurate']}/8, нарушений{baseline['violations']}/8. Method wall (сумма median4) {primary['median_method_seconds']:.6f} против{baseline['median_method_seconds']:.6f} с. Измеренный feature/head/calibration overhead {primary['inference_overhead_seconds']:.6f} с; full warm policy-cost {primary['total_warm_seconds']:.6f} с. Cost per accurate: {primary['cost_per_accurate']} против{baseline['cost_per_accurate']} с, включая стоимость неточных доставок.",
        '', f"Пропущенные возможности точного R128 с экономией≥10%: {a['missed_R128_opportunities']}. Default R256 может выдать результат даже при собственном p<q: это механизм fallback, а не гарантия точности. Risk q95 отдельно: {p['risk_q95']}. Production API по-прежнему заблокирован (8 потенциальных отказов, dispatch не выполнялся).",
        '', 'Результаты q=.5/.8 сохранены только как ранее зафиксированные диагностические сравнения; победивший q не выбирался. Модели/калибраторы/253 features/q/solver не менялись. Глобальная граница текущего MC point-forecast .837788412<.95 по прежнему отчёту означает, что текущий freeze не может выбрать MC даже при его реальном успехе.',
        '', '## Прогнозы и неопределённость', '',
        '|Профиль|Brier primary|Logloss primary|Brier all4|Ожидалось точных primary|Наблюдалось|Time MAPE|', '|---|---:|---:|---:|---:|---:|---:|']
    for profile,s in a['probability_and_time'].items():
        lines.append(f"|{profile}|{s['primary']['brier']:.6f}|{s['primary']['logloss']:.6f}|{s['all4_family_balanced']['brier']:.6f}|{s['predicted_expected_successes']:.3f}|{s['observed_primary_successes']}/8|{100*s['time_MAPE']:.1f}%|")
    lines += ['', 'Полные point/low/high p и cost, флаг supported стоимости R128, попадание времени в сохранённый диапазон и семейные SE proper scores находятся в analysis.json. Эмпирические ensemble-диапазоны не являются калиброванными доверительными интервалами. Для обобщения независимых единиц8, а не32: R-повторы детерминированы, MC-seeds сгруппированы по семейству. Этот объём позволяет выявить ошибки, но не подтвердить q95 надёжность.',
        '', 'Критические ошибки вероятностей R256: F0026 p=.9620, но E_grid=.01269; F0033 p=.2285, но E_grid=.000296. Поэтому risk-q95 пропускает реальную ошибку и отказывается от точной задачи. Совпадение суммарного ожидания≈7.026 с7 успехами не означает правильности индивидуальных прогнозов. R128 точен на7/8, но ни один его calibrated point не достигает q=.95. MC ожидал≈.583 primary-успеха на8, наблюдалось0: низкий Brier на полностью отрицательном пакете не подтверждает распознавание MC-успехов.',
        '', 'На этих8 MAPE времени: R128≈2.56%,R256≈1.73%,MC≈23.76%; сохранённые cost bands покрыли лишь1/8,2/8,2/8 фактических медиан соответственно. Они не являются откалиброванными интервалами; стоимость R128 в текущем registry помечена unsupported. Особый промах ресурсного сценария MC — F0035:≈71 с по path-update scaling против≈114 с фактически; стоимость зависит также от AST/вычисления forcing. Общая оценка2332.7 с оказалась близка к2448.6 с серии, но это не делает case-wise прогнозы достоверными.',
        '', 'У MC сохраняются seed-specific stderr/pointwise CI, signed mean residual и between-seed scatter. Случайный шум при M4096 и возможная временная дискретизация при h=1/128 различаются концептуально;4 seed не позволяют однозначно разделить их вклад. Discretization error bound отсутствует, total_error_certified=false. Pointwise CI не ограничивает полную ошибку33 точек; корректный tail certificate не устраняет sampling/discretization error. Даже4/4 наблюдаемых успеха дают двусторонний95% CP lower≈.398, а не доказательство p_new≥.95. Средний residual — только диагностика, не новая численная метка/агрегированный MC.',
        '', '## Следующий шаг', '',
        'Сначала использовать эти development-результаты в отдельной train-only ревизии: добавить dataset-qualified identity в ingestion/семейные группы, включить четыре MC Bernoulli outcomes на семью без псевдорепликации, сопоставимые full-wall targets и R128 targets. Повторить вложенный family-OOF с вероятностной калибровкой, прогнозом времени и неопределённостью; производственный guard пока сохранить. Этот этап возможен без новых solver-вызовов и здесь не выполнялся.',
        '', 'Если реальные MC-победы отсутствуют, массовый добор случайных семейств ради класса не обоснован. В train-only анализе следует проверить, есть ли вообще область признаков с физически полезным MC1: точность нового seed и выигрыш против самого быстрого точного R. Лишь при таком обосновании нужен отдельный небольшой preregistered train-пилот в этой области; менять M/h означает новый профиль с новым контрактом и бюджетом. Нельзя снижать q или создавать победителей искусственно. Gate20+5 MC-winner не нужен для регрессионно-вероятностной модели, но реальная поддержка MC-успехов/полезности и калибровки необходима.',
        '', 'После ревизии — новая заморозка и новые независимо отобранные holdout families с достаточной поддержкой каждого выбранного действия. Эти8 development, старые8 validation и test для настройки/финального подтверждения использовать нельзя. Универсального окончательного n нет; он зависит от частоты MC-action и заданного риска. Текущий пилот проверяет полезность профилей, а не завершает научную задачу трёхметодного селектора.',
        '', '## Проверки и файлы', '',
        'Независимо пересчитаны96 E_grid из safe AST/exact, сверены cfg/T/G/N/work/RNG/семейства/таймеры, bitwise R-повторы и unique1056 MC-point streams. Источники, модели, старые результаты и входы сверены SHA256. Сохранены plan/preflight/budget/frozen_predictions, raw96 records/requests, attempts.csv, resource traces, analysis.json, independent_verification.json, source snapshots и контроль completed-resume0 calls. Итоговый manifest и PROJECT_STATUS фиксируют завершение.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    out=Path(sys.argv[1]).resolve();a=analyze(out);markdown_report(out,a);print(json.dumps({k:v for k,v in a.items() if k not in ('cases','probability_and_time')}))
