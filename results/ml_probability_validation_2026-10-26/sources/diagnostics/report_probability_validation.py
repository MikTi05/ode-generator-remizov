"""Saved-only validation analysis: frozen choices, no tuning, fitting or solves."""
from __future__ import annotations
import argparse,csv,json,math,statistics,struct,hashlib,sys,time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnostics.mc_pilot_p2 import read_json,write_json,digest
from ml.probabilistic_selector import probability_scores
SOURCE=ROOT/'results/ml_probability_validation_preflight_2026-10-26';MODEL=ROOT/'results/ml_probability_selector_2026-10-26'
PROFILES=('R128','R256','MC1')


def time_summary(values):
    import numpy as np
    med=statistics.median(values);mad=statistics.median(abs(t-med) for t in values)
    return dict(raw=values,median=med,min=min(values),max=max(values),IQR=float(np.percentile(values,75)-np.percentile(values,25)),MAD=mad,MAD_fraction=mad/med)


def load_results(out):
    import numpy as np
    from all_methods_dataset_loader import load_all_methods_equations
    plan=read_json(out/'plan.json');ph=digest(out/'plan.json')
    assert plan['jobs']==read_json(SOURCE/'plan.json')['jobs'] and len(plan['jobs'])==96
    for n,h in plan['protected_sha256'].items():assert digest(ROOT/n)==h,n
    assert digest(out/'frozen_predictions.json')==plan['frozen_predictions_sha256']
    data=ROOT/plan['data_dir'];ids={c['equation_id'] for c in plan['cases']}
    eqs={e.equation_id:e for e in load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl') if e.equation_id in ids}
    requests=[read_json(p) for p in (out/'events').glob('*.request.json')]
    assert len(requests)==len({r['job']['key'] for r in requests})<=96
    assert all(r['plan_sha256']==ph and r['job'] in plan['jobs'] for r in requests)
    groups={pid:{p:[] for p in PROFILES} for pid in ids};attempts=[];raw=[]
    for j in plan['jobs']:
        path=out/'records'/f"{j['key']}.json"
        if not path.exists():
            attempts.append(dict(key=j['key'],equation_id=j['equation_id'],profile=j['profile'],episode=j['episode'],status='pending_or_not_run',E_grid=None,method_wall_seconds=None));continue
        r=read_json(path);z=r.get('result') or {};e=eqs[j['equation_id']];m=e.metadata
        assert r['job']==j and r['plan_sha256']==ph and r['config']==j['config']
        assert r['expression_hash']==m['expression_hash'] and r['family_id']==m['family_id'] and r['split']==m['split']=='validation'
        if z:
            assert z['problem_id']==j['equation_id'] and z['T']==j['expected_T']
            np.testing.assert_array_equal(z['x'],plan['target_x'])
        if r['status']=='completed':
            assert r['solver_called'] and z['success'] and z['failure_reason']==''
            f=np.asarray(z['f']);exact=np.asarray(e.y_exact(np.asarray(plan['target_x'])))
            assert f.shape==(33,) and np.isfinite(f).all() and np.array_equal(r['y_exact'],exact)
            error=float(np.max(abs(f-exact)));assert error==r['E_grid'] and (error<=.01)==r['accuracy_met']
            if j['profile']=='MC1':
                assert r['seed']==j['config']['seed']==z['seed_entropy']==248800060100+j['episode']
                assert z['M']==4096 and z['completed_paths']==[4096]*33 and z['dt_requested']==1/128 and 0<z['dt_actual']<=1/128
                assert z['n_path_steps']==z['n_normal_draws']==j['numerical_work']==j['config']['max_path_steps']
                assert z['n_steps']==j['expected_steps'] and z['forcing_abs_upper_bound']==j['forcing_G'] and z['tail_tolerance_met']
                assert z['backend']=='numba' and z['block_steps']==64 and z['compilation_seconds']==z['numba_import_seconds']==0
                assert z['rng_scheme']=='PCG64/SeedSequence/point_address_v1'
                keys=[[*struct.unpack('<8I',hashlib.sha256(b'mc_point_v1\0'+j['equation_id'].encode()+struct.pack('<d',0. if x==0 else x)).digest()),0] for x in plan['target_x']]
                assert z['point_spawn_keys']==keys
            else:
                n=int(j['profile'][1:]);assert z['n']==n and z['scheme']=='solve' and z['time_nodes']==513
                assert z['work_grid_points']==2081 and z['actual_dx']==.0125 and z['work_domain']==[-13.,13.]
                assert z['n_operator_applications']==n*512 and z['n_spline_builds']==n*512+1
        else:assert r.get('accuracy_met') is None or r['status']=='invalid_result'
        if r.get('method_wall_seconds') is not None:
            assert r['method_wall_seconds']>0 and math.isclose((r['timer_stop_ns']-r['timer_start_ns'])*1e-9,r['method_wall_seconds'],abs_tol=1e-12,rel_tol=0)
        groups[j['equation_id']][j['profile']].append(r);raw.append(r)
        attempts.append(dict(key=j['key'],equation_id=j['equation_id'],profile=j['profile'],episode=j['episode'],status=r['status'],E_grid=r.get('E_grid'),method_wall_seconds=r.get('method_wall_seconds'),seed=r.get('seed'),source=str(path.relative_to(ROOT)),source_sha256=digest(path)))
    cases=[]
    for c in plan['cases']:
        pid=c['equation_id'];row=dict(equation_id=pid,family_id=c['family_id'],profiles={})
        for p,rs in groups[pid].items():
            rs.sort(key=lambda r:r['job']['episode']);zero=next((r for r in rs if r['job']['episode']==0),None)
            if p!='MC1':
                fs=[r['result']['f'] for r in rs if r['status']=='completed'];assert all(f==fs[0] for f in fs) if fs else True
            valid_times=[r['method_wall_seconds'] for r in rs if r.get('method_wall_seconds') is not None]
            sessions={r.get('session') for r in rs};timing=time_summary(valid_times) if len(valid_times)==4 and len(sessions)==1 else None
            row['profiles'][p]=dict(attempts=len(rs),completed=sum(r['status']=='completed' for r in rs),accurate=sum(r['status']=='completed' and r['accuracy_met'] for r in rs),
                finite_accuracy_violations=sum(r['status']=='completed' and not r['accuracy_met'] for r in rs),
                execution_failures=sum(r['status']!='completed' for r in rs),errors=[r.get('E_grid') for r in rs],
                primary_status=zero['status'] if zero else 'not_run',primary_error=zero.get('E_grid') if zero else None,
                primary_accurate=bool(zero and zero['status']=='completed' and zero['accuracy_met']),
                primary_violation=bool(zero and zero['status']=='completed' and not zero['accuracy_met']),timing=timing,
                numerical_repeats_equal=(p!='MC1'),timing_cohort=list(sessions))
        cases.append(row)
    return plan,eqs,groups,cases,attempts,raw


def policy_metrics(cases,choices,overhead,baseline=None):
    selected=accurate=violations=failures=abstain=0;warm=0.;missing_cost=0;paired=[];regret=[]
    for c,p in zip(cases,choices):
        pid=c['equation_id'];warm+=overhead.get(pid,0.)
        if p=='abstain':abstain+=1;continue
        selected+=1;z=c['profiles'][p];accurate+=z['primary_accurate'];violations+=z['primary_violation'];failures+=z['primary_status']!='completed'
        if z['timing'] is not None:warm+=z['timing']['median']
        else:missing_cost+=1
        oracle=c.get('offline_oracle',{})
        if z['primary_accurate'] and z['timing'] and oracle.get('minimum_accurate_cost_seconds') is not None:
            best=oracle['minimum_accurate_cost_seconds'];delta=z['timing']['median']+overhead.get(pid,0.)-best
            regret.append(dict(equation_id=pid,seconds=delta,relative=delta/best))
        if baseline:
            b=c['profiles'][baseline]
            if z['primary_accurate'] and b['primary_accurate'] and z['timing'] and b['timing']:
                paired.append(dict(equation_id=pid,saved_seconds=b['timing']['median']-z['timing']['median']-overhead.get(pid,0.)))
    return dict(tasks=len(cases),actions=dict(Counter(choices)),selected=selected,abstain=abstain,accurate_primary=accurate,finite_accuracy_violations=violations,
        execution_failures=failures,delivery_accuracy_rate=accurate/len(cases),conditional_accuracy=accurate/selected if selected else None,
        coverage=selected/len(cases),warm_policy_seconds=warm if not missing_cost else None,missing_cost_cases=missing_cost,
        accuracy_constrained_regret=regret,regret_coverage=len(regret)/len(cases),
        common_accurate_comparison=paired,common_accurate_saved_seconds_sum=sum(r['saved_seconds'] for r in paired),
        evaluation='offline frozen policy, episode0 errors +full median4 wall +measured feature/inference overhead; not actual online solver dispatch')


def offline_oracle(case):
    own=case['profiles'];eligible=[p for p,z in own.items() if z['primary_accurate']]
    timed=[p for p in eligible if own[p]['timing'] is not None]
    complete=all(z['primary_status']=='completed' for z in own.values())
    fastest=min(timed,key=lambda p:own[p]['timing']['median']) if complete and len(timed)==len(eligible) and timed else None
    label='none' if not eligible and all(z['primary_status']=='completed' for z in own.values()) else 'unknown'
    R=[p for p in timed if p!='MC1'];bestR=min(R,key=lambda p:own[p]['timing']['median']) if R else None
    if len(eligible)==1 and all(z['primary_status']=='completed' for z in own.values()):label=eligible[0]
    elif fastest and all(z['primary_status']=='completed' for z in own.values()) and all(own[p]['timing']['MAD_fraction']<=.2 for p in timed):
        if 'MC1' not in timed:label=bestR
        elif not bestR:label='MC1'
        else:
            tmc=own['MC1']['timing'];tr=own[bestR]['timing']
            label='MC1' if tmc['median']<=.9*tr['median'] and tmc['max']<tr['min'] else bestR
    return dict(eligible_profiles=eligible,fastest_accurate_cost_profile=fastest,
        minimum_accurate_cost_seconds=own[fastest]['timing']['median'] if fastest else None,
        operational10pct_profile=label,scope='offline episode0 only; no true conditional-p oracle or training label')


def measure_overhead(plan,frozen):
    import numpy as np
    from ml.remizov_features import load_inputs
    from ml.probabilistic_selector import ProbabilisticSelector
    t=time.perf_counter();api=ProbabilisticSelector(MODEL/'registry.json');loading=time.perf_counter()-t
    inputs=load_inputs(ROOT/plan['data_dir']);idx={str(pid):i for i,pid in enumerate(inputs['ids'])};rows=[]
    with np.load(ROOT/plan['data_dir']/'all_methods_equations_v1_ml.npz',allow_pickle=False) as z:
        for c in plan['cases']:
            pid=c['equation_id'];values=z['X'][idx[pid]];start=time.perf_counter_ns()
            result=api.select_method(values,inputs['x_grid'],inputs['metadata'][pid],q=.95,profile_context=api.registry['profile_context'],timing_context=api.registry['timing_context'])
            wall=(time.perf_counter_ns()-start)*1e-9
            assert result['estimates']==frozen[pid]['estimates'] and result['action']=='abstain'
            rows.append(dict(equation_id=pid,feature_and_inference_seconds=wall,production_action=result['action'],status=result['status']))
    return dict(model_loading_seconds=loading,rows=rows,scope='post-series measurement, frozen weights and estimates unchanged; shared input loading excluded; no numerical call or fitting')


def report(out):
    import numpy as np
    from scipy.stats import beta
    from diagnostics.chernov_batch_control import lock_is_held
    plan,eqs,groups,cases,attempts,raw=load_results(out)
    frozen={r['equation_id']:r for r in read_json(out/'frozen_predictions.json')['rows']}
    assert all(not lock_is_held(ROOT/'results'/n) for n in ('.chernov_batch_controller.lock','.chernov_batch_compute.lock'))
    assert read_json(out/'active.json')['finished']
    overhead=measure_overhead(plan,frozen);cost_overhead={r['equation_id']:r['feature_and_inference_seconds'] for r in overhead['rows']}
    scores={};aggregates={};costtargets=read_json(MODEL/'full_cost_targets.json');accuracytargets=read_json(MODEL/'accuracy_targets.json')
    for p in PROFILES:
        score_rows=[];time_errors=[]
        for c in cases:
            pid=c['equation_id'];z=c['profiles'][p];rs=groups[pid][p];operational=[r for r in rs if r['status'] not in ('invalid_result','interrupted')]
            if p=='MC1':
                n=len(operational);k=sum(r['status']=='completed' and r['accuracy_met'] for r in operational)
                if n:
                    z['individual_CP95']=[float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]
                z['CP_scope']='individual descriptive seed probability, no simultaneous or95pct reliability claim'
            else:
                n=int(z['primary_status'] not in ('not_run','invalid_result','interrupted'));k=int(z['primary_accurate'])
            if n:score_rows.append((k,n,frozen[pid]['estimates'][p]['probability']['point']))
            if z['timing']:
                truth=z['timing']['median'];pred=frozen[pid]['estimates'][p]['cost']['point']
                time_errors.append(dict(equation_id=pid,observed_seconds=truth,predicted_seconds=pred,absolute_percentage_error=abs(pred/truth-1),absolute_log_error=abs(math.log(pred/truth))))
        observed=sum(z['profiles'][p]['completed'] for z in cases);correct=sum(z['profiles'][p]['accurate'] for z in cases)
        aggregates[p]=dict(requested=32,terminal=len([r for r in raw if r['profile']==p]),completed=observed,accurate_calls=correct,
            finite_accuracy_violations=sum(z['profiles'][p]['finite_accuracy_violations'] for z in cases),
            primary_accurate_equations=sum(z['profiles'][p]['primary_accurate'] for z in cases),primary_finite_violations=sum(z['profiles'][p]['primary_violation'] for z in cases),
            execution_failures=sum(z['profiles'][p]['execution_failures'] for z in cases))
        if score_rows:
            k,n,prob=zip(*score_rows);train=[r for r in accuracytargets if r['profile']==p];prior=sum(r['k'] for r in train)/sum(r['n'] for r in train)
            probability=dict(model=probability_scores(k,n,prob),constant_train_rate=prior,baseline=probability_scores(k,n,[prior]*len(prob)))
        else:probability=dict(status='no_complete_trials')
        traincost=float(np.exp(np.mean(np.log([r['seconds'] for r in costtargets if r['profile']==p]))))
        scores[p]=dict(probability=probability,time_errors=time_errors,
            cost_model=dict(cases=len(time_errors),median_APE=float(np.median([r['absolute_percentage_error'] for r in time_errors])) if time_errors else None,mean_absolute_log_error=float(np.mean([r['absolute_log_error'] for r in time_errors])) if time_errors else None),
            constant_cost_baseline=dict(train_geomean_seconds=traincost,median_APE=float(np.median([abs(traincost/r['observed_seconds']-1) for r in time_errors])) if time_errors else None))
    for case in cases:case['offline_oracle']=offline_oracle(case)
    baseline={p:policy_metrics(cases,[p]*8,{}) for p in ('R256','R128')}
    policies={'production_research_guard':policy_metrics(cases,['abstain']*8,cost_overhead)}
    for q in ('.5','.8','.95'):
        key=str(float(q));policies[key]={}
        for mode in ('point','model_spread'):
            choices=[frozen[c['equation_id']]['diagnostic_choices'][key][mode]['action'] for c in cases]
            metric=policy_metrics(cases,choices,cost_overhead,baseline='R256')
            b=baseline['R256'];metric['comparison_constant_R256']=dict(accurate_delivery_difference=metric['accurate_primary']-b['accurate_primary'],
                finite_violation_difference=metric['finite_accuracy_violations']-b['finite_accuracy_violations'],
                warm_seconds_saved=b['warm_policy_seconds']-metric['warm_policy_seconds'] if b['warm_policy_seconds'] is not None and metric['warm_policy_seconds'] is not None else None,
                abstention_difference=metric['abstain']-b['abstain'],claim='descriptive eight-family diagnostic; abstention and accuracy reported together, no population superiority claim')
            policies[key][mode]=metric
    samples=[json.loads(s) for f in (out/'workers').glob('*.rss.jsonl') for s in f.read_text().splitlines()]
    thermal=[json.loads(s) for s in (out/'thermal.jsonl').read_text().splitlines()]
    guardians=[read_json(f) for f in (out/'control').rglob('*.guardian.json')]
    assert all(g['finished'] and g['cleanup_success'] for g in guardians)
    resources=dict(peak_rss_mib=max(s['peak_rss_mib'] for s in samples),rss_samples=len(samples),thermal_samples=len(thermal),thermal_max=max(s['thermal_state'] for s in thermal),
        warmups=[read_json(f) for f in (out/'workers').glob('*.warmup.json')],guardians=guardians)
    assert resources['peak_rss_mib']<=512 and resources['thermal_max']<2
    sessions=[read_json(f) for f in (out/'sessions').glob('*.json')]
    result=dict(status='complete' if len(raw)==96 else 'incomplete',records=len(raw),pending=96-len(raw),requests=len(list((out/'events').glob('*.request.json'))),
        profile_outcomes=aggregates,prediction_scores=scores,baselines=baseline,policies=policies,resources=resources,sessions=sessions,
        method_wall_sum_seconds=sum(r.get('method_wall_seconds') or 0. for r in raw),controller_wall_sum_seconds=sum(s['wall_seconds'] for s in sessions),
        registry_sha256=digest(MODEL/'registry.json'),frozen_prediction_sha256=digest(out/'frozen_predictions.json'),policy_and_models_unchanged=True,
        validation_only=True,test_used=False,ML_fits=0,benchmark1000_calls=0,
        interpretation='one diagnostic packet of8 previously selected validation families; no policy selection/calibration/refitting on outcomes; MC observed success is not new-call reliability')
    write_json(out/'cases.json',cases,exclusive=True);write_json(out/'analysis.json',result,exclusive=True);write_json(out/'policy_overhead.json',overhead,exclusive=True)
    fields=sorted({k for row in attempts for k in row})
    with (out/'attempts.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(attempts)
    print(json.dumps(dict(status=result['status'],profile_outcomes=aggregates,baselines=baseline,policies=policies),sort_keys=True),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);report(ap.parse_args().out.resolve())
