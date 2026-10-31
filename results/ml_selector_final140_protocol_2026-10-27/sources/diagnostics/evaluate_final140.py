"""Fixed140 saved-only evaluation. Missing MC40 remains unmeasured, never none."""
from __future__ import annotations
import argparse,hashlib,json,math,struct,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from diagnostics.final140_protocol import validate_scope
from diagnostics.prepare_development8 import read_json,write_json,digest
from diagnostics.evaluate_final_selector_experiment import stratified_delta


def missing(status='not_run'):
    return dict(status=status,correct=None,delivered=None,violation=None,error=None,seconds=None,observed_spent_seconds=None)


def decode(record,job,plan_sha,exact):
    assert record['job']==job and record['plan_sha256']==plan_sha
    for name in ('equation_id','expression_hash','family_id','profile','config','split'):
        assert record[name]==job[name],name
    assert record['seed']==job['config'].get('seed')
    if record['status'] not in ('completed','solver_failure'):
        spent=record.get('method_wall_seconds')
        if spent is None:spent=record.get('external_elapsed_seconds')
        return dict(missing(record['status']),observed_spent_seconds=spent)
    assert record['peak_rss_mib']<=512
    z=record['result'];assert z['problem_id']==job['equation_id'] and z['T']==job['expected_T']
    t=record['method_wall_seconds'];assert t>0 and math.isclose(t,(record['timer_stop_ns']-record['timer_start_ns'])*1e-9,abs_tol=1e-12,rel_tol=0)
    if not z['success']:
        assert record['status']=='solver_failure' and z['failure_reason']
        if z['failure_reason'] in ('time_limit_exceeded','work_limit_exceeded'):
            return dict(missing('resource_censored'),observed_spent_seconds=t)
        return dict(status='solver_failure',correct=0,delivered=0,violation=0,error=None,seconds=t,observed_spent_seconds=t)
    assert record['status']=='completed' and not z['failure_reason'] and record['solver_called']
    assert z['x']==job['target_x'] and len(z['f'])==33 and np.isfinite(z['f']).all()
    np.testing.assert_array_equal(record['y_exact'],exact)
    error=float(np.max(abs(np.array(z['f'])-exact)));assert error==record['E_grid'] and (error<=.01)==record['accuracy_met']
    if job['profile']=='MC1':
        assert z['backend']=='numba' and z['M']==4096 and z['block_steps']==64 and z['dt_requested']==1/128 and 0<z['dt_actual']<=1/128
        assert z['seed_entropy']==job['config']['seed'] and z['n_steps']==job['expected_steps']
        assert z['n_path_steps']==z['n_normal_draws']==job['numerical_work'] and z['completed_paths']==[4096]*33
        assert z['forcing_abs_upper_bound']==job['forcing_G'] and z['tail_tolerance_met'] and z['compilation_seconds']==z['numba_import_seconds']==0
        expected=[[*struct.unpack('<8I',hashlib.sha256(b'mc_point_v1\0'+job['equation_id'].encode()+struct.pack('<d',0. if x==0 else x)).digest()),0] for x in job['target_x']]
        assert z['point_spawn_keys']==expected and z['rng_scheme']=='PCG64/SeedSequence/point_address_v1'
    else:
        assert z['n']==job['config']['n'] and z['n_operator_applications']==job['numerical_work'] and z['n_spline_builds']==job['numerical_work']+1
        assert z['work_grid_points']==2081 and z['time_nodes']==513 and z['actual_dx']==.0125 and z['work_domain']==[-13.,13.]
    return dict(status='completed',correct=int(error<=.01),delivered=1,violation=int(error>.01),error=error,seconds=t,observed_spent_seconds=t)


def summarize(rows,n):
    known=[r for r in rows if r['correct'] is not None];count=sum(r['correct'] for r in known);unknown=n-len(known)
    times=[r['cost'] for r in rows];cost=sum(times) if all(t is not None for t in times) else None
    return dict(requested_tasks=n,known_outcomes=len(known),confirmed_correct=count,unknown_outcomes=unknown,
        accurate_deliveries=count if not unknown else None,delivery_accuracy=count/n if not unknown else None,accuracy_bounds=[count/n,(count+unknown)/n],
        finite_violations=sum(r['violation'] or 0 for r in rows),refusals=sum(r['profile'] is None for r in rows),
        known_solver_failures=sum(r['status']=='solver_failure' for r in rows),unattempted=sum(r['status']=='not_run' for r in rows),
        total_warm_cost=cost,known_partial_cost=sum(t for t in times if t is not None),missing_times=sum(t is None for t in times),
        known_observed_method_spend=sum(r.get('observed_spent_seconds') or 0. for r in rows),
        cost_per_accurate=cost/count if cost is not None and not unknown and count else None,
        cost_scope='single realized full warm method wall; selector additionally charged frozen measured feature/inference proxy; not median4')


def build_metrics(plan,predictions,observations,bootstrap=True):
    by_key={(j['equation_id'],j['profile']):j['key'] for j in plan['jobs']};cases=[];paired=[];MC_rows=[]
    for c in plan['cases']:
        pid=c['equation_id'];profiles={}
        for p in ('R128','R256','MC1'):
            key=by_key.get((pid,p));profiles[p]=observations.get(key,missing()) if key else missing('not_measured')
        pred=predictions[pid]['prediction'];policies={}
        for name,p in [('selector',pred['selected_profile']),('constant_R128','R128'),('constant_R256','R256')]:
            overhead=pred['inference_seconds'] if name=='selector' else 0.
            if p is None:row=dict(status='abstain',correct=0,delivered=0,violation=0,error=None,seconds=0.)
            else:row=profiles[p]
            policies[name]=dict(row,profile=p,cost=row['seconds']+overhead if row['seconds'] is not None else None,ML_overhead=overhead)
        full=pid in plan['MC_equation_ids'];scope=('R128','R256','MC1') if full else ('R128','R256')
        known=all(profiles[p]['correct'] is not None for p in scope)
        eligible=[p for p in scope if profiles[p]['correct']==1 and profiles[p]['seconds'] is not None]
        fastest=min(eligible,key=lambda p:profiles[p]['seconds']) if known and eligible else None
        tie_adjusted=fastest
        eligible_R=[p for p in eligible if p!='MC1']
        if fastest=='MC1' and eligible_R:
            best_R=min(eligible_R,key=lambda p:profiles[p]['seconds'])
            if profiles['MC1']['seconds']>.9*profiles[best_R]['seconds']:tie_adjusted=best_R
        label='single_realization_fastest' if fastest else ('observed_no_accurate_delivery' if known else 'unknown')
        if full:
            MC_rows.append(dict(case=c,outcome=profiles['MC1'],prediction=pred['estimates']['MC1']))
        cases.append(dict(case=c,profiles=profiles,policies=policies,oracle_scope='three_profiles_measured_subset' if full else 'R_only;MC_unmeasured',
            fastest_profile=fastest,oracle_status=label,single_realization_10percent_tie_profile=tie_adjusted,
            full_three_profile_oracle_status=label if full else 'unknown_MC_not_measured',stable_time_winner_claim=False))
        paired.append(dict(cell=c['complexity_level']+'/'+c['regime'],**policies))
    primary={name:summarize([r[name] for r in paired],60) for name in ('selector','constant_R128','constant_R256')}
    comparisons={}
    if bootstrap and all(r[name]['correct'] is not None and r[name]['cost'] is not None for r in paired for name in primary):
        for base in ('constant_R128','constant_R256'):
            result=stratified_delta(paired,'selector',base)
            result['point_delta_correct_deliveries']=primary['selector']['confirmed_correct']-primary[base]['confirmed_correct']
            result['point_delta_seconds']=primary['selector']['total_warm_cost']-primary[base]['total_warm_cost']
            result['relative_cost_saving']=1-primary['selector']['total_warm_cost']/primary[base]['total_warm_cost']
            losses=sum(r[base]['correct']>r['selector']['correct'] for r in paired)
            result['lost_baseline_correct_deliveries']=losses
            result['gained_correct_deliveries']=sum(r['selector']['correct']>r[base]['correct'] for r in paired)
            denom=primary[base]['confirmed_correct'];result['baseline_correct_denominator']=denom
            result['conditional_loss_fraction']=losses/denom if denom else None
            result['individual_loss_upper_Bonferroni_two_comparisons']=min(1.,result['observed_baseline_correct_losses']+math.sqrt(math.log(40)/120))
            result['claim']='descriptive paired stratified bootstrap; conservative loss bound assumes independent families, not a guarantee of noninferiority'
            comparisons[base]=result
    scores={}
    for p,subset in [('R128',cases),('R256',cases),('MC1',[r for r in cases if r['case']['equation_id'] in plan['MC_equation_ids']])]:
        known=[r for r in subset if r['profiles'][p]['correct'] is not None]
        if known:
            ps=np.array([predictions[r['case']['equation_id']]['prediction']['estimates'][p]['probability_accuracy'] for r in known]);ys=np.array([r['profiles'][p]['correct'] for r in known]);cl=np.clip(ps,1e-10,1-1e-10)
            scores[p]=dict(known_outcomes=len(known),planned_families=len(subset),Brier=float(np.mean((ps-ys)**2)),logloss=float(np.mean(-ys*np.log(cl)-(1-ys)*np.log1p(-cl))))
        else:scores[p]=dict(known_outcomes=0,planned_families=len(subset),Brier=None,logloss=None)
        timed=[r for r in known if r['profiles'][p]['status']=='completed' and r['profiles'][p]['seconds']>0]
        ratios=[r['profiles'][p]['seconds']/predictions[r['case']['equation_id']]['prediction']['estimates'][p]['predicted_full_warm_seconds'] for r in timed]
        scores[p]['time_forecast']=dict(completed_calls=len(timed),median_actual_over_predicted=float(np.median(ratios)) if ratios else None,
            mean_absolute_log_ratio=float(np.mean(np.abs(np.log(ratios)))) if ratios else None,
            scope='single realized full warm call; no repeated-time precision or model tuning')
    mc_known=[r for r in MC_rows if r['outcome']['correct'] is not None];success=sum(r['outcome']['correct'] for r in mc_known)
    mc_uncertainty=None
    if len(mc_known)==20:
        eps=math.sqrt(math.log(40)/40);rate=success/20
        mc_uncertainty=dict(mean_success_fraction=rate,conservative_Hoeffding95_interval=[max(0.,rate-eps),min(1.,rate+eps)],
            assumptions='independent bounded family outcomes; cells may differ; not per-input or future-seed reliability',within_cell_bootstrap_used=False)
    return dict(primary=primary,paired_comparisons=comparisons,probability_scores=scores,cases=cases,
        MC_secondary=dict(planned_families=20,known_outcomes=len(mc_known),confirmed_successes=success,unknown_outcomes=20-len(mc_known),
            complete=len(mc_known)==20,
            profile_summary=summarize([dict(r['outcome'],profile='MC1',cost=r['outcome']['seconds']) for r in MC_rows],20),
            measured_three_profile_winner_counts=dict(Counter(r['single_realization_10percent_tie_profile'] or 'unknown_or_no_delivery'
                for r in cases if r['case']['equation_id'] in plan['MC_equation_ids'])),
            not_measured_families=40,measured_rows=MC_rows,uncertainty=mc_uncertainty,selected_MC_actions=0,
            full60_three_profile_oracle_available=False,conditional_MC_choice_quality_confirmed=False,
            limitation='one seed on each of20 fixed strata; no within-family reliability/timing spread; absent MC40 remains unknown'),
        fixed_N_primary_complete=all(z['known_outcomes']==60 and z['missing_times']==0 for z in primary.values()),
        production_blocked=True,model_or_policy_tuning=False,superiority_required_for_completion=False)


def analyze(out):
    from diagnostics.run_final140 import collect
    plan=read_json(out/'plan.json');validate_scope(plan)
    for n,h in plan['protected_sha256'].items():assert digest(ROOT/n)==h,n
    records,requests,budget,state=collect(out,plan)
    if not records and not requests:return dict(status='not_executed',required_calls=140,requests=0,records=0,final_evaluation_complete=False)
    from all_methods_dataset_loader import load_all_methods_equations
    data=ROOT/plan['data_dir'];eqs={e.equation_id:e for e in load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl')}
    ph=digest(out/'plan.json');observations={}
    for j in plan['jobs']:
        if j['key'] in records:
            exact=eqs[j['equation_id']].y_exact(np.array(j['target_x']))
            observations[j['key']]=decode(records[j['key']],j,ph,exact)
    pred={r['equation_id']:r for r in read_json(out/'frozen_predictions.json')['rows']};result=build_metrics(plan,pred,observations)
    accounting=len(records)==len(requests)==140 and not state['orphan_keys']
    main_complete=accounting and result['fixed_N_primary_complete']
    result.update(status=('complete' if result['MC_secondary']['complete'] else 'primary_complete_MC_secondary_incomplete') if main_complete else 'partial_or_censored',
        accounting_complete=accounting,records=len(records),requests=len(requests),physical_cap=140,
        final_evaluation_complete=main_complete,completion_scope='primary selector vs R baselines; MC completeness reported separately',model_unchanged=True)
    write_json(out/'analysis.json',result);return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);r=analyze(ap.parse_args().out.resolve())
    print(json.dumps({k:v for k,v in r.items() if k not in ('cases','MC_secondary')},ensure_ascii=False))
