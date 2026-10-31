"""Preregistered final-test evaluator. Reads saved records; never fits or solves."""
from __future__ import annotations
import argparse,json,math,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from diagnostics.prepare_development8 import read_json,write_json,digest


def stratified_delta(rows,a,b):
    groups=defaultdict(list)
    for r in rows:groups[r['cell']].append(r)
    assert len(groups)==20 and all(len(v)==3 for v in groups.values())
    rng=np.random.default_rng(24882760);values=[]
    for rs in groups.values():
        matrix=np.array([[r[a]['correct']-r[b]['correct'],r[a]['cost']-r[b]['cost']] for r in rs])
        values.append(matrix[rng.integers(0,3,size=(10000,3))].sum(axis=1))
    draws=np.sum(values,axis=0)/60
    loss=np.mean([r[b]['correct']>r[a]['correct'] for r in rows])
    return dict(stratified_family_bootstrap95=np.quantile(draws,[.025,.975],axis=0).T.tolist(),columns=['correct_delivery_delta','seconds_per_request_delta'],
        observed_baseline_correct_losses=loss,conservative_Hoeffding95_upper_loss=min(1.,float(loss+math.sqrt(math.log(20)/120))),
        assumptions='bootstrap descriptive for stratified fresh-family design; Hoeffding upper conditional on independent-family sampling; neither is per-input certification')


def analyze(out):
    plan=read_json(out/'plan.json');jobs=plan['jobs'];ph=digest(out/'plan.json')
    for n,h in plan['protected_sha256'].items():assert digest(ROOT/n)==h,n
    paths=list((out/'records').glob('*.json'));requests=list((out/'events').glob('*.request.json'))
    if len(paths)!=720:
        return dict(status='not_executed' if not paths and not requests else 'incomplete',records=len(paths),requests=len(requests),required=720,
            final_evaluation_complete=False,no_model_or_policy_tuning=True)
    assert len(requests)==720;records={};streams=set()
    for j in jobs:
        r=read_json(out/'records'/f"{j['key']}.json");assert r['job']==j and r['plan_sha256']==ph
        z=r.get('result') or {};error=None;delivered=False
        if z.get('success'):
            assert r['status']=='completed' and z['x']==j['target_x'] and len(z['f'])==33 and np.isfinite(z['f']).all()
            assert z['problem_id']==j['equation_id'] and z['T']==j['expected_T']
            error=float(np.max(abs(np.array(z['f'])-np.array(r['y_exact']))));assert error==r['E_grid'] and (error<=.01)==r['accuracy_met'];delivered=True
            if j['profile']=='MC1':
                assert z['seed_entropy']==j['config']['seed'] and z['n_steps']==j['expected_steps']
                assert z['n_path_steps']==z['n_normal_draws']==j['numerical_work'] and z['completed_paths']==[4096]*33 and z['tail_tolerance_met']
                for key in z['point_spawn_keys']:
                    address=(z['seed_entropy'],tuple(key));assert address not in streams;streams.add(address)
            else:
                assert z['n']==j['config']['n'] and z['n_operator_applications']==j['numerical_work'] and z['work_grid_points']==2081
        else:assert r['status'] in ('solver_failure','external_time_limit','external_memory_limit','worker_exception','invalid_result')
        seconds=r.get('method_wall_seconds')
        if seconds is not None:
            assert seconds>0 and math.isclose(seconds,(r['timer_stop_ns']-r['timer_start_ns'])*1e-9,abs_tol=1e-12,rel_tol=0)
        records[j['key']]=dict(raw=r,error=error,correct=bool(delivered and error<=.01),delivered=delivered,seconds=seconds)
    predicted={r['equation_id']:r['prediction'] for r in read_json(out/'frozen_predictions.json')['rows']};cases=[];rows=[]
    for c in plan['cases']:
        profile_results={}
        for p in ('R128','R256','MC1'):
            rs=[records[j['key']] for j in jobs if j['equation_id']==c['equation_id'] and j['profile']==p];assert len(rs)==4
            if p!='MC1':
                finite=[r for r in rs if r['delivered']]
                if finite:assert all(r['raw']['result']['f']==finite[0]['raw']['result']['f'] for r in finite)
            times=[r['seconds'] for r in rs];median=float(np.median(times)) if all(t is not None for t in times) else None
            profile_results[p]=dict(correct=[r['correct'] for r in rs],delivered=[r['delivered'] for r in rs],errors=[r['error'] for r in rs],times=times,median_seconds=median)
        pred=predicted[c['equation_id']];chosen=pred['selected_profile'];measure={}
        for policy,p in [('selector',chosen),('constant_R128','R128'),('constant_R256','R256')]:
            overhead=pred['inference_seconds'] if policy=='selector' else 0.
            z=profile_results.get(p);correct=int(z['correct'][0]) if z else 0;delivered=int(z['delivered'][0]) if z else 0
            cost=(z['median_seconds']+overhead if z and z['median_seconds'] is not None else (overhead if not z else None))
            measure[policy]=dict(profile=p,correct=correct,delivered=delivered,violation=int(delivered and not correct),abstain=int(p is None),cost=cost,
                primary_method_seconds=z['times'][0] if z else 0.,sensitivity_correct=z['correct'] if z else [False]*4)
        feasible=[p for p,z in profile_results.items() if z['correct'][0] and z['median_seconds'] is not None]
        oracle=min(feasible,key=lambda p:profile_results[p]['median_seconds']) if feasible else None
        rs=[p for p in feasible if p!='MC1']
        if oracle=='MC1' and rs:
            best=min(rs,key=lambda p:profile_results[p]['median_seconds'])
            if profile_results['MC1']['median_seconds']>.9*profile_results[best]['median_seconds']:oracle=best
        cases.append(dict(case=c,profiles=profile_results,policies=measure,primary_observed_fastest=oracle,prediction=pred))
        rows.append(dict(cell=c['complexity_level']+'/'+c['regime'],**measure))
    summary={}
    for name in ('selector','constant_R128','constant_R256'):
        correct=sum(r[name]['correct'] for r in rows);times=[r[name]['cost'] for r in rows];full=all(t is not None for t in times)
        summary[name]=dict(correct=correct,requests=60,violations=sum(r[name]['violation'] for r in rows),abstentions=sum(r[name]['abstain'] for r in rows),
            delivery_failures=sum(not r[name]['delivered'] and not r[name]['abstain'] for r in rows),delivery_accuracy=correct/60,
            total_warm_cost=sum(times) if full else None,cost_per_accurate=sum(times)/correct if full and correct else None,
            uncertainty_scope='4 seeds repeated within each family; primary episode0, no best seed')
    probability={}
    for p in ('R128','R256','MC1'):
        probs=np.array([c['prediction']['estimates'][p]['probability_accuracy'] for c in cases]);ys=np.array([sum(c['profiles'][p]['correct'])/4 for c in cases]);cl=np.clip(probs,1e-10,1-1e-10)
        probability[p]=dict(brier=float(np.mean(ys*(1-probs)**2+(1-ys)*probs**2)),logloss=float(np.mean(-ys*np.log(cl)-(1-ys)*np.log1p(-cl))),
            successful_calls=int(ys.sum()*4),trials=240,ever_success_families=sum(y>0 for y in ys),four_success_families=sum(y==1 for y in ys))
    complete=all(r['raw']['status'] in ('completed','solver_failure') for r in records.values())
    comparisons={}
    if all(r[p]['cost'] is not None for r in rows for p in ('selector','constant_R128','constant_R256')):
        comparisons={b:stratified_delta(rows,'selector',b) for b in ('constant_R128','constant_R256')}
    result=dict(status='complete' if complete else 'complete_with_invalid_or_censored_outcomes',final_evaluation_complete=complete,
        records=720,requests=720,policies=summary,profile_probabilities=probability,paired_comparisons=comparisons,cases=cases,
        choices=Counter(c['prediction']['selected_profile'] for c in cases),MC_selected_families=sum(c['prediction']['selected_profile']=='MC1' for c in cases),
        MC_choice_claim='unconfirmed when selected support absent/too small; observed seed success is not new-call reliability',
        models_policy_unchanged=True,production_remains_blocked=True,no_followup_pilot_or_retraining_by_this_evaluator=True,
        completion_does_not_require_ML_superiority_or_MC_winners=True)
    write_json(out/'final_analysis.json',result);return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);r=analyze(ap.parse_args().out.resolve())
    print(json.dumps({k:v for k,v in r.items() if k!='cases'},ensure_ascii=False))
