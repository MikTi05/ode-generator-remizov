"""Saved-only family-OOF/validation audit. Never fit, calibrate or solve an ODE."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.probabilistic_selector import PROFILES, choose, probability_scores

MODEL = ROOT / 'results/ml_probability_selector_2026-10-26'
VALIDATION = ROOT / 'results/ml_probability_validation_2026-10-26'
PREFLIGHT = ROOT / 'results/ml_probability_validation_preflight_2026-10-26'
EDGES = (0., .05, .1, .25, .5, .75, .8, .9, .95, 1.)
Q_VALUES = (.5, .8, .95)  # Original frozen grid, never expanded or optimized.


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, allow_nan=False)
        f.write('\n')


def cp_interval(k, n, alpha=.05):
    """Two-sided exact binomial interval. Not an interval over correlated seeds."""
    from scipy.stats import beta
    if not 0 <= k <= n or n < 1:
        raise ValueError('invalid counts')
    return [float(beta.ppf(alpha / 2, k, n-k+1)) if k else 0.,
            float(beta.ppf(1-alpha / 2, k+1, n-k)) if k < n else 1.]


def zero_failure_required(q=.95, confidence=.95):
    if not 0 < q < 1 or not 0 < confidence < 1:
        raise ValueError('invalid probability')
    return math.ceil(math.log1p(-confidence) / math.log(q))


def fixed_bins(rows, profile):
    result = []
    for i, (lo, hi) in enumerate(zip(EDGES, EDGES[1:])):
        own = [r for r in rows if lo <= r['estimates'][profile]['probability']['point']
               and (r['estimates'][profile]['probability']['point'] < hi
                    or i == len(EDGES)-2 and r['estimates'][profile]['probability']['point'] == hi)]
        if not own:
            continue
        observed = [r['observed'][profile]['k']/r['observed'][profile]['n'] for r in own]
        predicted = [r['estimates'][profile]['probability']['point'] for r in own]
        result.append(dict(bin=[lo, hi],families=len({r['family_id'] for r in own}),
            equations=len(own),mean_prediction=statistics.mean(predicted),observed_rate=statistics.mean(observed),
            successes=sum(r['observed'][profile]['k'] for r in own),trials=sum(r['observed'][profile]['n'] for r in own)))
    return result


def probability_audit(rows, profile, seed=2488):
    import numpy as np
    from scipy.stats import binom
    k=np.array([r['observed'][profile]['k'] for r in rows]);n=np.array([r['observed'][profile]['n'] for r in rows])
    point=np.array([r['estimates'][profile]['probability']['point'] for r in rows])
    baseline=np.array([r['baseline_predictions'][profile+':probability'] for r in rows])
    low=np.array([r['estimates'][profile]['probability']['low'] for r in rows])
    high=np.array([r['estimates'][profile]['probability']['high'] for r in rows])
    rate=k/n
    # Bootstrap whole families of FIXED predictions, without re-fitting models.
    grouped=defaultdict(list)
    for r, y, p, b in zip(rows,rate,point,baseline):
        loss=y*(1-p)**2+(1-y)*p*p
        bloss=y*(1-b)**2+(1-y)*b*b
        grouped[r['family_id']].append([y,p,p-y,loss-bloss])
    family=np.asarray([np.mean(grouped[g],axis=0) for g in sorted(grouped)])
    rng=np.random.default_rng(seed)
    draws=np.mean(family[rng.integers(0,len(family),size=(4000,len(family)))],axis=1)
    bins=fixed_bins(rows,profile)
    tail={}
    for q in Q_VALUES:
        ix=point>=q;count=int(ix.sum());success=int(k[ix].sum());trials=int(n[ix].sum())
        tail[str(q)]=dict(equations=count,families=len({r['family_id'] for r,i in zip(rows,ix) if i}),
            successes=success,trials=trials,observed_rate=float(np.mean(rate[ix])) if count else None,
            mean_prediction=float(np.mean(point[ix])) if count else None,
            descriptive_binomial_CP95=cp_interval(success,trials) if count and np.all(n[ix]==1) else None,
            scope='OOF dependent across fits / validation diagnostic; CP is descriptive, not prospective certification')
    mc_counts=[]
    if profile=='MC1':
        for r,kk,nn,pp in zip(rows,k,n,point):
            if kk:
                mc_counts.append(dict(equation_id=r['equation_id'],k=int(kk),n=int(nn),predicted_probability=float(pp),
                    P_count_at_least_observed_if_point_p_true=float(binom.sf(kk-1,nn,pp)),
                    individual_seed_CP95=cp_interval(int(kk),int(nn))))
    return dict(scores=probability_scores(k,n,point),constant_scores=probability_scores(k,n,baseline),
        calibration_bias_mean_p_minus_rate=float(np.mean(point-rate)),
        descriptive_ECE=sum(b['equations']*abs(b['mean_prediction']-b['observed_rate']) for b in bins)/len(rows),
        bins=bins,tail=tail,prediction_range=[float(point.min()),float(point.max())],
        bootstrap_model_spread_median_width=float(np.median(high-low)),
        fixed_prediction_family_bootstrap95=dict(observed_rate=np.quantile(draws[:,0],[.025,.975]).tolist(),
            bias=np.quantile(draws[:,2],[.025,.975]).tolist(),Brier_delta_model_minus_constant=np.quantile(draws[:,3],[.025,.975]).tolist()),
        resamples=4000,bootstrap_scope='conditional descriptive fixed-prediction family bootstrap; no refit, calibration, per-input bound or prospective coverage',
        positive_MC_count_checks=mc_counts,
        MC_count_check_scope='binomial predictive diagnostic, conditional on frozen point p; neither multiple-testing claim nor true-p coverage')


def time_audit(rows, profile):
    import numpy as np
    own=[r for r in rows if r['observed_cost'][profile] is not None]
    details=[]
    for r in own:
        e=r['estimates'][profile]['cost'];t=r['observed_cost'][profile]
        details.append(dict(equation_id=r['equation_id'],observed=t,predicted=e['point'],
            ratio=e['point']/t,APE=abs(e['point']/t-1),log_error=math.log(e['point']/t),
            model_spread=[e['low'],e['high']],target_in_model_spread=e['low']<=t<=e['high'],
            supported=e['supported'],constant_APE=abs(r['baseline_predictions'][profile+':cost']/t-1)))
    return dict(known_cases=len(own),missing_cases=len(rows)-len(own),details=details,
        median_APE=float(np.median([r['APE'] for r in details])) if details else None,
        median_signed_ratio=float(np.median([r['ratio'] for r in details])) if details else None,
        mean_absolute_log_error=float(np.mean([abs(r['log_error']) for r in details])) if details else None,
        constant_median_APE=float(np.median([r['constant_APE'] for r in details])) if details else None,
        target_in_uncalibrated_model_spread=sum(r['target_in_model_spread'] for r in details),
        coverage_scope='observed family cost target inside uncalibrated model spread; not predictive interval coverage for a fresh run')


def policy_audit(rows, actions, overhead=None):
    overhead=overhead or {};cost=sum(overhead.get(r['equation_id'],0.) for r in rows)
    successes=0.;violations=0.;selected=0;missing=0
    for r,p in zip(rows,actions):
        if p=='abstain':
            continue
        selected+=1;z=r['observed'][p];successes+=z['k']/z['n'];violations+=1-z['k']/z['n']
        t=r['observed_cost'][p]
        if t is None:missing+=1
        else:cost+=t
    return dict(requests=len(rows),actions=dict(Counter(actions)),selected=selected,
        accurate_deliveries=successes,violations=violations,abstentions=len(rows)-selected,
        delivery_accuracy=successes/len(rows),conditional_accuracy=successes/selected if selected else None,
        abstention_rate=1-selected/len(rows),known_selected_cost_cases=selected-missing,missing_selected_cost_cases=missing,
        warm_cost=cost if not missing else None,
        seconds_per_accurate_delivery=cost/successes if not missing and successes else None,
        denominator_scope='R deterministic primary; MC seed-averaged empirical deliveries; no imputed costs, no free failure or abstention')


def audit_oof(cv,targets):
    rows=cv['oof'];assert len(rows)==len({r['family_id'] for r in rows})==40
    index={(r['equation_id'],r['profile']):r for r in targets}
    byfold={f['fold']:f for f in cv['folds']}
    checks=0
    for row in rows:
        assert row['split']=='train'
        fold=byfold[row['fold']];held=set(fold['held_families'])
        assert row['family_id'] in held and row['equation_id'] in fold['held_ids']
        for head in fold['heads'].values():
            assert not held & set(head['families'])
        for p in PROFILES:
            target=index[row['equation_id'],p]
            assert target['split']=='train' and target['family_id']==row['family_id']
            assert row['observed'][p]=={k:target[k] for k in ('k','n')}
            checks+=1
    return dict(passed=True,families=40,folds=len(byfold),all_head_family_exclusions_verified=True,
        observed_target_matches=checks,OOF_scope='40 paired families only, not OOF over all103 R128 rows/90 families',
        dependence='OOF models share training data; saved fold head/scaler audit checked, no prospective confidence guarantee')


def validation_rows(frozen,cases,targets,costs):
    byid={r['equation_id']:r for r in frozen['rows']};result=[]
    for c in cases:
        pid=c['equation_id'];row=dict(equation_id=pid,family_id=c['family_id'],split='validation',
            estimates=byid[pid]['estimates'],observed={},observed_cost={},baseline_predictions={})
        for p in PROFILES:
            z=c['profiles'][p]
            row['observed'][p]=dict(k=z['accurate'] if p=='MC1' else int(z['primary_accurate']),n=4 if p=='MC1' else 1)
            row['observed_cost'][p]=z['timing']['median']
            train=[t for t in targets if t['profile']==p]
            row['baseline_predictions'][p+':probability']=sum(t['k'] for t in train)/sum(t['n'] for t in train)
            row['baseline_predictions'][p+':cost']=math.exp(statistics.mean(math.log(t['seconds']) for t in costs if t['profile']==p))
        result.append(row)
    return result


def acquisition_plan(registry,metadata_path,case_count=20):
    """IDs selected from input strata/hash alone; costs read only AFTER selection."""
    from scipy.stats import binom
    fitted=set().union(*(set(h['metadata']['families']) for h in registry['heads'].values()))
    # Only train metadata is retained. No test outcomes/features/predictions used.
    train=[]
    with metadata_path.open() as f:
        for line in f:
            m=json.loads(line)
            if m['split']=='train':
                train.append({k:m[k] for k in ('equation_id','family_id','expression_hash','regime','complexity_level','split')})
    families=defaultdict(list)
    for row in train:
        if row['family_id'] not in fitted:families[row['family_id']].append(row)
    hashkey=lambda s:hashlib.sha256(('probability_calibration_pilot_v1|'+s).encode()).hexdigest()
    strata=defaultdict(list)
    for family,rows in families.items():
        representative=min(rows,key=lambda r:hashkey(r['equation_id']))
        strata[representative['regime']+'|'+representative['complexity_level']].append(representative)
    for values in strata.values():values.sort(key=lambda r:hashkey(r['family_id']))
    order=sorted(strata,key=hashkey);chosen=[]
    while len(chosen)<case_count:
        for key in order:
            if strata[key] and len(chosen)<case_count:chosen.append(strata[key].pop(0))
        assert any(strata.values()) or len(chosen)==case_count
    factor=read(ROOT/'reports/ml_selector_contract/pilot_p2_2026-10-26/analysis/summary.json')['calibration']['coverage20']['median']
    costs=read(MODEL/'full_cost_targets.json')
    rtime={p:statistics.median(t['seconds'] for t in costs if t['profile']==p) for p in ('R128','R256')}
    pins={};mc_sum=0.
    for row in chosen:
        pid=row['equation_id'];raw=[]
        for episode in range(4):
            path=ROOT/f'results/mc1000_x0_2026-10-25/attempts/{pid}_seed{episode}.json'
            d=read(path);z=d['result']
            assert d['expression_hash']==row['expression_hash'] and d['metadata_identity']['split']=='train'
            assert d['status']=='completed' and z['success'] and z['M']==1024 and z['x']==[0.]
            assert z['dt_requested']==1/128 and z['backend']=='numba'
            assert z['compilation_seconds']==z['numba_import_seconds']==0
            raw.append(z['runtime_seconds']);pins[str(path.relative_to(ROOT))]=sha(path)
        one=statistics.mean(raw)*132*factor;mc_sum+=4*one
        row.update(estimated_MC1_one_call_seconds=one,saved_x0_N=z['n_steps'],saved_x0_T=z['T'],
            MC1_actual_work_per_call=33*4096*z['n_steps'])
    total=mc_sum+case_count*sum(rtime.values())
    return dict(execution_authorized=False,budget_approved=False,scope='bounded TRAIN calibration-support pilot, not a winner study or reliability certificate',
        original_train_families=len({m['family_id'] for m in train}),fitted_union_families=len(fitted),
        available_never_fit_families=len(families),proposed_new_families=case_count,
        remaining_never_fit_after_pilot=len(families)-case_count,
        selection='round-robin regime/complexity strata among never-fit original train families; fixed SHA256 order; one equation by fixed input ID hash; no accuracy/time outcome selection',
        cases=chosen,calls_per_case=dict(R128=1,R256=1,MC1=4),maximum_calls=6*case_count,
        seed_proposal=[248800070100+i for i in range(4)],seed_proposal_requires_full_nonreuse_check=True,
        stages=[dict(new_families=10,maximum_calls=60),dict(new_families=10,maximum_calls=60)],
        stopping='hard stop at20 families/120 calls or approved resource limit; no automatic extension to find successes; preserve all outcomes and censoring',
        experiment_contract='same frozen numerical profiles, grid33, wall contract, guardian/one worker/thread/RSS512; full identity/G/T/N/seeds/source/cap preflight required; no solver execution authorized here',
        R_repetition_scope='one deterministic accuracy measurement and one full warm time each; training/calibration cost target only, not median4 timing winner confirmation',
        optional_full_timing=dict(additional_R_repeats=3,extra_calls=6*case_count,total_calls=12*case_count,authorized=False),
        central_scenario_seconds=total,proposed_cap_seconds=math.ceil(2*total+300),MC_central_seconds=mc_sum,R_one_call_scenarios=rtime,
        cost_formula='sum4(mean saved x0 M1024 time *132 *P2 coverage20 factor)+20*(median R128 full wall+median R256 full wall); cap ceil(2*scenario+300)',
        cost_is_bound=False,cost_includes_future_setup=False,old_validation_budget_reusable=False,
        probability_at_least_one_new_MC_positive_family_at_p_3of40=float(binom.sf(0,case_count,3/40)),
        MC_support_scenario='3/40 ever-success train family rate, selected historical cohort; descriptive scenario only, not a forecast or stopping criterion',
        evidence_pins=pins)


def sample_size_audit():
    from scipy.stats import binom
    rare=3/40
    return dict(q=.95,confidence=.95,all_success_independent_required=zero_failure_required(),
        one_failure_independent_required=next(n for n in range(2,1000) if binom.cdf(1,n,.05)<=.05),
        unit='independent accepted family-level delivery under a frozen policy; seeds on one equation do not multiply independent families',
        minimum_MC_new_positive_families_for_exploratory_total10=7,
        ten_MC_positive_families_is='engineering diversity milestone, not a statistical guarantee or new hard training gate',
        MC_ever_success_rate_scenario=rare,MC_ever_success_family_CP95=cp_interval(3,40),
        expected_new_families_for7_more_positive=7/rare,
        probability_7plus_new_positive_in100=float(binom.sf(6,100,rare)),
        new_families_for80pct_probability_of7=next(n for n in range(7,10000) if binom.sf(6,n,rare)>=.8),
        MC_scenario_caveat='historical nonrandom40-family cohort; no exchangeability/constant rate assumption established',
        minimum_R128_additional_time_families_to_engineering8=4,
        R128_eight_family_rule='existing software support flag only; not calibration sample sufficiency',
        universal_minimum_calibration_families=None,
        why_not='depends on risk, selection coverage, bins and positive/negative diversity; 20-family pilot cannot certify q=.95',
        confidence_scope='59/59 is exact one-sided binomial lower bound >=.95 only under independent exchangeable post-freeze accepted outcomes; not each-input reliability and not a guarantee that collecting59 families yields59 successes')


def run(out):
    out=out.resolve();out.mkdir(parents=True,exist_ok=False);(out/'sources').mkdir()
    shutil.copyfile(ROOT/'PROJECT_STATUS.md',out/'PROJECT_STATUS.before.md')
    frozen_checks={}
    for folder in (MODEL,VALIDATION,PREFLIGHT):
        m=read(folder/'artifact_sha256.json')
        for name,h in m.items():assert sha(folder/name)==h,(folder.name,name)
        frozen_checks[folder.name]=len(m)
    cv=read(MODEL/'family_cv.json');targets=read(MODEL/'accuracy_targets.json');costs=read(MODEL/'full_cost_targets.json')
    frozen=read(VALIDATION/'frozen_predictions.json');cases=read(VALIDATION/'cases.json');registry=read(MODEL/'registry.json')
    source_pins={}
    for folder in (MODEL,VALIDATION,PREFLIGHT):
        source_pins[str((folder/'artifact_sha256.json').relative_to(ROOT))]=sha(folder/'artifact_sha256.json')
    for path in (MODEL/'family_cv.json',MODEL/'accuracy_targets.json',MODEL/'full_cost_targets.json',MODEL/'registry.json',
                 VALIDATION/'cases.json',VALIDATION/'frozen_predictions.json',VALIDATION/'analysis.json',
                 ROOT/'ml/probabilistic_selector.py',ROOT/'chernov_solver.py'):
        source_pins[str(path.relative_to(ROOT))]=sha(path)
    oof_check=audit_oof(cv,targets)
    val=validation_rows(frozen,cases,targets,costs)
    assert not {r['family_id'] for r in val}&{r['family_id'] for r in cv['oof']}
    checks={r['equation_id']:r for r in frozen['rows']}
    policy={};reasons={};traces=[]
    overhead={r['equation_id']:r['feature_and_inference_seconds'] for r in read(VALIDATION/'policy_overhead.json')['rows']}
    for name,rows,over in (('OOF',cv['oof'],{}),('validation',val,overhead)):
        policy[name]={};reasons[name]={}
        for q in Q_VALUES:
            key=str(q);policy[name][key]={};reasons[name][key]={}
            for mode,robust in (('point',False),('model_spread',True)):
                decisions=[choose(r['estimates'],q,robust=robust) for r in rows]
                actions=[r['action'] for r in decisions]
                policy[name][key][mode]=policy_audit(rows,actions,over)
                reasons[name][key][mode]=dict(Counter(d['reason'] for d in decisions))
                if name=='validation':
                    assert all(d==checks[r['equation_id']]['diagnostic_choices'][key][mode] for d,r in zip(decisions,rows))
                    if q==.95:
                        traces.extend(dict(equation_id=r['equation_id'],mode=mode,**d) for r,d in zip(rows,decisions))
        policy[name]['baselines']={p:policy_audit(rows,[p]*len(rows),over if False else {}) for p in PROFILES}
        policy[name]['production_guard']=policy_audit(rows,['abstain']*len(rows),over)
    # Primary validation policy accuracy must NOT use four deterministic repeats or MC seed averaging.
    original=read(VALIDATION/'analysis.json')
    for q in Q_VALUES:
        for mode in ('point','model_spread'):
            z=policy['validation'][str(q)][mode];old=original['policies'][str(q)][mode]
            assert z['accurate_deliveries']==old['accurate_primary'] and z['selected']==old['selected']
            assert math.isclose(z['warm_cost'],old['warm_policy_seconds'],abs_tol=1e-9)
    numerical=[]
    for c in cases:
        pid=c['equation_id'];rs=[]
        for p in ('R128','R256'):
            path=VALIDATION/'records'/f'{pid}_{p}_validation_episode0.json';r=read(path);z=r['result']
            rs.append(dict(profile=p,E_grid=r['E_grid'],boundary_state_max_abs=z['boundary_state_max_abs'],
                outside_query_values=z['outside_query_values'],total_error_certified=z['total_error_certified'],
                refinement_controls_performed=z['refinement_controls_performed']))
        numerical.append(dict(equation_id=pid,E128_over_E256=rs[0]['E_grid']/rs[1]['E_grid'],observations=rs))
    data=ROOT/read(VALIDATION/'plan.json')['data_dir']
    acquisition=acquisition_plan(registry,data/'all_methods_equations_v1_metadata.jsonl')
    source_pins.update(acquisition['evidence_pins'])
    source_pins[str((data/'all_methods_equations_v1_metadata.jsonl').relative_to(ROOT))]=sha(data/'all_methods_equations_v1_metadata.jsonl')
    result=dict(schema='saved_probability_selector_audit_v1',source_manifests_verified=frozen_checks,
        OOF_audit=oof_check,cohorts={name:{p:dict(probability=probability_audit(rows,p),time=time_audit(rows,p)) for p in PROFILES}
            for name,rows in (('OOF',cv['oof']),('validation',val))},policies=policy,reasons=reasons,traces=traces,
        numerical_diagnostics=numerical,original_q_grid=Q_VALUES,no_threshold_search=True,
        no_calibrator_fit=True,no_ML_fit=True,numerical_calls=0,test_outcomes_used=False,
        production_guard=dict(action_hardcoded='abstain',status='research_only',registry_deployment_ready=registry['deployment_ready'],
            validation_calibrated=registry['validation_calibrated'],R128_time_families=4,R128_time_supported=False,
            note='select_method always initializes abstain and never assigns diagnostic choice to action; changing registry alone would not enable deployment'))
    write(out/'analysis.json',result);write(out/'acquisition_plan.json',acquisition);write(out/'sample_size.json',sample_size_audit())
    write(out/'source_sha256.json',source_pins)
    write(out/'verification.json',dict(passed=True,source_manifests_verified=frozen_checks,OOF_family_target_checks=120,
        frozen_decisions_reconstructed=8*3*2,validation_costs_and_accuracy_match=True,
        train_only_acquisition=True,new_train_families=20,new_calls=0,ML_fits=0,calibrator_fits=0,test_used=False,
        bootstrap_is_fixed_prediction_resampling=True,parameters_selected_on_validation=False))
    for name in ('diagnostics/audit_probability_selector.py','tests/test_probability_selector_audit.py'):
        path=ROOT/name;dest=out/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    print(json.dumps(dict(passed=True,OOF_families=40,validation_families=8,new_calls=0,
        proposed_families=20,proposed_calls=120,scenario_seconds=acquisition['central_scenario_seconds'],
        proposed_unapproved_cap=acquisition['proposed_cap_seconds']),sort_keys=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out)
