"""Frozen-model retrospective validation and prospective design. No fit or ODE."""
from __future__ import annotations
import argparse,copy,csv,hashlib,math,os,shutil,statistics,sys,time
from collections import Counter
from pathlib import Path
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[n]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from diagnostics.audit_probability_selector import read,write,sha,MODEL
from ml.probabilistic_selector import load_head,estimates_from_heads,choose
from ml.nested_probability_calibration import apply_sigmoid,baseline_default_choice
from ml.remizov_features import coefficient_features,CERT_FIELDS
NESTED=ROOT/'results/ml_nested_calibration_2026-10-26'
VAL=ROOT/'results/ml_probability_validation_2026-10-26'
TIMING=ROOT/'results/ml_switch5_r_timing_2026-10-26'
DATA=ROOT/'data/all_methods_v1_1000_2026-10-18'
Q=(.5,.8,.95)


def snapshot(out):
    return {str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and out not in p.parents
        and not any(k in p.parts for k in ('.venv','.git','__pycache__','.pytest_cache','.DS_Store'))}


def input_metadata():
    """Retain TRAIN metadata and the8 explicitly requested validation inputs only."""
    allowed={c['equation_id'] for c in read(VAL/'plan.json')['cases']};metadata={}
    with (DATA/'all_methods_equations_v1_metadata.jsonl').open() as f:
        for line in f:
            m=__import__('json').loads(line)
            if m['split']!='train' and m['equation_id'] not in allowed:continue
            metadata[m['equation_id']]={**{k:m[k] for k in CERT_FIELDS},
                **{k:m[k] for k in ('equation_id','family_id','split','expression_hash','structure_signature','regime','complexity_level')},
                'serialized_ast':{k:m['serialized_ast'][k] for k in ('a_raw','b_raw','c_raw')}}
    return metadata


def input_coefficients(ids):
    with np.load(DATA/'all_methods_equations_v1_ml.npz',allow_pickle=False) as z:
        assert z['channel_order'].tolist()==['a','b','c','g']
        index={str(pid):i for i,pid in enumerate(z['equation_id'])}
        return z['X'][[index[pid] for pid in ids]],z['x_grid'].copy()


def matrix(ids,metadata,names):
    coefficients,x=input_coefficients(ids);features=[]
    for pid,c in zip(ids,coefficients):
        row=coefficient_features(c,x,metadata[pid]);assert list(row)==names
        features.append([row[n] for n in names])
    return np.asarray(features)


def leaf_box_bound(values,counts,scale,bias):
    start=0;low=[];high=[]
    for count in counts:
        group=values[start:start+count];assert len(group)==count
        low.append(float(min(group)));high.append(float(max(group)));start+=count
    assert start==len(values)
    lo=math.fsum(low);hi=math.fsum(high)
    lower=min(scale*lo,scale*hi)+bias;upper=max(scale*lo,scale*hi)+bias
    guard=1e-10*(1+abs(bias)+abs(scale)*math.fsum(abs(float(v)) for v in values))
    from scipy.special import expit
    return dict(scale=scale,bias=bias,tree_min_leaves=low,tree_max_leaves=high,
        guarded_logit_lower=lower-guard,guarded_logit_upper=upper+guard,
        probability_lower=float(np.nextafter(expit(lower-guard),0.)),
        probability_upper=float(np.nextafter(expit(upper+guard),1.)),rounding_guard=guard,
        argument='every routed leaf is bounded by its tree min/max; sum bound ignores cross-tree compatibility, so it is conservative')


def calibrate(estimates,calibrators):
    result=copy.deepcopy(estimates)
    for p in ('R128','R256'):
        for key in ('point','low','high'):
            result[p]['probability'][key]=float(apply_sigmoid(calibrators[p],[estimates[p]['probability'][key]])[0])
    return result


def policy_metrics(cases,actions,overhead):
    selected=correct=bad=abstain=0;method=0.;rows=[];fastest=0;feasible=0
    for c,p in zip(cases,actions):
        eligible=[m for m,z in c['profiles'].items() if z['primary_accurate']]
        best=min(eligible,key=lambda m:c['profiles'][m]['timing']['median']) if eligible else None
        feasible+=bool(eligible)
        if p=='abstain':abstain+=1;good=False;cost=0.
        else:
            z=c['profiles'][p];selected+=1;good=z['primary_accurate'];correct+=good;bad+=not good
            cost=z['timing']['median'];method+=cost
            fastest+=good and p==best
        rows.append(dict(equation_id=c['equation_id'],action=p,accurate=bool(good),fastest_accurate_profile=best,method_seconds=cost))
    warm=method+overhead
    return dict(requests=len(cases),actions=dict(Counter(actions)),selected=selected,correct=correct,violations=bad,abstentions=abstain,
        delivery_accuracy=correct/len(cases),conditional_accuracy=correct/selected if selected else None,
        coverage=selected/len(cases),observed_feasible_cases=feasible,fastest_accurate_choices=fastest,
        feasible_case_delivery_rate=correct/feasible if feasible else None,method_cost_seconds=method,
        measured_inference_seconds=overhead,warm_cost_seconds=warm,seconds_per_correct=warm/correct if correct else None,
        details=rows,scope='offline choices, saved median4 method wall, fixed episode0 accuracy; retrospective8, not new solver dispatch')


def experiment_budget(families,R128,R256,MC):
    central=4*families*(R128+R256+statistics.mean(MC))
    return dict(families=families,maximum_calls=12*families,calls_per_family=dict(R128=4,R256=4,MC1=4),
        central_scenario_seconds=central,proposed_cap_seconds=math.ceil(2*central+300),
        median_MC_sensitivity_seconds=4*families*(R128+R256+statistics.median(MC)),
        maximum_historical_MC_sensitivity_seconds=4*families*(R128+R256+max(MC)),
        formula='4*n*(matched train median R128 + matched train median R256 + coverage20 mean MC1 family median); cap ceil(2*scenario+300)',
        approved=False,execution_authorized=False,not_upper_bound=True,
        preflight_still_required='new input IDs/AST/topology uniqueness/analytic G,T,N/resources/seeds and case-specific budget BEFORE any solver call')


def run(out):
    out=out.resolve();out.mkdir(parents=True,exist_ok=False);(out/'sources').mkdir()
    shutil.copyfile(ROOT/'PROJECT_STATUS.md',out/'PROJECT_STATUS.before.md');write(out/'protected_before.json',snapshot(out))
    counts={}
    for folder in (MODEL,NESTED,VAL,TIMING):
        manifest=read(folder/'artifact_sha256.json')
        for n,h in manifest.items():assert sha(folder/n)==h,(folder.name,n)
        counts[folder.name]=len(manifest)
    for name,h in read(ROOT/'reports/ml_selector_contract/plan.json')['dataset_sha256'].items():
        assert sha(DATA/name)==h,name
    registry=read(NESTED/'registry.json');original=read(MODEL/'registry.json')
    protocol=dict(schema='frozen_generalization_diagnostic_v1',primary='nested final calibrated R128/R256 + original frozen MC/cost; baseline_default q=.95',
        registry_sha256=sha(NESTED/'registry.json'),policy_code_sha256=sha(ROOT/'ml/nested_probability_calibration.py'),
        thresholds=list(Q),no_tuning=True,no_fit=True,no_solver=True,test_used=False,
        retrospective8='previously exposed validation, after prior development; not independent final evaluation',
        comparator='constant_R256, same saved cohort; warm method cost + separately measured actual feature/inference overhead',
        production='blocked and reported separately; no API changes',
        prospective_scope='input-only never-fit TRAIN frame scan; true independent final cohort needs new families and historical exposure audit')
    write(out/'protocol.json',protocol)
    names=registry['feature_names'];calibrators={p:read(NESTED/registry['calibrators'][p]) for p in ('R128','R256')}
    began=time.perf_counter();heads={}
    for p in ('R128','R256','MC1'):
        for kind in ('probability','cost'):
            fresh=p in ('R128','R256') and kind=='probability'
            entry=registry['final_heads'][p] if fresh else original['heads'][p+':'+kind]
            heads[p+':'+kind]=load_head(entry,NESTED/'models' if fresh else MODEL/'models')
    load_seconds=time.perf_counter()-began
    metadata=input_metadata();valcases=read(VAL/'plan.json')['cases'];ids=[c['equation_id'] for c in valcases]
    coefficients,x=input_coefficients(ids);frozen=[]
    for c,values in zip(valcases,coefficients):
        pid=c['equation_id'];assert metadata[pid]['split']=='validation'
        t=time.perf_counter_ns();f=coefficient_features(values,x,metadata[pid]);assert list(f)==names
        raw=estimates_from_heads(heads,np.asarray([[f[n] for n in names]]),names)[0];cal=calibrate(raw,calibrators)
        primary_action=baseline_default_choice(cal,.95)
        elapsed=(time.perf_counter_ns()-t)*1e-9
        frozen.append(dict(equation_id=pid,family_id=c['family_id'],split='validation',raw=raw,calibrated=cal,
            primary_action=primary_action,feature_and_prediction_seconds=elapsed,choices={str(q):dict(baseline_default=baseline_default_choice(cal,q),
                risk_constraint=choose(cal,q)['action'],raw_baseline_default=baseline_default_choice(raw,q)) for q in Q}))
    # Freeze all predictions/decisions before joining stored validation errors/times.
    write(out/'retrospective_predictions.json',dict(rows=frozen,registry_sha256=protocol['registry_sha256'],model_loading_seconds=load_seconds,
        fit=False,policy_changed_after_outcomes=False))
    oldpred={r['equation_id']:r['estimates'] for r in read(VAL/'frozen_predictions.json')['rows']}
    assert all(r['raw']==oldpred[r['equation_id']] for r in frozen)
    cases=read(VAL/'cases.json');overhead=sum(r['feature_and_prediction_seconds'] for r in frozen)
    assert [c['equation_id'] for c in cases]==ids
    # Independently replay existing errors only; no canonical equation functions or solvers.
    raw_records=[]
    for path in (VAL/'records').glob('*.json'):
        r=read(path);z=r['result'];err=float(np.max(abs(np.asarray(z['f'])-r['y_exact'])))
        assert err==r['E_grid'] and (err<=.01)==r['accuracy_met'] and r['status']=='completed' and r['split']=='validation'
        raw_records.append(r)
    assert len(raw_records)==96
    policies={}
    for q in Q:
        policies[str(q)]={name:policy_metrics(cases,[r['choices'][str(q)][name] for r in frozen],overhead)
            for name in ('baseline_default','risk_constraint','raw_baseline_default')}
    policies['constant_R256']=policy_metrics(cases,['R256']*8,0.)
    policies['production_research_guard']=policy_metrics(cases,['abstain']*8,overhead)
    write(out/'retrospective_results.json',dict(primary_q=.95,policies=policies,old_cases_sha256=sha(VAL/'cases.json'),
        records_verified=96,actual_timing='saved full method medians; no new solver run',models_loaded_once_seconds=load_seconds,
        comparison_scope='retrospective exposed8 only; no selection of best q/version from these data'))
    fit=set().union(*(set(n['train_families']) for n in read(NESTED/'model_index.json').values()))
    fit|=set().union(*(set(e['metadata']['families']) for e in original['heads'].values()))
    pool=sorted(pid for pid,m in metadata.items() if m['split']=='train' and m['family_id'] not in fit)
    X=matrix(pool,metadata,names);est=estimates_from_heads(heads,X,names);pool_rows=[]
    for pid,e in zip(pool,est):
        cal=calibrate(e,calibrators)
        pool_rows.append(dict(equation_id=pid,family_id=metadata[pid]['family_id'],split='train',
            regime=metadata[pid]['regime'],complexity=metadata[pid]['complexity_level'],
            probability={p:cal[p]['probability']['point'] for p in ('R128','R256','MC1')},
            action_q95=baseline_default_choice(cal,.95),labels_read=False))
    assert len(pool)==250 and len({r['family_id'] for r in pool_rows})==50
    write(out/'never_fit_input_scan.json',dict(rows=pool_rows,families=50,equations=250,actions=dict(Counter(r['action_q95'] for r in pool_rows)),
        no_outcome_read=True,no_model_or_policy_change=True,not_a_test=True,
        warning='never fit is not historically unseen; existing dataset has old x0 experiments, so do not claim this as untouched final cohort'))
    from catboost import Pool
    reach={}
    for p in ('R128','R256','MC1'):
        head=heads[p+':probability'];model=head.models[0];values=model.get_leaf_values();counts_leaf=model.get_tree_leaf_counts();scale,bias=model.get_scale_and_bias()
        b=leaf_box_bound(values,counts_leaf,scale,float(bias));Z=head.scaler.transform(X)
        leaves=model.calc_leaf_indexes(Pool(Z,feature_names=names),thread_count=1).astype(np.int64)
        offset=np.concatenate([np.zeros(1,dtype=np.int64),np.cumsum(counts_leaf,dtype=np.int64)[:-1]])
        reconstructed=scale*np.sum(values[leaves+offset],axis=1)+bias
        actual=model.predict(Pool(Z,feature_names=names),prediction_type='RawFormulaVal',thread_count=1)
        assert np.allclose(reconstructed,actual,rtol=0,atol=1e-12)
        b['leaf_sum_reconstruction_max_error']=float(np.max(abs(reconstructed-actual)))
        b['calibrated_probability_upper']=float(apply_sigmoid(calibrators[p],[b['probability_upper']])[0]) if p!='MC1' else b['probability_upper']
        reach[p]=b
    assert reach['MC1']['calibrated_probability_upper']<.95
    write(out/'model_reachability.json',dict(heads=reach,MC_q95_action_impossible=True,
        proof_scope='global conservative leaf-sum bound for frozen point model and unchanged MC calibration; not a bound on TRUE MC accuracy',
        does_not_prove='MC1 is useless or Remizov dominates all equations; only that this policy cannot select MC at q=.95'))
    costs=read(MODEL/'full_cost_targets.json');cohort={r['equation_id']:r['cohort'] for r in csv.DictReader((ROOT/'reports/ml_selector_contract/pilot_manifest.csv').open())}
    mc=[r['seconds'] for r in costs if r['profile']=='MC1' and cohort[r['equation_id']]=='coverage']
    timing=read(TIMING/'analysis.json');r={p:statistics.median(c[p]['timing']['median'] for c in timing['cases']) for p in ('R128','R256')}
    pilot=experiment_budget(8,r['R128'],r['R256'],mc);formal=experiment_budget(59,r['R128'],r['R256'],mc)
    proposal=dict(execution_authorized=False,budget_approved=False,
        conclusion='no numerical final test can demonstrate three-action generalization of current q95 policy: MC action structurally unreachable',
        minimum_next_experiment=dict(purpose='fresh-family MC1 usefulness/futility + frozen-policy diagnostic, NOT final proof',**pilot),
        fresh_family_design=dict(generate_now=False,generation_version='all_methods_v1 unchanged',input_pool_master_seed=2488000901,
            proposed_pool_families=40,samples_per_family=1,select_only_original_train_role=True,
            selected_families=8,selection='fixed input-only SHA256 order balanced across L1..L4 and regimes; discard topology signatures shared with fit/calibration/revealed families before outcomes',
            identity='dataset SHA256 + family topology signature + canonical equation ID; IDs alone may repeat between generated cohorts',
            input_preflight='materialize8 inputs/G/T/N and freeze choices/budget BEFORE seeking execution approval; no existing test use',
            main_preservation='existing dataset/parameters/results remain unchanged; fresh input-only cohort in new directory'),
        evaluation_rules=dict(primary_policy='unchanged calibrated R256-default q=.95; baseline constant_R256; production guard reported separately',
            frozen_diagnostic_other_q=list(Q),primary_accuracy='episode0 fixed before calls; all4 trials separately, no best seed or aggregation into larger M',
            accuracy='E_grid33<=.01 for every return, exact after timing; compare deliveries, violations, abstentions, feasible/none and fastest observed accurate profile',
            timing='median4/ranges/IQR/MAD/full warm wall in one cohort;10% practical benefit and disjoint ranges; no dropped outliers',
            MC_usefulness='observed accurate MC with>=10% faster median than cheapest accurate R is a candidate only;4/4 does not certify new-call p>=.95',
            stop='hard budget/call cap, keep all failures; no extra seeds/families to hunt winners; do not change q/models on packet'),
        resources=dict(workers=1,threads=1,RSS_MiB=512,guardian=True,owner_lease_seconds=300,thermal_stop_state=2,immutable_results=True,persist_original_deadlines=True),
        seed_proposal=[248800080100+i for i in range(4)],seed_nonreuse_preflight_required=True,
        final_evidence_lower_bound=dict(**formal,conditional_only=True,
            estimand='if chosen risk goal is>=.95 accurate among independently selected outputs, one-sided95% lower bound under zero errors needs59 accepted independent family outcomes',
            missing='59 requests do not ensure59 accepted or correct; pooled59 does not validate rare MC choices; each MC-specific reliability claim needs adequate separately preregistered MC-selected support',
            switch_gate='before final numerical evaluation, nonzero policy switches/MC support must exist in a new frozen version; current version cannot meet MC gate',
            not_universal_minimum=True,with_one_failure_required=93,
            future_freeze='new data/calibration/MC model version require separate authorization; no such work now; final holdout must be fresh relative to pilot'),
        why_not_old20='already frozen final policy chooses R256 on20/20; that packet cannot show switch advantage',
        MC_data_first='real success/failure on independent families + comparable costs versus fastest accurate R; no artificial winners or old20+5 gate; do not force MC selection if profile has no genuine benefit')
    write(out/'prospective_experiment.json',proposal)
    pins={str((f/'artifact_sha256.json').relative_to(ROOT)):sha(f/'artifact_sha256.json') for f in (MODEL,NESTED,VAL,TIMING)}
    pins.update({str((DATA/n).relative_to(ROOT)):h for n,h in read(ROOT/'reports/ml_selector_contract/plan.json')['dataset_sha256'].items()})
    for p in ('R128','R256'):pins[str((NESTED/registry['calibrators'][p]).relative_to(ROOT))]=sha(NESTED/registry['calibrators'][p])
    for name in ('ml/probabilistic_selector.py','ml/nested_probability_calibration.py','ml/remizov_features.py','diagnostics/evaluate_selector_generalization.py'):
        pins[name]=sha(ROOT/name);dest=out/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    write(out/'source_sha256.json',pins)
    write(out/'verification.json',dict(passed=True,frozen_source_artifacts_verified=counts,validation_raw_errors_recomputed=96,
        original_raw8_predictions_exact=True,leaf_reconstruction_inputs=250,models_calibrators_policy_unchanged=True,
        no_model_fit=True,no_solver_calls=True,test_used=False,previous8_marked_retrospective=True,
        primary_policy_predeclared=True,never_fit_scan_no_labels=True,MC_structural_ceiling_verified=True))
    print(dict(primary=policies['0.95']['baseline_default'],reach_MC=reach['MC1']['probability_upper'],pool_actions=Counter(r['action_q95'] for r in pool_rows),pilot=pilot))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
