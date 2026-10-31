"""Train-only diagnostic probability/cost selector from immutable saved results."""
from __future__ import annotations
import argparse,hashlib,importlib.metadata,json,os,platform,shutil,sys,time
from pathlib import Path
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from diagnostics.train_method_selector import write,snapshot
from ml.remizov_features import load_inputs,VERSION as FEATURE_VERSION
from ml.selector_evidence import read,sha,CONTRACT
from ml.probabilistic_evidence import prepare
from ml.probabilistic_selector import VERSION,PROFILES,fit_head,save_head,estimates_from_heads,choose,probability_scores,ProbabilisticSelector

SEED=2488;Q_VALUES=(.5,.8,.95)


def evaluate_policy(rows,choices):
    selected=[(r,p) for r,p in zip(rows,choices) if p!='abstain']
    accurate=sum(r['observed'][p]['k']/r['observed'][p]['n'] for r,p in selected)
    costs=[r['observed_cost'][p] for r,p in selected if r['observed_cost'][p] is not None]
    return dict(tasks=len(rows),selected=len(selected),actions={p:choices.count(p) for p in (*PROFILES,'abstain')},
        empirical_seed_averaged_correct_deliveries=accurate,empirical_delivery_accuracy=accurate/len(rows),
        empirical_conditional_accuracy=accurate/len(selected) if selected else None,
        observed_cost_cases=len(costs),observed_cost_coverage=len(costs)/len(selected) if selected else None,
        saved_cost_sum_seconds=sum(costs) if len(costs)==len(selected) else None,
        total_actual_policy_wall=None,cost_regret=None,reason='saved-only family OOF; no executed policy or true conditional-p oracle; missing full cost never imputed')


def run(out):
    started=time.perf_counter();out.mkdir(parents=True,exist_ok=False)
    for name in ('models','sources'):(out/name).mkdir()
    shutil.copy2(ROOT/'PROJECT_STATUS.md',out/'PROJECT_STATUS.before.md')
    write(out/'protected_before.json',snapshot(out))
    protocol=dict(schema=VERSION,seed=SEED,source='saved only',new_solver_calls=0,
        objective='min predicted typical full warm wall subject to caller-supplied probability q; realized MC accuracy is unknowable before sampling',
        profiles=PROFILES,main_fixed_pair='R256_MC1; R128 is included with sparse-time warning, never silently excluded',
        targets='probability: actual k/n binomial sufficient statistics; cost: log observed full warm wall (median when4 available, single otherwise)',
        repetitions='MC P1/P2 four unique semantic seeds per equation; R deterministic repeats deduplicated; no synthetic winner labels or rows',
        training='only original train; 3-fold StratifiedGroupKFold by family, stratify MC ever-success3families; auxiliary R128 train cases exclude held families',
        model=dict(probability_loss='CrossEntropy',cost_loss='RMSE(log wall)',iterations=120,depth=2,learning_rate=.05,l2_leaf_reg=10,bootstrap_type='No',random_strength=0,threads=1),
        bootstrap_replicas=10,uncertainty='family bootstrap10/90 model-spread only, NOT calibrated probability bounds or CI',
        sparse_cost_head='less than8 train families: empirical log-time constant; no claimed learned timing regression',
        risk_sensitivity=list(Q_VALUES),thresholds_not_tuned=True,calibrator='not fit: no original validation outcomes',
        test='no test outcomes read for training/tuning/evaluation; no test predictions',
        fresh_F0047='known historical independent4-seed group reserved until weights freeze; post-hoc seen-family audit, not blind family validation',
        deployment='blocked; diagnostics expose both methods and uncertainty',
        gate_20plus5='belongs to direct MC-winner classification; not required to fit probability/cost heads; replaced by held-family calibration/cost/policy evidence')
    write(out/'protocol.json',protocol)
    write(out/'environment.json',dict(python=sys.version,platform=platform.platform(),versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','scikit-learn','catboost')},threads=1))
    inputs=load_inputs(ROOT/read(CONTRACT/'plan.json')['data_dir']);X=inputs['X'];names=inputs['feature_names'];bundle=prepare(inputs)
    assert X.shape==(1000,253) and np.isfinite(X).all()
    write(out/'data_audit.json',{k:bundle[k] for k in ('counts','raw_checks','exclusions')})
    write(out/'accuracy_targets.json',bundle['accuracy']);write(out/'full_cost_targets.json',bundle['cost'])
    trainindices=sorted({r['index'] for r in bundle['accuracy']})
    assert all(inputs['splits'][i]=='train' for i in trainindices)
    np.savez_compressed(out/'features_train.npz',X=X[trainindices],original_indices=np.array(trainindices),feature_names=np.array(names))
    write(out/'feature_schema.json',dict(version=FEATURE_VERSION,names=names,code_sha256=sha(ROOT/'ml/remizov_features.py'),input_grid_sha256=inputs['grid_sha256'],
        legacy_descriptors='pre_T/n-derived remain fixed legacy n128 descriptors, not actual numerical profile parameters',
        forbidden='ID/family/split/seed/exact/error/runtime/status/outputs absent from X'))
    aindex={(r['equation_id'],r['profile']):r for r in bundle['accuracy']};cindex={(r['equation_id'],r['profile']):r for r in bundle['cost']}
    paired=bundle['paired_ids'];idx={str(pid):i for i,pid in enumerate(inputs['ids'])}
    from sklearn.model_selection import StratifiedGroupKFold
    families=np.array([aindex[pid,'MC1']['family_id'] for pid in paired]);strata=np.array([int(aindex[pid,'MC1']['k']>0) for pid in paired])
    cv=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED);folds=[];oof=[]
    def fit_all(held,foldseed):
        heads={};baselines={}
        for profile in PROFILES:
            for kind,source in [('probability',bundle['accuracy']),('cost',bundle['cost'])]:
                rows=[r for r in source if r['profile']==profile and r['family_id'] not in held]
                key=profile+':'+kind;heads[key]=fit_head(X,rows,names,kind,seed=foldseed)
                baselines[key]=sum(r['k'] for r in rows)/sum(r['n'] for r in rows) if kind=='probability' else float(np.exp(np.mean(np.log([r['seconds'] for r in rows]))))
        return heads,baselines
    for fold,(train,hold) in enumerate(cv.split(np.zeros((40,1)),strata,families)):
        held=set(families[hold]);assert not held&set(families[train])
        heads,baseline=fit_all(held,SEED+fold*100)
        ids=[paired[i] for i in hold];est=estimates_from_heads(heads,X[[idx[p] for p in ids]],names)
        foldaudit=dict(fold=fold,held_families=sorted(held),held_ids=ids,positive_MC_held_families=int(sum(strata[hold])),heads={k:h.metadata for k,h in heads.items()},baselines=baseline)
        assert all(not held&set(h.metadata['families']) for h in heads.values());folds.append(foldaudit)
        for pid,estimates in zip(ids,est):
            oof.append(dict(equation_id=pid,family_id=aindex[pid,'MC1']['family_id'],split='train',fold=fold,estimates=estimates,
                observed={p:{k:aindex[pid,p][k] for k in ('k','n')} for p in PROFILES},
                observed_cost={p:cindex.get((pid,p),{}).get('seconds') for p in PROFILES},baseline_predictions=baseline))
        print(json.dumps(dict(fold=fold,held=len(ids),fit_complete=True)),flush=True)
    assert len(oof)==40 and len({r['equation_id'] for r in oof})==40
    metrics={}
    for p in PROFILES:
        k=[r['observed'][p]['k'] for r in oof];n=[r['observed'][p]['n'] for r in oof]
        pred=[r['estimates'][p]['probability']['point'] for r in oof];base=[r['baseline_predictions'][p+':probability'] for r in oof]
        known=[r for r in oof if r['observed_cost'][p] is not None]
        def costmetrics(base=False):
            predicted=np.array([r['baseline_predictions'][p+':cost'] if base else r['estimates'][p]['cost']['point'] for r in known]);truth=np.array([r['observed_cost'][p] for r in known])
            return dict(cases=len(known),mean_absolute_log_error=float(np.mean(abs(np.log(predicted/truth)))),median_absolute_percentage_error=float(np.median(abs(predicted/truth-1))))
        metrics[p]=dict(probability=probability_scores(k,n,pred),constant_probability=probability_scores(k,n,base),cost=costmetrics(),constant_cost=costmetrics(True))
    policies={}
    for q in Q_VALUES:
        policies[str(q)]={}
        for mode,robust in [('point',False),('model_spread',True)]:
            choices=[choose(r['estimates'],q,robust=robust)['action'] for r in oof]
            policies[str(q)][mode]=evaluate_policy(oof,choices)
        pairchoices=[choose({p:r['estimates'][p] for p in ('R256','MC1')},q)['action'] for r in oof]
        policies[str(q)]['fixed_R256_MC1_point']=evaluate_policy(oof,pairchoices)
    policies['constant_methods']={p:evaluate_policy(oof,[p]*40) for p in PROFILES}
    write(out/'family_cv.json',dict(folds=folds,oof=oof,metrics=metrics,policies=policies,
        original_validation_outcomes_used=False,test_used=False,calibration='descriptive held-train-family probability diagnostics only'))
    heads,_=fit_all(set(),SEED)
    entries={key:save_head(h,out/'models',key.replace(':','_'),names,X[[idx[p] for p in paired]]) for key,h in heads.items()}
    registry=dict(schema=VERSION,feature_version=FEATURE_VERSION,feature_code_sha256=sha(ROOT/'ml/remizov_features.py'),feature_names=names,input_grid_sha256=inputs['grid_sha256'],
        profile_context={p:read(CONTRACT/'plan.json')['profiles'][p] for p in PROFILES},
        timing_context=dict(scope='warm_end_to_end_method_wall_v1',status='historical_train_cohorts_not_uniformly_validated'),
        status='research_only_uncalibrated',deployment_ready=False,validation_calibrated=False,heads=entries,
        old_MC_winner_gate_applied=False,uncertainty='uncalibrated family-bootstrap model spread; no accuracy certificates')
    write(out/'registry.json',registry);frozen_hash=sha(out/'registry.json')
    # From here weights/policy are frozen. The four MC outcomes were not fit targets.
    pid='AMV1_F0047_S00';seen=estimates_from_heads(heads,X[[idx[pid]]],names)[0]
    fresh=[read(ROOT/p) for p in bundle['seed_audit_paths']]
    assert len({r['seed'] for r in fresh})==4
    freshk=sum(r['E_grid']<=.01 for r in fresh)
    write(out/'seen_family_fresh_seed_audit.json',dict(equation_id=pid,registry_sha256_before_outcomes=frozen_hash,estimates=seen,
        training_MC_counts={k:aindex[pid,'MC1'][k] for k in ('k','n')},fresh_k=freshk,fresh_n=4,
        fresh_errors=[r['E_grid'] for r in fresh],used_for_fit_or_calibration=False,
        scope='post-hoc known-history seed audit at a seen family; not blind family holdout',
        beta_binomial_predictive_interval_not_claimed=True,probability_scores=probability_scores([freshk],[4],[seen['MC1']['probability']['point']])))
    api=ProbabilisticSelector(out/'registry.json');data=ROOT/read(CONTRACT/'plan.json')['data_dir']
    with np.load(data/'all_methods_equations_v1_ml.npz',allow_pickle=False) as z:values=z['X'][idx[pid]]
    t=time.perf_counter();served=api.select_method(values,inputs['x_grid'],inputs['metadata'][pid],q=.95,profile_context=registry['profile_context'],timing_context=registry['timing_context']);inference=time.perf_counter()-t
    write(out/'serving_check.json',dict(result=served,feature_and_inference_seconds=inference,models_loaded_before_timer=True))
    # Planning uses validation identities/features only; no validation outcomes.
    valfamilies=sorted(set(inputs['families'][inputs['splits']=='validation']));rng=np.random.default_rng(SEED)
    chosen=sorted(rng.choice(valfamilies,size=8,replace=False));proposal=[]
    for family in chosen:
        ids=[str(pid) for pid,g,s in zip(inputs['ids'],inputs['families'],inputs['splits']) if g==family and s=='validation']
        pid=min(ids,key=lambda p:hashlib.sha256(('probability_validation_v1|'+p).encode()).hexdigest())
        proposal.append(dict(equation_id=pid,family_id=str(family),split='validation',expression_hash=inputs['metadata'][pid]['expression_hash']))
    write(out/'NEXT_VALIDATION.json',dict(execution_authorized=False,status='proposal_only',cases=proposal,
        selection='8 original validation families sampled with seed2488; one ID by stable hash; no outcomes or test used',
        main_pair='R256/MC1',calls_per_case=dict(R256=4,MC1=4),total_calls=64,
        optional_R128_cost_comparison=dict(extra_calls=32,total_three_profile_calls=96),
        purpose='first held-family probability/cost/policy diagnostic; NOT sufficient calibration guarantee; stop and report absent MC support if observed',
        model_and_policy_frozen_sha256=frozen_hash,thresholds=Q_VALUES,calibration_fit_requires_separate_design=True,
        no_test=True,no_benchmark1000=True))
    sourcefiles=['ml/probabilistic_evidence.py','ml/probabilistic_selector.py','ml/remizov_features.py','ml/selector_evidence.py','diagnostics/train_probabilistic_selector.py','requirements-ml.txt']
    for n in sourcefiles:
        dest=out/'sources'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    write(out/'source_sha256.json',{**bundle['source_sha256'],**{n:sha(ROOT/n) for n in sourcefiles}})
    write(out/'run_summary.json',dict(counts=bundle['counts'],metrics=metrics,selector_status=registry['status'],
        head_count=len(heads),MC_winner_classifier_trained=False,MC_probability_head_trained=True,
        fresh_seed_fit_excluded=True,original_validation_fit=False,test_fit_or_tuning=False,
        numerical_calls=0,benchmark1000_calls=0,pipeline_seconds=time.perf_counter()-started))
    print(json.dumps(read(out/'run_summary.json'),sort_keys=True),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);run(ap.parse_args().out.resolve())
