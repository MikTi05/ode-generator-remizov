"""Nested family OOF on saved original-train features/labels only. No ODEs."""
from __future__ import annotations
import argparse,copy,json,os,platform,shutil,sys,time
from pathlib import Path
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from diagnostics.audit_probability_selector import read,write,sha,MODEL
from ml.probabilistic_selector import fit_head,save_head,load_head,choose
from ml.nested_probability_calibration import (
    Q_VALUES,outer_blocks,inner_blocks,require_disjoint,fit_sigmoid,apply_sigmoid,
    metric_summary,paired_uncertainty,baseline_default_choice,
)


def snapshot(out):
    return {str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and out not in p.parents
            and not any(k in p.parts for k in ('.venv','.git','__pycache__','.pytest_cache','.DS_Store'))}


def policy_metrics(rows,actions):
    from scipy.stats import beta
    known_cost=0.;missing=0;accurate=0.;baseline=0;lost=0.;gained=0.;predicted_cost=0.;details=[]
    for row,p in zip(rows,actions):
        b=row['observed']['R256']['k'];baseline+=b
        z=row['observed'][p] if p!='abstain' else dict(k=0,n=1)
        s=z['k']/z['n'];accurate+=s;lost+=max(0.,b-s);gained+=max(0.,s-b)
        if p!='abstain':
            t=row['observed_cost'][p]
            if t is None:missing+=1
            else:known_cost+=t
            predicted_cost+=row['estimates_raw'][p]['cost']['point']
        details.append(dict(equation_id=row['equation_id'],family_id=row['family_id'],action=p,
            accurate=s,baseline_accurate=b,lost=max(0.,b-s),gained=max(0.,s-b)))
    selected=sum(p!='abstain' for p in actions)
    delta=np.array([d['accurate']-d['baseline_accurate'] for d in details])
    rng=np.random.default_rng(2488);draws=delta[rng.integers(0,len(rows),size=(4000,len(rows)))].mean(axis=1)
    return dict(tasks=len(rows),families=len({r['family_id'] for r in rows}),actions={p:actions.count(p) for p in ('R128','R256','MC1','abstain')},
        accurate_deliveries=accurate,baseline_accurate_deliveries=baseline,selected=selected,
        inaccurate_selected=selected-accurate,abstentions=len(rows)-selected,delivery_accuracy=accurate/len(rows),
        conditional_accuracy=accurate/selected if selected else None,lost_baseline_accurate=lost,gained_accurate=gained,
        no_observed_individual_delivery_loss=lost==0,aggregate_delivery_delta=accurate-baseline,
        conditional_family_bootstrap95_delivery_delta_rate=np.quantile(draws,[.025,.975]).tolist(),
        descriptive_one_sided95_upper_loss_rate_among_baseline_correct=float(beta.ppf(.95,lost+1,baseline-lost)) if lost<baseline else 1.,
        observed_method_cost_sum=known_cost if not missing else None,missing_method_cost_cases=missing,
        known_method_cost_sum=known_cost,predicted_method_cost_sum=predicted_cost,
        observed_seconds_per_accurate=known_cost/accurate if not missing and accurate else None,
        cost_scope='method-only historical full wall; complete cost null if missing; predicted cost separate; new inference overhead not measured',
        uncertainty_scope='fixed OOF bootstrap can degenerate at zero differences; not a noninferiority guarantee',details=details)


def run(out):
    started=time.perf_counter();out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    for name in ('models','nodes','calibrators','sources'):(out/name).mkdir()
    shutil.copyfile(ROOT/'PROJECT_STATUS.md',out/'PROJECT_STATUS.before.md')
    write(out/'protected_before.json',snapshot(out))
    frozen=read(MODEL/'artifact_sha256.json')
    for name,h in frozen.items():assert sha(MODEL/name)==h,name
    targets=read(MODEL/'accuracy_targets.json');cost=read(MODEL/'full_cost_targets.json');old=read(MODEL/'family_cv.json')
    with np.load(MODEL/'features_train.npz',allow_pickle=False) as data:
        X=data['X'];original_indices=data['original_indices'].tolist();names=data['feature_names'].tolist()
    assert X.shape==(103,253) and np.isfinite(X).all()
    local={index:i for i,index in enumerate(original_indices)}
    rows=[]
    for r in targets:
        assert r['split']=='train' and r['index'] in local
        row=copy.deepcopy(r);row['original_index']=row['index'];row['index']=local[row['index']];rows.append(row)
    byprofile={p:[r for r in rows if r['profile']==p] for p in ('R128','R256')}
    assert len(byprofile['R128'])==103 and len(byprofile['R256'])==40
    blocks=outer_blocks(byprofile['R128'],old['folds'])
    protocol=dict(schema='nested_family_R_calibration_v1',source='saved original-train feature/target artifacts only',
        seed=2488,outer_blocks=blocks,inner_folds=3,inner_split='StratifiedGroupKFold seed248800+outer*100+profile_index',
        outer_design='6 disjoint family blocks: original3 paired40 held sets +3 fixed-hash extra50-family blocks; original paired point fits reproduced exactly',
        base_model='original fixed CatBoost CrossEntropy120/depth2/lr.05/l2_leaf10/no bootstrap/random_strength0;10 family-bootstrap replicas retained',
        calibration='single prespecified nonnegative sigmoid a*logit(p)+b, L2=2 toward(a=1,b=0), equal family weight; no calibration/threshold/model search',
        outer_exclusion='all outer-held family data excluded from every inner scaler/model and calibrator fit',
        final_model='raw R heads refit on all real train; final calibrators on full raw family OOF; never used for outer evaluation',
        thresholds=list(Q_VALUES),primary_threshold=.95,
        policies=['original predicted-p constraint, abstain if no eligible','research baseline_default: R256 fallback; switch to cheaper p>=q R128/MC; q does not constrain fallback'],
        policy_selection='none; report all predetermined q and both policies; no production choice based on outer scores',
        MC_cost='reuse exact frozen original outer OOF40 heads; all profiles remain candidates; no MC retraining',
        evaluation='probabilities R128103/90families, R25640/40; policies only complete-label paired40; no invented R256 labels on auxiliary63',
        uncertainty='family-weighted proper scores,4000 fixed-OOF family bootstrap, outer-fold dispersion; no per-input reliability bounds',
        excluded='eight previous validation families and all test outcomes/features/predictions excluded from pipeline',
        production='original API unchanged, new artifacts research_only/deployment_ready=false',numerical_calls=0)
    write(out/'protocol.json',protocol)
    for filename in ('features_train.npz','accuracy_targets.json','full_cost_targets.json'):shutil.copyfile(MODEL/filename,out/filename)
    write(out/'environment.json',dict(python=sys.version,platform=platform.platform(),threads=1,
        thread_environment={n:os.environ[n] for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS')}))
    access=['features_train.npz','accuracy_targets.json','full_cost_targets.json','family_cv.json','registry.json']
    write(out/'data_access.json',dict(semantic_inputs=[str((MODEL/f).relative_to(ROOT)) for f in access],
        original_validation_outcomes_used=False,test_outcomes_used=False,canonical_mixed_split_dataset_loaded=False,
        final_old_model_predictions_scope='train-only identity/reload checks, never used as OOF scores or calibrator inputs'))
    node_index={};cal_index={};oof={p:[] for p in byprofile};cost_index={(r['equation_id'],r['profile']):r for r in cost}

    def fit_node(profile,train_rows,excluded,node,seed,role):
        require_disjoint(train_rows,excluded)
        head=fit_head(X,train_rows,names,'probability',seed=seed)
        assert not set(head.metadata['families'])&set(excluded)
        np.testing.assert_array_equal(head.scaler.mean_,X[[r['index'] for r in train_rows]].mean(axis=0))
        entry=save_head(head,out/'models',node,names,X)
        record=dict(node=node,profile=profile,role=role,seed=seed,excluded_families=sorted(excluded),
            train_equation_ids=[r['equation_id'] for r in train_rows],train_families=sorted({r['family_id'] for r in train_rows}),
            entry=entry)
        write(out/'nodes'/f'{node}.json',record);node_index[node]=record
        return head

    for block in blocks:
        fold=block['fold'];held=set(block['held_families'])
        for pi,profile in enumerate(('R128','R256')):
            own=byprofile[profile];hold=[r for r in own if r['family_id'] in held]
            if not hold:continue
            train_rows=[r for r in own if r['family_id'] not in held];inner_pred={};inner_audit=[]
            seed=248800+fold*100+pi
            for j,split in enumerate(inner_blocks(train_rows,seed)):
                inner_train=[train_rows[i] for i in split['train_indices']];inner_hold=[train_rows[i] for i in split['held_indices']]
                excluded=held|set(split['held_families']);node=f'outer{fold}_{profile}_inner{j}'
                head=fit_node(profile,inner_train,excluded,node,seed+j+1,'inner')
                p=head.predict(X[[r['index'] for r in inner_hold]],names)['point']
                for r,v in zip(inner_hold,p):
                    assert r['equation_id'] not in inner_pred
                    inner_pred[r['equation_id']]=float(v)
                inner_audit.append(dict(node=node,held_families=split['held_families'],equation_ids=[r['equation_id'] for r in inner_hold]))
            assert set(inner_pred)=={r['equation_id'] for r in train_rows}
            cal=fit_sigmoid(train_rows,[inner_pred[r['equation_id']] for r in train_rows],held)
            cal.update(outer_fold=fold,profile=profile,excluded_families=sorted(held),inner_blocks=inner_audit,inner_probabilities=inner_pred)
            cal_key=f'outer{fold}_{profile}';write(out/'calibrators'/f'{cal_key}.json',cal);cal_index[cal_key]=cal
            node=f'outer{fold}_{profile}_point';head=fit_node(profile,train_rows,held,node,2488+fold*100,'outer')
            raw=head.predict(X[[r['index'] for r in hold]],names);cal_p=apply_sigmoid(cal,raw['point'])
            prior=sum(r['k'] for r in train_rows)/len(train_rows)
            for k,r in enumerate(hold):
                oof[profile].append(dict(equation_id=r['equation_id'],family_id=r['family_id'],split='train',
                    original_index=r['original_index'],index=r['index'],profile=profile,k=r['k'],n=r['n'],outer_fold=fold,
                    raw=float(raw['point'][k]),calibrated=float(cal_p[k]),train_constant=prior,
                    raw_model_spread=[float(raw['low'][k]),float(raw['high'][k])],base_node=node,calibrator_key=cal_key))
            print(json.dumps(dict(outer=fold,profile=profile,held=len(hold),inner_fits=3,calibrator=[cal['slope'],cal['intercept']])),flush=True)
    for p in oof:
        assert len(oof[p])==len(byprofile[p])==len({r['equation_id'] for r in oof[p]})
        assert {r['equation_id'] for r in oof[p]}=={r['equation_id'] for r in byprofile[p]}
    index={(r['equation_id'],p):r for p,rs in oof.items() for r in rs};paired=[]
    for original in old['oof']:
        pid=original['equation_id'];raw=copy.deepcopy(original['estimates']);calibrated=copy.deepcopy(raw)
        for p in ('R128','R256'):
            obs=index[pid,p]
            assert obs['outer_fold']==original['fold'] and obs['raw']==raw[p]['probability']['point'],(pid,p)
            c=cal_index[obs['calibrator_key']]
            for key in ('point','low','high'):
                calibrated[p]['probability'][key]=float(apply_sigmoid(c,[raw[p]['probability'][key]])[0])
        paired.append(dict(equation_id=pid,family_id=original['family_id'],outer_fold=original['fold'],split='train',
            observed=original['observed'],observed_cost=original['observed_cost'],estimates_raw=raw,estimates_calibrated=calibrated))
    write(out/'oof.json',oof);write(out/'paired_oof.json',paired)
    metrics={};fold_metrics={}
    for profile,own in oof.items():
        metrics[profile]={kind:metric_summary(own,[r[key] for r in own]) for kind,key in
            (('raw','raw'),('calibrated','calibrated'),('train_constant','train_constant'))}
        metrics[profile]['uncertainty']=paired_uncertainty(own,[r['raw'] for r in own],[r['calibrated'] for r in own])
        fold_metrics[profile]={}
        for fold in sorted({r['outer_fold'] for r in own}):
            rs=[r for r in own if r['outer_fold']==fold]
            fold_metrics[profile][str(fold)]={kind:metric_summary(rs,[r[kind] for r in rs]) for kind in ('raw','calibrated')}
    policies={}
    for mode in ('raw','calibrated'):
        policies[mode]={}
        for q in Q_VALUES:
            e=[r['estimates_'+mode] for r in paired]
            policies[mode][str(q)]=dict(
                risk_constraint=policy_metrics(paired,[choose(z,q)['action'] for z in e]),
                baseline_default=policy_metrics(paired,[baseline_default_choice(z,q) for z in e]))
    policies['constant_R256']=policy_metrics(paired,['R256']*40)
    final={};final_cal={}
    for p,rs in byprofile.items():
        head=fit_node(p,rs,set(),f'final_{p}',2488,'final_not_evaluated_on_training')
        original_entry=read(MODEL/'registry.json')['heads'][p+':probability']
        restored=load_head(original_entry,MODEL/'models')
        for key,v in head.predict(X,names).items():np.testing.assert_array_equal(v,restored.predict(X,names)[key])
        final[p]=node_index[f'final_{p}']['entry']
        obs={r['equation_id']:r['raw'] for r in oof[p]}
        cal=fit_sigmoid(rs,[obs[r['equation_id']] for r in rs],())
        cal.update(profile=p,scope='final research model only; fit on full raw family OOF, never scored on its own fit outcomes',
            OOF_equation_ids=[r['equation_id'] for r in rs])
        write(out/'calibrators'/f'final_{p}.json',cal);final_cal[p]=cal
    registry=dict(schema='nested_family_R_sigmoid_research_v1',status='research_only',deployment_ready=False,
        original_validation_calibrated=False,final_heads=final,calibrators={p:f'calibrators/final_{p}.json' for p in final},
        feature_names=names,feature_count=253,original_model_registry=str((MODEL/'registry.json').relative_to(ROOT)),
        original_model_registry_sha256=sha(MODEL/'registry.json'),original_production_code_sha256=sha(ROOT/'ml/probabilistic_selector.py'),
        no_production_dispatch=True,uncertainty='nested family diagnostics only; no per-input reliability bounds',
        source_profile_and_timing_context='unchanged original registry; MC and cost heads not refitted',q_values=Q_VALUES)
    write(out/'registry.json',registry);write(out/'model_index.json',node_index)
    write(out/'metrics.json',dict(probability=metrics,outer_folds=fold_metrics,policies=policies,
        original_paired40_point_predictions_reproduced_exactly=True,threshold_or_calibrator_selection_on_outer=False,
        original_validation_or_test_used=False,numerical_calls=0,MC_refitted=False,
        final_models_raw_predictions_exactly_original=True,fit_nodes=len(node_index),wall_seconds=time.perf_counter()-started))
    sourcefiles=['ml/nested_probability_calibration.py','diagnostics/train_nested_probability_calibration.py','ml/probabilistic_selector.py','ml/remizov_features.py']
    pins={str((MODEL/f).relative_to(ROOT)):sha(MODEL/f) for f in ('artifact_sha256.json','features_train.npz','accuracy_targets.json','full_cost_targets.json','family_cv.json','registry.json')}
    for name in sourcefiles:
        dest=out/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest);pins[name]=sha(ROOT/name)
    write(out/'source_sha256.json',pins)
    print(json.dumps(dict(complete=True,nodes=len(node_index),R128_OOF=len(oof['R128']),R256_OOF=len(oof['R256']),
        metrics=metrics,wall_seconds=time.perf_counter()-started)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out)
