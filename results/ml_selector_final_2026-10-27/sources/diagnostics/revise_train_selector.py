"""Three-level family OOF revision using permitted saved TRAIN only; no solves."""
from __future__ import annotations
import argparse,copy,json,math,os,platform,shutil,sys,time
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from ml.revision_evidence import read,sha,fingerprint
from ml.revision_selector import (PROFILES,MODES,fit_point,fit_binomial_sigmoid,family_weights,family_blocks,predict_bundle,candidate_grid,choose_policy,ResearchRevision)
from diagnostics.prepare_development8 import write_json


def prepare_artifacts(out):
    from ml.revision_evidence import prepare
    from diagnostics.prepare_development8 import snapshot
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(ROOT/'PROJECT_STATUS.md',out/'PROJECT_STATUS.before.md')
    write_json(out/'protected_before.json',snapshot(out),exclusive=True)
    b=prepare(out)
    for name in ('accuracy','cost','audit','raw','quarantine','pins'):write_json(out/(name+'.json'),b[name],exclusive=True)
    np.savez_compressed(out/'features_train.npz',X=b['X'],feature_names=np.array(b['names']),equation_keys=np.array(sorted({r['equation_key'] for r in b['accuracy']})))
    write_json(out/'paired_keys.json',b['paired'],exclusive=True)
    print(json.dumps(b['audit']['counts']),flush=True)


def run(out):
    began=time.perf_counter();assert out.is_absolute() and not (out/'registry.json').exists()
    accuracy=read(out/'accuracy.json');cost=read(out/'cost.json');paired=set(read(out/'paired_keys.json'));audit=read(out/'audit.json')
    permitted=audit['original_dataset_sha256']
    assert all(r['split']=='train' and r['dataset_sha256']==permitted for r in accuracy+cost)
    with np.load(out/'features_train.npz',allow_pickle=False) as z:X=z['X'];names=z['feature_names'].tolist();keys=z['equation_keys'].tolist()
    assert X.shape==(103,253) and len(keys)==len(set(keys))==103 and np.isfinite(X).all()
    by_key={r['equation_key']:r for r in accuracy};all_groups={r['family_id'] for r in accuracy};aindex={(r['equation_key'],r['profile']):r for r in accuracy}
    cindex={(r['equation_key'],r['profile']):r for r in cost}
    assert set(keys)==set(by_key) and all(keys[r['index']]==r['equation_key'] for r in accuracy+cost)
    canon={r['canonical_family_id']:r['family_id'] for r in accuracy}
    old=read(ROOT/'results/ml_probability_selector_2026-10-26/family_cv.json')['folds']
    blocks=[{canon[g] for g in f['held_families']} for f in old];used=set().union(*blocks)
    extra=family_blocks(all_groups-used,salt='revision_auxiliary_outer_v1')
    for i in range(3):blocks[i]|=extra[i]
    assert set().union(*blocks)==all_groups and sum(map(len,blocks))==len(all_groups)
    candidates=candidate_grid()
    protocol=dict(schema='train_revision_nested_family_v1',created_before_fits=True,seed=248827,
        input_files=['accuracy.json','cost.json','features_train.npz','paired_keys.json'],
        excluded_from_fit_and_selection=['previous validation8','all test','development8','previous F0047 postfit independent MC seed audit'],
        identity='canonical dataset CSV SHA256 + equation_id + expression_hash; globally identical topology held together',
        profiles=list(PROFILES),features=253,point_model=dict(model='CatBoost',iterations=120,depth=2,learning_rate=.05,l2_leaf_reg=10,threads=1,hyperparameter_search=False),
        outer_groups=[sorted(g) for g in blocks],outer_design='same original paired40 folds + fixed-hash partition of auxiliary50 groups',
        calibration='for every evaluation/selection bundle: 3 inner family OOF raw predictions; calibrator fit only within that bundle train; all ancestor-held families excluded from all descendants',
        policy_tuning='outer policy chosen only on separately nested calibrated inner3 OOF; no outer-held outcome reaches its model/calibrator/policy selection',
        probability='real k/n binomial likelihood; equal equation mass within topology, real n trial likelihood; regularized sigmoid L2=2, no artificial observations',
        cost='real full warm method wall only; median within cohort then equal-cohort median per equation/profile; missing costs never imputed as observations',
        R_cost_challenger='pooled log(seconds/n) regressor across real R128/R256 observations; n is known profile context, not a new input feature; linear-n scaling is a model assumption',
        R_error_challenger='pooled log(n*E_grid) regressor + profile-specific family-OOF residual ECDF; 1/n mean-scaling is a model assumption, not an enforced numerical theorem',
        candidates=candidates,policy_gate='inner empirical no individual delivery loss versus constant_R256, then max correct and min PREDICTED cost; not a population safety guarantee',
        evaluation='probability OOF on all labelled original train; policy accuracy paired40; observed cost on common complete-cost subset only, other totals remain unknown',
        final_selection='all-train nested-calibrated OOF selects final policy; this final selection is not the outer unbiased score; primary score is outer-specific nested selected policy',
        uncertainty='family residual spread, fixed-prediction family bootstrap; no per-input accuracy certificate',
        production_API='unchanged and blocked',numerical_calls=0,test_predictions=0,development_predictions=0)
    write_json(out/'protocol.json',protocol,exclusive=True)
    write_json(out/'environment.json',dict(python=sys.version,platform=platform.platform(),threads={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS')}),exclusive=True)
    node_index={};bundle_index={}

    def fit_raw(train,excluded,prefix):
        assert not train&excluded and train|excluded==all_groups
        heads={};paths={}
        train_a=[r for r in accuracy if r['family_id'] in train];train_c=[r for r in cost if r['family_id'] in train]
        for p in PROFILES:
            own=[r for r in train_a if r['profile']==p];key=p+'_prob';path=out/'models'/prefix/key
            heads[key]=fit_point(X,own,names,'probability',excluded,path);paths[key]=str(path.relative_to(out))
        data={
            'R_cost':[dict(r,target=math.log(r['seconds']/int(r['profile'][1:]))) for r in train_c if r['profile']!='MC1'],
            'MC_cost':[dict(r,target=math.log(r['seconds'])) for r in train_c if r['profile']=='MC1'],
            'R_error':[dict(r,target=math.log(max(r['error'],1e-14)*int(r['profile'][1:]))) for r in train_a if r['profile']!='MC1']}
        for key,rows in data.items():
            path=out/'models'/prefix/key;heads[key]=fit_point(X,rows,names,'regression',excluded,path);paths[key]=str(path.relative_to(out))
        for key,h in heads.items():
            node_index[prefix+'/'+key]=dict(path=paths[key],prefix=prefix,head=key,train_families=sorted(train),excluded_families=sorted(excluded),metadata=h.meta)
        return heads,paths

    def calibrated_bundle(train,held,prefix):
        assert train.isdisjoint(held);excluded=all_groups-train;raw_oof={};sub_audit=[]
        for j,subheld in enumerate(family_blocks(train,salt='revision_calibration_'+prefix)):
            subtrain=train-subheld;heads,paths=fit_raw(subtrain,all_groups-subtrain,prefix+f'/calibration{j}')
            own=[k for k in keys if by_key[k]['family_id'] in subheld];ix=[by_key[k]['index'] for k in own]
            values={p:h.predict(X[ix],names) for p,h in heads.items()}
            for i,k in enumerate(own):
                assert k not in raw_oof;raw_oof[k]={p:float(v[i]) for p,v in values.items()}
            sub_audit.append(dict(held_families=sorted(subheld),paths=paths,held_keys=own))
        train_a=[r for r in accuracy if r['family_id'] in train];train_c=[r for r in cost if r['family_id'] in train]
        calibration=dict(sigmoid={},error_cdf={},time_residuals={},excluded_families=sorted(excluded),subfolds=sub_audit,raw_OOF=raw_oof)
        for p in PROFILES:
            own=[r for r in train_a if r['profile']==p]
            calibration['sigmoid'][p]=fit_binomial_sigmoid(own,[raw_oof[r['equation_key']][p+'_prob'] for r in own],excluded)
            if p!='MC1':
                residuals=[math.log(max(r['error'],1e-14))-(raw_oof[r['equation_key']]['R_error']-math.log(int(p[1:]))) for r in own]
                calibration['error_cdf'][p]=dict(residuals=residuals,weights=family_weights(own).tolist(),equation_keys=[r['equation_key'] for r in own],families=sorted({r['family_id'] for r in own}),excluded_families=sorted(excluded))
            own=[r for r in train_c if r['profile']==p];residuals=[]
            for r in own:
                value=raw_oof[r['equation_key']]['MC_cost' if p=='MC1' else 'R_cost']+(0 if p=='MC1' else math.log(int(p[1:])))
                residuals.append(math.log(r['seconds'])-value)
            assert residuals
            calibration['time_residuals'][p]=dict(q10=float(np.quantile(residuals,.1)),q90=float(np.quantile(residuals,.9)),residuals=residuals,
                equation_keys=[r['equation_key'] for r in own],families=sorted({r['family_id'] for r in own}),excluded_families=sorted(excluded))
        heads,paths=fit_raw(train,excluded,prefix+'/point')
        own=[k for k in keys if by_key[k]['family_id'] in held];ix=[by_key[k]['index'] for k in own]
        variants=predict_bundle(heads,calibration,X[ix] if ix else X[:1],names) if own else []
        rows=[dict(equation_key=k,equation_id=by_key[k]['equation_id'],family_id=by_key[k]['family_id'],index=by_key[k]['index'],variants=v) for k,v in zip(own,variants)]
        record=dict(prefix=prefix,train_families=sorted(train),excluded_families=sorted(excluded),held_families=sorted(held),paths=paths,calibration=calibration)
        dest=out/'bundles'/prefix;dest.mkdir(parents=True,exist_ok=False);write_json(dest/'bundle.json',record,exclusive=True);bundle_index[prefix]=record
        print(json.dumps(dict(bundle=prefix,train_families=len(train),held=len(held),elapsed_seconds=time.perf_counter()-began)),flush=True)
        return rows,record

    outer=[];selection=[]
    for fold,held in enumerate(blocks):
        train=all_groups-held;inner=[]
        for j,innerheld in enumerate(family_blocks(train,salt='revision_policy_outer'+str(fold))):
            rows,_=calibrated_bundle(train-innerheld,innerheld,f'outer{fold}/policy_inner{j}');inner.extend(r for r in rows if r['equation_key'] in paired)
        policy,scores=choose_policy(inner,aindex,candidates)
        frozen=dict(fold=fold,selected_policy=policy,selection_scores=scores,inner_predictions=inner,outer_held_families=sorted(held),selected_before_outer_evaluation=True)
        write_json(out/f'outer{fold}_selection.json',frozen,exclusive=True);selection.append(frozen)
        rows,_=calibrated_bundle(train,held,f'outer{fold}/evaluation')
        for r in rows:r.update(outer_fold=fold,nested_selected_policy=policy)
        outer.extend(rows)
    assert len(outer)==103 and {r['equation_key'] for r in outer}==set(keys)
    write_json(out/'outer_oof.json',outer,exclusive=True)
    final_policy,final_scores=choose_policy([r for r in outer if r['equation_key'] in paired],aindex,candidates)
    write_json(out/'final_policy_selection.json',dict(policy=final_policy,scores=final_scores,scope='all-train OOF final selection, not primary unbiased OOF evaluation'),exclusive=True)
    _,final=calibrated_bundle(all_groups,set(),'final')
    registry=dict(schema='train_revision_nested_family_v1',status='research_only',deployment_ready=False,production_blocked=True,
        feature_names=names,feature_code_sha256=sha(ROOT/'ml/remizov_features.py'),original_dataset_sha256=permitted,
        final_heads=final['paths'],calibration=final['calibration'],final_policy=final_policy,
        candidates_preserved=list(PROFILES),accuracy_certificate=False,independent_validation=False,
        selected_on_train_only=True,development_validation_test_used_for_selection=False)
    write_json(out/'registry.json',registry,exclusive=True)
    write_json(out/'model_index.json',node_index,exclusive=True);write_json(out/'bundle_index.json',bundle_index,exclusive=True)
    model=ResearchRevision(out);inference=[]
    for k in sorted(paired):
        i=by_key[k]['index'];start=time.perf_counter_ns();z=model.select_features(X[i:i+1]);seconds=(time.perf_counter_ns()-start)*1e-9
        assert z[0]['action']=='abstain' and not z[0]['accuracy_certificate']
        inference.append(dict(equation_key=k,seconds=seconds,diagnostic_choice=z[0]['diagnostic_choice']))
    write_json(out/'inference_timing.json',dict(rows=inference,scope='warm one-input final-head inference/policy only; feature extraction measured separately; proxy for outer policy overhead'),exclusive=True)
    write_json(out/'run_summary.json',dict(training_seconds=time.perf_counter()-began,nodes=len(node_index),bundles=len(bundle_index),outer_cases=len(outer),outer_families=len(all_groups),
        final_policy=final_policy,numerical_calls=0,production_blocked=True,development_validation_test_fit=False),exclusive=True)
    print(json.dumps(read(out/'run_summary.json')),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);ap.add_argument('--action',choices=('prepare','train'),default='train');a=ap.parse_args()
    (prepare_artifacts if a.action=='prepare' else run)(a.out.resolve())
