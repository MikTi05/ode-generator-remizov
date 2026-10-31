"""Reproducible observed-fastest pipeline using saved data only.

No trajectory generation, numerical methods, benchmark or synthetic examples.
Full classifier is gated; train-only family CV is explicitly diagnostic.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib.metadata,json,os,platform,shutil,sys,time
from collections import Counter
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from ml.remizov_features import load_inputs, VERSION
from ml.selector_labels import LABELS,LABEL_VERSION,support,training_gate,fingerprint
from ml.selector_evidence import load_saved_evidence,read,sha,check,CONTRACT
from ml.method_selector import train_only_diagnostic,fit_catboost,predict_model,classification_metrics,retrospective_policy,MethodSelector

SEED=2488

def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return value.item()
    return value


def write(path,value):
    with Path(path).open('x') as f:json.dump(clean(value),f,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False);f.write('\n')


def jsonl(path,rows):
    with Path(path).open('x') as f:
        for r in rows:f.write(json.dumps(clean(r),sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')


def snapshot(out):
    return {str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and not p.is_relative_to(out) and not any(k in p.parts for k in ('.venv','.git','__pycache__','.pytest_cache','.DS_Store'))}


def save_fitted(out,prefix,scaler,model,names,X):
    modelpath=out/'models'/f'{prefix}.cbm';scalepath=out/'models'/f'{prefix}_scaler.npz'
    check(not modelpath.exists() and not scalepath.exists(),'immutable model path')
    model.save_model(str(modelpath));np.savez_compressed(scalepath,mean=scaler.mean_,scale=scaler.scale_,var=scaler.var_)
    from catboost import CatBoostClassifier
    loaded=CatBoostClassifier();loaded.load_model(str(modelpath))
    before,p,_=predict_model(scaler,model,X,names);after,q,_=predict_model(scaler,loaded,X,names)
    check(np.array_equal(before,after) and np.allclose(p,q,rtol=0,atol=1e-12),'CBM reload mismatch')
    from sklearn.preprocessing import StandardScaler
    restored=StandardScaler()
    with np.load(scalepath,allow_pickle=False) as z:
        restored.mean_=z['mean'];restored.scale_=z['scale'];restored.var_=z['var'];restored.n_features_in_=len(names)
    check(np.array_equal(restored.transform(X),scaler.transform(X)),'scaler reload mismatch')
    return dict(model=dict(path=str(modelpath.relative_to(out)),sha256=sha(modelpath)),scaler=dict(path=str(scalepath.relative_to(out)),sha256=sha(scalepath)),classes=list(model.classes_),reload_max_probability_diff=float(np.max(abs(p-q))))


def run(out):
    began=time.perf_counter();out.mkdir(parents=True,exist_ok=False)
    for name in ('models','sources'):(out/name).mkdir()
    shutil.copy2(ROOT/'PROJECT_STATUS.md',out/'PROJECT_STATUS.before.md')
    write(out/'protected_before.json',snapshot(out))
    protocol=dict(schema='observed_selector_training_protocol_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        user_authorization='Current request authorizes ML from saved evidence only; no numerical calls or benchmark1000',seed=SEED,
        pair='R128_zero_commonT versus MC1_single4096_h128; fixed diagnostic pair, no R256 oracle-min',
        label_type=LABEL_VERSION,primary_episode=0,threshold=.01,unknown_excluded_from_fit=True,
        split='original700/150/150, family140/30/30; no relabeling held-out splits',
        full_gate='>=20 MC-winner train families,>=5 validation MC-winner families; R class and validation labels required',
        diagnostic='4-fold stratified family CV within known original train only; fixed200 trees, no model selection',
        catboost=dict(iterations=200,depth=3,learning_rate=.05,l2_leaf_reg=5.,bootstrap_type='No',thread_count=1,random_seed=SEED,loss_function='MultiClass'),
        preprocessing='StandardScaler fit on labeled train only; inside each diagnostic fold separately',
        test='evaluate once only after full model freeze; diagnostic never predicts/evaluates original validation/test',
        uncertainty='500 whole-family bootstrap draws of fixed OOF predictions; not prospective validation',
        no_synthetic_training_examples=True,no_new_numerical_calls=True,no_benchmark1000=True,
        historical_test_exposure='Existing numerical audits inspected test historically; not a pristine prospective holdout')
    write(out/'protocol.json',protocol)
    write(out/'environment.json',dict(python=sys.version,platform=platform.platform(),thread_count=1,
        versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','scikit-learn','catboost','pandas','joblib')},
        thread_environment={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS')}))
    data=ROOT/read(CONTRACT/'plan.json')['data_dir'];t=time.perf_counter_ns();inputs=load_inputs(data);feature_seconds=(time.perf_counter_ns()-t)*1e-9
    X=inputs['X'];names=inputs['feature_names'];check(X.shape==(1000,253) and len(names)==253 and np.isfinite(X).all(),'feature schema mismatch')
    bundle=load_saved_evidence(inputs);rows=bundle['rows'];check([r['equation_id'] for r in rows]==list(inputs['ids']),'feature-target order mismatch')
    check(not any(e['issues'] for e in bundle['evidence'].values()),'invalid saved evidence; stop before fitting')
    schema=dict(feature_version=VERSION,feature_names=names,feature_code_sha256=sha(ROOT/'ml/remizov_features.py'),
        feature_schema_sha256=fingerprint(dict(version=VERSION,names=names)),input_grid_sha256=inputs['grid_sha256'],input_grid_points=256,feature_count=253,
        matrix_sha256=hashlib.sha256(X.tobytes()).hexdigest(),pre_T_semantics='fixed legacy descriptors using n128,tail1e-8; NOT actual solver T or profile parameters',
        forbidden=['equation_id','family_id','split','seed','exact','y_shape','solver_output','error','runtime','status','CI'],
        allowed='given a,b,c,g samples; certificate whitelist; coefficient a/b/c AST whitelist; g is the given forcing')
    write(out/'feature_schema.json',schema)
    # Numerical features physically separated from identities, targets and costs.
    np.savez_compressed(out/'features.npz',X=X,feature_names=np.array(names))
    jsonl(out/'feature_index.jsonl',[dict(index=i,equation_id=r['equation_id'],expression_hash=r['expression_hash'],family_id=r['family_id'],split=r['split']) for i,r in enumerate(rows)])
    write(out/'profile_pair.json',dict(profile_pair_id=bundle['pair_id'],pair=bundle['pair']))
    jsonl(out/'labels.jsonl',rows);write(out/'evidence.json',bundle['evidence'])
    jsonl(out/'execution.jsonl',[{k:v for k,v in r.items() if k not in ('seconds','timing_scope','timing_cohort')} for r in bundle['execution']])
    jsonl(out/'timing.jsonl',[{k:r[k] for k in ('method','profile','equation_id','episode','status','seconds','timing_scope','timing_cohort','source','source_sha256')} for r in bundle['execution']])
    # Explain incompatible artifacts instead of silently pooling them.
    write(out/'excluded_sources.json',dict(historical_mass_R='taper/legacyT/mixedworkers: different profile',MC1000='x0 only, not grid33',
        joint_controls='4 control_* jobs use tail1e-8 and/or taper; primary zero/commonT only',joint_MC0='M1024 is not MC1',
        P1_MC0_MC2='different M/h profiles; not extra MC1 samples',final_grid5_MC='different seed protocol; not P1/P2 episode0',R256='different n, pending profile selection; no oracle-min pooling',mean4='different estimator, diagnostics only'))
    gate=training_gate(rows);write(out/'training_gate.json',gate)
    baselines={}
    for split in ('train','validation','test'):
        own=[r for r in rows if r['split']==split]
        baselines[split]={m:retrospective_policy(own,bundle['evidence'],[m]*len(own)) for m in ('remizov','monte_carlo')}
        baselines[split]['offline_oracle']=retrospective_policy(own,bundle['evidence'],[r['label'] if r['label']!='unknown' else 'none' for r in own])
        baselines[split]['offline_oracle']['abstain_due_to_unknown']=sum(r['label']=='unknown' for r in own)
    write(out/'baselines.json',baselines)
    registry=dict(schema='method_selector_registry_v1',label_type=LABEL_VERSION,profile_pair_id=bundle['pair_id'],
        feature_version=VERSION,feature_code_sha256=schema['feature_code_sha256'],feature_names=names,input_grid_sha256=inputs['grid_sha256'],
        timing_context=dict(pair=bundle['pair_id'],scope='saved_P1_P2_MC1_and_historical_zero_R128;not_uniform_timing'),
        selector_status='insufficient_evidence',blocking_reasons=gate['reasons'],model=None,scaler=None,classes=[],diagnostic=None)
    diagnostic=train_only_diagnostic(X,rows,names)
    if diagnostic['model'] is not None:
        model=diagnostic.pop('model');scaler=diagnostic.pop('scaler');idx=diagnostic['fit_indices']
        artifact=save_fitted(out,'diagnostic_R_vs_none',scaler,model,names,X[idx]);registry['diagnostic']=artifact
        diagnostic['artifacts']=artifact;diagnostic['feature_means_fit_on_labeled_train_only']=scaler.mean_.tolist()
        write(out/'diagnostic_cv.json',diagnostic)
    else:
        diagnostic.pop('model');write(out/'diagnostic_cv.json',diagnostic)
    if gate['allowed']:
        ti=np.array([i for i,r in enumerate(rows) if r['split']=='train' and r['label']!='unknown']);vi=np.array([i for i,r in enumerate(rows) if r['split']=='validation' and r['label']!='unknown'])
        y=np.array([r['label'] for r in rows]);scaler,model,seconds=fit_catboost(X[ti],y[ti],names,validation=(X[vi],y[vi]))
        registry.update(selector_status='trained',blocking_reasons=[],**save_fitted(out,'observed_fastest',scaler,model,names,X[ti]))
        write(out/'full_fit.json',dict(train_indices=ti.tolist(),validation_indices=vi.tolist(),train_seconds=seconds,tree_count=model.tree_count_))
        # Freeze weights/policy BEFORE touching test target values for metrics.
        write(out/'registry.json',registry)
        test=np.array([i for i,r in enumerate(rows) if r['split']=='test' and r['label']!='unknown'],dtype=int)
        if len(test):
            yp,proba,seconds=predict_model(scaler,model,X[test],names)
            write(out/'test_evaluation.json',dict(metrics=classification_metrics(y[test],yp),indices=test.tolist(),inference_seconds=seconds,registry_sha256=sha(out/'registry.json'),status='single_frozen_test_evaluation'))
        else:write(out/'test_evaluation.json',dict(status='no_known_test_labels',metrics=None))
    else:
        write(out/'registry.json',registry);write(out/'test_evaluation.json',dict(status='not_evaluated_no_full_model',metrics=None,known_test_labels=gate['support']['test']['known']))
    # Serving a diagnostic registry must return abstain, never a pretend R/MC model.
    api=MethodSelector(out/'registry.json')
    with np.load(data/'all_methods_equations_v1_ml.npz',allow_pickle=False) as z:values=z['X'][0]
    example=api.select_method(values,inputs['x_grid'],inputs['metadata'][str(inputs['ids'][0])],profile_pair_id=bundle['pair_id'],timing_context=registry['timing_context'])
    write(out/'serving_check.json',example)
    source_files=['ml/selector_labels.py','ml/selector_evidence.py','ml/method_selector.py','ml/remizov_features.py','diagnostics/train_method_selector.py','tests/test_method_selector.py','requirements-ml.txt']
    for n in source_files:
        dest=out/'sources'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    sources={**bundle['source_sha256'],**{n:sha(ROOT/n) for n in source_files}}
    write(out/'source_sha256.json',sources)
    audit=dict(rows=len(rows),features=len(names),support=gate['support'],label_reasons=dict(Counter(r['reason'] for r in rows)),
        original_families={s:len(set(inputs['families'][inputs['splits']==s])) for s in ('train','validation','test')},
        independent_saved_record_checks=len(bundle['execution']),invalid_evidence=0,known_labels_all_original_train=True,
        blocked_two_accurate=[r['equation_id'] for r in rows if r['reason']=='timing_unknown'],
        numerical_calls=0,MC_calls=0,R_calls=0,benchmark1000_calls=0,
        selector_status=registry['selector_status'],diagnostic_status=diagnostic['status'],
        feature_extraction_seconds=feature_seconds,pipeline_wall_seconds=time.perf_counter()-began)
    write(out/'data_audit.json',audit)
    print(json.dumps(clean(audit),ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.out.resolve())
