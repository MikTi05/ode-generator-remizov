"""Preflight and unapproved budget proposal only. No numerical run or fitting API."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,shutil,statistics,sys,time
from pathlib import Path
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
MODEL=ROOT/'results/ml_probability_selector_2026-10-26';CONTRACT=ROOT/'reports/ml_selector_contract'
from diagnostics.train_method_selector import write,snapshot
from ml.selector_evidence import read,sha


def budget_proposal(cases,r128,r256,factors):
    central=factors['coverage20']['median'];mc=sum(c['x0_mean_solver_seconds']*132*central*4 for c in cases)
    R128=32*r128;R256=32*r256;total=mc+R128+R256
    return dict(approved=False,approval_evidence=None,scope_calls=96,MC_calls=32,R128_calls=32,R256_calls=32,
        MC_scenario_seconds=mc,R128_scenario_seconds=R128,R256_scenario_seconds=R256,
        total_scenario_seconds=total,total_scenario_hours=total/3600,proposed_series_cap_seconds=math.ceil(2*total+300),
        formula='cap=ceil(2*central_scenario+300); no automatic extensions',
        MC_formula='mean4 saved x0 M1024 runtime *33 grid points *4 paths ratio *P2 coverage20 median calibration; four calls',
        R_formula='median saved full method wall per profile *32; numerical work n*512 and2081 points unchanged',
        calibration_factor=central,calibration_source='reports/ml_selector_contract/pilot_p2_2026-10-26/analysis/summary.json',
        MC_sensitivity_seconds=[sum(c['x0_mean_solver_seconds']*132*factors['all40'][bound]*4 for c in cases) for bound in ('min','max')],
        confidence_interval=False,upper_bound=False,includes_future_setup_overhead=False,
        caveat='x0 operation scaling and old timing cohorts are scenarios; no E_grid33 inference; stop on cap rather than expanding it')


def build_jobs(cases,contract,budget):
    jobs=[]
    for case in cases:
        pid=case['equation_id'];block_templates={}
        for profile in ('R128','R256','MC1'):
            cfg=dict(contract['profiles'][profile]['config'])
            if profile=='MC1':
                work=33*4096*case['N'];seconds=case['MC1_scenario_one_call_seconds'];watchdog=math.ceil(max(120,6*seconds+15))
                cfg.update(max_path_steps=work,max_runtime_seconds=watchdog)
            else:
                cfg['T']=case['T'];work=int(profile[1:])*512;seconds=budget[profile+'_scenario_seconds']/32;watchdog=120 if profile=='R128' else 164
            block_templates[profile]=dict(equation_id=pid,family_id=case['family_id'],split='validation',expression_hash=case['expression_hash'],profile=profile,
                config=cfg,expected_T=case['T'],forcing_G=case['G'],expected_steps=case['N'] if profile=='MC1' else 0,
                numerical_work=work,work_unit='path_updates' if profile=='MC1' else 'S_applications',target_x=contract['target_x'],
                scenario_seconds=seconds,proposed_watchdog_seconds=watchdog,execution_authorized=False)
        for episode in range(4):
            block=[]
            for profile in ('R128','R256','MC1'):
                job=dict(block_templates[profile],episode=episode,key=f'{pid}_{profile}_validation_episode{episode}')
                job['config']=dict(job['config'])
                if profile=='MC1':job['config']['seed']=248800060100+episode
                block.append(job)
            shift=int(hashlib.sha256(f'{pid}|{episode}'.encode()).hexdigest(),16)%3
            jobs.extend(block[shift:]+block[:shift])
    assert len(jobs)==len({j['key'] for j in jobs})==96
    return jobs


def plan(out):
    import numpy as np
    from ml.remizov_features import load_inputs
    from ml.probabilistic_selector import ProbabilisticSelector,estimates_from_heads,choose
    from all_methods_dataset_loader import load_all_methods_equations
    from all_methods_bounds import certify_all_methods_forcing
    from mc_solver import auto_T
    from diagnostics.chernov_batch_control import lock_is_held
    out.mkdir(parents=True,exist_ok=False);(out/'sources').mkdir()
    shutil.copy2(ROOT/'PROJECT_STATUS.md',out/'PROJECT_STATUS.before.md');write(out/'protected_before.json',snapshot(out))
    for name,h in read(MODEL/'artifact_sha256.json').items():assert sha(MODEL/name)==h,name
    for name,h in read(MODEL/'source_sha256.json').items():assert sha(ROOT/name)==h,name
    for name,h in read(MODEL/'additional_sources_sha256.json').items():assert sha(ROOT/name)==h,name
    contract=read(CONTRACT/'plan.json');design=read(MODEL/'VALIDATION_DESIGN.json');proposal=read(MODEL/'NEXT_VALIDATION.json')
    registry=read(MODEL/'registry.json');registryhash=sha(MODEL/'registry.json')
    assert registryhash==design['frozen_registry_sha256']==proposal['model_and_policy_frozen_sha256']
    assert design['cases']==proposal['cases'] and len(design['cases'])==8 and design['full_three_profile_calls']==96
    inputs=load_inputs(ROOT/contract['data_dir']);meta=inputs['metadata'];index={str(pid):i for i,pid in enumerate(inputs['ids'])}
    valfamilies=sorted(set(inputs['families'][inputs['splits']=='validation']));chosen=sorted(np.random.default_rng(2488).choice(valfamilies,size=8,replace=False))
    assert [c['family_id'] for c in design['cases']]==chosen
    fitted=set()
    for h in registry['heads'].values():
        fitted.update(h['metadata']['families'])
        assert all(inputs['splits'][i]=='train' for i in h['metadata']['indices'])
    assert not fitted&set(chosen)
    for filename in ('accuracy_targets.json','full_cost_targets.json'):
        assert all(r['split']=='train' and r['family_id'] not in chosen for r in read(MODEL/filename))
    for c in design['cases']:
        m=meta[c['equation_id']];assert all(m[k]==c[k] for k in ('equation_id','expression_hash','family_id','split')) and c['split']=='validation'
        candidates=[str(pid) for pid,g,s in zip(inputs['ids'],inputs['families'],inputs['splits']) if g==c['family_id'] and s=='validation']
        assert c['equation_id']==min(candidates,key=lambda p:hashlib.sha256(('probability_validation_v1|'+p).encode()).hexdigest())
    data=ROOT/contract['data_dir'];eqs={e.equation_id:e for e in load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl')}
    oldplan=read(ROOT/'results/mc1000_x0_2026-10-25/plan.json');pins={};cases=[]
    for name,h in oldplan['protected_sha256'].items():assert sha(ROOT/name)==h,name
    calibration=read(CONTRACT/'pilot_p2_2026-10-26/analysis/summary.json')['calibration']
    for c in design['cases']:
        pid=c['equation_id'];e=eqs[pid];cert=certify_all_methods_forcing(e.remizov_problem,e.metadata);G=cert.forcing_bound.abs_upper_bound
        T=auto_T(e.remizov_problem.lambda_value,e.remizov_problem.c_abs_upper_bound,G,1e-4);N=math.ceil(T*128);times=[]
        for episode in range(4):
            path=ROOT/f'results/mc1000_x0_2026-10-25/attempts/{pid}_seed{episode}.json';r=read(path);z=r['result']
            assert r['expression_hash']==c['expression_hash'] and r['metadata_identity']['split']=='validation'
            assert r['status']=='completed' and z['success'] and z['M']==1024 and z['x']==[0.]
            assert z['T']==T and z['n_steps']==N and r['forcing_bound']['abs_upper_bound']==G
            assert z['n_path_steps']==1024*N and z['dt_requested']==1/128 and z['backend']=='numba'
            assert z['compilation_seconds']==z['numba_import_seconds']==0 and z['runtime_seconds']>0
            times.append(z['runtime_seconds']);pins[str(path.relative_to(ROOT))]=sha(path)
        cases.append(dict(c,T=T,G=G,N=N,x0_mean_solver_seconds=statistics.mean(times),
            MC1_scenario_one_call_seconds=statistics.mean(times)*132*calibration['coverage20']['median']))
    cost=read(MODEL/'full_cost_targets.json');rmed={p:statistics.median(r['seconds'] for r in cost if r['profile']==p) for p in ('R128','R256')}
    budget=budget_proposal(cases,rmed['R128'],rmed['R256'],calibration);jobs=build_jobs(cases,contract,budget)
    pids={c['equation_id'] for c in cases};existing=[];conflicts=[];seedset=set(design['proposed_MC_seed_group'])
    rawpaths=list((ROOT/'results').glob('*/records/*.json'))+list(CONTRACT.glob('*/records/*.json'))
    requestpaths=list((ROOT/'results').glob('*/events/*.request.json'))+list(CONTRACT.glob('*/events/*.request.json'))
    for path in rawpaths+requestpaths:
        r=read(path);j=r.get('job',{});pid=r.get('equation_id',j.get('equation_id',j.get('problem_id')))
        if pid not in pids:continue
        cfg=j.get('config',r.get('config',{}));grid=j.get('target_x',(r.get('result') or {}).get('x'))
        compatible_profile=j.get('profile') in ('R128','R256','MC1') or (cfg.get('n') in (128,256) and cfg.get('boundary')=='zero' and cfg.get('tail_eps')==1e-4) or (cfg.get('M')==4096 and cfg.get('dt')==1/128)
        if compatible_profile and grid==contract['target_x']:existing.append(str(path.relative_to(ROOT)))
        if j.get('config',{}).get('seed') in seedset:conflicts.append(str(path.relative_to(ROOT)))
    assert not existing and not conflicts
    locks={n:lock_is_held(ROOT/'results'/n) for n in ('.chernov_batch_controller.lock','.chernov_batch_compute.lock')}
    assert not any(locks.values())
    api=ProbabilisticSelector(MODEL/'registry.json');t=time.perf_counter();predictions=estimates_from_heads(api.heads,inputs['X'][[index[c['equation_id']] for c in cases]],inputs['feature_names']);inference=time.perf_counter()-t
    frozen=[]
    for c,est in zip(cases,predictions):
        frozen.append(dict(equation_id=c['equation_id'],expression_hash=c['expression_hash'],estimates=est,
            production_action='abstain',production_status=registry['status'],
            diagnostic_choices={str(q):dict(point=choose(est,q),model_spread=choose(est,q,robust=True)) for q in design['thresholds']}))
    write(out/'frozen_predictions.json',dict(registry_sha256=registryhash,policy_code_sha256=sha(ROOT/'ml/probabilistic_selector.py'),rows=frozen,validation_outcomes_used=False,
        batch_prediction_seconds=inference,feature_precomputation_and_model_loading_outside_timer=True))
    manifest=[]
    for c in cases:manifest.append({k:c[k] for k in ('equation_id','family_id','split','expression_hash','T','G','N','MC1_scenario_one_call_seconds')})
    with (out/'manifest.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
    execution=dict(schema='probability_validation96_proposal_v1',execution_authorized=False,scope_requested_by_user=True,
        status='blocked_before_numerical_execution',blockers=['no_approved_total_compute_budget'],
        source_design_sha256=sha(MODEL/'VALIDATION_DESIGN.json'),registry_sha256=registryhash,policy_code_sha256=sha(ROOT/'ml/probabilistic_selector.py'),
        cases=manifest,jobs=jobs,budget=budget,resources_proposed=dict(workers=1,threads=1,rss_mib=512,owner_lease_seconds=300,thermal_stop_at=2,RSS_sampling_seconds=.25,thermal_sampling_seconds=10,guardians=True),
        order='fixed manifest order;episode0..3;SHA256(ID|episode)%3 cyclic R128/R256/MC1',
        evaluation=dict(primary='all predeclared q=.5/.8/.95, no post-hoc best-q selection',
            policies='production API abstain reported separately from frozen point/model-spread diagnostic choices',
            baselines=['constant_R256 main train-selected baseline','constant_R128 secondary faster-profile baseline'],
            accuracy='canonical E_grid33<=.01;episode0 primary;all4 MC outcomes for probability scoring and CP, not reliability95',
            probability='proper binomial Brier/logloss and support; no calibration or refitting on these8 families',
            time='full warm median4, ranges/IQR/MAD; cost/error statistics include inaccurate calls',
            policy_cost='one chosen method per task:feature+inference+median4 method; no free abstain speed claim; entire96-call research cost separate',
            censoring='missing/error/timeout separate, no invented E_grid or completed time',
            independent_test='eight validation families are diagnostic; final separate family test only after freeze, no test outcome tuning'),
        numerical_calls=0,model_fits=0,benchmark1000_calls=0)
    write(out/'plan.json',execution);write(out/'budget.json',budget)
    write(out/'checks.json',dict(manifest_passed=True,models_and_policy_frozen=True,model_registry_sha256=registryhash,
        source_model_artifacts_checked=len(read(MODEL/'artifact_sha256.json')),training_validation_family_overlap=[],fit_only_original_train=True,
        canonical_dataset_hashes_verified=True,analytic_G_T_rechecked=True,existing_matching_grid33_records_or_requests=existing,
        proposed_seed_conflicts=conflicts,locks=locks,budget_approved=False,only_blocker='missing explicit total-budget approval',
        no_validation_accuracy_outcomes_used=True,x0_used_only_for_runtime_and_G_T_not_accuracy=True,
        numerical_calls=0,model_fits=0,loaded_CBM_files=54))
    write(out/'budget_sources_sha256.json',pins)
    shutil.copy2(Path(__file__),out/'sources'/Path(__file__).name)
    print(json.dumps(dict(status=execution['status'],budget=budget,checks='all technical preflight checks passed; budget not approved')),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);plan(ap.parse_args().out.resolve())
