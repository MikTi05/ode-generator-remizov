"""One final prospective test: materialize inputs/manifest/budget, NEVER solve."""
from __future__ import annotations
import csv,hashlib,json,math,os,statistics,sys,time
from collections import Counter,defaultdict
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from ml.revision_evidence import read,sha,ORIGINAL,CONTRACT
from ml.final_selector import FinalResearchSelector
from diagnostics.prepare_development8 import write_json,utc

SEEDS=[248800100100+i for i in range(4)]
LEVELS=('L1','L2','L3','L4')
REGIMES=('constant','slow_varying','moderate_varying','oscillatory','high_contrast')


def select(rows,excluded,dataset_sha):
    chosen=[]
    for level in LEVELS:
        for regime in REGIMES:
            pool=[r for r in rows if r['split']=='test' and r['complexity_level']==level and r['regime']==regime and r['structure_signature'] not in excluded]
            def order(r):return hashlib.sha256(f"final60_v1|{dataset_sha}|{r['structure_signature']}|{r['equation_id']}".encode()).hexdigest()
            pool.sort(key=order)
            if len(pool)<3:raise ValueError('insufficient fresh test topologies: '+level+'/'+regime)
            chosen.extend(dict(r,selection_sha256=order(r)) for r in pool[:3])
    assert len(chosen)==len({r['structure_signature'] for r in chosen})==60
    return chosen


def prepare(release):
    from src.all_methods_dataset_v1 import AllMethodsConfig,generate_dataset,write_dataset,validate_dataset_artifacts
    from all_methods_dataset_loader import load_all_methods_equations
    from all_methods_bounds import certify_all_methods_forcing
    from mc_solver import auto_T,MCConfig
    from chernov_solver import ChernovConfig
    from diagnostics.remizov_r0 import NUMERICAL
    freeze=read(release/'release_freeze.json');model=FinalResearchSelector(release/'model')
    assert freeze['model_registry_sha256']==sha(release/'model/registry.json') and freeze['model_manifest_sha256']==sha(release/'model/artifact_sha256.json')
    out=release/'final_experiment';resuming=out.exists()
    if resuming:assert (out/'preregistration.json').exists() and not (out/'plan.json').exists(),'only interrupted input preparation may resume'
    else:out.mkdir(exist_ok=False)
    original=read(ORIGINAL/'all_methods_equations_v1_report.json')['configuration'];cfg=dict(original,seed=2488001001,n_families=800,samples_per_family=1)
    excluded=set();history=[]
    for root in (ROOT/'data',ROOT/'results',ROOT/'reports'):
        for path in root.rglob('*metadata.jsonl'):
            if release in path.parents:continue
            signatures=set()
            for line in path.open():
                row=json.loads(line)
                if 'structure_signature' in row:signatures.add(row['structure_signature'])
            if signatures:excluded|=signatures;history.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),topologies=len(signatures)))
    protocol=dict(created_utc=utc(),model_frozen_before_inputs=True,release_freeze_sha256=sha(release/'release_freeze.json'),
        generator_config=cfg,selected_families=60,only_fresh_test_role=True,quota_per_complexity_regime_cell=3,
        selection='input-only SHA256(final60_v1|dataset_csv_sha|structure_signature|canonicalID);3 per20cell; no prediction/error/time selection',
        exclusions='all known dataset topology identities, including original test identity only; no old held-out outcomes/features used',
        no_automatic_pool_expansion=True,no_family_replacement=True,no_more_pilots=True,numerical_calls_now=0,
        MC_seeds=SEEDS,primary_episode=0,maximum_calls=720,planned_profiles=['R128','R256','MC1'])
    if not resuming:write_json(out/'preregistration.json',protocol,exclusive=True)
    else:
        previous=read(out/'preregistration.json');assert previous['generator_config']==cfg and previous['release_freeze_sha256']==sha(release/'release_freeze.json')
    config=AllMethodsConfig(**cfg)
    if not (out/'input_pool').exists():
        dataset=generate_dataset(config);write_dataset(dataset,out/'input_pool')
    valid=validate_dataset_artifacts(out/'input_pool',expected_config=config);assert valid['valid']
    if not (out/'input_validation.json').exists():write_json(out/'input_validation.json',valid,exclusive=True)
    ds=sha(out/'input_pool/all_methods_equations_v1.csv')
    rows=[json.loads(l) for l in (out/'input_pool/all_methods_equations_v1_metadata.jsonl').open()];selected=select(rows,excluded,ds)
    ids={r['equation_id'] for r in selected};dest=out/'test_inputs';dest.mkdir(exist_ok=resuming)
    with (out/'input_pool/all_methods_equations_v1.csv').open(newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;subset=[r for r in reader if r['equation_id'] in ids]
    if not (dest/'all_methods_equations_v1.csv').exists():
        with (dest/'all_methods_equations_v1.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(subset)
    if not (dest/'all_methods_equations_v1_metadata.jsonl').exists():
        with (dest/'all_methods_equations_v1_metadata.jsonl').open('x') as f:
            for line in (out/'input_pool/all_methods_equations_v1_metadata.jsonl').open():
                if json.loads(line)['equation_id'] in ids:f.write(line)
    eqs={e.equation_id:e for e in load_all_methods_equations(dest/'all_methods_equations_v1.csv',dest/'all_methods_equations_v1_metadata.jsonl')}
    with np.load(out/'input_pool/all_methods_equations_v1_ml.npz',allow_pickle=False) as z:
        index={str(v):i for i,v in enumerate(z['equation_id'])};coeff=z['X'][[index[r['equation_id']] for r in selected]];x=z['x_grid']
        assert all(str(z['split'][index[r['equation_id']]])=='test' for r in selected)
    # Budget rates are TRAIN-only. Final-test outcomes do not exist.
    training_raw=read(release/'training/raw.json');rates=[]
    groups=defaultdict(list)
    for r in training_raw:
        if r['profile']=='MC1' and r['full_seconds'] is not None:groups[r['equation_key']].append(r['full_seconds']/(33*4096*r['N']))
    rates=[statistics.median(v) for v in groups.values()];rate=statistics.median(rates)
    cases=[];predictions=[];previous_predictions={}
    if (out/'frozen_predictions.json').exists():
        previous=read(out/'frozen_predictions.json');assert previous['release_registry_sha256']==sha(release/'model/registry.json')
        previous_predictions={r['equation_id']:r['prediction'] for r in previous['rows']};assert set(previous_predictions)==ids
    for r,values in zip(selected,coeff):
        pid=r['equation_id'];e=eqs[pid];problem=e.remizov_problem;G=certify_all_methods_forcing(problem,e.metadata).forcing_bound.abs_upper_bound
        T=auto_T(problem.lambda_value,problem.c_abs_upper_bound,G,1e-4);N=math.ceil(128*T);assert 0<N<=100000
        pred=previous_predictions.get(pid) or model.predict(values,x,r);identity=f"{ds}|{pid}|{r['expression_hash']}"
        work=33*4096*N;mc=max(pred['estimates']['MC1']['predicted_full_warm_seconds'],work*rate)
        cases.append(dict(equation_id=pid,family_id=r['family_id'],split='test',dataset_sha256=ds,expression_hash=r['expression_hash'],structure_signature=r['structure_signature'],
            identity=identity,complexity_level=r['complexity_level'],regime=r['regime'],selection_sha256=r['selection_sha256'],G=G,T=T,N=N,MC_work=work,
            MC_scenario_seconds=mc,R128_scenario_seconds=pred['estimates']['R128']['predicted_full_warm_seconds'],R256_scenario_seconds=pred['estimates']['R256']['predicted_full_warm_seconds']))
        predictions.append(dict(equation_id=pid,identity=identity,prediction=pred))
    for root in (ROOT/'results',ROOT/'reports'):
        for path in list(root.rglob('records/*.json'))+list(root.rglob('*.request.json')):
            if release in path.parents:continue
            record=read(path);j=record.get('job',{});assert j.get('config',record.get('config',{})).get('seed') not in SEEDS,path
    scenario=4*sum(c['MC_scenario_seconds']+c['R128_scenario_seconds']+c['R256_scenario_seconds'] for c in cases)
    cap=math.ceil(2*scenario+300)
    budget=dict(approved=False,execution_authorized=False,scenario_seconds=scenario,proposed_cap_seconds=cap,maximum_calls=720,
        MC_rate_train_family_median=rate,train_rate_range=[min(rates),max(rates)],
        formula='4*sum(R128 predicted full wall+R256 predicted full wall+max(MC predicted wall,33*4096*N*TRAIN median seconds/update)); cap=ceil(2*scenario+300)',
        not_a_proven_upper_bound=True,no_automatic_extension=True,
        max_train_rate_sensitivity_seconds=4*sum(c['R128_scenario_seconds']+c['R256_scenario_seconds']+c['MC_work']*max(rates) for c in cases))
    if not (out/'budget_proposal.json').exists():write_json(out/'budget_proposal.json',budget,exclusive=True)
    else:assert read(out/'budget_proposal.json')==budget
    if not (out/'frozen_predictions.json').exists():
        write_json(out/'frozen_predictions.json',dict(release_registry_sha256=sha(release/'model/registry.json'),before_solver=True,rows=predictions,
            choices=Counter(r['prediction']['selected_profile'] for r in predictions),no_selection_or_tuning_on_predictions=True),exclusive=True)
    contract=read(CONTRACT/'plan.json');jobs=[]
    for i,c in enumerate(cases):
        for episode in range(4):
            block=[]
            for profile in ('R128','R256','MC1'):
                config=dict(contract['profiles'][profile]['config']);mc=profile=='MC1';seconds=c['MC_scenario_seconds'] if mc else c[profile+'_scenario_seconds'];watchdog=math.ceil(max(120,6*seconds+15)) if mc else (120 if profile=='R128' else 164)
                work=c['MC_work'] if mc else int(profile[1:])*512
                if mc:config.update(seed=SEEDS[episode],max_path_steps=work,max_runtime_seconds=watchdog);MCConfig(**config).validate()
                else:config['T']=c['T'];ChernovConfig(**config).validate()
                block.append(dict(key=f"{c['equation_id']}_{profile}_final_episode{episode}",equation_id=c['equation_id'],family_id=c['family_id'],split='test',identity=c['identity'],
                    expression_hash=c['expression_hash'],profile=profile,episode=episode,config=config,expected_T=c['T'],forcing_G=c['G'],expected_steps=c['N'] if mc else 0,
                    numerical_work=work,target_x=contract['target_x'],proposed_watchdog_seconds=watchdog,scenario_seconds=seconds))
            shift=(i+episode)%3;jobs.extend(block[shift:]+block[:shift])
    sources=list(NUMERICAL)+['diagnostics/run_final_selector_experiment.py','diagnostics/evaluate_final_selector_experiment.py','diagnostics/prepare_final_selector_experiment.py',
        'diagnostics/confirm_f0047_timing.py','diagnostics/remizov_r0.py','diagnostics/mc_pilot_p2.py','diagnostics/chernov_batch_control.py','diagnostics/mc_production_x0.py',
        'diagnostics/joint_benchmark_pilot.py','diagnostics/run_chernov_batch.py','ml/final_selector.py','ml/revision_selector.py','ml/remizov_features.py']
    pins={n:sha(ROOT/n) for n in sources}
    for folder in (release/'model',out/'input_pool',dest):
        pins.update({str(p.relative_to(ROOT)):sha(p) for p in folder.rglob('*') if p.is_file()})
    pins[str((release/'release_freeze.json').relative_to(ROOT))]=sha(release/'release_freeze.json')
    pins[str((out/'frozen_predictions.json').relative_to(ROOT))]=sha(out/'frozen_predictions.json')
    plan=dict(schema='final_independent60_v1',execution_authorized=False,only_after_separate_user_budget_approval=True,
        created_utc=utc(),data_dir=str(dest.relative_to(ROOT)),cases=cases,jobs=jobs,MC_seed_entropy=SEEDS,target_x=contract['target_x'],protected_sha256=pins,
        resources=dict(compute_workers=1,rss_mib=512,series_seconds=cap,owner_lease_seconds=300,startup_seconds=90,poll_seconds=.05,monitor_seconds=.25,thermal_seconds=10,thermal_stop_at=2),
        experiment_role='single final independent test, never another preliminary pilot',maximum_calls=720,primary_episode=0,
        completion='all720 terminal outcomes valid and fully accounted, frozen choices unchanged, independent checks/report complete; superiority/MC selection NOT required',
        no_replacement_or_extra_seeds=True,no_policy_or_model_tuning_after_inputs=True,production_blocked=True)
    write_json(out/'plan.json',plan,exclusive=True)
    with (out/'manifest.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(cases[0]));w.writeheader();w.writerows(cases)
    write_json(out/'independence.json',dict(passed=True,history=history,excluded_topologies=sorted(excluded),selected_topologies=[r['structure_signature'] for r in selected],
        disjoint=True,input_only_selection=True,model_frozen_before_input_generation=True,no_old_test_outcomes_used=True,new_final_outcomes_do_not_exist=True),exclusive=True)
    write_json(out/'approval_request.json',dict(execution_authorized=False,approved=False,plan_sha256=sha(out/'plan.json'),maximum_calls=720,series_seconds=cap,rss_mib=512,workers=1,threads=1,
        requested_action='one final independent60-family experiment;R128/R256/MC1x4; no tuning; preserve every result',no_solver_started=True),exclusive=True)
    for n in ('records','events','workers','sessions'):(out/n).mkdir()
    from diagnostics.run_final_selector_experiment import preflight
    check=preflight(out);write_json(out/'preflight.json',check,exclusive=True)
    print(dict(prepared=True,calls=720,families=60,scenario_seconds=scenario,proposed_cap_seconds=cap,choices=Counter(r['prediction']['selected_profile'] for r in predictions),executed=0))


if __name__=='__main__':prepare(Path(sys.argv[1]).resolve())
