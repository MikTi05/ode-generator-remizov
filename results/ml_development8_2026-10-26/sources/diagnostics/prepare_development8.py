"""Input-only preregistration for the authorized fresh development8 pilot."""
from __future__ import annotations
import copy,csv,json,math,os,shutil,statistics,sys,time
from collections import Counter
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnostics.confirm_f0047_timing import (read_json,write_json,digest,snapshot,NUMERICAL,thermal,lock_is_held,utc,THREADS)
GENERAL=ROOT/'results/ml_selector_generalization_plan_2026-10-26'
NESTED=ROOT/'results/ml_nested_calibration_2026-10-26'
MODEL=ROOT/'results/ml_probability_selector_2026-10-26'
CONTRACT=ROOT/'reports/ml_selector_contract'
SEEDS=[248800080100+i for i in range(4)]


def select_cases(rows,rules,dataset_sha,excluded):
    candidates=[r for r in rows if r['split']=='train' and r['structure_signature'] not in excluded]
    def order(r):
        import hashlib
        return hashlib.sha256(f"fresh_MC8_v1|{dataset_sha}|{r['structure_signature']}|{r['equation_id']}".encode()).hexdigest()
    selected=[];used=set();regimes=Counter()
    for _ in range(rules['quota_per_complexity']):
        for level in rules['complexity_order']:
            pool=[r for r in candidates if r['complexity_level']==level and r['structure_signature'] not in used]
            if not pool:raise ValueError('insufficient independent topologies: '+level)
            r=min(pool,key=lambda r:(regimes[r['regime']],order(r)))
            selected.append(dict(r,selection_sha256=order(r)));used.add(r['structure_signature']);regimes[r['regime']]+=1
    assert len(selected)==rules['target_families']==8
    return selected


def freeze_predictions(out,cases):
    import numpy as np
    from ml.remizov_features import coefficient_features,CERT_FIELDS
    from ml.probabilistic_selector import load_head,estimates_from_heads,choose
    from ml.nested_probability_calibration import apply_sigmoid,baseline_default_choice
    registry=read_json(NESTED/'registry.json');old=read_json(MODEL/'registry.json');heads={}
    for p in ('R128','R256','MC1'):
        for kind in ('probability','cost'):
            nested=p!='MC1' and kind=='probability'
            heads[p+':'+kind]=load_head(registry['final_heads'][p] if nested else old['heads'][p+':'+kind],(NESTED if nested else MODEL)/'models')
    ids=[r['equation_id'] for r in cases];metadata={}
    for r in cases:
        m={k:r[k] for k in CERT_FIELDS};m['serialized_ast']={k:r['serialized_ast'][k] for k in ('a_raw','b_raw','c_raw')};metadata[r['equation_id']]=m
    with np.load(out/'input_pool/all_methods_equations_v1_ml.npz',allow_pickle=False) as z:
        index={str(v):i for i,v in enumerate(z['equation_id'])};ix=[index[v] for v in ids]
        assert all(str(z['split'][i])=='train' for i in ix)
        coeff=z['X'][ix];x=z['x_grid'];assert z['channel_order'].tolist()==['a','b','c','g']
    features=[];feature_seconds=[]
    for r,values in zip(cases,coeff):
        start=time.perf_counter_ns();f=coefficient_features(values,x,metadata[r['equation_id']]);feature_seconds.append((time.perf_counter_ns()-start)*1e-9)
        assert list(f)==registry['feature_names'];features.append([f[n] for n in registry['feature_names']])
    X=np.asarray(features);rows=[]
    for i,r in enumerate(cases):
        t=time.perf_counter_ns();raw=estimates_from_heads(heads,X[i:i+1],registry['feature_names'])[0];cal=copy.deepcopy(raw)
        for p in ('R128','R256'):
            c=read_json(NESTED/registry['calibrators'][p])
            for k in ('point','low','high'):cal[p]['probability'][k]=float(apply_sigmoid(c,[raw[p]['probability'][k]])[0])
        choices={str(q):dict(baseline_default=baseline_default_choice(cal,q),risk=choose(cal,q)['action']) for q in (.5,.8,.95)}
        rows.append(dict(equation_id=r['equation_id'],family_id=r['family_id'],identity=r['identity'],raw=raw,calibrated=cal,choices=choices,
            feature_seconds=feature_seconds[i],inference_seconds=(time.perf_counter_ns()-t)*1e-9))
    np.savez_compressed(out/'frozen_features.npz',X=X,equation_id=np.asarray(ids))
    write_json(out/'frozen_predictions.json',dict(created_utc=utc(),before_any_solver=True,q=.95,rows=rows,fit_calls=0,
        primary='baseline_default',registry_sha256=digest(NESTED/'registry.json'),old_registry_sha256=digest(MODEL/'registry.json'),
        input_channels=['a','b','c','g'],exact_or_derivative_targets_used=False),exclusive=True)
    return rows


def prepare(out):
    import numpy as np
    from all_methods_dataset_loader import load_all_methods_equations
    from all_methods_bounds import certify_all_methods_forcing
    from mc_optimized import prepare_all_methods_mc
    from mc_solver import MCConfig,auto_T
    from chernov_solver import ChernovConfig
    from src.all_methods_dataset_v1 import AllMethodsConfig,validate_dataset_artifacts
    assert out.is_absolute() and out.exists() and not (out/'plan.json').exists()
    protected=snapshot(out);write_json(out/'protected_before.json',protected,exclusive=True)
    for folder in (GENERAL,NESTED,MODEL,ROOT/'results/remizov_r0_2026-10-26',CONTRACT/'sentinel_p1_2026-10-26',CONTRACT/'pilot_p2_2026-10-26'):
        for name,h in read_json(folder/'artifact_sha256.json').items():assert digest(folder/name)==h,(folder,name)
    old=read_json(ROOT/'results/remizov_r0_2026-10-26/plan.json')
    for n in NUMERICAL:assert digest(ROOT/n)==old['protected_sha256'][n],n
    rules=read_json(GENERAL/'sampling_rules.json');contract=read_json(CONTRACT/'plan.json');data=ROOT/contract['data_dir'];pool=out/'input_pool'
    original=read_json(data/'all_methods_equations_v1_report.json')['configuration'];config=dict(original,seed=rules['pool_seed'],n_families=40,samples_per_family=1)
    assert read_json(pool/'all_methods_equations_v1_report.json')['configuration']==config
    valid=validate_dataset_artifacts(pool,expected_config=AllMethodsConfig(**config));assert valid['valid'];write_json(out/'input_validation.json',valid,exclusive=True)
    # Leakage audit uses topology/identity fields only, never historical outcomes or held-out features.
    excluded=set(rules['minimum_known_excluded_topology_signatures']);historical={};history_files=[]
    for root in (ROOT/'data',ROOT/'results',ROOT/'reports'):
        for file in root.rglob('*metadata.jsonl'):
            if out in file.parents:continue
            count=0
            for line in file.open():
                m=json.loads(line)
                if m.get('split')!='train':continue
                sig=m.get('structure_signature');eh=m.get('expression_hash')
                if sig:excluded.add(sig);historical[eh]=sig;count+=1
            if count:history_files.append(dict(path=str(file.relative_to(ROOT)),sha256=digest(file),train_rows=count))
    # Every recorded all_methods experiment must resolve to a known excluded topology.
    unresolved=[];exposure_paths=[];all_records=[]
    for root in (ROOT/'results',ROOT/'reports'):
        for file in root.rglob('records/*.json'):
            if out in file.parents:continue
            r=read_json(file);j=r.get('job',{});eh=r.get('expression_hash',j.get('expression_hash'));pid=r.get('equation_id',j.get('equation_id',''))
            if isinstance(pid,str) and pid.startswith('AMV1_') and r.get('split',j.get('split'))=='train' and eh not in historical:unresolved.append(dict(path=str(file.relative_to(ROOT)),expression_hash=eh))
            cfg=j.get('config',r.get('config',{}))
            if cfg.get('seed') in SEEDS:raise AssertionError('seed entropy already used: '+str(file))
            if eh in historical:exposure_paths.append(str(file.relative_to(ROOT)))
    for root in (ROOT/'results',ROOT/'reports'):
        for file in root.rglob('*.request.json'):
            if out in file.parents:continue
            r=read_json(file);assert r.get('job',{}).get('config',{}).get('seed') not in SEEDS,str(file)
    assert not unresolved,unresolved[:5]
    rows=[json.loads(l) for l in (pool/'all_methods_equations_v1_metadata.jsonl').open()]
    # Dataset hash is the canonical CSV bytes; all four artifacts additionally pinned separately.
    dataset_sha=digest(pool/'all_methods_equations_v1.csv');selected=select_cases(rows,rules,dataset_sha,excluded)
    ids={r['equation_id'] for r in selected};dest=out/'training_inputs';dest.mkdir()
    with (pool/'all_methods_equations_v1.csv').open(newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;subset=[r for r in reader if r['equation_id'] in ids]
    with (dest/'all_methods_equations_v1.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(subset)
    with (dest/'all_methods_equations_v1_metadata.jsonl').open('x') as f:
        for line in (pool/'all_methods_equations_v1_metadata.jsonl').open():
            if json.loads(line)['equation_id'] in ids:f.write(line)
    eqs={e.equation_id:e for e in load_all_methods_equations(dest/'all_methods_equations_v1.csv',dest/'all_methods_equations_v1_metadata.jsonl')};assert len(eqs)==8
    # Case-specific work scaling from saved TRAIN MC runs; includes full method wall, not native solver time.
    targets=read_json(MODEL/'full_cost_targets.json');rates=[];historical_work=[]
    for target in targets:
        if target['profile']!='MC1':continue
        assert target['split']=='train'
        record=read_json(ROOT/target['sources'][0]);work=record['job']['numerical_work'];rate=target['seconds']/work
        rates.append(rate);historical_work.append(dict(equation_id=target['equation_id'],work=work,seconds=target['seconds'],seconds_per_update=rate))
    cases=[]
    for r in selected:
        e=eqs[r['equation_id']];m=e.metadata;p=e.remizov_problem;G=certify_all_methods_forcing(p,m).forcing_bound.abs_upper_bound
        prepared=prepare_all_methods_mc(m);assert prepared.forcing_bound.abs_upper_bound==G
        T=auto_T(p.lambda_value,p.c_abs_upper_bound,G,1e-4);N=math.ceil(T*128);assert N<=100000
        work=33*4096*N
        r.update(identity=f"{dataset_sha}|{r['structure_signature']}|{r['equation_id']}")
        cases.append(dict(equation_id=r['equation_id'],family_id=r['family_id'],split='train',identity=r['identity'],expression_hash=r['expression_hash'],
            structure_signature=r['structure_signature'],complexity_level=r['complexity_level'],regime=r['regime'],selection_sha256=r['selection_sha256'],G=G,T=T,N=N,
            MC_work=work,MC_scenario_seconds=work*statistics.median(rates)))
    scenario=4*sum(12.321290521+24.688813375+c['MC_scenario_seconds'] for c in cases)
    budget=dict(approved_cap_seconds=5460,maximum_calls=96,central_seconds=scenario,MC_seconds_per_update_median=statistics.median(rates),
        MC_rate_range=[min(rates),max(rates)],historical_train=historical_work,upper_bound=False,formula='4*sum(train timing R128median+R256median+case33*4096*N*median TRAIN full MC seconds/update)',
        stop_at_original_cap=True,no_automatic_extension=True)
    write_json(out/'budget_preflight.json',budget,exclusive=True)
    assert scenario<5460,'case-specific central scenario exceeds approved cap'
    write_json(out/'identity_preflight.json',dict(passed=True,dataset_sha256=dataset_sha,dataset_sha_definition='SHA256 canonical CSV bytes',historical_metadata=history_files,
        excluded_signatures=sorted(excluded),original_minimum_exclusions=len(rules['minimum_known_excluded_topology_signatures']),train_record_exposures=len(exposure_paths),unresolved_train_exposures=unresolved,
        selected=cases,only_train_selected=True,selection_outcome_blind=True,seed_entropy_nonreuse=True,generated_heldout_roles_not_evaluated=True),exclusive=True)
    predictions=freeze_predictions(out,selected)
    jobs=[]
    for case in cases:
        for episode in range(4):
            block=[]
            for profile in ('R128','R256','MC1'):
                cfg=dict(contract['profiles'][profile]['config']);mc=profile=='MC1';seconds=case['MC_scenario_seconds'] if mc else (12.321290521 if profile=='R128' else 24.688813375)
                watchdog=math.ceil(max(120,6*seconds+15)) if mc else (120 if profile=='R128' else 164)
                work=case['MC_work'] if mc else int(profile[1:])*512
                if mc:cfg.update(seed=SEEDS[episode],max_path_steps=work,max_runtime_seconds=watchdog);MCConfig(**cfg).validate()
                else:cfg['T']=case['T'];ChernovConfig(**cfg).validate()
                block.append(dict(key=f"{case['equation_id']}_{profile}_development_episode{episode}",equation_id=case['equation_id'],family_id=case['family_id'],split='train',
                    identity=case['identity'],expression_hash=case['expression_hash'],profile=profile,episode=episode,config=cfg,target_x=contract['target_x'],forcing_G=case['G'],expected_T=case['T'],
                    expected_steps=case['N'] if mc else 0,numerical_work=work,work_unit='path_updates' if mc else 'S_applications',scenario_seconds=seconds,proposed_watchdog_seconds=watchdog))
            # Preregistered cyclic order balances profile position across episodes and families.
            shift=(cases.index(case)+episode)%3;jobs.extend(block[shift:]+block[:shift])
    for n in ('.chernov_batch_controller.lock','.chernov_batch_compute.lock'):assert not lock_is_held(ROOT/'results'/n)
    sensor=thermal();assert sensor['thermal_state'] is not None and sensor['thermal_state']<2
    assert shutil.disk_usage(ROOT).free>=512*1024**2
    for n in ('records','events','workers','sessions','sources'):(out/n).mkdir()
    sources=list(NUMERICAL)+['diagnostics/prepare_development8.py','diagnostics/run_development8.py','diagnostics/confirm_f0047_timing.py','diagnostics/remizov_r0.py','diagnostics/mc_pilot_p2.py',
        'diagnostics/chernov_batch_control.py','diagnostics/mc_production_x0.py','diagnostics/joint_benchmark_pilot.py','diagnostics/run_chernov_batch.py','ml/probabilistic_selector.py','ml/nested_probability_calibration.py','ml/remizov_features.py']
    pins={n:digest(ROOT/n) for n in sources}
    for folder in (GENERAL,NESTED,MODEL):
        for n,h in read_json(folder/'artifact_sha256.json').items():pins[str((folder/n).relative_to(ROOT))]=h
        pins[str((folder/'artifact_sha256.json').relative_to(ROOT))]=digest(folder/'artifact_sha256.json')
    for folder in (pool,dest):
        pins.update({str(p.relative_to(ROOT)):digest(p) for p in folder.iterdir() if p.is_file()})
    pins[str((out/'frozen_predictions.json').relative_to(ROOT))]=digest(out/'frozen_predictions.json')
    for n in sources:
        target=out/'sources'/n;target.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(ROOT/n,target)
    authorization='User approves5460 seconds, maximum96 calls: 8 fresh development TRAIN topology families, R128/R256/MC1x4, one worker/thread RSS<=512MiB; no validation/test, model/q changes or fitting.'
    (out/'AUTHORIZATION.txt').write_text(authorization+'\n')
    plan=dict(schema='fresh_development96_execution_v1',created_utc=utc(),execution_authorized=True,authorization=authorization,cases=cases,jobs=jobs,data_dir=str(dest.relative_to(ROOT)),target_x=contract['target_x'],
        protected_sha256=pins,resources=dict(compute_workers=1,rss_mib=512,series_seconds=5460,owner_lease_seconds=300,startup_seconds=90,poll_seconds=.05,monitor_seconds=.25,thermal_seconds=10,thermal_stop_at=2),
        scenario_seconds=scenario,evaluation=read_json(GENERAL/'prospective_experiment.json')['evaluation_rules'],primary_episode=0,MC_seed_entropy=SEEDS,
        resume='immutable completed, never redispatch requested keys, original persistent epoch+monotonic deadlines, no replacement or extra calls',
        development_only=True,no_test=True,no_validation=True,no_ML_fit=True,numerical_warmup_calls=0)
    from diagnostics.run_development8 import validate_scope
    validate_scope(plan);write_json(out/'plan.json',plan,exclusive=True)
    write_json(out/'preflight.json',dict(passed=True,thermal=sensor,scenario_seconds=scenario,budget_approved=True,cap=5460,calls=96,selected_ids=[c['equation_id'] for c in cases],
        predictions_before_calls=True,choices=Counter(r['choices']['0.95']['baseline_default'] for r in predictions)),exclusive=True)
    print(json.dumps(read_json(out/'preflight.json')),flush=True)


if __name__=='__main__':prepare(Path(sys.argv[1]).resolve())
