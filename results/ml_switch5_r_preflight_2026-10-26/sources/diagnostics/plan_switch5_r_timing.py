"""Prepare the five frozen TRAIN OOF-switch timing cases; NEVER execute a solver."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.metadata,json,math,os,platform,shutil,statistics,subprocess,sys
from collections import Counter
from pathlib import Path
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnostics.audit_probability_selector import read,write,sha

NESTED=ROOT/'results/ml_nested_calibration_2026-10-26'
MODEL=ROOT/'results/ml_probability_selector_2026-10-26'
CONTRACT=ROOT/'reports/ml_selector_contract'
IDS=('AMV1_F0016_S01','AMV1_F0043_S00','AMV1_F0119_S00','AMV1_F0153_S01','AMV1_F0172_S03')
GRID=[-1+i/16 for i in range(33)]


def snapshot(out):
    return {str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and out not in p.parents
            and not any(k in p.parts for k in ('.venv','.git','__pycache__','.pytest_cache','.DS_Store'))}


def selected_inputs(data,dest):
    """Unmodified canonical rows, selected to avoid loading held-out equations."""
    dest.mkdir()
    csv_name='all_methods_equations_v1.csv';metadata_name='all_methods_equations_v1_metadata.jsonl'
    with (data/csv_name).open(newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;selected=[r for r in reader if r['equation_id'] in IDS]
    assert len(selected)==5 and {r['equation_id'] for r in selected}==set(IDS)
    with (dest/csv_name).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(selected)
    lines=[]
    with (data/metadata_name).open() as f:
        for line in f:
            row=json.loads(line)
            if row['equation_id'] in IDS:
                assert row['split']=='train';lines.append(line)
    assert len(lines)==5
    with (dest/metadata_name).open('x') as f:f.writelines(lines)
    return dest/csv_name,dest/metadata_name


def budget(cases,overhead_family_medians,full_r128_targets):
    overhead=statistics.median(overhead_family_medians)
    r128=sum(c['historical_R128_native_seconds']+overhead for c in cases)*4
    r256=sum(c['historical_R256_full_seconds'] for c in cases)*4
    scenario=r128+r256
    alternate=20*statistics.median(full_r128_targets)+r256
    return dict(approved=False,approval_evidence=None,maximum_calls=40,R128_calls=20,R256_calls=20,MC_calls=0,
        central_scenario_seconds=scenario,R128_scenario_seconds=r128,R256_scenario_seconds=r256,
        proposed_cap_seconds=math.ceil(2*scenario+300),cap_formula='ceil(2*central_scenario+300); fixed40 calls, no automatic extension',
        central_formula='4*sum(same-ID saved native R128+family-median measured external factory overhead) +4*sum(same-ID saved full R256)',
        external_R128_overhead_seconds=overhead,external_overhead_families=4,
        full_wall_only_R128_generic_alternative_seconds=alternate,
        previous_generic_scenario_seconds=20*(statistics.median(full_r128_targets)+statistics.median(
            r['seconds'] for r in read(NESTED/'full_cost_targets.json') if r['profile']=='R256')),
        previous_generic_cap_seconds=read(NESTED/'NEXT_STEPS.json')['immediate_research_option']['proposed_cap_seconds'],
        includes_all_inaccurate_calls=True,confidence_interval=False,upper_bound=False,
        startup_IO_guardian_margin='2x scenario +300s covers planned setup/monitor/serialization/thermal overhead, not a proved bound',
        old_native_times_are_not_observed_full_cost=True,old_budget_reusable=False,
        caveat='old cohorts/hardware conditions and sparse R128 factory overhead; actual new full times remain unknown')


def build_jobs(cases,contract,b):
    jobs=[]
    for episode in range(4):
        # Rotate case order; exactly two R128-first and two R256-first pairs per ID.
        for i in range(5):
            c=cases[(i+episode)%5];pid=c['equation_id']
            phase=int(hashlib.sha256(('switch5_R40|'+pid).encode()).hexdigest(),16)%2
            profiles=('R128','R256') if (phase+episode)%2==0 else ('R256','R128')
            for p in profiles:
                cfg=dict(contract['profiles'][p]['config'],T=c['T']);reference=c['references'][p]
                seconds=c['historical_R128_native_seconds']+b['external_R128_overhead_seconds'] if p=='R128' else c['historical_R256_full_seconds']
                jobs.append(dict(key=f'{pid}_{p}_switch5_timing_episode{episode}',equation_id=pid,
                    family_id=c['family_id'],split='train',expression_hash=c['expression_hash'],profile=p,episode=episode,
                    config=cfg,seed=None,target_x=GRID,forcing_G=c['G'],expected_T=c['T'],
                    reference_path=reference['source'],reference_sha256=reference['source_sha256'],
                    reference_E_grid=reference['E_grid'],expected_accurate=reference['accurate'],
                    expected_f_float64_sha256=reference['f_float64_sha256'],
                    numerical_work=int(p[1:])*512,work_unit='S_applications',scenario_seconds=seconds,
                    proposed_watchdog_seconds=120 if p=='R128' else 164,execution_authorized=False))
    assert len(jobs)==len({j['key'] for j in jobs})==40
    assert Counter(j['profile'] for j in jobs)=={'R128':20,'R256':20}
    return jobs


def plan(out):
    import numpy as np
    from all_methods_dataset_loader import load_all_methods_equations
    from all_methods_bounds import certify_all_methods_forcing
    from mc_solver import auto_T  # Analytic horizon only. No MC sampling/function call.
    from chernov_solver import ChernovConfig
    from ml.nested_probability_calibration import baseline_default_choice
    from diagnostics.chernov_batch_control import lock_is_held
    from diagnostics.joint_benchmark_pilot import thermal
    from diagnostics.remizov_r0 import NUMERICAL
    out=out.resolve();out.mkdir(parents=True,exist_ok=False);(out/'sources').mkdir()
    shutil.copyfile(ROOT/'PROJECT_STATUS.md',out/'PROJECT_STATUS.before.md');write(out/'protected_before.json',snapshot(out))
    verified={}
    for folder in (NESTED,MODEL,ROOT/'results/remizov_r0_2026-10-26'):
        manifest=read(folder/'artifact_sha256.json')
        for name,h in manifest.items():assert sha(folder/name)==h,(folder.name,name)
        verified[folder.name]=len(manifest)
    # joint_grid100 predates artifact manifests; its selected records are pinned
    # by the immutable original ML package, not by a nonexistent new manifest.
    historical_pins=read(MODEL/'source_sha256.json')
    step=read(NESTED/'NEXT_STEPS.json')['immediate_research_option']
    assert step['equation_ids']==list(IDS) and step['maximum_calls']==40 and not step['approved']
    metrics=read(NESTED/'metrics.json');m=metrics['policies']['calibrated']['0.95']['baseline_default']
    assert [r['equation_id'] for r in m['details'] if r['action']=='R128']==list(IDS)
    assert m['accurate_deliveries']==m['baseline_accurate_deliveries']==33 and m['lost_baseline_accurate']==0
    pair={r['equation_id']:r for r in read(NESTED/'paired_oof.json') if r['equation_id'] in IDS}
    frozen=[]
    for pid in IDS:
        r=pair[pid];assert r['split']=='train' and r['outer_fold']==1
        assert baseline_default_choice(r['estimates_calibrated'],.95)=='R128'
        frozen.append(dict(equation_id=pid,family_id=r['family_id'],outer_fold=1,q=.95,
            policy='baseline_default',action='R128',estimates=r['estimates_calibrated'],
            evidence_scope='frozen outer-fold1 decision; not a final-model or production dispatch'))
    write(out/'frozen_OOF_decisions.json',frozen)
    contract=read(CONTRACT/'plan.json');data=ROOT/contract['data_dir'];r0=ROOT/'results/remizov_r0_2026-10-26'
    pins={}
    for name,h in contract['dataset_sha256'].items():assert sha(data/name)==h; pins[str((data/name).relative_to(ROOT))]=h
    for name in NUMERICAL:
        h=read(r0/'plan.json')['protected_sha256'][name];assert sha(ROOT/name)==h; pins[name]=h
    csv_path,meta_path=selected_inputs(data,out/'training_inputs')
    eqs={e.equation_id:e for e in load_all_methods_equations(csv_path,meta_path)}
    targets=read(NESTED/'accuracy_targets.json');cost=read(NESTED/'full_cost_targets.json');cases=[]
    certs=[]
    for pid in IDS:
        e=eqs[pid];metadata=e.metadata;assert metadata['split']=='train' and metadata['family_id']==pair[pid]['family_id']
        cert=certify_all_methods_forcing(e.remizov_problem,metadata);G=cert.forcing_bound.abs_upper_bound;p=e.remizov_problem
        T=auto_T(p.lambda_value,p.c_abs_upper_bound,G,1e-4);exact=np.asarray(e.y_exact(np.asarray(GRID)));references={}
        for prof in ('R128','R256'):
            target=next(r for r in targets if r['equation_id']==pid and r['profile']==prof)
            assert target['split']=='train' and target['expression_hash']==metadata['expression_hash'] and len(target['sources'])==1
            source=target['sources'][0];assert sha(ROOT/source)==historical_pins[source]
            r=read(ROOT/source);j=r['job'];z=r['result'];cfg=dict(contract['profiles'][prof]['config'],T=T)
            assert r['status']=='completed' and z['success'] and z['failure_reason']==''
            assert j.get('equation_id',j.get('problem_id'))==pid and z['problem_id']==pid
            assert r['family_id']==metadata['family_id'] and r['split']=='train' and r['expression_hash']==metadata['expression_hash']
            assert j['config']==cfg and z['n']==cfg['n'] and z['T']==T and z['scheme']=='solve'
            assert z['time_nodes']==513 and z['work_grid_points']==2081 and z['actual_dx']==.0125 and z['work_domain']==[-13.,13.]
            assert z['n_operator_applications']==int(prof[1:])*512 and z['n_spline_builds']==int(prof[1:])*512+1
            np.testing.assert_array_equal(z['x'],GRID);np.testing.assert_array_equal(r['y_exact'],exact)
            f=np.asarray(z['f']);assert f.shape==(33,) and np.isfinite(f).all()
            error=float(np.max(abs(f-exact)))
            assert error==r.get('E_grid',r.get('E_inf'))==target['errors'][0]
            assert (error<=.01)==r['accuracy_met']==bool(target['k'])
            ChernovConfig(**cfg).validate();pins[source]=sha(ROOT/source)
            references[prof]=dict(source=source,source_sha256=sha(ROOT/source),E_grid=error,accurate=error<=.01,
                config=cfg,full_method_wall_seconds=r.get('method_wall_seconds'),native_runtime_seconds=z['runtime_seconds'],
                f_float64_sha256=hashlib.sha256(f.astype('<f8').tobytes()).hexdigest(),
                exact_float64_sha256=hashlib.sha256(exact.astype('<f8').tobytes()).hexdigest())
            if prof=='R256':
                assert r['forcing_G']==G and r['effective_T']==T
                assert r['method_wall_seconds']>z['runtime_seconds']>0
                assert math.isclose(r['method_wall_seconds'],(r['timer_stop_ns']-r['timer_start_ns'])*1e-9,abs_tol=1e-12,rel_tol=0)
        assert references['R128']['full_method_wall_seconds'] is None
        c=next(r for r in cost if r['equation_id']==pid and r['profile']=='R256')
        assert c['scope']=='warm_end_to_end_method_wall_v1' and c['seconds']==references['R256']['full_method_wall_seconds']
        cases.append(dict(equation_id=pid,family_id=metadata['family_id'],split='train',expression_hash=metadata['expression_hash'],G=G,T=T,
            references=references,historical_R128_native_seconds=references['R128']['native_runtime_seconds'],historical_R256_full_seconds=c['seconds']))
        certs.append(dict(equation_id=pid,version=cert.version,serialized_ast_sha256=cert.serialized_ast_sha256,G=G,T=T,rounding=cert.rounding,
            coefficient_bounds=cert.coefficient_bounds,term_bounds=cert.term_bounds,support=list(cert.support)))
    assert [c['equation_id'] for c in cases if not c['references']['R128']['accurate']]==['AMV1_F0172_S03']
    assert [c['equation_id'] for c in cases if not c['references']['R256']['accurate']]==['AMV1_F0172_S03']
    overheads=[];overhead_sources=[];full128=[]
    for c in cost:
        if c['profile']!='R128':continue
        delta=[];full128.append(c['seconds'])
        for name in c['sources']:
            assert sha(ROOT/name)==historical_pins[name]
            r=read(ROOT/name);assert r['split']=='train' and r['status']=='completed'
            assert r.get('profile',r.get('job',{}).get('profile'))=='R128'
            assert math.isclose(r['method_wall_seconds'],(r['timer_stop_ns']-r['timer_start_ns'])*1e-9,abs_tol=1e-12,rel_tol=0)
            delta.append(r['method_wall_seconds']-r['result']['runtime_seconds']);assert delta[-1]>0
            pins[name]=sha(ROOT/name)
        overheads.append(statistics.median(delta));overhead_sources.append(dict(equation_id=c['equation_id'],source_paths=c['sources'],full_minus_native=delta,family_median=overheads[-1]))
    b=budget(cases,overheads,full128);jobs=build_jobs(cases,contract,b)
    assert math.isclose(sum(j['scenario_seconds'] for j in jobs),b['central_scenario_seconds'],abs_tol=1e-9)
    request_keys={j['key'] for j in jobs};existing=[];collisions=[]
    paths=list((ROOT/'results').glob('*/records/*.json'))+list(CONTRACT.glob('*/records/*.json'))+list((ROOT/'results').glob('*/events/*.request.json'))+list(CONTRACT.glob('*/events/*.request.json'))
    for path in paths:
        # Filenames narrow the scan to these5 TRAIN IDs; do not deserialize held-out data.
        if not any(pid in path.name for pid in IDS):continue
        r=read(path);j=r.get('job',{});pid=r.get('equation_id',j.get('equation_id',j.get('problem_id')))
        if j.get('key',path.stem) in request_keys:collisions.append(str(path.relative_to(ROOT)))
        z=r.get('result') or {};cfg=r.get('config',j.get('config',{}))
        if pid in IDS and z.get('x')==GRID and cfg.get('n') in (128,256):
            existing.append(dict(source=str(path.relative_to(ROOT)),equation_id=pid,n=cfg['n'],full_method_wall=r.get('method_wall_seconds') is not None,status=r.get('status')))
    assert not collisions
    write(out/'existing_results.json',dict(records_or_requests=existing,proposed_key_collisions=collisions,
        decision='old records remain accuracy references; do not substitute old R256 times into a new timing cohort'))
    sensor=thermal();locks={n:lock_is_held(ROOT/'results'/n) for n in ('.chernov_batch_controller.lock','.chernov_batch_compute.lock')}
    assert not any(locks.values()) and sensor['thermal_state'] is not None and sensor['thermal_state']<2
    assert shutil.disk_usage(ROOT).free>=256*1024**2
    for folder in (NESTED,MODEL):pins[str((folder/'artifact_sha256.json').relative_to(ROOT))]=sha(folder/'artifact_sha256.json')
    for name in ('registry.json','metrics.json','paired_oof.json','NEXT_STEPS.json','calibrators/outer1_R128.json','calibrators/outer1_R256.json'):
        pins[str((NESTED/name).relative_to(ROOT))]=sha(NESTED/name)
    sources=['diagnostics/plan_switch5_r_timing.py','diagnostics/remizov_r0.py','diagnostics/chernov_batch_control.py',
        'diagnostics/run_probability_validation.py','diagnostics/joint_benchmark_pilot.py','ml/nested_probability_calibration.py','ml/probabilistic_selector.py']+list(NUMERICAL)
    for name in sources:
        pins[name]=sha(ROOT/name);dest=out/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    execution=dict(schema='frozen_switch5_R40_preflight_v1',execution_authorized=False,status='awaiting_budget_approval',
        blockers=['no_approved_budget'],no_solver_calls_before_approval=True,data_dir=contract['data_dir'],target_x=GRID,
        frozen_OOF_decisions_sha256=sha(out/'frozen_OOF_decisions.json'),nested_registry_sha256=sha(NESTED/'registry.json'),
        cases=cases,jobs=jobs,budget=b,protected_sha256=pins,
        resources_proposed=dict(workers=1,threads=1,rss_mib=512,owner_lease_seconds=300,startup_seconds=90,
            monitor_seconds=.25,thermal_seconds=10,thermal_stop_at=2,series_cap_seconds=b['proposed_cap_seconds'],guardian=True),
        timing=dict(clock='perf_counter_ns',scope='warm_end_to_end_method_wall_v1',
            includes='certify analytic G + auto_T + config + fresh constructor + solve',
            excludes=['shared imports','shared loading','canonical exact/scoring','reference comparison','serialization','ML inference'],
            warmup='shared imports/load only; no extra numerical/JIT/MC warmup or dropped first call'),
        order='episode0..3; rotate cases by episode; alternate paired profile order with fixed ID-hash phase; exactly2 pairs each order per ID',
        seeds='none: deterministic Remizov; repeated timings are not independent accuracy trials; no MC/RNG seeds',
        resumption=dict(new_result_directory='results/ml_switch5_r_timing_2026-10-26',immutable_completed=True,
            persist_original_epoch_and_monotonic_deadline=True,maximum_physical_calls=40,requested_attempt_consumes_slot=True,
            no_automatic_retry_or_overwrite=True,completed_resume_new_calls=0,
            interrupted_unresolved_request='preserve partial/unknown; do not replace automatically within this40-job design'),
        runner_requirement='validate a dedicated R-only40 scope before execution; existing8-call F0047 and96-call validation drivers must not be invoked directly',
        critical_stop=['identity/hash/config/G/T/source mismatch','unexpected deterministic output or failure','RSS/wall/watchdog/thermal/lease breach'],
        known_failure_policy='F0172 both profiles inaccurate is expected; retain all8 timings/results, never label it an accurate-method win',
        evaluation=dict(primary_scope='five frozen outer-fold1 OOF-switch TRAIN cases, not final-model or new-family generalization',
            accuracy='E_grid33<=.01 every return; expected4/5 accurate profiles; exact output/repeat/reference equality, no tuning',
            cost='per-case median4 and sums R128 versus constant_R256, include inaccurate calls; accurate4 subset separately; method-only cost, inference separate',
            variability='all4 times,min/max,IQR,MAD,paired episode differences; do not remove outliers or add repeats',
            supported_speedup='both accurate, median_R128<=.9*median_R256,max_R128<min_R256, both MAD/median<=.2; otherwise tie/uncertain, no policy replacement',
            global40='local time delta attributable to5 switches only;4 other R256 full costs still missing, no measured global40 percentage or online overhead claim',
            test_validation_MC_ML='no other equations/splits, no MC, no benchmark1000, no model/policy/calibrator changes'),
        numerical_work=dict(S_applications=3932160,spline_builds=3932200,work_grid_points=2081,call_count=40))
    write(out/'plan.json',execution);write(out/'budget.json',b);write(out/'certificates.json',certs)
    write(out/'overhead_sources.json',overhead_sources)
    write(out/'preflight.json',dict(passed=True,solver_calls=0,MC_calls=0,ML_fits=0,identities_verified=5,
        saved_reference_errors_recomputed=10,accurate_reference_cases=4,known_inaccurate_case='AMV1_F0172_S03',
        G_T_configs_verified=True,OOF_scope_verified=True,source_manifests=verified,locks=locks,thermal=sensor,
        budget_approved=False,ready_to_execute=False,blocking_condition='budget_not_approved',validation_or_test_outcomes_used=False))
    with (out/'manifest.csv').open('x',newline='') as f:
        fields=['equation_id','family_id','split','expression_hash','G','T','R128_E_grid','R256_E_grid','R128_native_seconds','R256_full_seconds','expected_accurate']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for c in cases:writer.writerow({**{k:c[k] for k in fields[:6]},'R128_E_grid':c['references']['R128']['E_grid'],
            'R256_E_grid':c['references']['R256']['E_grid'],'R128_native_seconds':c['historical_R128_native_seconds'],
            'R256_full_seconds':c['historical_R256_full_seconds'],'expected_accurate':c['references']['R128']['accurate']})
    write(out/'approval_request.json',dict(approved=False,approval_evidence=None,plan_sha256=sha(out/'plan.json'),
        proposed_cap_seconds=b['proposed_cap_seconds'],maximum_calls=40,R128_calls=20,R256_calls=20,MC_calls=0,
        workers=1,threads=1,RSS_MiB=512,known_accuracy_failures=['AMV1_F0172_S03 both profiles exceed.01'],
        scope='approve execution separately; preparation/current request does not authorize numerical calls'))
    write(out/'environment.json',dict(python=sys.version,platform=platform.platform(),executable=sys.executable,
        versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','sympy')},thermal=sensor,
        thread_environment={n:os.environ[n] for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS')}))
    power={}
    for name,cmd in [('settings',['pmset','-g','custom']),('source',['pmset','-g','batt'])]:
        r=subprocess.run(cmd,capture_output=True,text=True);power[name]=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
    write(out/'hardware_power_preflight.json',power)
    print(json.dumps(dict(preflight_passed=True,solver_calls=0,known_inaccurate='F0172/S03',
        scenario_seconds=b['central_scenario_seconds'],proposed_cap_seconds=b['proposed_cap_seconds'],approved=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);plan(parser.parse_args().out)
