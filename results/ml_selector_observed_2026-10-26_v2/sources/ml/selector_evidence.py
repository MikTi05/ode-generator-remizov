"""Read and verify saved grid33 R128/MC1 evidence; no solver execution.

Historical taper/x0/MC0/MC2 and diagnostic mean4 are deliberately outside this
fixed pair. All1000 input identities still receive a target, including unknown.
"""
from __future__ import annotations
from dataclasses import asdict
import hashlib,json,math,struct
from pathlib import Path
import numpy as np
from ml.selector_labels import Evidence, GRID, TIMING_SCOPE, fingerprint, label_pair

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'reports/ml_selector_contract'
P1=CONTRACT/'sentinel_p1_2026-10-26'
P2=CONTRACT/'pilot_p2_2026-10-26'
R_FOLDER=ROOT/'results/joint_grid100_2026-10-25'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(condition,reason):
    if not condition:raise ValueError(reason)


def verify_manifest(folder):
    for name,h in read(folder/'artifact_sha256.json').items():check(sha(folder/name)==h,'changed frozen artifact: '+str(folder/name))


def validate_record(r,job,plan_hash,metadata,exact,method):
    """Reject wrong identities/configuration and stale or fabricated errors."""
    check(r['job']==job and r['plan_sha256']==plan_hash,'job_or_plan_mismatch')
    pid=metadata['equation_id'];z=r.get('result') or {};cfg=job['config']
    check(r['expression_hash']==metadata['expression_hash'] and r['family_id']==metadata['family_id'] and r['split']==metadata['split'],'identity_mismatch')
    check(job.get('equation_id',job.get('problem_id'))==pid,'job_identity_mismatch')
    # A finite positive runtime never converts an execution failure to a label.
    if r['status']!='completed':return None
    check(z.get('problem_id')==pid and z.get('success') is True and z.get('failure_reason')=='','invalid_execution')
    x=np.asarray(z.get('x'));f=np.asarray(z.get('f'),dtype=float)
    check(x.shape==f.shape==(33,) and np.array_equal(x,GRID) and np.isfinite(f).all(),'invalid_grid_or_output')
    check(np.array_equal(r['y_exact'],exact),'noncanonical_exact')
    error=float(np.max(np.abs(f-exact)))
    check(error==r.get('E_grid',r.get('E_inf')) and bool(error<=.01)==r['accuracy_met'],'error_or_accuracy_mismatch')
    check(math.isfinite(z['T']) and z['T']>0,'invalid_T')
    if method=='remizov':
        expected={'n':128,'boundary':'zero','num_t':513,'dx':.0125,'padding':12.,'interp':'cubic','tail_eps':1e-4,'t_grading':'quadratic','n_jobs':1,'max_work_grid_points':100000,'n_steps':32768,'start_method':'spawn','taper_width':None}
        check(all(cfg.get(k)==v for k,v in expected.items()),'unsupported_remizov_profile')
        check(z['n']==128 and z['T']==cfg['T'],'R_result_profile_mismatch')
    else:
        expected={'M':4096,'dt':1/128,'T':None,'chunk':1024,'max_batch_paths':1024,'tail_eps':1e-4,'stream_id':0,'max_steps':100000}
        check(all(cfg.get(k)==v for k,v in expected.items()),'unsupported_MC_profile')
        check(r['config']==cfg and r['seed']==cfg['seed']==z['seed_entropy'],'MC_seed_or_config_mismatch')
        check(cfg['seed']==248800000000+job['stage']*10000+100+job['episode'],'unfrozen_seed_schedule')
        check(z['M']==4096 and z['backend']=='numba' and z['completed_paths']==[4096]*33,'incomplete_paths')
        check(z['T']==job['expected_T'] and z['forcing_abs_upper_bound']==job['forcing_G'],'MC_horizon_or_G_mismatch')
        check(z['n_steps']==job['expected_steps'] and z['dt_requested']==1/128 and 0<z['dt_actual']<=1/128,'MC_steps_mismatch')
        check(z['n_path_steps']==z['n_normal_draws']==job['numerical_work']==cfg['max_path_steps']==33*4096*z['n_steps'],'MC_work_mismatch')
        check(z['tail_tolerance_met'] and z['compilation_seconds']==z['numba_import_seconds']==0.,'MC_tail_or_warmup_mismatch')
        check(z['rng_scheme']=='PCG64/SeedSequence/point_address_v1','wrong_RNG')
        keys=[[*struct.unpack('<8I',hashlib.sha256(b'mc_point_v1\0'+pid.encode()+struct.pack('<d',0. if x==0 else x)).digest()),0] for x in GRID]
        check(z['point_spawn_keys']==keys,'wrong_point_RNG_addresses')
        check(math.isclose((r['timer_stop_ns']-r['timer_start_ns'])*1e-9,r['method_wall_seconds'],rel_tol=0,abs_tol=1e-12),'timer_mismatch')
    return error


def load_saved_evidence(inputs):
    from all_methods_dataset_loader import load_all_methods_equations
    contract=read(CONTRACT/'plan.json');data=ROOT/contract['data_dir'];pins={}
    for name,h in contract['dataset_sha256'].items():
        check(sha(data/name)==h,'canonical_dataset_changed');pins[str((data/name).relative_to(ROOT))]=h
    for folder in (P1,P2):verify_manifest(folder)
    eqs={e.equation_id:e for e in load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl')}
    pair=dict(schema='selector_fixed_pair_v1',label_type='observed_fastest_v1',remizov=contract['profiles']['R128'],monte_carlo=contract['profiles']['MC1'],target_x=list(GRID),eps=.01,episode=0,
        rule='sole accurate; both fail none; both pass median4 with10%+disjoint ranges, tie R',timing_scope=TIMING_SCOPE,
        resource_policy='frozen per-equation guard limits retained; completed full calls only',
        profile_status='diagnostic_R128_MC1_pair; final_R128_vs_R256_selection_still_pending')
    pair_id=fingerprint(pair);evidence={};execution=[]
    for method,folder,profile in [('remizov',R_FOLDER,'R128'),('monte_carlo',P1,'MC1'),('monte_carlo',P2,'MC1')]:
        plan=read(folder/'plan.json');ph=sha(folder/'plan.json');pins[str((folder/'plan.json').relative_to(ROOT))]=ph
        for name,h in plan['protected_sha256'].items():
            check(sha(ROOT/name)==h,'source_or_dataset_changed: '+name);pins[name]=h
        jobs={j['key']:j for j in plan['jobs'] if (method=='remizov' and j.get('method')=='R' and j.get('stage')=='primary') or (method=='monte_carlo' and j.get('profile')=='MC1')}
        grouped={}
        for key,job in jobs.items():
            pid=job.get('equation_id',job.get('problem_id'));grouped.setdefault(pid,[]).append((key,job))
        for pid,items in grouped.items():
            m=inputs['metadata'][pid];e=Evidence(method,pid,m['expression_hash'],m['family_id'],m['split'],profile)
            check(eqs[pid].metadata['expression_hash']==m['expression_hash'],'input_join_identity_mismatch')
            exact=np.asarray(eqs[pid].y_exact(np.asarray(GRID)));own=[]
            for key,job in sorted(items,key=lambda x:x[1].get('episode',0)):
                path=folder/'records'/f'{key}.json'
                if not path.exists():continue
                pins[str(path.relative_to(ROOT))]=sha(path);r=read(path);episode=job.get('episode',0)
                try:error=validate_record(r,job,ph,m,exact,method)
                except (ValueError,KeyError,TypeError,OverflowError) as ex:
                    e.issues.append(str(ex));error=None;r={**r,'status':'invalid_result'}
                z=r.get('result') or {};scope=TIMING_SCOPE if method=='monte_carlo' else 'historical_solver_runtime_excludes_external_factory'
                wall=r.get('method_wall_seconds') if method=='monte_carlo' else z.get('runtime_seconds')
                cohort=fingerprint(dict(environment_sha256=sha(folder/'environment.json'),session=r.get('session'),workers=1,threads=1))
                row=dict(method=method,profile=profile,equation_id=pid,episode=episode,status=r['status'],error=error,config=job['config'],seed=job['config'].get('seed'),result=z,y_exact=r.get('y_exact'),seconds=wall,timing_scope=scope,timing_cohort=cohort,source=str(path.relative_to(ROOT)),source_sha256=sha(path),plan_sha256=ph)
                own.append(row);execution.append(row)
            e.records=own;e.source=[r['source'] for r in own];e.source_sha256=[r['source_sha256'] for r in own]
            zero=next((r for r in own if r['episode']==0),None)
            if zero:
                e.status=zero['status'];e.error=zero['error'];e.T=zero['result'].get('T')
            if e.issues:e.status='invalid_result';e.error=None
            e.times=[r['seconds'] for r in own if r['status']=='completed' and r['seconds'] is not None]
            e.timing_scope=own[0]['timing_scope'] if own else None
            cohorts={r['timing_cohort'] for r in own};e.timing_cohort=next(iter(cohorts)) if len(cohorts)==1 else None
            e.deterministic_repeats_equal=(method=='remizov' and len(own)==4 and all(r['status']=='completed' and r['result']['f']==own[0]['result']['f'] for r in own))
            check((pid,method) not in evidence,'duplicate_equation_method_group');evidence[pid,method]=e
    rows=[]
    for pid in map(str,inputs['ids']):
        m=inputs['metadata'][pid]
        for method,profile in [('remizov','R128'),('monte_carlo','MC1')]:
            evidence.setdefault((pid,method),Evidence(method,pid,m['expression_hash'],m['family_id'],m['split'],profile))
        row=label_pair(evidence[pid,'remizov'],evidence[pid,'monte_carlo'],pair_id)
        row.update(complexity=m['complexity_level'],regime=m['regime'],dataset_sha256=contract['dataset_sha256'],
            evidence_keys=[pid+':remizov',pid+':monte_carlo'])
        rows.append(row)
    return dict(rows=rows,evidence={pid+':'+method:asdict(e) for (pid,method),e in evidence.items()},execution=execution,pair=pair,pair_id=pair_id,source_sha256=pins)
