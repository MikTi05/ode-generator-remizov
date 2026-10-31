"""Dataset-qualified, profile-checked saved evidence. No numerical solving."""
from __future__ import annotations
import csv,hashlib,json,math,statistics,struct
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from ml.selector_evidence import ROOT,CONTRACT,read,sha,verify_manifest
from ml.remizov_features import coefficient_features,CERT_FIELDS

GRID=[-1+i/16 for i in range(33)]
ORIGINAL=ROOT/'data/all_methods_v1_1000_2026-10-18'
DEVELOPMENT=ROOT/'results/ml_development8_2026-10-26'
SOURCES=(
    ROOT/'results/joint_grid100_2026-10-25',ROOT/'results/final_grid5_2026-10-25',
    CONTRACT/'sentinel_p1_2026-10-26',CONTRACT/'pilot_p2_2026-10-26',
    ROOT/'results/remizov_r0_2026-10-26',ROOT/'results/f0047_observed_timing_2026-10-26',
    ROOT/'results/f0047_three_profile_2026-10-26',ROOT/'results/ml_switch5_r_timing_2026-10-26',DEVELOPMENT)


def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def equation_key(dataset_sha,pid,expression_hash):
    if len(dataset_sha)!=64 or len(expression_hash)!=64:raise ValueError('full dataset and expression SHA256 required')
    return fingerprint(dict(dataset_sha256=dataset_sha,equation_id=pid,expression_hash=expression_hash))


def profile_of(config,contract):
    """Resource ceilings do not change a completed physical numerical profile."""
    if config.get('n') in (128,256):
        p='R'+str(config['n']);expected=contract['profiles'][p]['config']
        if set(config)!=set(expected) or any(config[k]!=v for k,v in expected.items() if k!='T'):return None
        return p
    p='MC1';expected=contract['profiles'][p]['config'];guards={'seed','max_path_steps','max_runtime_seconds'}
    if set(config)!=set(expected) or any(config[k]!=v for k,v in expected.items() if k not in guards):return None
    return p


def semantic_MC_key(row):
    """Same seed/address/config is a replay; clocks and resource caps are not trials."""
    return fingerprint(dict(equation_key=row['equation_key'],profile=row['profile'],T=row['T'],N=row['N'],
        seed=row['seed'],stream=row['stream_id'],rng=row['rng_scheme'],point_spawn_keys=row['point_spawn_keys']))


def collapse_accuracy(rows):
    grouped=defaultdict(list)
    for r in rows:grouped[r['equation_key'],r['profile']].append(r)
    targets=[];dedup=[]
    for (key,p),rs in sorted(grouped.items()):
        base={k:rs[0][k] for k in ('equation_key','dataset_sha256','equation_id','expression_hash','family_id','canonical_family_id','split','profile')}
        if p!='MC1':
            if any(r['f']!=rs[0]['f'] or r['T']!=rs[0]['T'] for r in rs):raise ValueError('conflicting deterministic repeats: '+key)
            unique=[rs[0]];semantics='one deterministic equation/profile; repeated timings are not labels'
        else:
            trials={}
            for r in rs:
                semantic=semantic_MC_key(r)
                if semantic in trials:
                    old=trials[semantic]
                    if r['f']!=old['f'] or r['error']!=old['error']:raise ValueError('conflicting MC replay: '+semantic)
                    dedup.append(dict(semantic_key=semantic,kept=old['source'],duplicate=r['source'],equation_key=key))
                else:trials[semantic]=r
            unique=list(trials.values());semantics='unique semantic seeds are repeated observations, one equation row'
        targets.append(dict(base,k=sum(r['accurate'] for r in unique),n=len(unique),errors=[r['error'] for r in unique],
            error=unique[0]['error'] if p!='MC1' else None,sources=[r['source'] for r in rs],
            unique_sources=[r['source'] for r in unique],semantic_trials=[semantic_MC_key(r) for r in unique] if p=='MC1' else [],trial_semantics=semantics))
    return targets,dedup


def subset_inputs(data,ids,dest):
    dest.mkdir()
    with (data/'all_methods_equations_v1.csv').open(newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=[r for r in reader if r['equation_id'] in ids]
    with (dest/'all_methods_equations_v1.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    with (dest/'all_methods_equations_v1_metadata.jsonl').open('x') as f:
        for line in (data/'all_methods_equations_v1_metadata.jsonl').open():
            r=json.loads(line)
            if r['equation_id'] in ids:
                assert r['split']=='train';f.write(line)


def prepare(out):
    from all_methods_dataset_loader import load_all_methods_equations
    from all_methods_bounds import certify_all_methods_forcing
    from mc_solver import auto_T
    contract=read(CONTRACT/'plan.json');pins={};original_sha=sha(ORIGINAL/'all_methods_equations_v1.csv')
    for n,h in contract['dataset_sha256'].items():
        assert sha(ORIGINAL/n)==h;pins[str((ORIGINAL/n).relative_to(ROOT))]=h
    metadata={}
    # Only TRAIN metadata/coefficients become semantic model inputs.
    for line in (ORIGINAL/'all_methods_equations_v1_metadata.jsonl').open():
        r=json.loads(line)
        if r['split']=='train':metadata[r['equation_id']]=r
    original_ids=set(metadata);development_meta={}
    for line in (DEVELOPMENT/'training_inputs/all_methods_equations_v1_metadata.jsonl').open():
        r=json.loads(line);development_meta[r['equation_id']]=r
    dev_sha=sha(DEVELOPMENT/'input_pool/all_methods_equations_v1.csv')
    raw=[];audit=[];quarantine=[];inventory=[];known_ids=set()
    for folder in SOURCES:
        if (folder/'artifact_sha256.json').exists():verify_manifest(folder)
        plan=read(folder/'plan.json');ph=sha(folder/'plan.json');pins[str((folder/'plan.json').relative_to(ROOT))]=ph
        for name,h in plan.get('protected_sha256',{}).items():
            assert sha(ROOT/name)==h,name
        development=folder==DEVELOPMENT
        records=sorted((folder/'records').glob('*.json'));inventory.append(dict(folder=str(folder.relative_to(ROOT)),records=len(records),scope='audited development quarantine' if development else 'original train candidate'))
        for path in records:
            r=read(path);j=r.get('job',{});pid=r.get('equation_id',j.get('equation_id',j.get('problem_id')))
            if pid not in (development_meta if development else metadata):
                audit.append(dict(source=str(path.relative_to(ROOT)),reason='not permitted original train identity'));continue
            cfg=j.get('config',r.get('config',{}));profile=profile_of(cfg,contract)
            if not profile:
                audit.append(dict(source=str(path.relative_to(ROOT)),reason='different numerical profile',M=cfg.get('M'),n=cfg.get('n'),dt=cfg.get('dt')));continue
            if r.get('status')!='completed' or not (r.get('result') or {}).get('success'):
                audit.append(dict(source=str(path.relative_to(ROOT)),reason='unknown/censored execution',status=r.get('status')));continue
            m=(development_meta if development else metadata)[pid];ds=dev_sha if development else original_sha;z=r['result']
            assert r['expression_hash']==m['expression_hash'] and r['family_id']==m['family_id'] and r['split']=='train'
            if 'config' in r:assert r['config']==cfg
            if development:assert j['identity']==f"{ds}|{m['structure_signature']}|{pid}"
            assert r['plan_sha256']==ph and z['problem_id']==pid and j in plan['jobs'],str(path)
            assert z['x']==GRID and np.asarray(z['f']).shape==(33,) and np.isfinite(z['f']).all() and not z['failure_reason']
            error=float(np.max(abs(np.asarray(z['f'])-np.asarray(r['y_exact']))))
            assert error==r.get('E_grid',r.get('E_inf')) and (error<=.01)==r['accuracy_met']
            row=dict(equation_key=equation_key(ds,pid,m['expression_hash']),dataset_sha256=ds,equation_id=pid,expression_hash=m['expression_hash'],
                family_id=m['structure_signature'],canonical_family_id=m['family_id'],split='train',profile=profile,T=z['T'],f=z['f'],y_exact=r['y_exact'],
                error=error,accurate=error<=.01,source=str(path.relative_to(ROOT)),source_sha256=sha(path),seed=None,full_seconds=None,
                timing_cohort=r.get('session'),folder=str(folder.relative_to(ROOT)),config=cfg,eligibility='development_quarantine' if development else 'permitted_original_train')
            if profile=='MC1':
                assert z['backend']=='numba' and z['block_steps']==64 and z['M']==4096 and z['completed_paths']==[4096]*33
                assert z['n_path_steps']==z['n_normal_draws']==33*4096*z['n_steps'] and z['dt_requested']==1/128 and 0<z['dt_actual']<=1/128
                assert z['n_path_steps']<=cfg['max_path_steps'] and z['n_steps']<=cfg['max_steps'] and z['tail_tolerance_met']
                assert z['rng_scheme']=='PCG64/SeedSequence/point_address_v1' and z['seed_entropy']==cfg['seed']
                keys=[[*struct.unpack('<8I',hashlib.sha256(b'mc_point_v1\0'+pid.encode()+struct.pack('<d',0. if x==0 else x)).digest()),0] for x in GRID]
                assert z['point_spawn_keys']==keys
                row.update(N=z['n_steps'],seed=z['seed_entropy'],stream_id=z['stream_id'],rng_scheme=z['rng_scheme'],point_spawn_keys=z['point_spawn_keys'],G=z['forcing_abs_upper_bound'])
                # Previously reserved post-fit seed audit remains excluded from fit/tuning.
                if folder.name=='f0047_three_profile_2026-10-26':row['eligibility']='previous_postfit_seed_audit_quarantine'
            else:
                n=int(profile[1:]);assert z['n']==n and z['T']==cfg['T'] and z['scheme']=='solve'
                assert z['n_operator_applications']==n*512 and z['n_spline_builds']==n*512+1 and z['work_grid_points']==2081 and z['time_nodes']==513
                assert z['actual_dx']==.0125 and z['work_domain']==[-13.,13.]
            if 'method_wall_seconds' in r:
                seconds=r['method_wall_seconds'];assert seconds>0 and math.isclose(seconds,(r['timer_stop_ns']-r['timer_start_ns'])*1e-9,abs_tol=1e-12,rel_tol=0)
                row['full_seconds']=seconds
            else:row['time_exclusion']='native runtime or task wall includes scoring; never a full method-wall target'
            pins[row['source']]=row['source_sha256']
            if row['eligibility']=='permitted_original_train':raw.append(row);known_ids.add(pid)
            else:quarantine.append(row)
    # Previous validation/test and legacy profiles are inventoried without accessing outcomes.
    for folder in sorted((ROOT/'results').iterdir()):
        if not folder.is_dir() or folder in SOURCES or folder==out:continue
        records=list((folder/'records').glob('*.json'))
        if records:inventory.append(dict(folder=str(folder.relative_to(ROOT)),records=len(records),scope='excluded without outcome access: previous validation, x0 or different profile'))
    for name,reason in [('joint_pilot_2026-10-25','legacy x0/T-rounded/scheme budgets'),('mc1000_x0_2026-10-25','x0 only'),('chernov_solve1000_2026-10-18','legacy taper different profile'),('mc_optimized_backend_2026-10-25','M64/1024 or different dataset/backend')]:
        inventory.append(dict(folder='results/'+name,scope='excluded without outcome access: '+reason))
    # Reconstruct exact/G/T only for whitelisted original TRAIN cases. Never solve.
    subset_inputs(ORIGINAL,known_ids,out/'training_inputs')
    eqs={e.equation_id:e for e in load_all_methods_equations(out/'training_inputs/all_methods_equations_v1.csv',out/'training_inputs/all_methods_equations_v1_metadata.jsonl')}
    horizons={}
    for pid,e in eqs.items():
        p=e.remizov_problem;G=certify_all_methods_forcing(p,e.metadata).forcing_bound.abs_upper_bound
        horizons[pid]=(G,auto_T(p.lambda_value,p.c_abs_upper_bound,G,1e-4),e.y_exact(np.asarray(GRID)))
    for r in raw:
        G,T,exact=horizons[r['equation_id']]
        assert r['T']==T and np.array_equal(r['y_exact'],exact),r['source']
        if r['profile']=='MC1':assert r['G']==G and r['N']==math.ceil(T*128)
    accuracy,duplicates=collapse_accuracy(raw)
    # Preserve development repeated-seed identities for audit only, never targets.
    quarantine_targets,quarantine_duplicates=collapse_accuracy(quarantine)
    index_keys=sorted({r['equation_key'] for r in accuracy});index={k:i for i,k in enumerate(index_keys)}
    by_key={r['equation_key']:r for r in raw};ids=[by_key[k]['equation_id'] for k in index_keys]
    with np.load(ORIGINAL/'all_methods_equations_v1_ml.npz',allow_pickle=False) as z:
        lookup={str(pid):i for i,pid in enumerate(z['equation_id'])};ix=[lookup[pid] for pid in ids]
        assert all(str(z['split'][i])=='train' for i in ix)
        coeff=z['X'][ix];grid=z['x_grid'];assert list(z['channel_order'])==['a','b','c','g']
    feature_rows=[]
    for pid,values in zip(ids,coeff):
        m=metadata[pid];allowed={k:m[k] for k in CERT_FIELDS};allowed['serialized_ast']={k:m['serialized_ast'][k] for k in ('a_raw','b_raw','c_raw')}
        feature_rows.append(coefficient_features(values,grid,allowed))
    names=list(feature_rows[0]);X=np.asarray([[r[n] for n in names] for r in feature_rows]);assert X.shape==(len(ids),253)
    for r in accuracy:r['index']=index[r['equation_key']]
    grouped=defaultdict(list)
    for r in raw:
        if r['full_seconds'] is not None:grouped[r['equation_key'],r['profile']].append(r)
    costs=[]
    for (key,p),rs in sorted(grouped.items()):
        base={k:rs[0][k] for k in ('equation_key','dataset_sha256','equation_id','expression_hash','family_id','canonical_family_id','split','profile')}
        # Equal cohort weight: repeated timing blocks never multiply equations.
        cohorts=defaultdict(list)
        for r in rs:cohorts[r['timing_cohort']].append(r['full_seconds'])
        seconds=statistics.median(statistics.median(ts) for ts in cohorts.values())
        costs.append(dict(base,index=index[key],seconds=seconds,times=[r['full_seconds'] for r in rs],cohort_medians={k:statistics.median(v) for k,v in cohorts.items()},sources=[r['source'] for r in rs],scope='warm_end_to_end_method_wall_v1'))
    paired=sorted(k for k in index_keys if all(any(r['equation_key']==k and r['profile']==p for r in accuracy) for p in ('R128','R256','MC1')))
    counts={p:dict(equations=sum(r['profile']==p for r in accuracy),families=len({r['family_id'] for r in accuracy if r['profile']==p}),
        successes=sum(r['k'] for r in accuracy if r['profile']==p),trials=sum(r['n'] for r in accuracy if r['profile']==p),full_cost_equations=sum(r['profile']==p for r in costs)) for p in ('R128','R256','MC1')}
    return dict(X=X,names=names,accuracy=accuracy,cost=costs,paired=paired,raw=raw,quarantine=quarantine,
        audit=dict(counts=counts,permitted_rows=len(raw),duplicates=duplicates,quarantine_duplicates=quarantine_duplicates,quarantine_counts=Counter(r['eligibility'] for r in quarantine),
            quarantined_equation_profiles=len(quarantine_targets),excluded_records=audit,inventory=inventory,
            original_dataset_sha256=original_sha,development_dataset_sha256=dev_sha,canonical_ID_collisions=sorted(set(development_meta)&original_ids),
            identity='SHA256 canonical CSV bytes + equation_id + expression_hash',family_groups='global topology structure_signature; same topology always one fold across cohorts',
            dev8_and_previous_seed_audit_used_for_fit_or_tuning=False,validation_test_outcomes_read=False,
            full_time_missing_not_imputed=True,selection_bias='historical selected TRAIN cohorts, adaptive acquisition and few MC-positive families; pooled seed rates not population rates'),pins=pins)


def timing_contexts(raw):
    full_folders={r['folder'] for r in raw if r['full_seconds'] is not None};contexts=[];pins={};reference=None
    for folder in SOURCES:
        env=read(folder/'environment.json');plan=read(folder/'plan.json');assert plan['resources']['compute_workers']==1
        threads=env.get('thread_environment',env.get('threads',{}));assert isinstance(threads,dict) and threads and all(str(v)=='1' for v in threads.values())
        signature=dict(python=env['python'],numpy=env['versions']['numpy'],scipy=env['versions']['scipy'])
        if reference is None:reference=signature
        assert signature==reference
        full=str(folder.relative_to(ROOT)) in full_folders
        if full:assert env.get('platform')=='macOS-26.6.2-arm64-arm-64bit'
        contexts.append(dict(folder=str(folder.relative_to(ROOT)),environment_sha256=sha(folder/'environment.json'),signature=signature,
            platform=env.get('platform'),hardware_metadata_missing='platform' not in env,full_method_training_times_used=full,
            numba=env['versions'].get('numba'),threads=threads,workers=1))
        pins[str((folder/'environment.json').relative_to(ROOT))]=sha(folder/'environment.json')
    return dict(passed=True,contexts=contexts,different_sessions_retained=True,
        scope='full warm method-wall targets share recorded platform/library/thread context; old native-only cohorts lack platform metadata and contribute accuracy only; not a matched prospective timing experiment'),pins


def legacy_inventory(out):
    """Audit configs and acquisition roles without opening held-out outcomes."""
    entries=[];pins={}
    for folder in sorted((ROOT/'results').iterdir()):
        if not folder.is_dir() or folder==out:continue
        name=folder.name
        if folder in SOURCES:scope='audited raw candidate source (development8 quarantined)'
        elif 'validation' in name:scope='previous validation: outcome access prohibited'
        elif name.startswith(('ml_','remizov_ml','scientific_report','remizov_example','remizov_technical')):scope='derived ML/report/artifact; never a new independent solver observation'
        else:scope='legacy acquisition, other dataset, x0 or incompatible numerical profile; not permission to add labels'
        entry=dict(folder=str(folder.relative_to(ROOT)),scope=scope)
        plan_path=folder/'plan.json'
        if plan_path.exists() and folder not in SOURCES:
            plan=read(plan_path);pins[str(plan_path.relative_to(ROOT))]=sha(plan_path)
            configs={k:plan[k] for k in ('baseline_config','old_config','new_config','r_base_config','mc_base_config') if k in plan}
            if configs:entry['declared_configs']=configs
            if name=='joint_pilot_2026-10-25':
                attempts=plan['attempts'];entry['datasets']=sorted({a['identity']['dataset_sha256'] for a in attempts})
                entry['permitted_training_attempts']=sum(a['identity'].get('training_eligible',False) for a in attempts)
                entry['scope']='different small20 dataset; development-exposed acquisition, declared ML_training_eligible=false; not original train'
                spec=read(folder/'analysis_spec.json');assert spec['ML_training_eligible'] is False
                pins[str((folder/'analysis_spec.json').relative_to(ROOT))]=sha(folder/'analysis_spec.json')
        entries.append(entry)
    return dict(scope='directory inventory plus source plans/configs; no previous validation/test or legacy development outcomes opened',entries=entries),pins
