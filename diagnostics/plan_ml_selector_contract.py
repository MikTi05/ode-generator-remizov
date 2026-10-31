"""Build an inert benchmark specification from local SAVED evidence only.

No solver imports, simulations, fitting, dataset generation or executable run
subcommands. The generated jobs are conditional specifications, never a queue
submitted to a worker. Outputs must not already exist.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/all_methods_v1_1000_2026-10-18'
J = ROOT/'results/joint_grid100_2026-10-25'
F = ROOT/'results/final_grid5_2026-10-25'
R = ROOT/'reports/ml_selector_readiness'
SENTINELS = ['AMV1_F0047_S00', 'AMV1_F0005_S02', 'AMV1_F0042_S00',
             'AMV1_F0059_S04', 'AMV1_F0125_S04']
FORCED = SENTINELS+['AMV1_F0134_S03','AMV1_F0136_S01','AMV1_F0139_S00',
                   'AMV1_F0141_S00','AMV1_F0184_S01']
INPUTS = {}


def digest(p):
    p = Path(p); value = hashlib.sha256(p.read_bytes()).hexdigest()
    INPUTS[str(p.relative_to(ROOT))] = value
    return value


def read(p):
    digest(p)
    return json.loads(Path(p).read_text())


def rows(p):
    digest(p)
    with Path(p).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write(p, value):
    payload=json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n'
    with p.open('x') as stream: stream.write(payload)


def csvwrite(p, records):
    with p.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,list(records[0]));writer.writeheader();writer.writerows(records)


def rank(role, pid):
    return hashlib.sha256(f'2488|selector_contract_v1|{role}|{pid}'.encode()).hexdigest()


def seed(stage, profile, episode):
    # Seed is global per stage/profile/episode. Existing point_address_v1 adds
    # equation ID and coordinate, so different equations have different streams.
    return 248800000000 + stage*10000 + profile*100 + episode


def build(out):
    out=out.resolve();out.mkdir(parents=True,exist_ok=True)
    names=['plan.json','pilot_manifest.csv','pilot_jobs.jsonl','benchmark_requirements.csv',
           'cost_forecast.csv','benchmark_cost_scenarios.csv','planning_checks.json','sources_sha256.json']
    if any((out/n).exists() for n in names):raise FileExistsError('refusing to overwrite generated plan')
    # Every requested audit file is consumed completely; never use an ID-only
    # join or inspect validation/test outcomes for profile selection.
    audit=read(R/'audit_summary.json');verification=read(R/'verification.json')
    digest(R/'AUDIT_REPORT.md');status_text=(ROOT/'PROJECT_STATUS.md').read_text()
    digest(ROOT/'PROJECT_STATUS.md');assert status_text
    audited={r['equation_id']:r for r in rows(R/'audit_by_equation.csv')}
    digest(DATA/'all_methods_equations_v1_metadata.jsonl')
    meta={m['equation_id']:m for m in map(json.loads,(DATA/'all_methods_equations_v1_metadata.jsonl').read_text().splitlines())}
    passport=read(DATA/'all_methods_equations_v1_report.json')
    for name,h in passport['output_sha256'].items():assert digest(DATA/name)==h
    assert len(meta)==len(audited)==1000 and set(meta)==set(audited)
    assert audit['historical_remizov']['successful']==997 and audit['coverage']['same_grid33_pairs']==100
    for name,h in verification['artifact_sha256'].items():assert digest(R/name)==h
    train={p:m for p,m in meta.items() if m['split']=='train'}
    jp=read(J/'plan.json'); fp=read(F/'plan.json')
    cp=read(ROOT/'results/mc1000_x0_2026-10-25/plan.json')
    pre={r['problem_id']:r for r in cp['preflight']}
    joint={r['problem_id']:r for r in jp['selection']}
    assert set(joint)<=set(train)
    for pid,p in pre.items():assert p['expression_hash']==meta[pid]['expression_hash']
    # Outcomes are accessed below only through the train subset.
    selected=[];used=set()
    def add(pid,cohort,reason):
        assert pid in train and train[pid]['family_id'] not in used
        used.add(train[pid]['family_id']);selected.append((pid,cohort,reason))
    for pid in FORCED:
        add(pid,'stress','sentinel_final5' if pid in SENTINELS else
            ('historical_worker_exception' if pid in FORCED[5:8] else 'historical_taper_artifact'))
    for accuracy in ('True','False'):
        candidates=[p for p in joint if audited[p]['remizov_accuracy_pass']==accuracy]
        count=0
        for pid in sorted(candidates,key=lambda p:rank('stress_'+accuracy,p)):
            if train[pid]['family_id'] in used:continue
            add(pid,'stress','historical_R_taper_'+('pass' if accuracy=='True' else 'finite_inaccurate'))
            count+=1
            if count==5:break
        assert count==5
    cells=sorted({(m['complexity_level'],m['regime']) for m in train.values()})
    for cell in cells:
        candidates=[p for p in joint if joint[p]['cohort']=='representative' and
                    tuple(joint[p]['cell'])==cell and train[p]['family_id'] not in used]
        if not candidates:
            candidates=[p for p,m in train.items() if (m['complexity_level'],m['regime'])==cell and m['family_id'] not in used]
        add(min(candidates,key=lambda p:rank('core',p)),'coverage','one_train_family_per_complexity_regime_cell')
    assert len(selected)==len(used)==40 and len(cells)==20
    grid=[-1+j/16 for j in range(33)]
    profiles={
        'R128':dict(method='remizov',implementation='chernov_solver.ChernovSolver.solve',config=dict(jp['r_base_config'],n=128)),
        'R256':dict(method='remizov',implementation='chernov_solver.ChernovSolver.solve',config=dict(jp['r_base_config'],n=256)),
    }
    for code,M,h in [('MC0',1024,1/128),('MC1',4096,1/128),('MC2',4096,1/256)]:
        profiles[code]=dict(method='monte_carlo',implementation='mc_optimized.solve_monte_carlo_optimized',
            backend='numba',block_steps=64,aggregation='none_single_full_call',
            config=dict(cp['config'],M=M,dt=h,seed=None,max_runtime_seconds=None))
    manifest=[];requirements=[];forecasts=[];jobs=[]
    final_ids={t['problem_id'] for t in fp['tasks']}
    for pid,m in meta.items():
        have128=pid in joint; have256=pid in final_ids
        haveMC0=pid in joint
        requirements.append(dict(equation_id=pid,family_id=m['family_id'],split=m['split'],expression_hash=m['expression_hash'],
            R128_numeric_reusable=have128,R256_numeric_reusable=have256,MC0_historical_episode_available=haveMC0,
            MC1_four_historical_seeds=pid in final_ids,MC2_historical_available=False,
            R128_numeric_missing=not have128,R256_numeric_missing=not have256,
            MC0_historical_grid33_missing=not haveMC0,
            final_fresh_R_timing_calls_if_selected=4,final_fresh_MC_calls_if_selected=4,
            final_seed0_accuracy_requires_new_MC=True,
            numerical_reuse_does_not_supply_final_timing=True))
    oldseeds=set(cp['seeds']+jp['seeds']+fp['seeds'])
    def job(pid,profile,stage,episode,condition):
        spec=profiles[profile];p=pre[pid];cfg=dict(spec['config']);N=0
        if profile.startswith('MC'):
            N=max(1,math.ceil(p['T']/cfg['dt']));cfg.update(seed=seed(stage,int(profile[-1]),episode),
                max_path_steps=33*cfg['M']*N,max_runtime_seconds=None)
            assert cfg['seed'] not in oldseeds and N<=cfg['max_steps']
        else:cfg['T']=p['T']
        return dict(key=f'{pid}_{profile}_stage{stage}_episode{episode}',equation_id=pid,
            expression_hash=meta[pid]['expression_hash'],family_id=meta[pid]['family_id'],split='train',
            profile=profile,stage=stage,episode=episode,condition=condition,
            target_x=grid,config=cfg,expected_T=p['T'],forcing_G=p['G'],expected_steps=N,
            numerical_work=33*cfg['M']*N if N else cfg['n']*512,
            work_unit='path_updates' if N else 'S_applications',
            execution_authorized=False)
    for pid,cohort,reason in selected:
        m=train[pid];p=pre[pid]
        row=dict(equation_id=pid,family_id=m['family_id'],split='train',expression_hash=m['expression_hash'],
            complexity=m['complexity_level'],regime=m['regime'],cohort=cohort,reason=reason,
            sentinel=pid in SENTINELS,historical_R_status=audited[pid]['remizov_status'],
            historical_R_error=audited[pid]['remizov_error'],historical_R128_grid33=pid in joint,
            historical_R256_grid33=pid in final_ids,historical_MC0_grid33=pid in joint,T=p['T'],G=p['G'])
        manifest.append(row)
        for profile,available in [('R128',pid in joint),('R256',pid in final_ids)]:
            if not available:jobs.append(job(pid,profile,0,0,'pilot_authorized;numeric_cell_missing'))
        for profile in ('MC0','MC1','MC2'):
            for episode in range(4):
                jobs.append(job(pid,profile,1 if pid in SENTINELS else 2,episode,
                    'pilot_authorized' if pid in SENTINELS else f'pilot_authorized;selected_MC_profile=={profile};stage1_gate_allows_stage2'))
    # Costs are operation-scaled SCENARIOS anchored to this equation, not
    # predictions of accuracy or a claim of equal throughput at finer h.
    for row in manifest:
        pid=row['equation_id'];baseline=None;source=None
        if pid in joint:
            path=J/'records'/f'{pid}_MC0.json';v=read(path)
            assert v['expression_hash']==meta[pid]['expression_hash']
            baseline=v['result']['runtime_seconds'];source=str(path.relative_to(ROOT))
        else:
            vals=[]
            for i in range(4):
                path=ROOT/'results/mc1000_x0_2026-10-25/attempts'/f'{pid}_seed{i}.json';v=read(path)
                assert v['expression_hash']==meta[pid]['expression_hash']
                vals.append(v['result']['runtime_seconds'])
            baseline=33*statistics.median(vals);source='x0_median4_times33_scenario'
        for profile in ('MC0','MC1','MC2'):
            cfg=profiles[profile]['config'];n=math.ceil(pre[pid]['T']/cfg['dt'])
            scaled=baseline*(cfg['M']/1024)*(n/pre[pid]['N'])
            forecasts.append(dict(equation_id=pid,profile=profile,split='train',sentinel=row['sentinel'],
                baseline_seconds=baseline,baseline_source=source,baseline_path_steps=33*1024*pre[pid]['N'],
                candidate_path_steps=33*cfg['M']*n,scenario_seconds_one_call=scaled,
                scenario_seconds_four_calls=4*scaled,confidence_interval=False,
                includes_external_preprocessing=False,throughput_at_new_h_measured=profile=='MC0' and pid in joint))
    # Resource allocation proposals are derived from work-scaled scenarios;
    # do not silently inherit 60s/120s historical limits.
    forecast_index={(r['equation_id'],r['profile']):r for r in forecasts}
    x0_costs={r['equation_id']:r for r in rows(ROOT/'results/mc1000_x0_2026-10-25/analysis/equations.csv')}
    benchmark_cost=[]
    for code in ('MC0','MC1','MC2'):
        cfg=profiles[code]['config'];total=0.;work=0
        for pid in meta:
            base=float(audited[pid]['mc_runtime']) if pid in joint else float(x0_costs[pid]['mc_solver_seconds'])*33/4
            n=math.ceil(pre[pid]['T']/cfg['dt'])
            total+=base*(cfg['M']/1024)*(n/pre[pid]['N']);work+=33*cfg['M']*n
        benchmark_cost.append(dict(profile=code,equations=1000,
            one_call_each_scenario_seconds=total,four_calls_each_scenario_seconds=4*total,
            one_call_each_work=work,work_unit='path_updates',
            individual59_all_success_MC_scenario_seconds=59*total,
            source='direct_grid100_solver_wall_else_x0_mean4_times33;operation_scaled',
            measured_forecast=False,includes_preprocessing=False))
    rmean=statistics.mean(float(audited[p]['pair_remizov_runtime']) for p in joint)
    for code,n in [('R128',128),('R256',256)]:
        total=1000*rmean*n/128
        benchmark_cost.append(dict(profile=code,equations=1000,
            one_call_each_scenario_seconds=total,four_calls_each_scenario_seconds=4*total,
            one_call_each_work=1000*n*512,work_unit='S_applications',
            individual59_all_success_MC_scenario_seconds=None,
            source='joint100_mean_solver_wall;linear_n_scenario',
            measured_forecast=False,includes_preprocessing=False))
    for j in jobs:
        if j['profile'].startswith('MC'):
            t=forecast_index[(j['equation_id'],j['profile'])]['scenario_seconds_one_call']
        else:t=12.344684562529437*(int(j['profile'][1:])/128)
        j['scenario_solver_seconds']=t
        j['proposed_watchdog_seconds']=math.ceil(max(120.,6*t+15))
        if j['profile'].startswith('MC'):j['config']['max_runtime_seconds']=j['proposed_watchdog_seconds']
    active_mc_worst=max(sum(j['scenario_solver_seconds'] for j in jobs if j['profile'].startswith('MC') and
        (j['stage']==1 or j['profile']==profile)) for profile in ('MC0','MC1','MC2'))
    stage1=sum(j['scenario_solver_seconds'] for j in jobs if j['stage']==1)
    pilotR=sum(j['scenario_solver_seconds'] for j in jobs if j['stage']==0)
    plan=dict(schema='ml_selector_contract_v1',date='2026-10-25',execution_authorized=False,
        data_dir=str(DATA.relative_to(ROOT)),dataset_sha256=passport['output_sha256'],
        mathematical_object='bounded_whole_line_resolvent',target_x=grid,error_threshold=.01,
        error_metric='max_absolute_grid_error',profiles=profiles,
        recommended_track='A_fixed_profiles_observed_fastest',reliability_requirement_active=False,
        reliability_candidate=dict(success_probability=.95,confidence=.95,
            zero_failure_minimum_trials_individual=math.ceil(math.log(.05)/math.log(.95)),
            zero_failure_minimum_trials_simultaneous1000_one_sided=math.ceil(math.log(.05/1000)/math.log(.95)),
            zero_failure_minimum_trials_simultaneous1000_two_sided=math.ceil(math.log(.025/1000)/math.log(.95))),
        pilot=dict(equations=40,families=40,coverage_cells=20,cohorts=dict(Counter(r['cohort'] for r in manifest)),
            sentinels=SENTINELS,independent_repeats=4,aggregation='none',
            stage1_new_MC_calls=60,stage2_new_MC_calls_max=140,new_MC_calls_max=200,
            stage0_missing_R_calls=sum(j['stage']==0 for j in jobs),
            conditional_job_rows=len(jobs),conditional_rows_are_not_all_to_be_executed=True,
            stage1_scenario_seconds=stage1,missing_R_scenario_seconds=pilotR,
            maximum_branch_MC_scenario_seconds=active_mc_worst,
            proposed_stage1_compute_cap_seconds=math.ceil(2*(stage1+pilotR)+300),
            stage2_compute_cap='freeze from stage1 actual timing before authorizing stage2',
            frozen_selection_rule='SHA256 rank2488;mandatory10+pass5+inaccurate5+cell20;train-only distinct families'),
        gates=dict(stage1_required_MC_calls=60,MC_max_operational_failures_per20=1,
            MC_expensive_cost_ratio_cap=8.,MC_expensive_zero_success_median_error_ratio_cutoff=.8,
            MC_zero_all_profiles_action='stop_before_stage2;negative_control_requires_new_decision',
            MC_selection_order=['max_observed_successes','lower_stable_median_warm_cost_with10pct_margin','MC0_MC1_MC2_priority'],
            R_selection_order=['max_delivered_accurate_answers_of40','smaller_n'],
            stage2_profile_count=1,stage2_profile_switching=False,
            requires_complete_stage1_before_accuracy_gates=True),
        timing=dict(cost='warm_end_to_end_method_wall',clock='perf_counter_ns',repeats_final=4,
            compilation='warm_once_per_process_report_separately',preprocessing='G/T/AST/constructor included per method call',
            exact_scoring='after_timer',concurrency=1,threads=1,
            tie_relative_margin=.10,noisy_MAD_over_median_threshold=.20,
            tie_rule='prefer_remizov_with_tie_flag',minimum_repeat_times_needed=4),
        final_seed_schedule={code:[seed(3,idx,k) for k in range(4)] for idx,code in enumerate(('MC0','MC1','MC2'))},
        final_benchmark=dict(equations=1000,per_method_calls=4000,combined_calls=8000,
            historical_numeric_reuse_allowed=True,historical_final_timing_reuse_allowed=False,
            profile_choice_train_only=True,validation_selects_ML_only=True,test_once_after_freeze=True),
        current_numeric_coverage=dict(R128=100,R256=5,MC0=100,MC1_four_seeds=5,MC2=0),
        notes=['operation-scaled timing scenarios are not bounds or new measurements',
               'selection contains stress enrichment; no population accuracy inference',
               'pilot seeds disjoint from final seeds; no best-seed selection',
               'no coupling, h512 or aggregate algorithm silently introduced'])
    assert set(s for seeds in plan['final_seed_schedule'].values() for s in seeds).isdisjoint(
        {j['config'].get('seed') for j in jobs}|oldseeds)
    assert len({j['key'] for j in jobs})==len(jobs)
    assert all(j['split']=='train' and j['equation_id'] in train for j in jobs)
    assert sum(j['stage']==1 for j in jobs)==60
    assert sum(j['stage']==2 for j in jobs)==420  # alternatives, only140 active
    assert sum(r['cohort']=='coverage' for r in manifest)==20
    csvwrite(out/'pilot_manifest.csv',manifest);csvwrite(out/'benchmark_requirements.csv',requirements)
    csvwrite(out/'cost_forecast.csv',forecasts)
    csvwrite(out/'benchmark_cost_scenarios.csv',benchmark_cost)
    with (out/'pilot_jobs.jsonl').open('x') as stream:
        for j in jobs:stream.write(json.dumps(j,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
    write(out/'plan.json',plan)
    write(out/'planning_checks.json',dict(equation_rows_read=len(audited),pilot_train_only=True,
        pilot_unique_families=40,coverage_cells=20,exact_forced_ids_present=True,
        source_audit_artifact_hashes_valid=True,all_dataset_hashes_valid=True,
        seeds_disjoint_from_historical_and_final=True,conditional_jobs=len(jobs),
        new_solver_calls=0,new_mc_paths=0,new_model_fits=0))
    for n in ['chernov_solver.py','mc_solver.py','mc_optimized.py','mc_ast_kernels.py','all_methods_bounds.py',
              'all_methods_dataset_loader.py','ml/remizov_features.py','ml/quality_prototype.py']:
        digest(ROOT/n)
    digest(Path(__file__))
    write(out/'sources_sha256.json',INPUTS)
    print(json.dumps(dict(pilot=plan['pilot'],reliability=plan['reliability_candidate']),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    build(ap.parse_args().out)
