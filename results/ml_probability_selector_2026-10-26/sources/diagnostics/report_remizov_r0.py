"""Saved-only verification, profile selection and observed-fastest analysis of R0."""
from __future__ import annotations
import argparse,csv,json,math,statistics,sys
from collections import Counter,defaultdict
from dataclasses import asdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnostics.mc_pilot_p2 import digest,read_json,write_json,snapshot
from diagnostics.chernov_batch_control import lock_is_held
from ml.selector_labels import Evidence,GRID,TIMING_SCOPE,fingerprint,label_pair,support,training_gate

CONTRACT=ROOT/'reports/ml_selector_contract'
V2=ROOT/'results/ml_selector_observed_2026-10-26_v2'
P1=CONTRACT/'sentinel_p1_2026-10-26'
P2=CONTRACT/'pilot_p2_2026-10-26'
JOINT=ROOT/'results/joint_grid100_2026-10-25'
FINAL=ROOT/'results/final_grid5_2026-10-25'


def verified_source(folder):
    for name,h in read_json(folder/'artifact_sha256.json').items():
        assert digest(folder/name)==h,(folder,name)


def read_r(path,pid,profile,e,expected,plan_hash=None):
    r=read_json(path);j=r['job'];z=r['result'];m=e.metadata
    assert r['status']=='completed' and z['success'] and z['failure_reason']==''
    assert r['equation_id']==pid if 'equation_id' in r else j['problem_id']==pid
    assert r['expression_hash']==m['expression_hash'] and r['family_id']==m['family_id'] and r['split']==m['split']=='train'
    assert j['config']==expected and z['problem_id']==pid and z['scheme']=='solve'
    assert z['n']==expected['n'] and z['T']==expected['T'] and z['time_nodes']==513
    assert z['work_grid_points']==2081 and z['actual_dx']==.0125 and z['work_domain']==[-13.,13.]
    assert z['n_operator_applications']==expected['n']*512 and z['n_spline_builds']==expected['n']*512+1
    exact=np.asarray(e.y_exact(np.asarray(GRID)));f=np.asarray(z['f'])
    assert np.array_equal(z['x'],GRID) and np.array_equal(r['y_exact'],exact)
    assert f.shape==(33,) and np.isfinite(f).all()
    error=float(np.max(abs(f-exact)))
    assert error==r.get('E_grid',r.get('E_inf')) and (error<=.01)==r['accuracy_met']
    if plan_hash:
        assert r['plan_sha256']==plan_hash and r['solver_called'] and j['profile']==profile
        assert r['config']==expected and r['seed'] is None and r['forcing_G']==j['forcing_G']
        assert r['effective_T']==j['expected_T']==expected['T']
        assert math.isclose((r['timer_stop_ns']-r['timer_start_ns'])*1e-9,r['method_wall_seconds'],abs_tol=1e-12,rel_tol=0)
        assert r['method_wall_seconds']>z['runtime_seconds']>0
    return r,error


def main(out):
    from all_methods_dataset_loader import load_all_methods_equations
    for folder in (P1,P2,V2):verified_source(folder)
    plan=read_json(out/'plan.json');ph=digest(out/'plan.json');contract=read_json(CONTRACT/'plan.json')
    frozen=[json.loads(s) for s in (CONTRACT/'pilot_jobs.jsonl').read_text().splitlines()]
    jobs={j['key']:j for j in frozen if j['stage']==0}
    assert {j['key']:j for j in plan['jobs']}==jobs and len(jobs)==38
    for name,h in plan['protected_sha256'].items():assert digest(ROOT/name)==h,name
    data=ROOT/contract['data_dir']
    for name,h in contract['dataset_sha256'].items():assert digest(data/name)==h,name
    eqs={e.equation_id:e for e in load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl')}
    manifest=list(csv.DictReader((CONTRACT/'pilot_manifest.csv').open()))
    ids=[r['equation_id'] for r in manifest];assert len(ids)==len(set(ids))==40
    assert all(eqs[r['equation_id']].metadata['expression_hash']==r['expression_hash'] and eqs[r['equation_id']].metadata['family_id']==r['family_id'] for r in manifest)
    paths=list((out/'records').glob('*.json'));assert len(paths)==38 and {p.stem for p in paths}==set(jobs)
    requests=[read_json(p) for p in (out/'events').glob('*.request.json')]
    amendment=read_json(out/'resume_amendment.json')
    assert amendment['original_plan_sha256']==ph
    assert digest(ROOT/'diagnostics/resume_remizov_r0.py')==amendment['resume_driver_sha256']
    wanted=Counter(jobs.keys())+Counter(amendment['permitted_retry_keys'])
    assert len(requests)==39 and Counter(x['job']['key'] for x in requests)==wanted
    assert all(x['plan_sha256']==ph and x['job']==jobs[x['job']['key']] for x in requests)
    interruptions=[read_json(p) for p in (out/'events').glob('*.interrupted.json')]
    assert len(interruptions)==1 and interruptions[0]['reason']=='interrupted'
    guardians=[read_json(p) for p in (out/'control').rglob('*.guardian.json')]
    assert all(x['finished'] and x['cleanup_success'] for x in guardians)
    assert sum(x['reason']=='owner_lease_expired' for x in guardians)==1
    assert not lock_is_held(ROOT/'results/.chernov_batch_controller.lock')
    assert not lock_is_held(ROOT/'results/.chernov_batch_compute.lock')
    assert read_json(out/'active.json')['finished']
    assert not read_json(out/'pending.json')['keys']
    thermals=[json.loads(s) for s in (out/'thermal.jsonl').read_text().splitlines()]
    assert thermals and all(x['thermal_state'] is None or x['thermal_state']<2 for x in thermals)
    records={};observations=[]
    for row in manifest:
        pid=row['equation_id'];e=eqs[pid];records[pid]={}
        for profile in ('R128','R256'):
            newkey=f'{pid}_{profile}_stage0_episode0'
            if newkey in jobs:
                path=out/'records'/f'{newkey}.json';new=True
            elif profile=='R128':
                path=JOINT/'records'/f'{pid}_R.json';new=False
            else:
                path=FINAL/'records'/f'{pid}_R256.json';new=False
            assert path.exists(),(pid,profile)
            cfg=dict(contract['profiles'][profile]['config'],T=float(row['T']))
            r,error=read_r(path,pid,profile,e,cfg,ph if new else None)
            records[pid][profile]=(r,error,path,new)
            observations.append(dict(equation_id=pid,family_id=row['family_id'],profile=profile,source=str(path.relative_to(ROOT)),source_sha256=digest(path),new_R0=new,E_grid=error,accurate=error<=.01,method_wall_seconds=r.get('method_wall_seconds') if new else None,old_solver_runtime_seconds=None if new else r['result']['runtime_seconds']))
    assert len(observations)==80 and sum(r['new_R0'] for r in observations)==38
    counts={p:sum(records[pid][p][1]<=.01 for pid in ids) for p in ('R128','R256')}
    selected='R256' if counts['R256']>counts['R128'] else 'R128'
    olde=read_json(V2/'evidence.json');mce={pid:Evidence(**olde[pid+':monte_carlo']) for pid in ids}
    assert all(e.status=='completed' and len(e.records)==4 and e.T==float(next(r['T'] for r in manifest if r['equation_id']==pid)) for pid,e in mce.items())
    pair_rows={};details=[]
    for profile in ('R128','R256'):
        pair=dict(label_type='observed_fastest_v1',R=profile,MC='MC1',grid=list(GRID),eps=.01,episode=0,selected=(profile==selected),timer=TIMING_SCOPE,rule='accuracy first; both-pass median4 same-cohort 10% and disjoint range; tie R')
        pair.update(remizov=contract['profiles'][profile],monte_carlo=contract['profiles']['MC1'],
            dataset_sha256=contract['dataset_sha256'],
            implementation_sha256={k:v for k,v in plan['protected_sha256'].items() if k in ('chernov_solver.py','mc_solver.py','mc_optimized.py','mc_ast_kernels.py','all_methods_bounds.py')},
            common_horizon='analytic G + auto_T tail_eps1e-4; exact per-job T retained in source records')
        pair_id=fingerprint(pair);rows=[]
        for pid in ids:
            r,err,path,new=records[pid][profile];m=eqs[pid].metadata
            er=Evidence('remizov',pid,m['expression_hash'],m['family_id'],'train',profile,
                status='completed',error=err,T=r['result']['T'],
                times=[r['method_wall_seconds']] if new else [r['result']['runtime_seconds']],
                timing_scope=TIMING_SCOPE if new else 'historical_solver_runtime_excludes_external_factory',
                timing_cohort='R0_session' if new else 'old_R_session',
                source=[str(path.relative_to(ROOT))],source_sha256=[digest(path)])
            label=label_pair(er,mce[pid],pair_id);rows.append(label)
            details.append(dict(equation_id=pid,profile=profile,R_E_grid=err,MC1_episode0_E_grid=mce[pid].error,label=label['label'],reason=label['reason'],R_source=er.source[0],MC_sources=mce[pid].source))
        pair_rows[profile]=rows
        write_json(out/f'labels_{profile.lower()}_pilot40.json',dict(pair=pair,profile_pair_id=pair_id,rows=rows),exclusive=True)
    selected_rows=pair_rows[selected]
    allrows=[]
    chosen={r['equation_id']:r for r in selected_rows}
    for pid in sorted(k[:-12] for k in olde if k.endswith(':monte_carlo')):
        if pid in chosen:allrows.append(chosen[pid]);continue
        m=eqs[pid].metadata
        allrows.append(dict(equation_id=pid,family_id=m['family_id'],split=m['split'],expression_hash=m['expression_hash'],profile_pair_id=selected_rows[0]['profile_pair_id'],label_type='observed_fastest_v1',episode=0,label='unknown',reason='missing_selected_R_and_MC1',r_profile=selected,mc_profile='MC1'))
    assert len(allrows)==1000
    write_json(out/'R_observations.json',observations,exclusive=True)
    write_json(out/'pair_comparisons.json',details,exclusive=True)
    write_json(out/'selected_labels_1000.json',dict(profile=selected,rows=allrows,support=support(allrows),gate=training_gate(allrows)),exclusive=True)
    # Saved experiments are separate budgets and cohorts. These are accuracy candidates,
    # never extra training rows or cross-cohort timing comparisons.
    candidates=[];archive_counts=Counter();accurate_ids=defaultdict(set)
    for label,folder,glob in [('joint_MC0',JOINT,'*_MC*.json'),('final_MC4096',FINAL,'*_MC*.json'),('P1_MC0_MC1_MC2',P1,'*MC*.json'),('P2_MC1',P2,'*MC*.json')]:
        for path in (folder/'records').glob(glob):
            r=read_json(path);j=r['job'];pid=j.get('equation_id',j.get('problem_id'))
            if pid not in eqs or r.get('status')!='completed' or not r.get('result',{}).get('success'):continue
            if label=='joint_MC0' and j.get('method')!='MC':continue
            assert r['expression_hash']==eqs[pid].metadata['expression_hash']
            assert r['family_id']==eqs[pid].metadata['family_id']
            z=r['result'];grid=np.asarray(z.get('x'));f=np.asarray(z.get('f'))
            if grid.shape!=(33,) or not np.array_equal(grid,GRID) or f.shape!=(33,) or not np.isfinite(f).all():continue
            exact=np.asarray(eqs[pid].y_exact(np.asarray(GRID)));err=float(np.max(abs(f-exact)))
            assert err==r.get('E_grid',r.get('E_inf'))
            archive_counts[label]+=1
            if err>.01:continue
            accurate_ids[label].add(pid)
            same=pid in records and z['T']==records[pid]['R128'][0]['result']['T']
            relation=None
            if same:
                relation='R128_fail' if records[pid]['R128'][1]>.01 else 'R128_pass'
                relation+=',R256_fail' if records[pid]['R256'][1]>.01 else ',R256_pass'
            candidates.append(dict(archive=label,equation_id=pid,profile=j.get('profile',j['config'].get('M')),episode=j.get('episode',j.get('seed_index')),E_grid=err,source=str(path.relative_to(ROOT)),common_T_grid_identity_with_pilot_R=same,R_relation=relation,timing_comparable=False))
    write_json(out/'saved_MC_accuracy_candidates.json',dict(archive_grid33_completed=dict(archive_counts),archive_accurate_distinct_ids={k:len(v) for k,v in accurate_ids.items()},accurate_calls=candidates),exclusive=True)
    sensitivity=[];all_mc_records=[r for e in mce.values() for r in e.records]
    for pid,e in mce.items():
        own=e.records;passes=sum(r['error']<=.01 for r in own)
        exact=np.asarray(own[0]['y_exact']);mean=np.mean([r['result']['f'] for r in own],axis=0)
        rs,re,path,new=records[pid][selected]
        sensitivity.append(dict(equation_id=pid,primary_error=e.error,passes4=passes,
            mean4_error=float(np.max(abs(mean-exact))),mean4_cost_seconds=sum(e.times),
            MC1_wall_median=statistics.median(e.times),MC1_wall_min=min(e.times),MC1_wall_max=max(e.times),
            R_selected_error=re,R_saved_solver_seconds=rs['result']['runtime_seconds'],
            MC_to_R_saved_native_ratio_DIAGNOSTIC_ONLY=statistics.median(e.times)/rs['result']['runtime_seconds'],
            valid_cost_winner=False,reason='R has one time; historical scope/cohort incompatible; median4 matched R required'))
    primary_candidates=[x['equation_id'] for x in sensitivity if x['primary_error']<=.01]
    absence=dict(MC1_primary_accurate=sum(x['primary_error']<=.01 for x in sensitivity),MC1_primary_total=40,
        MC1_accurate_calls=sum(r['error']<=.01 for r in all_mc_records),MC1_calls=160,
        any_seed_accurate_ids=[x['equation_id'] for x in sensitivity if x['passes4']],
        all_four_accurate_ids=[x['equation_id'] for x in sensitivity if x['passes4']==4],
        mean4_accurate=sum(x['mean4_error']<=.01 for x in sensitivity),mean4_is_new_estimator_not_labels=True,
        primary_cost_only_candidates=primary_candidates,
        primary_MC_sole_accurate=sum(x['primary_error']<=.01 and x['R_selected_error']>.01 for x in sensitivity),
        valid_MC_winners=sum(x['label']=='monte_carlo' for x in selected_rows),
        exploratory_candidates=[x for x in sensitivity if x['passes4']],
        gate=dict(required_train=20,required_validation=5,available_train=0,available_validation=0,
            maximum_primary_candidate_winners_in_measured40=len(primary_candidates),
            remaining_unmeasured_train_families=100,unmeasured_validation_families=30,
            rate_forecast_valid=False,reason='train-selected biased pilot; P1 also selected MC profile; no representative prevalence estimate'),
        recommendation='One preregistered8-call matched-timing block on AMV1_F0047_S00:4 selected-R +4 MC1; no class-hunting benchmark expansion',
        minimum_next_step_new_calls=8,full_benchmark_not_authorized=True)
    write_json(out/'MC_absence_diagnostics.json',absence,exclusive=True)
    write_json(out/'MC1_seed_sensitivity.json',sensitivity,exclusive=True)
    historical_cost=read_json(P2/'analysis/summary.json')['full1000_cost']
    write_json(out/'full1000_cost_reference.json',dict(source=str((P2/'analysis/summary.json').relative_to(ROOT)),
        scenarios=[x for x in historical_cost if x['R_profile']==selected],
        limitations='Saved scenarios, not a measured future benchmark or CI; native R scope excludes preparation; no guarantee of MC-winner support',
        recommended=False,reason='Current MC1 accuracy/candidate support does not justify mass computation solely to manufacture20+5 class support'),exclusive=True)
    new=[read_json(p) for p in paths];wall=sum(r['method_wall_seconds'] for r in new)
    peak=max(r['peak_rss_mib'] for r in new)
    summary=dict(new_R0_completed_cells=38,physical_attempts=39,interrupted_attempts=1,
        interrupted_external_seconds=interruptions[0]['wall_seconds'],previous_R_cells=42,total_R_cells=80,R_accurate=counts,selected_R_profile=selected,
        profile_rule='maximum finite E_grid<=0.01 among40; tie R128',MC1_calls=160,
        pilot_label_counts={p:dict(Counter(x['label'] for x in pair_rows[p])) for p in pair_rows},
        selected_support=support(allrows),gate=training_gate(allrows),
        new_R0_method_wall_seconds=wall,new_R0_max_peak_rss_mib=peak,thermal_samples=len(thermals),
        MC_candidates=len(candidates),MC_accurate_by_archive={k:len(v) for k,v in accurate_ids.items()},
        no_new_MC=True,no_new_ML_fit=True,no_benchmark1000=True)
    write_json(out/'analysis.json',summary,exclusive=True)
    print(json.dumps(summary,sort_keys=True),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);main(ap.parse_args().out.resolve())
