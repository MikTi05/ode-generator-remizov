"""Saved-only P2/combined MC1 analysis; no solver invocations or fitting."""
from __future__ import annotations
import argparse,csv,json,math,statistics,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnostics.report_mc_sentinel_p1 import read,sha,csvread,csvwrite,write,summary
CONTRACT=ROOT/'reports/ml_selector_contract';P1=CONTRACT/'sentinel_p1_2026-10-26'

def report(out):
    import numpy as np
    from scipy.stats import beta
    from all_methods_dataset_loader import load_all_methods_equations
    from diagnostics.chernov_batch_control import lock_is_held
    plan=read(out/'plan.json');ph=sha(out/'plan.json')
    frozen=[json.loads(s) for s in (CONTRACT/'pilot_jobs.jsonl').read_text().splitlines()]
    expected={j['key']:j for j in frozen if j['stage']==2 and j['profile']=='MC1'}
    assert {j['key']:j for j in plan['jobs']}==expected and len(expected)==140
    for name,h in plan['protected_sha256'].items():assert sha(ROOT/name)==h,name
    for name,h in read(P1/'artifact_sha256.json').items():assert sha(P1/name)==h,name
    manifest={r['equation_id']:r for r in csvread(CONTRACT/'pilot_manifest.csv')}
    assert len(manifest)==40 and len({r['family_id'] for r in manifest.values()})==40
    data=ROOT/plan['data_dir'];eqs={e.equation_id:e for e in load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl')}
    records=[];attempts=[];byid={};raw_checks=0
    requests=[read(p) for p in (out/'events').glob('*.request.json')]
    assert len(requests)==140 and len({r['job']['key'] for r in requests})==140
    p2files=list((out/'records').glob('*.json'));assert len(p2files)==140
    for stage,folder in [(1,P1),(2,out)]:
        pl=read(folder/'plan.json');pinned=sha(folder/'plan.json')
        jobs={j['key']:j for j in pl['jobs'] if j['profile']=='MC1'}
        own=[]
        for path in (folder/'records').glob('*.json'):
            r=read(path)
            if r['profile']!='MC1':continue
            j=jobs[path.stem];pid=j['equation_id'];e=eqs[pid];m=e.metadata;z=r.get('result') or {}
            assert r['job']==j and r['plan_sha256']==pinned
            assert j['stage']==stage and j['split']==r['split']==m['split']=='train'
            assert r['expression_hash']==j['expression_hash']==m['expression_hash']==manifest[pid]['expression_hash']
            assert r['family_id']==j['family_id']==m['family_id']
            assert r['config']==j['config'] and r['seed']==j['config']['seed']==248800000000+stage*10000+100+j['episode']
            assert j['config']['M']==4096 and j['config']['dt']==1/128
            assert r['status']=='completed' and r['solver_called'] and z['success']
            grid=np.asarray(plan['target_x']);exact=np.asarray(e.y_exact(grid))
            np.testing.assert_array_equal(z['x'],grid);np.testing.assert_array_equal(r['y_exact'],exact)
            f=np.asarray(z['f']);assert f.shape==(33,) and np.isfinite(f).all()
            assert np.isfinite(z['stderr']).all() and (np.asarray(z['stderr'])>=0).all()
            err=float(np.max(abs(f-exact)));assert err==r['E_grid'] and bool(err<=.01)==r['accuracy_met']
            assert z['completed_paths']==[4096]*33 and z['M']==4096
            assert z['n_path_steps']==z['n_normal_draws']==j['numerical_work']==33*4096*j['expected_steps']
            assert z['T']==j['expected_T'] and z['n_steps']==j['expected_steps']
            assert z['forcing_abs_upper_bound']==j['forcing_G'] and z['tail_tolerance_met']
            assert z['dt_requested']==1/128 and z['dt_actual']<=1/128
            assert z['seed_entropy']==j['config']['seed'] and z['rng_scheme']=='PCG64/SeedSequence/point_address_v1'
            assert z['backend']=='numba' and z['compilation_seconds']==z['numba_import_seconds']==0.
            assert math.isclose((r['timer_stop_ns']-r['timer_start_ns'])*1e-9,r['method_wall_seconds'],abs_tol=1e-12,rel_tol=0)
            assert r['method_wall_seconds']>z['runtime_seconds']>0
            r['_stage']=stage;r['_path']=str(path.relative_to(ROOT));records.append(r);own.append(r);byid.setdefault(pid,[]).append(r);raw_checks+=1
            attempts.append(dict(stage=stage,key=j['key'],equation_id=pid,expression_hash=m['expression_hash'],family_id=m['family_id'],cohort=manifest[pid]['cohort'],episode=j['episode'],seed=r['seed'],E_grid=err,accuracy_met=r['accuracy_met'],method_wall_seconds=r['method_wall_seconds'],solver_seconds=z['runtime_seconds'],path_updates=z['n_path_steps'],peak_rss_mib=r['peak_rss_mib'],source=r['_path']))
        assert len(own)==(20 if stage==1 else 140)
    assert len(records)==160 and len(byid)==40
    cells=[];aggregates={}
    for pid in manifest:
        rs=sorted(byid[pid],key=lambda r:r['job']['episode']);assert len(rs)==4
        assert [r['job']['episode'] for r in rs]==list(range(4))
        assert len({r['seed'] for r in rs})==4
        n=4;k=sum(r['accuracy_met'] for r in rs);es=summary([r['E_grid'] for r in rs]);ts=summary([r['method_wall_seconds'] for r in rs])
        lo=float(beta.ppf(.025,k,n-k+1)) if k else 0.;hi=float(beta.ppf(.975,k+1,n-k)) if k<n else 1.
        row=dict(equation_id=pid,family_id=manifest[pid]['family_id'],stage=rs[0]['_stage'],cohort=manifest[pid]['cohort'],complexity=manifest[pid]['complexity'],regime=manifest[pid]['regime'],trials=n,accuracy_pass=k,episode0_pass=rs[0]['accuracy_met'],error_min=es['min'],error_median=es['median'],error_max=es['max'],wall_median=ts['median'],wall_MAD=ts['mad'],wall_IQR=ts['iqr'],wall_min=ts['min'],wall_max=ts['max'],probability_lower95=lo,probability_upper95=hi,reliability95_confirmed=False)
        cells.append(row)
        fs=np.asarray([r['result']['f'] for r in rs]);mean=fs.mean(axis=0);exact=np.asarray(rs[0]['y_exact']);err=float(np.max(abs(mean-exact)))
        aggregates[pid]=dict(kind='diagnostic_mean4_NOT_single_call',f=mean.tolist(),y_exact=exact.tolist(),E_grid=err,accuracy_met=err<=.01,cost_seconds_sum_calls=sum(r['method_wall_seconds'] for r in rs),total_paths_per_point=16384,independent_aggregate_replicates=1,reliability95_confirmed=False,seeds=[r['seed'] for r in rs])
    comparisons=[]
    for pid in manifest:
        mc=next(c for c in cells if c['equation_id']==pid)
        for code,folder,suffix,n in [('R128','joint_grid100_2026-10-25','R',128),('R256','final_grid5_2026-10-25','R256',256)]:
            path=ROOT/'results'/folder/'records'/f'{pid}_{suffix}.json'
            row=dict(equation_id=pid,profile=code,available=path.exists(),reason='saved_compatible' if path.exists() else 'missing_not_run',R_E_grid=None,R_accurate=None,MC1_episode0_pass=mc['episode0_pass'],MC1_passes4=mc['accuracy_pass'],MC1_median_E=mc['error_median'],source=str(path.relative_to(ROOT)) if path.exists() else '')
            if path.exists():
                r=read(path);z=r['result'];cfg=r['job']['config'];exact=np.asarray(eqs[pid].y_exact(np.asarray(plan['target_x'])))
                assert r['expression_hash']==manifest[pid]['expression_hash'] and r['split']=='train'
                assert r['job']['problem_id']==z['problem_id']==pid and r['family_id']==manifest[pid]['family_id']
                assert np.asarray(z['f']).shape==exact.shape==(33,)
                assert cfg['n']==n and cfg['boundary']=='zero' and cfg['interp']=='cubic'
                assert cfg['num_t']==513 and cfg['dx']==.0125 and cfg['padding']==12 and cfg['tail_eps']==1e-4
                assert cfg['t_grading']=='quadratic' and z['T']==byid[pid][0]['job']['expected_T']
                np.testing.assert_array_equal(z['x'],plan['target_x']);np.testing.assert_array_equal(r['y_exact'],exact)
                assert r['status']=='completed' and z['success'] and np.isfinite(z['f']).all()
                err=float(np.max(abs(np.asarray(z['f'])-exact)));assert err==r['E_inf'] and bool(err<=.01)==r['accuracy_met']
                row.update(R_E_grid=err,R_accurate=err<=.01)
            comparisons.append(row)
    rsummary={}
    for code in ['R128','R256']:
        rows=[r for r in comparisons if r['profile']==code and r['available']]
        rsummary[code]=dict(available=len(rows),missing=40-len(rows),R_accurate=sum(r['R_accurate'] for r in rows),MC_episode0_accurate_same_tasks=sum(r['MC1_episode0_pass'] for r in rows),both=sum(r['R_accurate'] and r['MC1_episode0_pass'] for r in rows),R_only=sum(r['R_accurate'] and not r['MC1_episode0_pass'] for r in rows),MC_only=sum(not r['R_accurate'] and r['MC1_episode0_pass'] for r in rows),neither=sum(not r['R_accurate'] and not r['MC1_episode0_pass'] for r in rows),median_R_error=summary([r['R_E_grid'] for r in rows])['median'],median_MC_error_same_tasks=summary([r['MC1_median_E'] for r in rows])['median'])
    baseline={r['equation_id']:float(r['scenario_seconds_one_call']) for r in csvread(CONTRACT/'cost_forecast.csv') if r['profile']=='MC1'}
    factors=[dict(equation_id=c['equation_id'],stage=c['stage'],cohort=c['cohort'],measured_median_seconds=c['wall_median'],original_scenario_seconds=baseline[c['equation_id']],factor=c['wall_median']/baseline[c['equation_id']]) for c in cells]
    fs={}
    for label,rows in [('all40',factors),('P1',[r for r in factors if r['stage']==1]),('P2',[r for r in factors if r['stage']==2]),('coverage20',[r for r in factors if r['cohort']=='coverage']),('stress20',[r for r in factors if r['cohort']=='stress'])]:fs[label]=summary([r['factor'] for r in rows])
    old={r['profile']:r for r in csvread(CONTRACT/'benchmark_cost_scenarios.csv')}
    # Coverage has one train representative per frozen complexity/regime cell.
    # This is a descriptive calibration, not a random population sample or CI.
    central=fs['coverage20']['median'];low=min(r['factor'] for r in factors);high=max(r['factor'] for r in factors)
    forecasts=[]
    for label,factor in [('central_coverage_median',central),('sensitivity_min40',low),('sensitivity_max40',high)]:
        for repeats in (1,4):
            mc=float(old['MC1']['one_call_each_scenario_seconds'])*repeats*factor
            for rcode in ['R128','R256']:
                rv=float(old[rcode]['one_call_each_scenario_seconds'])*repeats
                forecasts.append(dict(scenario=label,repeats_per_equation=repeats,MC_profile='MC1',R_profile=rcode,calibration_factor=factor,MC_seconds=mc,R_seconds_legacy_scenario=rv,total_seconds=mc+rv,total_hours=(mc+rv)/3600,confidence_interval=False,includes_future_overhead=False))
    groups={}
    for label,rs in [('P1', [r for r in records if r['_stage']==1]),('P2',[r for r in records if r['_stage']==2]),('combined',records),('coverage',[r for r in records if manifest[r['equation_id']]['cohort']=='coverage']),('stress',[r for r in records if manifest[r['equation_id']]['cohort']=='stress'])]:
        ids={r['equation_id'] for r in rs};cs=[c for c in cells if c['equation_id'] in ids]
        groups[label]=dict(equations=len(ids),calls=len(rs),accurate_calls=sum(r['accuracy_met'] for r in rs),episode0_pass=sum(c['episode0_pass'] for c in cs),any_pass=sum(c['accuracy_pass']>0 for c in cs),all4_pass=sum(c['accuracy_pass']==4 for c in cs),error=summary([r['E_grid'] for r in rs]),wall=summary([r['method_wall_seconds'] for r in rs]),method_seconds_sum=sum(r['method_wall_seconds'] for r in rs))
    guardians=[read(p) for p in (out/'control').rglob('*.guardian.json')];assert guardians and all(g['finished'] and g['cleanup_success'] for g in guardians)
    assert read(out/'active.json')['finished']
    locks={n:lock_is_held(ROOT/'results'/n) for n in ('.chernov_batch_controller.lock','.chernov_batch_compute.lock')};assert not any(locks.values())
    thermal=[json.loads(s) for s in (out/'thermal.jsonl').read_text().splitlines()]
    sessions=[read(p) for p in (out/'sessions').glob('*.json')]
    result=dict(schema='MC1_P1_P2_saved_analysis_v1',groups=groups,P2_confirmed_calls=140,combined_confirmed_calls=160,raw_error_checks=raw_checks,R_comparison=rsummary,mean4_diagnostic_accurate=sum(r['accuracy_met'] for r in aggregates.values()),calibration=fs,full1000_cost=forecasts,peak_rss_mib=max(r['peak_rss_mib'] for r in records if r['_stage']==2),controller_wall_seconds=sum(r['wall_seconds'] for r in sessions),thermal_samples=len(thermal),thermal_states=dict(Counter(str(r['thermal_state']) for r in thermal)),max_within_equation_wall_MAD_fraction=max(c['wall_MAD']/c['wall_median'] for c in cells),P2_path_updates=sum(r['result']['n_path_steps'] for r in records if r['_stage']==2),warmups=[read(p) for p in (out/'workers').glob('*.warmup.json')],guardians_finished=True,locks_free=True,new_R_calls=0,new_benchmark1000_calls=0,new_model_fits=0,selection_bias='P1 selects MC1; all40 biased train pilot; neither accuracy nor reliability extrapolates to1000; no fastest label certified')
    dest=out/'analysis';dest.mkdir(exist_ok=False)
    csvwrite(dest/'attempts.csv',attempts);csvwrite(dest/'equations.csv',cells);csvwrite(dest/'remizov_comparison.csv',comparisons);csvwrite(dest/'cost_calibration.csv',factors);csvwrite(dest/'benchmark1000_cost_scenarios.csv',forecasts)
    write(dest/'mean4_diagnostic.json',aggregates);write(dest/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('full1000_cost','warmups','calibration')},ensure_ascii=False,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);report(ap.parse_args().out.resolve())
