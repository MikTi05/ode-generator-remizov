"""Saved-only accuracy, matched-time and resource audit of approved R-only40."""
from __future__ import annotations
import argparse,csv,datetime,hashlib,json,math,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnostics.run_switch5_r_timing import read_json,write_json,digest,SOURCE,validate_scope,lock_is_held


def times(values):
    import numpy as np
    median=statistics.median(values);mad=statistics.median(abs(v-median) for v in values)
    return dict(raw=values,median=median,min=min(values),max=max(values),IQR=float(np.percentile(values,75)-np.percentile(values,25)),MAD=mad,MAD_fraction=mad/median)


def aggregate(cases):
    result={}
    for name,own in [('all5',cases),('both_accurate',[c for c in cases if c['R128']['accurate'] and c['R256']['accurate']])]:
        r=sum(c['R128']['timing']['median'] for c in own);b=sum(c['R256']['timing']['median'] for c in own)
        accurate=sum(c['R128']['accurate'] for c in own);baseline=sum(c['R256']['accurate'] for c in own)
        result[name]=dict(cases=len(own),R128_median_cost=r,R256_median_cost=b,saved_seconds=b-r,
            relative_saved=1-r/b if b else None,speedup_factor=b/r if r else None,
            accurate_deliveries=accurate,baseline_accurate_deliveries=baseline,
            lost_baseline_deliveries=sum(c['R256']['accurate'] and not c['R128']['accurate'] for c in own),
            method_seconds_per_accurate=r/accurate if accurate else None,
            baseline_method_seconds_per_accurate=b/baseline if baseline else None,
            scope='one delivery per case via median4 method wall; excludes ML inference; inaccurate calls included in all5 cost')
    result['global40']=dict(measured_total_cost=None,measured_percent_saving=None,
        attributable_switch_cost_delta_seconds=result['all5']['saved_seconds'],
        reason='four other R256 costs missing; unchanged35 actions cancel in local delta but no complete comparable global denominator')
    return result


def report(out):
    import numpy as np
    from all_methods_dataset_loader import load_all_methods_equations
    plan=read_json(out/'plan.json');validate_scope(plan);ph=digest(out/'plan.json')
    assert plan['source_plan_sha256']==digest(SOURCE/'plan.json')
    for name,h in plan['protected_sha256'].items():assert digest(ROOT/name)==h,name
    data=out/'training_inputs';eqs={e.equation_id:e for e in load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl')}
    requests=[read_json(p) for p in (out/'events').glob('*.request.json')]
    assert len(requests)==len({r['job']['key'] for r in requests})==40
    assert all(r['job'] in plan['jobs'] and r['plan_sha256']==ph for r in requests)
    records={p.stem:read_json(p) for p in (out/'records').glob('*.json')}
    assert set(records)=={j['key'] for j in plan['jobs']} and len({r['session'] for r in records.values()})==len({r['worker_pid'] for r in records.values()})==1
    cases=[];attempts=[];operators=0;full=0.
    for c in plan['cases']:
        pid=c['equation_id'];exact=np.asarray(eqs[pid].y_exact(np.asarray(plan['target_x'])));case=dict(equation_id=pid,family_id=c['family_id'])
        own={}
        for p in ('R128','R256'):
            rs=sorted((r for r in records.values() if r['equation_id']==pid and r['profile']==p),key=lambda r:r['job']['episode'])
            assert len(rs)==4 and [r['job']['episode'] for r in rs]==list(range(4))
            for r in rs:
                j=r['job'];z=r['result'];assert j in plan['jobs'] and r['plan_sha256']==ph and r['config']==j['config']
                assert r['status']=='completed' and r['solver_called'] and z['success'] and z['failure_reason']==''
                assert r['split']==eqs[pid].metadata['split']=='train' and r['expression_hash']==eqs[pid].metadata['expression_hash']
                np.testing.assert_array_equal(z['x'],plan['target_x']);np.testing.assert_array_equal(r['y_exact'],exact)
                error=float(np.max(abs(np.asarray(z['f'])-exact)))
                assert error==r['E_grid']==j['reference_E_grid'] and (error<=.01)==r['accuracy_met']==j['expected_accurate']
                assert hashlib.sha256(np.asarray(z['f'],dtype='<f8').tobytes()).hexdigest()==j['expected_f_float64_sha256']
                assert z['n']==j['config']['n'] and z['scheme']=='solve' and z['T']==j['expected_T']==r['effective_T']
                assert r['forcing_G']==j['forcing_G'] and z['time_nodes']==513 and z['work_grid_points']==2081 and z['actual_dx']==.0125 and z['work_domain']==[-13.,13.]
                assert z['n_operator_applications']==j['numerical_work'] and z['n_spline_builds']==j['numerical_work']+1
                assert math.isclose((r['timer_stop_ns']-r['timer_start_ns'])*1e-9,r['method_wall_seconds'],abs_tol=1e-12,rel_tol=0)
                assert r['reference_array_identical'] and r['seed'] is None
                operators+=z['n_operator_applications'];full+=r['method_wall_seconds']
                attempts.append(dict(key=j['key'],equation_id=pid,profile=p,episode=j['episode'],status=r['status'],E_grid=error,accurate=r['accuracy_met'],method_wall_seconds=r['method_wall_seconds'],source_sha256=digest(out/'records'/f"{j['key']}.json")))
            assert all(r['result']['f']==rs[0]['result']['f'] for r in rs)
            case[p]=dict(E_grid=rs[0]['E_grid'],errors=[r['E_grid'] for r in rs],accurate=rs[0]['accuracy_met'],timing=times([r['method_wall_seconds'] for r in rs]))
            own[p]=rs
        r=case['R128']['timing'];b=case['R256']['timing']
        case['median_saved_seconds']=b['median']-r['median'];case['relative_saved']=1-r['median']/b['median']
        case['disjoint_time_ranges']=r['max']<b['min'];case['MAD_within20pct']=max(r['MAD_fraction'],b['MAD_fraction'])<=.2
        case['paired_episodes']=[dict(episode=i,saved_seconds=own['R256'][i]['method_wall_seconds']-own['R128'][i]['method_wall_seconds'],
            relative_saved=1-own['R128'][i]['method_wall_seconds']/own['R256'][i]['method_wall_seconds']) for i in range(4)]
        both=case['R128']['accurate'] and case['R256']['accurate']
        case['supported_accurate_speedup']=bool(both and r['median']<=.9*b['median'] and case['disjoint_time_ranges'] and case['MAD_within20pct'])
        case['classification']='R128_accurate_speedup' if case['supported_accurate_speedup'] else 'neither_profile_accurate' if not case['R128']['accurate'] and not case['R256']['accurate'] else 'tie_or_uncertain'
        cases.append(case)
    assert operators==3932160
    sessions=[read_json(p) for p in (out/'sessions').glob('*.json')];assert len(sessions)==1 and sessions[0]['reason'] is None
    active=read_json(out/'active.json');budget=read_json(out/'budget.json')
    assert active['finished'] and active['finished_monotonic']<=budget['deadline_monotonic']
    assert datetime.datetime.fromisoformat(sessions[0]['finished_utc']).timestamp()<=budget['deadline_epoch']
    assert sessions[0]['wall_seconds']<=1766 and not read_json(out/'pending.json')['keys']
    samples=[json.loads(s) for p in (out/'workers').glob('*.rss.jsonl') for s in p.read_text().splitlines()]
    ts=[json.loads(s) for s in (out/'thermal.jsonl').read_text().splitlines()]
    guardians=[read_json(p) for p in (out/'control').rglob('*.guardian.json')]
    assert all(g['finished'] and g['cleanup_success'] for g in guardians)
    assert max(s['peak_rss_mib'] for s in samples)<=512 and all(t['thermal_state'] is not None and t['thermal_state']<2 for t in ts)
    assert all(not lock_is_held(ROOT/'results'/n) for n in ('.chernov_batch_controller.lock','.chernov_batch_compute.lock'))
    a=aggregate(cases)
    result=dict(passed=True,complete=True,records=40,requests=40,cases=cases,aggregate=a,
        accuracy=dict(R128_accurate_calls=sum(r['accuracy_met'] for r in records.values() if r['profile']=='R128'),
            R256_accurate_calls=sum(r['accuracy_met'] for r in records.values() if r['profile']=='R256'),
            expected_shared_failed_case='AMV1_F0172_S03',accuracy_lost_relative_baseline=0),
        resources=dict(controller_wall_seconds=sessions[0]['wall_seconds'],method_wall_seconds=full,peak_RSS_MiB=max(s['peak_rss_mib'] for s in samples),
            RSS_samples=len(samples),thermal_samples=len(ts),maximum_thermal_state=max(t['thermal_state'] for t in ts),
            guardian_cleanup=True,workers=1,threads=1,operator_applications=operators,setup=[read_json(p) for p in (out/'workers').glob('*.setup.json')]),
        MC_calls=0,ML_fits=0,validation_test_used=False,policy_changed=False,
        inference_overhead_included=False,production_unblocked=False)
    write_json(out/'analysis.json',result,exclusive=True)
    with (out/'attempts.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(attempts[0]));w.writeheader();w.writerows(attempts)
    print(json.dumps(dict(passed=True,aggregate=a,resources=result['resources'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);report(p.parse_args().out.resolve())
