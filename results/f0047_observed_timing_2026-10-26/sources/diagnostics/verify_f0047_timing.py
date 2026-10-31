"""Independent audit from saved arrays and timer intervals; never solve/fit."""
from __future__ import annotations
import argparse,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnostics.mc_pilot_p2 import read_json,write_json,digest,snapshot


def verify(out):
    import numpy as np
    from all_methods_dataset_loader import load_all_methods_equations
    plan=read_json(out/'plan.json');data=ROOT/plan['data_dir'];ph=digest(out/'plan.json')
    e=next(e for e in load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl') if e.equation_id=='AMV1_F0047_S00')
    exact=np.asarray(e.y_exact(np.asarray(plan['target_x'])))
    times={'R256':[],'MC1':[]};errors={};pids=set();sessions=set();arrays=[];intervals=[]
    for j in plan['jobs']:
        r=read_json(out/'records'/f"{j['key']}.json");z=r['result']
        assert r['status']=='completed' and r['job']==j and r['plan_sha256']==ph
        assert r['equation_id']==z['problem_id']==e.equation_id and r['expression_hash']==e.metadata['expression_hash']
        assert r['config']==read_json(ROOT/j['reference_path'])['job']['config']
        assert np.array_equal(z['x'],plan['target_x']) and np.array_equal(exact,r['y_exact'])
        assert np.array_equal(z['f'],read_json(ROOT/j['reference_path'])['result']['f'])
        err=float(np.max(abs(np.asarray(z['f'])-exact)));assert err==r['E_grid'] and (err<=.01)==r['accuracy_met']
        elapsed=(r['timer_stop_ns']-r['timer_start_ns'])*1e-9
        assert elapsed==r['method_wall_seconds'] and elapsed>0
        times[j['profile']].append(elapsed);errors[j['profile'],j['episode']]=err
        pids.add(r['worker_pid']);sessions.add(r['session']);intervals.append((r['timer_start_ns'],r['timer_stop_ns']))
        if j['profile']=='R256':arrays.append(z['f'])
        else:assert r['seed']==248800010100+j['episode'] and z['compilation_seconds']==0
    assert len(pids)==len(sessions)==1 and all(x==arrays[0] for x in arrays)
    assert all(b<a2 for (a,b),(a2,b2) in zip(intervals,intervals[1:]))
    tr=statistics.median(times['R256']);tm=statistics.median(times['MC1'])
    stable=all(statistics.median(abs(t-statistics.median(ts)) for t in ts)/statistics.median(ts)<=.2 for ts in times.values())
    rp=errors['R256',0]<=.01;mp=errors['MC1',0]<=.01;tie=False
    if rp and mp:
        if not stable:want='unknown'
        elif tm<=.9*tr and max(times['MC1'])<min(times['R256']):want='monte_carlo'
        elif tr<=.9*tm and max(times['R256'])<min(times['MC1']):want='remizov'
        else:want='remizov';tie=True
    else:want='remizov' if rp else ('monte_carlo' if mp else 'none')
    label=read_json(out/'label.json');assert label['label']==want and label['tie']==tie
    before=read_json(out/'protected_before.json');after=snapshot(out)
    changed={k for k,h in before.items() if after.get(k)!=h};assert changed<= {'PROJECT_STATUS.md'},changed
    additions=set(after)-set(before)
    assert additions<= {'diagnostics/report_f0047_timing.py','diagnostics/verify_f0047_timing.py'},additions
    for n,h in plan['protected_sha256'].items():assert digest(ROOT/n)==h,n
    requests=[read_json(p) for p in (out/'events').glob('*.request.json')]
    assert len(requests)==8 and {r['job']['key'] for r in requests}=={j['key'] for j in plan['jobs']}
    assert all(r['plan_sha256']==ph for r in requests)
    old=read_json(ROOT/'results/remizov_r0_2026-10-26/selected_labels_1000.json')['rows']
    new=read_json(out/'selected_labels_1000.json')['rows']
    assert len(old)==len(new)==1000 and [a['equation_id'] for a,b in zip(old,new) if a!=b]==[e.equation_id]
    result=dict(passed=True,raw_results=8,requested_calls=8,equations=[e.equation_id],same_worker=True,same_session=True,
        canonical_errors_recomputed=8,all_reference_arrays_identical=True,R_repeats_identical=True,stable=stable,
        independent_label=want,tie=tie,R_median=tr,MC_median=tm,R_over_MC_ratio=tr/tm,
        disjoint_MC_faster=max(times['MC1'])<min(times['R256']),old_paths_checked=len(before),old_paths_changed=sorted(changed),
        allowed_new_code=sorted(additions),new_MC_accuracy_trials_are_replays=True,no_solver_or_ML_fit=True)
    write_json(out/'independent_verification.json',result,exclusive=True)
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);verify(ap.parse_args().out.resolve())
