"""Independent saved-result audit for the fixed train-only grid100 protocol."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def verify(out):
    import numpy as np
    from diagnostics.run_chernov_batch import read_json,digest,write_json
    from diagnostics.chernov_batch_control import lock_is_held
    from diagnostics.joint_grid100 import controls_pass,primary_rows
    from diagnostics.grid100_statistics import adaptive_ids,aggregate_mc
    plan=read_json(out/'plan.json');ph=digest(out/'plan.json')
    protected=read_json(out/'protected_before.json');changed=[n for n,h in protected.items() if not (ROOT/n).is_file() or digest(ROOT/n)!=h]
    assert not changed,changed
    for n,h in plan['protected_sha256'].items():assert digest(ROOT/n)==h,n
    records={p.stem:read_json(p) for p in (out/'records').glob('*.json')}
    decision=read_json(out/'adaptive_selection.json')
    assert decision['selected']==adaptive_ids(primary_rows(out,plan),12)
    jobs=plan['jobs']+decision['jobs'];by_key={j['key']:j for j in jobs}
    assert len(jobs)==len(by_key)
    assert all(t['split']=='train' for t in plan['selection'])
    assert len({t['problem_id'] for t in plan['selection']})==100
    tasks={t['problem_id']:t for t in plan['selection']};primary=0;work=0
    for key,r in records.items():
        j=r['job'];assert j==by_key[key] and ph==r['plan_sha256']
        assert r['task_wall_seconds']<=j['watchdog_seconds'] and r['peak_rss_mib']<=512
        if r['status']!='completed':continue
        z=r['result'];assert z['success'] and z['x']==plan['target_x'] and len(z['f'])==33
        exact=np.asarray(r['y_exact']);error=float(np.max(abs(np.asarray(z['f'])-exact)))
        assert error==r['E_inf'] and bool(error<=.01)==r['accuracy_met']
        assert r['split']=='train'
        if j['stage']=='control':continue
        t=tasks[j['problem_id']];assert r['expression_hash']==t['expression_hash']
        assert z['T']==t['T']
        if j['method']=='MC':
            assert z['M']==1024 and z['completed_paths']==[1024]*33 and z['backend']=='numba'
            assert z['tail_tolerance_met'] and z['n_path_steps']==z['n_normal_draws']==t['mc_path_steps']
            assert z['forcing_abs_upper_bound']==t['G'] and z['seed_entropy']==plan['seeds'][j['seed_index']]
            work+=z['n_path_steps']
        else:
            assert j['config']['boundary']=='zero' and j['config']['interp']=='cubic'
            assert z['n_operator_applications']==128*512 and z['time_nodes']==513
        primary+=j['stage']=='primary'
    assert controls_pass(out,plan)
    control_bitwise=all(np.asarray(records[j['key']]['result']['f'],dtype=np.float64).tobytes()==
        np.asarray(read_json(ROOT/j['reference'])['result']['f'],dtype=np.float64).tobytes() for j in plan['jobs'][:4])
    assert control_bitwise
    stats=read_json(out/'analysis/simultaneous_statistics.json')
    for pid,s in stats.items():
        ms=[records[k] for i in range(4) if (k:=pid+f'_MC{i}') in records and records[k]['status']=='completed']
        for m in ms:np.testing.assert_array_equal(m['y_exact'],records[pid+'_R']['y_exact'])
        recomputed=aggregate_mc(ms if len(ms)==4 else [records[pid+'_MC0']])
        np.testing.assert_array_equal(recomputed['f'],s['final']['f'])
        np.testing.assert_array_equal(recomputed['normal_simultaneous_half'],s['final']['normal_simultaneous_half'])
        assert recomputed['normal_z']>3.17 and recomputed['total_accuracy_certified'] is False
    guardians=[read_json(p) for p in (out/'control').rglob('*.guardian.json')]
    assert all(g['finished'] and g.get('cleanup_success') for g in guardians)
    locks={n:lock_is_held(ROOT/'results'/n) for n in ('.chernov_batch_controller.lock','.chernov_batch_compute.lock')};assert not any(locks.values())
    assert read_json(out/'active.json')['finished']
    value=dict(protected_files_checked=len(protected),protected_files_changed=changed,terminal=len(records),planned=len(jobs),
        primary_completed=primary,MC_path_steps_completed=work,selection_train_only=True,controls_reproduced=True,controls_bitwise_equal=control_bitwise,
        adaptive_rule_reproduced=True,simultaneous_bands_reproduced=True,all_guardians_finished=True,locks_held=locks,
        primary_profile='cubic/zero common analyticT; historical control profile isolated',
        numeric_schemes_changed=False,ML_retrained=False,additional900_run=False,
        frozen_sources_sha256={n:digest(ROOT/n) for n in plan['protected_sha256'] if not n.startswith('data/')})
    write_json(out/'verification.json',value);print(json.dumps({k:v for k,v in value.items() if k!='frozen_sources_sha256'},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);verify(ap.parse_args().out.resolve())
