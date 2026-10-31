"""Independent saved-array, clock and candidate audit; no solves or ML fits."""
from __future__ import annotations
import argparse,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnostics.mc_pilot_p2 import read_json,write_json,digest,snapshot


def verify(out):
    import numpy as np
    from all_methods_dataset_loader import load_all_methods_equations
    plan=read_json(out/'plan.json');data=ROOT/plan['data_dir'];ph=digest(out/'plan.json')
    eqs={e.equation_id:e for e in load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl')};e=eqs[plan['equation_id']]
    exact=np.asarray(e.y_exact(np.asarray(plan['target_x'])));groups={p:[] for p in ('R128','R256','MC1')};raw=[]
    oldseeds={json.loads(s)['config'].get('seed') for s in (ROOT/'reports/ml_selector_contract/pilot_jobs.jsonl').read_text().splitlines()}
    for j in plan['jobs']:
        r=read_json(out/'records'/f"{j['key']}.json");z=r['result'];assert r['status']=='completed' and r['job']==j and r['plan_sha256']==ph and z['problem_id']==e.equation_id
        assert r['expression_hash']==e.metadata['expression_hash'] and r['split']=='train'
        assert np.array_equal(z['x'],plan['target_x']) and np.array_equal(r['y_exact'],exact)
        err=float(np.max(abs(np.asarray(z['f'])-exact)));assert err==r['E_grid'] and (err<=.01)==r['accuracy_met']
        assert (r['timer_stop_ns']-r['timer_start_ns'])*1e-9==r['method_wall_seconds']>0
        if j['profile']=='MC1':assert r['seed']==248800040100+j['episode'] and r['seed'] not in oldseeds and z['compilation_seconds']==0
        else:assert np.asarray(z['f'],dtype='<f8').tobytes()==np.asarray(read_json(ROOT/j['reference_path'])['result']['f'],dtype='<f8').tobytes()
        groups[j['profile']].append(r);raw.append(r)
    assert len(raw)==12 and all(len(rs)==4 for rs in groups.values())
    assert len({r['worker_pid'] for r in raw})==len({r['session'] for r in raw})==1
    assert all(a['timer_stop_ns']<b['timer_start_ns'] for a,b in zip(raw,raw[1:]))
    timing={p:statistics.median(r['method_wall_seconds'] for r in rs) for p,rs in groups.items()}
    zero={p:next(r for r in rs if r['job']['episode']==0) for p,rs in groups.items()}
    mc_pass=zero['MC1']['E_grid']<=.01
    analysis=read_json(out/'analysis.json')
    for p in ('R128','R256'):
        rp=zero[p]['E_grid']<=.01;rt=[r['method_wall_seconds'] for r in groups[p]];mt=[r['method_wall_seconds'] for r in groups['MC1']]
        stable=all(statistics.median(abs(t-statistics.median(ts)) for t in ts)/statistics.median(ts)<=.2 for ts in (rt,mt));tie=False
        if rp and mc_pass:
            if not stable:want='unknown'
            elif timing['MC1']<=.9*timing[p] and max(mt)<min(rt):want='monte_carlo'
            elif timing[p]<=.9*timing['MC1'] and max(rt)<min(mt):want='remizov'
            else:want='remizov';tie=True
        else:want='remizov' if rp else ('monte_carlo' if mc_pass else 'none')
        assert analysis['pair_labels_new_independent_block'][p]['label']==want
        assert analysis['pair_labels_new_independent_block'][p]['tie']==tie
    cheapest_R=min((p for p in ('R128','R256') if zero[p]['E_grid']<=.01),key=lambda p:timing[p])
    assert analysis['best_accurate_R']==cheapest_R and analysis['fastest_observed_profile']==cheapest_R
    assert analysis['independent_accuracy']['k']==sum(r['E_grid']<=.01 for r in groups['MC1'])
    candidates={r['equation_id']:r for r in read_json(out/'train_candidate_rows.json')};assert len(candidates)==40
    candidate_checks=[]
    for pid,stage in [('AMV1_F0005_S02',1),('AMV1_F0136_S01',2)]:
        folder=ROOT/'reports/ml_selector_contract'/('sentinel_p1_2026-10-26' if stage==1 else 'pilot_p2_2026-10-26');rs=[]
        for ep in range(4):
            r=read_json(folder/'records'/f'{pid}_MC1_stage{stage}_episode{ep}.json');exact_case=np.asarray(eqs[pid].y_exact(np.asarray(plan['target_x'])))
            assert float(np.max(abs(np.asarray(r['result']['f'])-exact_case)))==r['E_grid'];rs.append(r)
        row=candidates[pid];assert row['MC1_errors']==[r['E_grid'] for r in rs]
        assert row['MC1_successes4']==sum(r['E_grid']<=.01 for r in rs) and row['MC1_primary_error']==rs[0]['E_grid']
        assert row['MC1_median_wall']==statistics.median(r['method_wall_seconds'] for r in rs)
        assert row['matched_R128_MC1_8_scenario_seconds']==4*(row['R128_native_seconds']+row['MC1_median_wall'])
        assert not row['cost_scope_comparable'];candidate_checks.append(pid)
    for name,h in read_json(out/'candidate_sources_sha256.json').items():assert digest(ROOT/name)==h,name
    before=read_json(out/'protected_before.json');after=snapshot(out)
    changed={n for n,h in before.items() if after.get(n)!=h};assert changed<= {'PROJECT_STATUS.md'},changed
    additions=set(after)-set(before)
    assert additions<= {'diagnostics/report_f0047_three_profile.py','diagnostics/screen_train_MC_candidates.py','diagnostics/verify_f0047_three_profile.py','tests/test_f0047_three_analysis.py'},additions
    for name,h in plan['protected_sha256'].items():assert digest(ROOT/name)==h,name
    requests=[read_json(p) for p in (out/'events').glob('*.request.json')]
    assert len(requests)==12 and len({r['job']['key'] for r in requests})==12 and all(r['plan_sha256']==ph for r in requests)
    result=dict(passed=True,results=12,requests=12,unique_equations=[e.equation_id],same_worker=True,canonical_errors_recomputed=12,
        independent_MC_trials=4,independent_MC_passes=sum(r['accuracy_met'] for r in groups['MC1']),fresh_primary_accurate=mc_pass,
        fastest_accurate_profile=cheapest_R,median_seconds=timing,candidate_checks=candidate_checks,candidate_source_files=len(read_json(out/'candidate_sources_sha256.json')),
        old_paths_checked=len(before),old_paths_changed=sorted(changed),allowed_new_code=sorted(additions),no_solver_or_ML_fit=True)
    write_json(out/'independent_verification.json',result,exclusive=True);print(json.dumps(result,sort_keys=True),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);verify(ap.parse_args().out.resolve())
