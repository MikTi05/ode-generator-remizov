"""Saved-only three-profile analysis. Old seeds/labels stay in their cohorts."""
from __future__ import annotations
import argparse,csv,json,math,statistics,sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnostics.mc_pilot_p2 import read_json,write_json,digest
from diagnostics.report_f0047_timing import stats
from diagnostics.chernov_batch_control import lock_is_held
from ml.selector_labels import Evidence,fingerprint,label_pair,TIMING_SCOPE
PID='AMV1_F0047_S00';CONTRACT=ROOT/'reports/ml_selector_contract'
R0=ROOT/'results/remizov_r0_2026-10-26';PREVIOUS=ROOT/'results/f0047_observed_timing_2026-10-26'


def report(out):
    import numpy as np
    from scipy.stats import beta
    from all_methods_dataset_loader import load_all_methods_equations
    plan=read_json(out/'plan.json');ph=digest(out/'plan.json')
    for name,h in plan['protected_sha256'].items():assert digest(ROOT/name)==h,name
    jobs=plan['jobs'];assert len(jobs)==12 and Counter(j['profile'] for j in jobs)=={'R128':4,'R256':4,'MC1':4}
    requests=[read_json(p) for p in (out/'events').glob('*.request.json')]
    assert len(requests)==12 and {r['job']['key'] for r in requests}=={j['key'] for j in jobs}
    assert all(r['plan_sha256']==ph and r['job'] in jobs for r in requests)
    assert not list((out/'events').glob('*.interrupted.json'))
    assert len(list((out/'records').glob('*.json')))==12 and read_json(out/'active.json')['finished'] and not read_json(out/'pending.json')['keys']
    assert all(not lock_is_held(ROOT/'results'/n) for n in ('.chernov_batch_controller.lock','.chernov_batch_compute.lock'))
    data=ROOT/plan['data_dir']
    eqs={e.equation_id:e for e in load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl')}
    e=eqs[PID];m=e.metadata;exact=np.asarray(e.y_exact(np.asarray(plan['target_x'])))
    groups={p:[] for p in ('R128','R256','MC1')};rows=[];raw=[]
    for j in jobs:
        path=out/'records'/f"{j['key']}.json";r=read_json(path);z=r['result'];cfg=j['config']
        assert r['status']=='completed' and r['job']==j and r['plan_sha256']==ph and r['solver_called']
        assert r['equation_id']==z['problem_id']==PID and r['expression_hash']==m['expression_hash'] and r['family_id']==m['family_id'] and r['split']=='train'
        assert cfg==r['config'] and z['success'] and z['failure_reason']=='' and z['T']==j['expected_T']
        assert np.array_equal(z['x'],plan['target_x']) and np.array_equal(r['y_exact'],exact)
        f=np.asarray(z['f']);assert f.shape==(33,) and np.isfinite(f).all()
        err=float(np.max(abs(f-exact)));assert err==r['E_grid'] and (err<=.01)==r['accuracy_met']
        assert math.isclose((r['timer_stop_ns']-r['timer_start_ns'])*1e-9,r['method_wall_seconds'],rel_tol=0,abs_tol=1e-12)
        assert r['method_wall_seconds']>z['runtime_seconds']>0
        if j['profile']!='MC1':
            n=int(j['profile'][1:]);ref=read_json(ROOT/j['reference_path'])
            assert cfg==ref['job']['config'] and z['n']==n and z['scheme']=='solve'
            assert z['time_nodes']==513 and z['work_grid_points']==2081 and z['actual_dx']==.0125 and z['work_domain']==[-13.,13.]
            assert z['n_operator_applications']==n*512 and z['n_spline_builds']==n*512+1
            assert f.astype('<f8').tobytes()==np.asarray(ref['result']['f'],dtype='<f8').tobytes()
            assert r['array_identical_to_reference']
        else:
            assert r['new_independent_MC_seed'] and r['seed']==cfg['seed']==z['seed_entropy']==248800040100+j['episode']
            assert z['M']==cfg['M']==4096 and z['completed_paths']==[4096]*33 and z['backend']=='numba'
            assert z['dt_requested']==cfg['dt']==1/128 and 0<z['dt_actual']<=1/128 and z['block_steps']==64
            assert z['n_path_steps']==z['n_normal_draws']==cfg['max_path_steps']==j['numerical_work']
            assert z['n_steps']==j['expected_steps'] and z['compilation_seconds']==z['numba_import_seconds']==0
            assert z['tail_tolerance_met'] and z['forcing_abs_upper_bound']==j['forcing_G'] and z['rng_scheme']=='PCG64/SeedSequence/point_address_v1'
        groups[j['profile']].append(r);raw.append(r)
        rows.append(dict(profile=j['profile'],episode=j['episode'],seed=r['seed'],E_grid=err,accurate=err<=.01,method_wall_seconds=r['method_wall_seconds'],solver_seconds=z['runtime_seconds'],source=str(path.relative_to(ROOT)),source_sha256=digest(path)))
    assert len({r['worker_pid'] for r in raw})==len({r['session'] for r in raw})==1
    assert all(a['timer_stop_ns']<b['timer_start_ns'] for a,b in zip(raw,raw[1:]))
    for p in groups:
        groups[p].sort(key=lambda r:r['job']['episode']);assert [r['job']['episode'] for r in groups[p]]==list(range(4))
        if p!='MC1':assert all(r['result']['f']==groups[p][0]['result']['f'] for r in groups[p])
    timing={p:stats([r['method_wall_seconds'] for r in rs]) for p,rs in groups.items()}
    cohort=fingerprint(dict(environment_sha256=digest(out/'environment.json'),session=raw[0]['session'],worker_pid=raw[0]['worker_pid'],threads=1))
    evid={}
    for p,rs in groups.items():
        ev=Evidence('monte_carlo' if p=='MC1' else 'remizov',PID,m['expression_hash'],m['family_id'],'train',p,status='completed',error=rs[0]['E_grid'],T=rs[0]['result']['T'],times=[r['method_wall_seconds'] for r in rs],timing_scope=TIMING_SCOPE,timing_cohort=cohort,deterministic_repeats_equal=p!='MC1',source=[str((out/'records'/f"{r['job']['key']}.json").relative_to(ROOT)) for r in rs])
        ev.source_sha256=[digest(ROOT/n) for n in ev.source];evid[p]=ev
    comparisons={}
    for p in ('R128','R256'):
        pairid=fingerprint(dict(R=p,MC='MC1',contract_sha256=plan['protected_sha256']['reports/ml_selector_contract/plan.json'],block_plan_sha256=ph,MC_seed_group=4,label='observed_fastest_v1'))
        comparisons[p]=label_pair(evid[p],evid['MC1'],pairid)
    accurate_R=[p for p in ('R128','R256') if evid[p].eligibility=='pass']
    bestR=min(accurate_R,key=lambda p:(timing[p]['median'],int(p[1:]))) if accurate_R else None
    selected=comparisons[bestR] if bestR else None
    stable=all(v['MAD_over_median']<=.2 for v in timing.values())
    fastest_observed_profile='MC1' if selected and selected['label']=='monte_carlo' else (bestR if selected and selected['label']=='remizov' else 'unknown')
    if not stable:fastest_observed_profile='unknown'
    k=sum(r['accuracy_met'] for r in groups['MC1']);lo=float(beta.ppf(.025,k,5-k)) if k else 0.;hi=float(beta.ppf(.975,k+1,4-k)) if k<4 else 1.
    old=read_json(PREVIOUS/'analysis.json')
    thermal=[json.loads(s) for s in (out/'thermal.jsonl').read_text().splitlines()]
    samples=[json.loads(s) for f in (out/'workers').glob('*.rss.jsonl') for s in f.read_text().splitlines()]
    guardians=[read_json(f) for f in (out/'control').rglob('*.guardian.json')]
    assert len(guardians)==1 and guardians[0]['finished'] and guardians[0]['cleanup_success']
    assert max(s['peak_rss_mib'] for s in samples)<512 and all(s['thermal_state']<2 for s in thermal)
    warm=[read_json(f) for f in (out/'workers').glob('*.warmup.json')];assert len(warm)==1 and warm[0]['extra_solver_calls']==0 and warm[0]['numba_threads']==1
    result=dict(equation_id=PID,calls=12,profiles=timing,errors={p:[r['E_grid'] for r in rs] for p,rs in groups.items()},
        pair_labels_new_independent_block=comparisons,best_accurate_R=bestR,fastest_observed_profile=fastest_observed_profile,all_timings_stable=stable,
        MC_speed_ratios={p:timing[p]['median']/timing['MC1']['median'] for p in ('R128','R256')},
        MC_time_reduction_vs_R={p:1-timing['MC1']['median']/timing[p]['median'] for p in ('R128','R256')},
        MC_disjoint_faster_than={p:timing['MC1']['max']<timing[p]['min'] for p in ('R128','R256')},
        independent_accuracy=dict(k=k,n=4,CP_two_sided95=[lo,hi],fixed_before_results=True,reliable_p_ge095_certified=False),
        old_replay_block=dict(source=str(PREVIOUS.relative_to(ROOT)),label=old['label']['label'],MC_errors=[r['E_grid'] for r in old['attempts'] if r['profile']=='MC1'],new_accuracy_trials_in_old_replay=0,unchanged=True),
        method_wall_sum_seconds=sum(r['method_wall_seconds'] for r in raw),sessions=[read_json(f) for f in (out/'sessions').glob('*.json')],
        resources=dict(peak_rss_mib=max(s['peak_rss_mib'] for s in samples),rss_samples=len(samples),thermal_samples=len(thermal),thermal_max=max(s['thermal_state'] for s in thermal),warmup=warm[0]),
        old_labels_preserved=True,global_train_selected_R_profile='R256',no_ML=True,no_other_equations=True,no_benchmark1000=True)
    write_json(out/'analysis.json',result,exclusive=True)
    write_json(out/'timing_evidence.json',{p:asdict(e) for p,e in evid.items()},exclusive=True)
    with (out/'attempts.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({k:result[k] for k in ('profiles','errors','pair_labels_new_independent_block','fastest_observed_profile','independent_accuracy')}),flush=True)
    return eqs


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);report(ap.parse_args().out.resolve())
