"""Saved-only accuracy, replay, timing and label audit of the eight-call block."""
from __future__ import annotations
import argparse,csv,json,math,statistics,sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnostics.mc_pilot_p2 import read_json,write_json,digest
from diagnostics.chernov_batch_control import lock_is_held
from ml.selector_labels import Evidence,label_pair,fingerprint,TIMING_SCOPE,training_gate,support
PID='AMV1_F0047_S00';R0=ROOT/'results/remizov_r0_2026-10-26'


def stats(values):
    import numpy as np
    med=statistics.median(values);mad=statistics.median(abs(x-med) for x in values)
    return dict(raw=values,n=len(values),median=med,min=min(values),max=max(values),IQR=float(np.percentile(values,75)-np.percentile(values,25)),MAD=mad,MAD_over_median=mad/med)


def report(out):
    import numpy as np
    from scipy.stats import beta
    from all_methods_dataset_loader import load_all_methods_equations
    plan=read_json(out/'plan.json');ph=digest(out/'plan.json');jobs=plan['jobs']
    assert len(jobs)==8 and Counter(j['profile'] for j in jobs)=={'R256':4,'MC1':4}
    for n,h in plan['protected_sha256'].items():assert digest(ROOT/n)==h,n
    data=ROOT/plan['data_dir']
    e=next(e for e in load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl') if e.equation_id==PID)
    exact=np.asarray(e.y_exact(np.asarray(plan['target_x'])));m=e.metadata
    requests=[read_json(p) for p in (out/'events').glob('*.request.json')]
    assert len(requests)==8 and len({r['job']['key'] for r in requests})==8
    assert len(list((out/'records').glob('*.json')))==8
    assert not list((out/'events').glob('*.interrupted.json'))
    assert read_json(out/'active.json')['finished'] and not read_json(out/'pending.json')['keys']
    assert all(not lock_is_held(ROOT/'results'/n) for n in ('.chernov_batch_controller.lock','.chernov_batch_compute.lock'))
    rows=[];byprofile={'R256':[],'MC1':[]};raw={}
    for j in jobs:
        path=out/'records'/f"{j['key']}.json";r=read_json(path);z=r['result'];cfg=j['config']
        assert r['job']==j and r['plan_sha256']==ph and r['status']=='completed' and r['solver_called']
        assert r['config']==cfg and r['equation_id']==z['problem_id']==PID
        assert r['expression_hash']==m['expression_hash']==plan['expression_hash'] and r['family_id']==m['family_id'] and r['split']==m['split']=='train'
        assert z['success'] and z['failure_reason']=='' and z['T']==j['expected_T']
        assert np.array_equal(z['x'],plan['target_x']) and np.array_equal(r['y_exact'],exact)
        f=np.asarray(z['f']);assert f.shape==(33,) and np.isfinite(f).all()
        error=float(np.max(abs(f-exact)));assert error==r['E_grid'] and bool(error<=.01)==r['accuracy_met']
        assert r['method_wall_seconds']>z['runtime_seconds']>0
        assert math.isclose((r['timer_stop_ns']-r['timer_start_ns'])*1e-9,r['method_wall_seconds'],rel_tol=0,abs_tol=1e-12)
        ref=read_json(ROOT/j['reference_path'])
        assert r['reference_sha256']==digest(ROOT/j['reference_path']) and np.array_equal(z['f'],ref['result']['f']) and r['array_identical_to_reference']
        if j['profile']=='R256':
            assert cfg['n']==z['n']==256 and cfg['boundary']=='zero' and cfg['num_t']==z['time_nodes']==513
            assert cfg['dx']==z['actual_dx']==.0125 and z['scheme']=='solve' and z['work_grid_points']==2081
            assert z['n_operator_applications']==131072 and z['n_spline_builds']==131073
        else:
            assert cfg['seed']==248800010100+j['episode']==r['seed']==z['seed_entropy']
            assert cfg['M']==z['M']==4096 and z['completed_paths']==[4096]*33
            assert cfg['dt']==z['dt_requested']==1/128 and 0<z['dt_actual']<=1/128
            assert z['n_path_steps']==z['n_normal_draws']==j['numerical_work']==cfg['max_path_steps']
            assert z['n_steps']==j['expected_steps'] and z['backend']=='numba' and z['block_steps']==64
            assert z['compilation_seconds']==z['numba_import_seconds']==0 and z['tail_tolerance_met']
            assert z['rng_scheme']=='PCG64/SeedSequence/point_address_v1' and z['forcing_abs_upper_bound']==j['forcing_G']
        row=dict(key=j['key'],profile=j['profile'],episode=j['episode'],seed=r['seed'],E_grid=error,accurate=error<=.01,
            method_wall_seconds=r['method_wall_seconds'],solver_seconds=z['runtime_seconds'],reference_identical=True,
            source=str(path.relative_to(ROOT)),source_sha256=digest(path))
        rows.append(row);byprofile[j['profile']].append(r);raw[j['key']]=r
    assert len({r['session'] for r in raw.values()})==len({r['worker_pid'] for r in raw.values()})==1
    assert [r['job'] for r in sorted(requests,key=lambda x:jobs.index(x['job']))]==jobs
    # Nanosecond intervals are disjoint and in the preregistered order.
    assert all(raw[a['key']]['timer_stop_ns']<raw[b['key']]['timer_start_ns'] for a,b in zip(jobs,jobs[1:]))
    cohort=fingerprint(dict(environment_sha256=digest(out/'environment.json'),session=next(iter(raw.values()))['session'],worker_pid=next(iter(raw.values()))['worker_pid'],threads=1))
    evidence=[];statistics_by_profile={}
    for profile,method in [('R256','remizov'),('MC1','monte_carlo')]:
        own=sorted(byprofile[profile],key=lambda r:r['job']['episode']);assert [r['job']['episode'] for r in own]==list(range(4))
        times=[r['method_wall_seconds'] for r in own];statistics_by_profile[profile]=stats(times)
        identical=all(r['result']['f']==own[0]['result']['f'] for r in own)
        if profile=='R256':assert identical
        ev=Evidence(method,PID,m['expression_hash'],m['family_id'],m['split'],profile,status='completed',error=own[0]['E_grid'],T=own[0]['result']['T'],times=times,
            timing_scope=TIMING_SCOPE,timing_cohort=cohort,deterministic_repeats_equal=profile=='R256' and identical,
            source=[str((out/'records'/f"{r['job']['key']}.json").relative_to(ROOT)) for r in own])
        ev.source_sha256=[digest(ROOT/p) for p in ev.source];evidence.append(ev)
    pair=read_json(R0/'labels_r256_pilot40.json');newlabel=label_pair(*evidence,pair['profile_pair_id'])
    tR=statistics_by_profile['R256']['median'];tMC=statistics_by_profile['MC1']['median']
    thermo=[json.loads(s) for s in (out/'thermal.jsonl').read_text().splitlines()]
    samples=[json.loads(s) for p in (out/'workers').glob('*.rss.jsonl') for s in p.read_text().splitlines()]
    guardians=[read_json(p) for p in (out/'control').rglob('*.guardian.json')]
    assert len(guardians)==1 and guardians[0]['finished'] and guardians[0]['cleanup_success']
    assert max(s['peak_rss_mib'] for s in samples)<512 and all(s['thermal_state']==0 or s['thermal_state']==1 for s in thermo)
    warm=[read_json(p) for p in (out/'workers').glob('*.warmup.json')];assert len(warm)==1 and warm[0]['extra_solver_calls']==0 and warm[0]['numba_threads']==1
    old=read_json(R0/'selected_labels_1000.json');updated=[]
    for row in old['rows']:
        updated.append(dict(row,**newlabel) if row['equation_id']==PID else row)
    assert sum(a!=b for a,b in zip(old['rows'],updated))==1
    known=[r for r in updated if r['label']!='unknown'];gate=training_gate(updated)
    k=sum(r['accuracy_met'] for r in byprofile['MC1']);lower=float(beta.ppf(.025,k,5-k)) if k else 0.
    summary=dict(calls=8,equation_id=PID,profiles=statistics_by_profile,attempts=rows,label=newlabel,
        R_over_MC_median_ratio=tR/tMC,MC_time_reduction_fraction=1-tMC/tR,median_seconds_saved=tR-tMC,
        method_wall_sum_seconds=sum(x['method_wall_seconds'] for x in rows),
        controller_sessions=[read_json(p) for p in (out/'sessions').glob('*.json')],
        reliability=dict(MC_observed_passes=k,trials=4,replayed_original_P1_seeds=True,additional_independent_accuracy_trials=0,original_four_seed_CP_lower95=lower,certified_p_ge_095=False),
        resources=dict(peak_rss_mib=max(s['peak_rss_mib'] for s in samples),rss_samples=len(samples),thermal_samples=len(thermo),thermal_max=max(s['thermal_state'] for s in thermo),warmup=warm[0]),
        selected_support=support(updated),gate=gate,no_other_equations=True,no_ML_fit=True,no_benchmark1000=True)
    write_json(out/'analysis.json',summary,exclusive=True)
    write_json(out/'label.json',newlabel,exclusive=True)
    write_json(out/'timing_evidence.json',[asdict(e) for e in evidence],exclusive=True)
    write_json(out/'selected_labels_1000.json',dict(profile='R256',rows=updated,support=support(updated),gate=gate,changed_equation=PID,source_labels_sha256=digest(R0/'selected_labels_1000.json')),exclusive=True)
    with (out/'attempts.csv').open('x',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(dict(label=newlabel,profiles=statistics_by_profile,ratio=tR/tMC,peak_rss=summary['resources']['peak_rss_mib'],gate=gate['reasons'])),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);report(ap.parse_args().out.resolve())
