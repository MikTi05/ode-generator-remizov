"""Verified train-only sufficient statistics, separate accuracy and full-cost data.

No solver calls; probability trials are semantic seeds, never timing replays.
"""
from __future__ import annotations
import json,statistics
from pathlib import Path
from ml.selector_evidence import ROOT,CONTRACT,load_saved_evidence,read,sha,verify_manifest


def prepare(inputs):
    from diagnostics.report_remizov_r0 import read_r
    from all_methods_dataset_loader import load_all_methods_equations
    r0=ROOT/'results/remizov_r0_2026-10-26';old=ROOT/'results/f0047_observed_timing_2026-10-26';fresh=ROOT/'results/f0047_three_profile_2026-10-26'
    for folder in (r0,old,fresh):verify_manifest(folder)
    bundle=load_saved_evidence(inputs);assert not any(e['issues'] for e in bundle['evidence'].values())
    pins=dict(bundle['source_sha256']);contract=read(CONTRACT/'plan.json');data=ROOT/contract['data_dir']
    eqs={e.equation_id:e for e in load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl')}
    index={str(pid):i for i,pid in enumerate(inputs['ids'])};accuracy={};cost={};raw_checks=len(bundle['execution'])
    def base(pid,profile):
        m=inputs['metadata'][pid];assert m['split']=='train'
        return dict(equation_id=pid,family_id=m['family_id'],split='train',index=index[pid],profile=profile,expression_hash=m['expression_hash'])
    for key,e in bundle['evidence'].items():
        if e['status']!='completed':continue
        pid=e['equation_id'];p=e['profile']
        if p=='R128':
            accuracy[pid,p]=dict(base(pid,p),k=int(e['error']<=.01),n=1,sources=e['source'],errors=[e['error']],trial_semantics='one deterministic equation')
        elif p=='MC1':
            rs=e['records'];assert len(rs)==4 and [r['episode'] for r in rs]==list(range(4))
            assert len({r['seed'] for r in rs})==4 and all(r['error'] is not None for r in rs)
            accuracy[pid,p]=dict(base(pid,p),k=sum(r['error']<=.01 for r in rs),n=4,sources=e['source'],errors=[r['error'] for r in rs],trial_semantics='four unique P1/P2 semantic seeds')
            cost[pid,p]=dict(base(pid,p),seconds=statistics.median(e['times']),times=e['times'],sources=e['source'],cohort=e['timing_cohort'],scope=e['timing_scope'])
    r0plan=read(r0/'plan.json');r0hash=sha(r0/'plan.json')
    for obs in read(r0/'R_observations.json'):
        pid=obs['equation_id'];p=obs['profile'];path=ROOT/obs['source'];assert sha(path)==obs['source_sha256']
        cfg=dict(contract['profiles'][p]['config'],T=read(path)['result']['T'])
        r,err=read_r(path,pid,p,eqs[pid],cfg,r0hash if obs['new_R0'] else None);raw_checks+=1
        if (pid,p) in accuracy:assert accuracy[pid,p]['errors']==[err]
        accuracy[pid,p]=dict(base(pid,p),k=int(err<=.01),n=1,sources=[obs['source']],errors=[err],trial_semantics='one deterministic equation')
        pins[obs['source']]=sha(path)
        if obs['new_R0']:
            cost[pid,p]=dict(base(pid,p),seconds=r['method_wall_seconds'],times=[r['method_wall_seconds']],sources=[obs['source']],cohort=r['session'],scope='warm_end_to_end_method_wall_v1')
    # Add one full R256 timing case; deterministic accuracy already counted once.
    for profile,folder in [('R256',old),('R128',fresh)]:
        rs=[];paths=[]
        for path in (folder/'records').glob(f'*_{profile}_*.json'):
            r=read(path);assert r['status']=='completed' and r['profile']==profile and r['array_identical_to_reference']
            assert r['method_wall_seconds']>r['result']['runtime_seconds']>0
            rs.append(r);paths.append(str(path.relative_to(ROOT)));pins[paths[-1]]=sha(path)
        assert len(rs)==4 and len({r['session'] for r in rs})==1
        pid='AMV1_F0047_S00';times=[r['method_wall_seconds'] for r in rs]
        cost[pid,profile]=dict(base(pid,profile),seconds=statistics.median(times),times=times,sources=paths,cohort=rs[0]['session'],scope='warm_end_to_end_method_wall_v1')
    for row in accuracy.values():assert 0<=row['k']<=row['n'] and inputs['splits'][row['index']]=='train'
    for row in cost.values():assert row['seconds']>0 and row['scope']=='warm_end_to_end_method_wall_v1'
    paired=sorted(pid for pid,p in accuracy if p=='MC1')
    assert len(paired)==40 and all((pid,'R256') in accuracy and (pid,'R128') in accuracy for pid in paired)
    counts={p:dict(equations=sum(q==p for _,q in accuracy),successes=sum(r['k'] for (_,q),r in accuracy.items() if q==p),trials=sum(r['n'] for (_,q),r in accuracy.items() if q==p),cost_equations=sum(q==p for _,q in cost)) for p in ('R128','R256','MC1')}
    assert counts['MC1']['trials']==160 and counts['MC1']['successes']==7
    held_fresh=[str(p.relative_to(ROOT)) for p in sorted((fresh/'records').glob('*MC1*.json'))];assert len(held_fresh)==4
    return dict(accuracy=list(accuracy.values()),cost=list(cost.values()),paired_ids=paired,counts=counts,source_sha256=pins,
        raw_checks=raw_checks,seed_audit_paths=held_fresh,
        exclusions=dict(MC_replay_trials=0,MC_fresh_group='reserved post-fit seed audit; never fit/calibrate/tune',
            old_final_MC20='selected earlier cohort not pooled with P1/P2',MC0_MC2='different profiles',legacy_taper_x0='different target/profile',
            R_native_time='never used as full wall target',deterministic_R_repeats='one accuracy trial per equation/profile'))
