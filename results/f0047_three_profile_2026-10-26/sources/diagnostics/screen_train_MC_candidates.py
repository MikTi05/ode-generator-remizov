"""Exploratory saved-only train candidate/cost audit. Never solve or fit."""
from __future__ import annotations
import argparse,csv,json,statistics,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnostics.mc_pilot_p2 import read_json,write_json,digest
CONTRACT=ROOT/'reports/ml_selector_contract';R0=ROOT/'results/remizov_r0_2026-10-26'


def screen(out):
    import numpy as np
    from scipy.stats import beta
    from all_methods_dataset_loader import load_all_methods_equations
    contract=read_json(CONTRACT/'plan.json');data=ROOT/contract['data_dir']
    eqs={e.equation_id:e for e in load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl')}
    observations=read_json(R0/'R_observations.json');R={};pins={}
    for obs in observations:
        path=ROOT/obs['source'];assert digest(path)==obs['source_sha256'];r=read_json(path);z=r['result'];pid=obs['equation_id'];e=eqs[pid]
        assert r['split']==e.metadata['split']=='train' and r['expression_hash']==e.metadata['expression_hash']
        exact=np.asarray(e.y_exact(np.asarray(contract['target_x'])));err=float(np.max(abs(np.asarray(z['f'])-exact)))
        assert err==obs['E_grid']==r.get('E_grid',r.get('E_inf'))
        assert z['n']==int(obs['profile'][1:]) and r['job']['config']==dict(contract['profiles'][obs['profile']]['config'],T=z['T'])
        assert np.array_equal(z['x'],contract['target_x']) and np.array_equal(r['y_exact'],exact)
        R[pid,obs['profile']]=dict(error=err,solver_seconds=z['runtime_seconds'],T=z['T']);pins[obs['source']]=digest(path)
    groups=defaultdict(list)
    for folder in (CONTRACT/'sentinel_p1_2026-10-26',CONTRACT/'pilot_p2_2026-10-26'):
        for name,h in read_json(folder/'artifact_sha256.json').items():assert digest(folder/name)==h,(folder,name)
        for path in (folder/'records').glob('*MC1*.json'):
            r=read_json(path);j=r['job'];z=r['result'];pid=r['equation_id'];e=eqs[pid]
            assert r['status']=='completed' and z['success'] and j['profile']=='MC1'
            assert r['split']==e.metadata['split']=='train' and r['expression_hash']==e.metadata['expression_hash']
            assert r['family_id']==e.metadata['family_id']
            assert z['M']==j['config']['M']==4096 and z['dt_requested']==j['config']['dt']==1/128
            assert z['T']==R[pid,'R128']['T']==R[pid,'R256']['T'] and z['n_steps']==j['expected_steps']
            assert r['seed']==j['config']['seed']==z['seed_entropy'] and z['backend']=='numba' and z['block_steps']==64
            assert z['rng_scheme']=='PCG64/SeedSequence/point_address_v1' and z['completed_paths']==[4096]*33
            exact=np.asarray(e.y_exact(np.asarray(contract['target_x'])));err=float(np.max(abs(np.asarray(z['f'])-exact)))
            assert np.array_equal(z['x'],contract['target_x']) and np.array_equal(r['y_exact'],exact) and err==r['E_grid']
            assert r['method_wall_seconds']>z['runtime_seconds']>0
            groups[pid].append(r);pins[str(path.relative_to(ROOT))]=digest(path)
    assert len(groups)==40 and sum(map(len,groups.values()))==160
    legacy=defaultdict(list);oldpins=read_json(R0/'protected_before.json')
    for path in (ROOT/'results/final_grid5_2026-10-25/records').glob('*MC*.json'):
        name=str(path.relative_to(ROOT));assert digest(path)==oldpins[name];r=read_json(path);j=r['job'];z=r['result'];pid=j['problem_id'];e=eqs[pid]
        assert r['status']=='completed' and z['success'] and r['split']=='train' and r['expression_hash']==e.metadata['expression_hash']
        assert j['config']['M']==z['M']==4096 and j['config']['dt']==z['dt_requested']==1/128 and z['T']==R[pid,'R128']['T']
        assert z['rng_scheme']=='PCG64/SeedSequence/point_address_v1' and z['block_steps']==64
        exact=np.asarray(e.y_exact(np.asarray(contract['target_x'])));err=float(np.max(abs(np.asarray(z['f'])-exact)))
        assert np.array_equal(z['x'],contract['target_x']) and err==r['E_inf']
        legacy[pid].append(dict(seed=j['config']['seed'],error=err));pins[name]=digest(path)
    rows=[]
    for pid,rs in sorted(groups.items()):
        rs.sort(key=lambda r:r['job']['episode']);assert [r['job']['episode'] for r in rs]==list(range(4))
        errors=[r['E_grid'] for r in rs];k=sum(e<=.01 for e in errors);mc=statistics.median(r['method_wall_seconds'] for r in rs)
        r128=R[pid,'R128'];r256=R[pid,'R256']
        eligible_R=[p for p in ('R128','R256') if R[pid,p]['error']<=.01]
        best=min(eligible_R,key=lambda p:R[pid,p]['solver_seconds']) if eligible_R else None
        cost_potential=best is None or mc<=.9*R[pid,best]['solver_seconds']
        older=legacy.get(pid,[]);assert not {r['seed'] for r in rs}&{r['seed'] for r in older}
        row=dict(equation_id=pid,family_id=eqs[pid].metadata['family_id'],split='train',
            R128_E_grid=r128['error'],R256_E_grid=r256['error'],R128_native_seconds=r128['solver_seconds'],R256_native_seconds=r256['solver_seconds'],
            MC1_primary_error=errors[0],MC1_errors=errors,MC1_successes4=k,MC1_median_wall=mc,
            descriptive_k_over_4=k/4,exploratory_CP95=[float(beta.ppf(.025,k,5-k)) if k else 0.,float(beta.ppf(.975,k+1,4-k)) if k<4 else 1.],
            legacy_final_distinct_seeds=older,combined_old_MC1_passes=k+sum(r['error']<=.01 for r in older),combined_old_MC1_trials=4+len(older),
            fastest_saved_accurate_R=best,saved_cost_potential=cost_potential,
            MC_min_error=min(errors),near_threshold_exploratory=min(errors)<=.015,
            needed_R128_full_seconds_for_MC_10pct=mc/.9,needed_R128_slowdown_vs_native=mc/(.9*r128['solver_seconds']),
            cost_scope_comparable=False,
            MC_only4_scenario_seconds=4*mc,matched_R128_MC1_8_scenario_seconds=4*(r128['solver_seconds']+mc),
            matched_three_profiles12_scenario_seconds=4*(r128['solver_seconds']+r256['solver_seconds']+mc),
            matched_cheapest_accurate_R_MC1_scenario_seconds=4*(R[pid,best]['solver_seconds']+mc) if best else 4*mc)
        rows.append(row)
    hits=[r for r in rows if r['MC1_successes4']]
    near=[r for r in rows if not r['MC1_successes4'] and r['near_threshold_exploratory'] and r['saved_cost_potential']]
    near.sort(key=lambda r:(r['MC_min_error'],r['matched_cheapest_accurate_R_MC1_scenario_seconds']))
    analysis=dict(scope='40 original train equations;160 MC1 P1/P2 +20 older final4096;80 R; saved-only',
        accurately_observed_candidates=hits,near_threshold_unconfirmed=near,
        MC_already_accurate_and_cost_potential=[r['equation_id'] for r in hits if r['saved_cost_potential']],
        primary_eligibility_not_lucky_seed=True,independent_new_F0047_seeds_excluded_from_old_train_counts=True,
        rate_interpretation='Individual k/4 and CP are exploratory after case/profile selection, not calibrated probability of a useful next block or family prevalence; no FWE coverage claim.',
        cost_interpretation='Scenarios from old R solver-only and MC warm times in different cohorts, not valid labels, forecasts with CI or bounds. Full method R prep/setup overhead is additional.',
        study_value_interpretation='Useful win needs accurate fixed primary AND >=10pct/disjoint matched-time gain against fastest accurate R; accuracy alone or lucky extra seed does not meet it.',
        no_other_solves=True,no_ML=True)
    write_json(out/'train_candidates.json',analysis,exclusive=True)
    write_json(out/'train_candidate_rows.json',rows,exclusive=True)
    write_json(out/'candidate_sources_sha256.json',pins,exclusive=True)
    with (out/'train_candidates.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})
    print(json.dumps(dict(hits=hits,near_unconfirmed=near),sort_keys=True),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);screen(ap.parse_args().out.resolve())
