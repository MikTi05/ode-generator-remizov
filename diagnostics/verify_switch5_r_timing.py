"""Independent verification of raw R-only40 records and aggregate savings."""
from __future__ import annotations
import argparse,math,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnostics.audit_probability_selector import read,write,sha


def verify(out):
    import numpy as np
    from all_methods_dataset_loader import load_all_methods_equations
    plan=read(out/'plan.json');analysis=read(out/'analysis.json');ph=sha(out/'plan.json')
    source=ROOT/'results/ml_switch5_r_preflight_2026-10-26'
    assert plan['jobs']==read(source/'plan.json')['jobs'] and plan['resources']['series_seconds']==1766
    assert sha(source/'plan.json')==plan['source_plan_sha256']
    for n,h in plan['protected_sha256'].items():assert sha(ROOT/n)==h,n
    records=[read(p) for p in (out/'records').glob('*.json')];requests=[read(p) for p in (out/'events').glob('*.request.json')]
    assert len(records)==len(requests)==40 and len({r['job']['key'] for r in requests})==40
    ordered=sorted(records,key=lambda r:r['timer_start_ns'])
    assert [r['job'] for r in ordered]==plan['jobs']
    assert all(a['timer_stop_ns']<=b['timer_start_ns'] for a,b in zip(ordered,ordered[1:]))
    data=out/'training_inputs';eqs={e.equation_id:e for e in load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl')}
    assert len(eqs)==5 and all(e.metadata['split']=='train' for e in eqs.values())
    for r in records:
        j=r['job'];z=r['result'];e=eqs[r['equation_id']]
        exact=np.asarray(e.y_exact(np.asarray(plan['target_x'])));error=float(np.max(abs(np.asarray(z['f'])-exact)))
        assert r['status']=='completed' and r['solver_called'] and r['plan_sha256']==ph and j in plan['jobs']
        assert r['expression_hash']==e.metadata['expression_hash'] and r['family_id']==e.metadata['family_id'] and r['split']=='train'
        assert r['config']==j['config'] and z['T']==j['expected_T'] and z['n']==j['config']['n']
        assert error==r['E_grid']==j['reference_E_grid'] and (error<=.01)==r['accuracy_met']
        np.testing.assert_array_equal(r['y_exact'],exact);np.testing.assert_array_equal(z['x'],plan['target_x'])
        ref=read(ROOT/j['reference_path']);assert sha(ROOT/j['reference_path'])==j['reference_sha256']
        np.testing.assert_array_equal(z['f'],ref['result']['f'])
        assert math.isclose((r['timer_stop_ns']-r['timer_start_ns'])*1e-9,r['method_wall_seconds'],rel_tol=0,abs_tol=1e-12)
    totals={p:0. for p in ('R128','R256')};good_totals={p:0. for p in ('R128','R256')};good_cases=0
    for c in analysis['cases']:
        for p in ('R128','R256'):
            rs=[r for r in records if r['equation_id']==c['equation_id'] and r['profile']==p]
            assert len(rs)==4 and all(r['result']['f']==rs[0]['result']['f'] for r in rs)
            median=statistics.median(r['method_wall_seconds'] for r in rs)
            assert median==c[p]['timing']['median'];totals[p]+=median
        if c['R128']['accurate'] and c['R256']['accurate']:
            good_cases+=1
            for p in totals:good_totals[p]+=c[p]['timing']['median']
            assert c['supported_accurate_speedup']==(c['R128']['timing']['median']<=.9*c['R256']['timing']['median']
                and c['R128']['timing']['max']<c['R256']['timing']['min']
                and max(c[p]['timing']['MAD_fraction'] for p in totals)<=.2)
        else:assert c['equation_id']=='AMV1_F0172_S03' and c['classification']=='neither_profile_accurate'
    assert good_cases==4
    for name,sums,correct in [('all5',totals,4),('both_accurate',good_totals,4)]:
        z=analysis['aggregate'][name]
        assert math.isclose(z['saved_seconds'],sums['R256']-sums['R128'],abs_tol=1e-12)
        assert math.isclose(z['relative_saved'],1-sums['R128']/sums['R256'],abs_tol=1e-15)
        assert z['lost_baseline_deliveries']==0 and z['accurate_deliveries']==correct
        assert math.isclose(z['method_seconds_per_accurate'],sums['R128']/correct,abs_tol=1e-12)
    assert analysis['aggregate']['global40']['measured_percent_saving'] is None
    assert sum(r['result']['n_operator_applications'] for r in records)==3932160
    assert len({r['worker_pid'] for r in records})==len({r['session'] for r in records})==1
    assert read(out/'active.json')['finished'] and not read(out/'pending.json')['keys']
    write(out/'independent_verification.json',dict(passed=True,raw_records=40,unique_requests=40,canonical_errors_recomputed=40,
        actual_execution_order_equals_manifest=True,reference_arrays_identical=True,median_savings_and_cost_per_success_recomputed=True,
        expected_accurate_cases=4,known_shared_inaccurate_case='AMV1_F0172_S03',workers=1,
        model_policy_numerical_sources_unchanged=True,numerical_calls_in_verifier=0,MC_calls=0,ML_fits=0,validation_test_used=False))
    print(dict(passed=True,records=40,errors=40,savings=analysis['aggregate']['all5']['relative_saved']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);verify(p.parse_args().out.resolve())
