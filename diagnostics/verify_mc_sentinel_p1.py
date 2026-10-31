"""Independently verify saved P1 answers, frozen inputs, costs and cleanup.

No MC calls or fitting. Exact functions are evaluated only on the fixed grid.
Run after reporting, then update PROJECT_STATUS and finalize preservation hash.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(p):
    with Path(p).open(newline='') as stream:return list(csv.DictReader(stream))


def verify(out):
    import numpy as np
    from scipy.stats import beta
    from all_methods_dataset_loader import load_all_methods_equations
    from diagnostics.chernov_batch_control import lock_is_held
    from diagnostics.report_mc_sentinel_p1 import choose_profile
    plan=read(out/'plan.json');ph=sha(out/'plan.json');jobs={j['key']:j for j in plan['jobs']}
    original={j['key']:j for j in map(json.loads,(ROOT/'reports/ml_selector_contract/pilot_jobs.jsonl').read_text().splitlines()) if j['stage']==1}
    assert jobs==original and len(jobs)==60
    assert plan['execution_authorized'] and plan['no_R'] and plan['no_stage2'] and plan['no_ML']
    assert sha(out/'AUTHORIZATION.txt')==plan['authorization_sha256']
    for name,h in plan['protected_sha256'].items():assert sha(ROOT/name)==h,name
    data=ROOT/plan['data_dir'];eqs={e.equation_id:e for e in load_all_methods_equations(
        data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl')}
    records={p.stem:read(p) for p in (out/'records').glob('*.json')}
    requests=[read(p) for p in (out/'events').glob('*.request.json')]
    assert len(requests)<=60 and len({r['job']['key'] for r in requests})==len(requests)
    assert set(records)<=set(jobs)
    expected_profiles=Counter();solves=0;errors=0
    for key,r in records.items():
        j=jobs[key];e=eqs[j['equation_id']];m=e.metadata
        assert r['job']==j and r['plan_sha256']==ph
        assert m['expression_hash']==r['expression_hash']==j['expression_hash']
        assert m['family_id']==r['family_id'] and m['split']==r['split']=='train'
        assert r['config']==j['config'] and r['seed']==j['config']['seed']
        expected_profiles[j['profile']]+=1;solves+=bool(r.get('solver_called'))
        if r['status']!='completed':continue
        z=r['result'];grid=np.asarray(plan['target_x']);exact=e.y_exact(grid)
        np.testing.assert_array_equal(z['x'],grid)
        np.testing.assert_array_equal(r['y_exact'],exact)
        error=float(np.max(abs(np.asarray(z['f'])-exact)))
        assert error==r['E_grid'] and bool(error<=.01)==r['accuracy_met'];errors+=1
        assert z['completed_paths']==[j['config']['M']]*33
        assert z['n_normal_draws']==z['n_path_steps']==j['numerical_work']
        assert z['T']==j['expected_T'] and z['n_steps']==j['expected_steps']
        assert z['backend']=='numba' and z['compilation_seconds']==z['numba_import_seconds']==0.
        assert r['method_wall_seconds']>z['runtime_seconds']>0
    summary=read(out/'analysis/summary.json');cells=table(out/'analysis/errors.csv')
    assert len(cells)==15
    for c in cells:
        rs=[r for r in records.values() if r['equation_id']==c['equation_id'] and r['profile']==c['profile']]
        good=[r for r in rs if r['status']=='completed'];k=sum(r['accuracy_met'] for r in good);n=len(rs)
        assert k==int(c['accuracy_pass']) and len(good)==int(c['completed'])
        lo=float(beta.ppf(.025,k,n-k+1)) if k else 0.
        hi=float(beta.ppf(.975,k+1,n-k)) if k<n else 1.
        assert abs(lo-float(c['probability_lower95']))<1e-14 and abs(hi-float(c['probability_upper95']))<1e-14
        if good:
            assert float(np.median([r['E_grid'] for r in good]))==float(c['error_median'])
            assert float(np.median([r['method_wall_seconds'] for r in good]))==float(c['wall_median'])
    assert choose_profile(summary['profiles'],plan['gates'],len(records)==60)==summary['decision']
    for value in read(out/'analysis/aggregates.json').values():
        assert value['independent_aggregate_replicates']==1 and not value['reliability95_confirmed']
    guardians=[read(p) for p in (out/'control').rglob('*.guardian.json')]
    assert guardians and all(g['finished'] and g['cleanup_success'] for g in guardians)
    assert read(out/'active.json')['finished']
    locks={n:lock_is_held(ROOT/'results'/n) for n in ('.chernov_batch_controller.lock','.chernov_batch_compute.lock')}
    assert not any(locks.values()),locks
    before=read(out/'protected_before.json')
    changed=[n for n,h in before.items() if not (ROOT/n).is_file() or sha(ROOT/n)!=h]
    assert set(changed)<= {'PROJECT_STATUS.md'},changed
    result=dict(planned=60,physical_requests=len(requests),terminal=len(records),confirmed_solver_calls=solves,
        independent_exact_error_checks=errors,profiles=dict(expected_profiles),frozen_jobs_equal=True,
        sources_and_data_unchanged=True,statistical_and_gate_checks=True,mean4_scope_separate=True,
        all_guardians_finished=True,locks_held=locks,protected_paths=len(before),changed=changed,
        new_R_calls=0,new_stage2_calls=0,new_model_fits=0,verification_solver_calls=0,
        result_sha256={p.name:sha(p) for p in (out/'records').glob('*.json')})
    with (out/'independent_verification.json').open('x') as stream:
        json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='result_sha256'},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    verify(ap.parse_args().out.resolve())
