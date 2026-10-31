"""Independent saved-only P2 verification; never dispatches numerical work."""
import argparse,csv,hashlib,json,sys,struct
from datetime import datetime
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(out):
    import numpy as np
    from scipy.stats import beta
    from all_methods_dataset_loader import load_all_methods_equations
    from diagnostics.chernov_batch_control import lock_is_held
    contract=ROOT/'reports/ml_selector_contract';plan=read(out/'plan.json');ph=sha(out/'plan.json')
    source=[json.loads(s) for s in (contract/'pilot_jobs.jsonl').read_text().splitlines()]
    expected={j['key']:j for j in source if j['stage']==2 and j['profile']=='MC1'}
    assert {j['key']:j for j in plan['jobs']}==expected and len(expected)==140
    assert plan['total_new_calls']==140 and plan['execution_authorized'] and plan['no_R'] and plan['no_ML'] and plan['no_benchmark1000']
    assert sha(out/'AUTHORIZATION.txt')==plan['authorization_sha256']
    for name,h in plan['protected_sha256'].items():assert sha(ROOT/name)==h,name
    p1=contract/'sentinel_p1_2026-10-26'
    for name,h in read(p1/'artifact_sha256.json').items():assert sha(p1/name)==h,name
    requests=[read(p) for p in (out/'events').glob('*.request.json')]
    assert len(requests)==140 and {r['job']['key'] for r in requests}==set(expected)
    for r in requests:assert r['job']==expected[r['job']['key']] and r['plan_sha256']==ph
    data=ROOT/plan['data_dir'];eqs={e.equation_id:e for e in load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl')}
    records={p.stem:read(p) for p in (out/'records').glob('*.json')};assert set(records)==set(expected)
    work=0
    for key,j in expected.items():
        r=records[key];e=eqs[j['equation_id']];m=e.metadata;z=r['result']
        assert r['status']=='completed' and r['solver_called'] and z['success']
        assert z['problem_id']==j['equation_id']
        assert r['job']==j and r['config']==j['config'] and r['plan_sha256']==ph
        assert r['expression_hash']==m['expression_hash']==j['expression_hash']
        assert r['family_id']==m['family_id']==j['family_id'] and m['split']==r['split']=='train'
        assert r['seed']==z['seed_entropy']==248800020100+j['episode']
        x=np.asarray([-1+i/16 for i in range(33)]);exact=np.asarray(e.y_exact(x));np.testing.assert_array_equal(z['x'],x);np.testing.assert_array_equal(r['y_exact'],exact)
        assert np.isfinite(z['f']).all() and np.isfinite(z['stderr']).all()
        error=float(np.max(np.abs(np.asarray(z['f'])-exact)));assert error==r['E_grid'] and bool(error<=.01)==r['accuracy_met']
        assert z['M']==4096 and z['completed_paths']==[4096]*33 and z['dt_requested']==1/128 and z['dt_actual']<=1/128
        assert z['T']==j['expected_T'] and z['n_steps']==j['expected_steps']
        assert z['n_path_steps']==z['n_normal_draws']==j['numerical_work'];work+=z['n_path_steps']
        assert z['forcing_abs_upper_bound']==j['forcing_G'] and z['tail_tolerance_met']
        assert z['backend']=='numba' and z['compilation_seconds']==z['numba_import_seconds']==0
    assert work==plan['total_path_updates']==59047329792
    power=read(out/'hardware_power_before.json')
    assert power['before_first_P2_call'] and datetime.fromisoformat(power['captured_utc']).timestamp()<min(datetime.fromisoformat(r['started_utc']).timestamp() for r in records.values())
    budget=read(out/'budget.json');assert budget['deadline_epoch']-budget['started_epoch']==18599
    assert max(datetime.fromisoformat(r['finished_utc']).timestamp() for r in records.values())<budget['deadline_epoch']
    assert max(r['peak_rss_mib'] for r in records.values())<=512
    warmups=[read(p) for p in (out/'workers').glob('*.warmup.json')]
    assert len(warmups)==1 and all(w['numba_threads']==1 and w['no_MC_calls_in_warmup'] for w in warmups)
    thermal=[json.loads(s) for s in (out/'thermal.jsonl').read_text().splitlines()]
    assert thermal and all(t['thermal_state'] is None or t['thermal_state']<2 for t in thermal)
    combined=list(records.values())+[read(p) for p in (p1/'records').glob('*.json') if read(p)['profile']=='MC1']
    assert len(combined)==160 and len({r['equation_id'] for r in combined})==40
    streams=set()
    for r in combined:
        expected_keys=[]
        for x in [-1+i/16 for i in range(33)]:
            address=b'mc_point_v1\0'+r['equation_id'].encode('utf-8')+struct.pack('<d',0. if x==0 else x)
            key=[*struct.unpack('<8I',hashlib.sha256(address).digest()),0]
            expected_keys.append(key);stream=(r['seed'],tuple(key));assert stream not in streams;streams.add(stream)
        assert r['result']['point_spawn_keys']==expected_keys
    assert len(streams)==5280
    summary=read(out/'analysis/summary.json');assert summary['groups']['combined']['accurate_calls']==sum(r['accuracy_met'] for r in combined)
    with (out/'analysis/equations.csv').open() as f:cells=list(csv.DictReader(f))
    assert len(cells)==40
    for c in cells:
        rs=[r for r in combined if r['equation_id']==c['equation_id']];assert len(rs)==4
        k=sum(r['accuracy_met'] for r in rs);assert k==int(c['accuracy_pass'])
        lo=float(beta.ppf(.025,k,5-k)) if k else 0.;hi=float(beta.ppf(.975,k+1,4-k)) if k<4 else 1.
        assert abs(lo-float(c['probability_lower95']))<1e-14 and abs(hi-float(c['probability_upper95']))<1e-14
        assert float(c['error_median'])==float(np.median([r['E_grid'] for r in rs]))
        assert float(c['wall_median'])==float(np.median([r['method_wall_seconds'] for r in rs]))
    guardians=[read(p) for p in (out/'control').rglob('*.guardian.json')]
    assert guardians and all(g['finished'] and g['cleanup_success'] for g in guardians)
    assert read(out/'active.json')['finished'] and not read(out/'pending.json')['keys']
    assert not any(lock_is_held(ROOT/'results'/n) for n in ['.chernov_batch_controller.lock','.chernov_batch_compute.lock'])
    before=read(out/'protected_before.json');changed=[n for n,h in before.items() if not (ROOT/n).is_file() or sha(ROOT/n)!=h]
    assert set(changed)<= {'PROJECT_STATUS.md'},changed
    result=dict(passed=True,planned_P2=140,physical_requests=140,confirmed_P2_solver_calls=140,completed=140,independent_exact_checks=140,combined_MC1_calls=160,combined_equations=40,path_updates=work,P1_all_artifacts_unchanged=True,source_and_dataset_pins_valid=True,frozen_jobs_equal=True,independent_rng_streams_verified=5280,CP_intervals_verified=True,guardians_finished=True,locks_free=True,protected_paths=len(before),changed_existing=changed,resource_and_deadline_checks=True,worker_startups=1,verification_solver_calls=0,new_R_calls=0,new_model_fits=0,new_benchmark1000_calls=0,result_sha256={p.name:sha(p) for p in (out/'records').glob('*.json')})
    with (out/'independent_verification.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='result_sha256'},indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);verify(ap.parse_args().out.resolve())
