"""Independent saved-only audit; neither solvers nor ML fitting are invoked."""
from __future__ import annotations
import argparse,json,math,statistics,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnostics.mc_pilot_p2 import read_json,write_json,digest,snapshot
from diagnostics.chernov_batch_control import lock_is_held
MODEL=ROOT/'results/ml_probability_selector_2026-10-26';SOURCE=ROOT/'results/ml_probability_validation_preflight_2026-10-26'


def verify(out):
    import numpy as np
    from all_methods_dataset_loader import load_all_methods_equations
    plan=read_json(out/'plan.json');analysis=read_json(out/'analysis.json');cases=read_json(out/'cases.json');ph=digest(out/'plan.json')
    assert plan['jobs']==read_json(SOURCE/'plan.json')['jobs'] and plan['resources']['series_seconds']==5691
    assert digest(MODEL/'registry.json')==plan['registry_sha256'] and digest(out/'frozen_predictions.json')==plan['frozen_predictions_sha256']
    for n,h in plan['protected_sha256'].items():assert digest(ROOT/n)==h,n
    data=ROOT/plan['data_dir'];ids={c['equation_id'] for c in plan['cases']}
    eqs={e.equation_id:e for e in load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl') if e.equation_id in ids}
    expected={j['key']:j for j in plan['jobs']};records={p.stem:read_json(p) for p in (out/'records').glob('*.json')}
    requests=[read_json(p) for p in (out/'events').glob('*.request.json')]
    assert len(requests)==len({r['job']['key'] for r in requests})<=96 and set(records)<=set(expected)
    assert all(r['plan_sha256']==ph and r['job']==expected[r['job']['key']] for r in requests)
    bycase={pid:{p:[] for p in ('R128','R256','MC1')} for pid in ids};errors=0
    for key,r in records.items():
        j=expected[key];pid=j['equation_id'];e=eqs[pid];z=r.get('result') or {}
        assert r['job']==j and r['plan_sha256']==ph and r['split']==e.metadata['split']=='validation'
        assert r['expression_hash']==e.metadata['expression_hash'] and r['config']==j['config']
        if r['status']=='completed':
            exact=np.asarray(e.y_exact(np.asarray(plan['target_x'])));error=float(np.max(abs(np.asarray(z['f'])-exact)))
            assert np.array_equal(r['y_exact'],exact) and np.array_equal(z['x'],plan['target_x'])
            assert error==r['E_grid'] and (error<=.01)==r['accuracy_met'];errors+=1
            assert z['T']==j['expected_T'] and z['success']
            if j['profile']=='MC1':
                assert r['seed']==248800060100+j['episode']==z['seed_entropy']
                assert z['n_path_steps']==j['numerical_work'] and z['completed_paths']==[4096]*33 and z['compilation_seconds']==0
            else:
                assert z['n']==j['config']['n'] and z['n_operator_applications']==j['numerical_work']
        if r.get('method_wall_seconds') is not None:
            assert math.isclose((r['timer_stop_ns']-r['timer_start_ns'])*1e-9,r['method_wall_seconds'],rel_tol=0,abs_tol=1e-12)
        bycase[pid][j['profile']].append(r)
    for p in ('R128','R256','MC1'):
        own=[r for r in records.values() if r['profile']==p];table=analysis['profile_outcomes'][p]
        assert table['terminal']==len(own) and table['accurate_calls']==sum(r['status']=='completed' and r['accuracy_met'] for r in own)
        assert table['finite_accuracy_violations']==sum(r['status']=='completed' and not r['accuracy_met'] for r in own)
    for case in cases:
        for p,rs in bycase[case['equation_id']].items():
            own=case['profiles'][p];zero=next((r for r in rs if r['job']['episode']==0),None)
            assert own['primary_accurate']==bool(zero and zero['status']=='completed' and zero['accuracy_met'])
            assert own['primary_violation']==bool(zero and zero['status']=='completed' and not zero['accuracy_met'])
            times=[r['method_wall_seconds'] for r in rs if r.get('method_wall_seconds') is not None]
            if own['timing']:assert len(times)==4 and own['timing']['median']==statistics.median(times)
            if p!='MC1':
                fs=[r['result']['f'] for r in rs if r['status']=='completed'];assert all(f==fs[0] for f in fs) if fs else True
    frozen={r['equation_id']:r for r in read_json(out/'frozen_predictions.json')['rows']}
    overhead={r['equation_id']:r['feature_and_inference_seconds'] for r in read_json(out/'policy_overhead.json')['rows']}
    def check_policy(metric,choices,overheads):
        correct=violations=selected=0;cost=0.;known=True
        for case,choice in zip(cases,choices):
            cost+=overheads.get(case['equation_id'],0.)
            if choice=='abstain':continue
            selected+=1;own=case['profiles'][choice];correct+=own['primary_accurate'];violations+=own['primary_violation']
            if own['timing']:cost+=own['timing']['median']
            else:known=False
        assert metric['accurate_primary']==correct and metric['finite_accuracy_violations']==violations and metric['selected']==selected
        assert metric['delivery_accuracy_rate']==correct/8 and metric['actions']==dict(Counter(choices))
        if known:assert math.isclose(metric['warm_policy_seconds'],cost,rel_tol=0,abs_tol=1e-9)
        else:assert metric['warm_policy_seconds'] is None
    for q in (.5,.8,.95):
        for mode in ('point','model_spread'):
            choices=[frozen[c['equation_id']]['diagnostic_choices'][str(q)][mode]['action'] for c in cases]
            check_policy(analysis['policies'][str(q)][mode],choices,overhead)
    for p in ('R256','R128'):check_policy(analysis['baselines'][p],[p]*8,{})
    check_policy(analysis['policies']['production_research_guard'],['abstain']*8,overhead)
    if analysis['status']=='complete':
        assert len(records)==len(requests)==96 and not read_json(out/'pending.json')['keys']
        assert len({r['worker_pid'] for r in records.values()})==len({r['session'] for r in records.values()})==1
    sessions=[read_json(p) for p in (out/'sessions').glob('*.json')]
    assert sum(s['wall_seconds'] for s in sessions)<=5691
    assert all(not lock_is_held(ROOT/'results'/n) for n in ('.chernov_batch_controller.lock','.chernov_batch_compute.lock'))
    before=read_json(out/'protected_before.json');after=snapshot(out)
    changed={n for n,h in before.items() if after.get(n)!=h};assert changed<= {'PROJECT_STATUS.md'},changed
    write_json(out/'independent_verification.json',dict(passed=True,records=len(records),requests=len(requests),canonical_errors_recomputed=errors,
        frozen_models_policy_predictions_unchanged=True,all_policy_accuracy_and_costs_recomputed=True,
        approved_wall_cap_seconds=5691,maximum_calls=96,old_paths_checked=len(before),old_paths_changed=sorted(changed),
        numerical_calls_in_verifier=0,model_fits=0,test_used=False,benchmark1000_calls=0),exclusive=True)
    print(dict(passed=True,records=len(records),requests=len(requests),errors_verified=errors,previous_paths=len(before)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);verify(ap.parse_args().out.resolve())
