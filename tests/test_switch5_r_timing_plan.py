"""Read saved train references and test inert40-call scope; never run solvers."""
import ast,math,statistics
from collections import Counter
from diagnostics.plan_switch5_r_timing import ROOT,NESTED,CONTRACT,IDS,read,budget,build_jobs


def inputs():
    targets=read(NESTED/'accuracy_targets.json');costs=read(NESTED/'full_cost_targets.json');cases=[];overhead=[]
    for pid in IDS:
        references={}
        for p in ('R128','R256'):
            t=next(r for r in targets if r['equation_id']==pid and r['profile']==p);path=t['sources'][0];r=read(ROOT/path)
            references[p]=dict(source=path,source_sha256='unit_not_a_real_file_pin',E_grid=t['errors'][0],accurate=bool(t['k']),f_float64_sha256='unit')
            if p=='R128':native=r['result']['runtime_seconds'];T=r['result']['T'];family=t['family_id'];expression=t['expression_hash']
        full=next(r['seconds'] for r in costs if r['equation_id']==pid and r['profile']=='R256')
        cases.append(dict(equation_id=pid,family_id=family,split='train',expression_hash=expression,T=T,G=1.,references=references,
            historical_R128_native_seconds=native,historical_R256_full_seconds=full))
    for c in costs:
        if c['profile']=='R128':
            rs=[read(ROOT/n) for n in c['sources']]
            overhead.append(statistics.median(r['method_wall_seconds']-r['result']['runtime_seconds'] for r in rs))
    b=budget(cases,overhead,[c['seconds'] for c in costs if c['profile']=='R128'])
    return cases,b,read(CONTRACT/'plan.json')


def test_manifest40_only_frozen5_train_and_no_authorization():
    cases,b,c=inputs();jobs=build_jobs(cases,c,b)
    assert len(jobs)==len({j['key'] for j in jobs})==40
    assert Counter(j['profile'] for j in jobs)=={'R128':20,'R256':20}
    assert {j['equation_id'] for j in jobs}==set(IDS)
    assert all(j['split']=='train' and j['seed'] is None and not j['execution_authorized'] for j in jobs)
    assert not b['approved'] and b['approval_evidence'] is None and b['MC_calls']==0


def test_alternation_balances_order_within_every_case():
    cases,b,c=inputs();jobs=build_jobs(cases,c,b)
    first=[]
    for i in range(0,40,2):
        a,z=jobs[i:i+2]
        assert a['equation_id']==z['equation_id'] and a['episode']==z['episode']
        first.append((a['equation_id'],a['profile']))
    assert all(first.count((pid,'R128'))==first.count((pid,'R256'))==2 for pid in IDS)


def test_known_inaccurate_case_is_preserved_for_all8_calls():
    cases,b,c=inputs();jobs=build_jobs(cases,c,b)
    failed=[j for j in jobs if not j['expected_accurate']]
    assert len(failed)==8 and {j['equation_id'] for j in failed}=={'AMV1_F0172_S03'}


def test_total_work_and_configs_are_fixed_by_contract():
    cases,b,c=inputs();jobs=build_jobs(cases,c,b)
    assert sum(j['numerical_work'] for j in jobs)==3932160
    for j in jobs:
        assert j['config']==dict(c['profiles'][j['profile']]['config'],T=j['expected_T'])
        assert j['config']['n_jobs']==1 and j['config']['boundary']=='zero'


def test_budget_includes_inaccurate_jobs_and_never_imputes_observed_R128_full_cost():
    cases,b,c=inputs();jobs=build_jobs(cases,c,b)
    assert math.isclose(sum(j['scenario_seconds'] for j in jobs),b['central_scenario_seconds'],abs_tol=1e-9)
    assert b['proposed_cap_seconds']==math.ceil(2*b['central_scenario_seconds']+300)
    assert b['includes_all_inaccurate_calls'] and b['old_native_times_are_not_observed_full_cost']
    assert not b['upper_bound'] and not b['confidence_interval']


def test_planner_has_no_numerical_ML_or_worker_launch_calls():
    tree=ast.parse((ROOT/'diagnostics/plan_switch5_r_timing.py').read_text())
    calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for n in ast.walk(tree)
           if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
    assert not calls&{'solve','solve_streaming','solve_monte_carlo_optimized','compile_backend','fit','fit_head','worker','new_control','GuardedProcess'}


def test_older_references_have_frozen_per_file_pins_even_without_a_folder_manifest():
    from diagnostics.plan_switch5_r_timing import MODEL,sha
    pins=read(MODEL/'source_sha256.json');targets=read(NESTED/'accuracy_targets.json')
    for pid in IDS:
        for p in ('R128','R256'):
            target=next(r for r in targets if r['equation_id']==pid and r['profile']==p)
            assert all(sha(ROOT/path)==pins[path] for path in target['sources'])
