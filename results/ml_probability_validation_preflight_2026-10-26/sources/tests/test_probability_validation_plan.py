"""Planning checks on saved inputs only: no solver calls or model fitting."""
import ast,math,statistics
from collections import Counter
from diagnostics.plan_probability_validation import ROOT,MODEL,CONTRACT,read,budget_proposal,build_jobs


def saved_inputs():
    design=read(MODEL/'VALIDATION_DESIGN.json');cases=[]
    calibration=read(CONTRACT/'pilot_p2_2026-10-26/analysis/summary.json')['calibration']
    for c in design['cases']:
        rs=[read(ROOT/f"results/mc1000_x0_2026-10-25/attempts/{c['equation_id']}_seed{i}.json") for i in range(4)]
        mean=statistics.mean(r['result']['runtime_seconds'] for r in rs)
        cases.append(dict(c,T=rs[0]['result']['T'],G=rs[0]['forcing_bound']['abs_upper_bound'],N=rs[0]['result']['n_steps'],x0_mean_solver_seconds=mean,MC1_scenario_one_call_seconds=mean*132*calibration['coverage20']['median']))
    rows=read(MODEL/'full_cost_targets.json');r={p:statistics.median(x['seconds'] for x in rows if x['profile']==p) for p in ('R128','R256')}
    budget=budget_proposal(cases,r['R128'],r['R256'],calibration)
    return cases,budget,read(CONTRACT/'plan.json')


def test_scope96_and_all_authorization_flags_remain_inert():
    cases,budget,contract=saved_inputs();jobs=build_jobs(cases,contract,budget)
    assert len(jobs)==len({j['key'] for j in jobs})==96
    assert Counter(j['profile'] for j in jobs)=={'R128':32,'R256':32,'MC1':32}
    assert {j['equation_id'] for j in jobs}=={c['equation_id'] for c in cases}
    assert all(j['split']=='validation' and not j['execution_authorized'] for j in jobs)
    assert budget['approved'] is False and budget['approval_evidence'] is None


def test_work_limits_and_profiles_are_exact_not_default_path_budget():
    cases,budget,contract=saved_inputs();jobs=build_jobs(cases,contract,budget)
    for j in jobs:
        cfg=j['config']
        if j['profile']=='MC1':
            assert cfg['M']==4096 and cfg['dt']==1/128 and cfg['T'] is None
            assert cfg['seed']==248800060100+j['episode']
            assert cfg['max_path_steps']==j['numerical_work']==33*4096*j['expected_steps']
            assert cfg['max_steps']>=j['expected_steps'] and cfg['max_runtime_seconds']==j['proposed_watchdog_seconds']
        else:
            assert cfg==dict(contract['profiles'][j['profile']]['config'],T=j['expected_T'])
            assert j['numerical_work']==cfg['n']*512


def test_cap_derived_from_all96_attempts_including_inaccurate_calls():
    cases,budget,contract=saved_inputs();jobs=build_jobs(cases,contract,budget)
    assert math.isclose(sum(j['scenario_seconds'] for j in jobs),budget['total_scenario_seconds'],abs_tol=1e-9)
    assert budget['proposed_series_cap_seconds']==math.ceil(2*budget['total_scenario_seconds']+300)
    assert not budget['confidence_interval'] and not budget['upper_bound']


def test_planner_cannot_call_compute_or_fit():
    tree=ast.parse((ROOT/'diagnostics/plan_probability_validation.py').read_text())
    calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
    assert not calls&{'solve','solve_monte_carlo','solve_monte_carlo_optimized','compile_backend','fit','run','worker'}
