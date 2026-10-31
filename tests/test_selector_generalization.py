"""Saved-only accounting and mathematical bound checks; no fabricated ML fit."""
import ast
from pathlib import Path
import numpy as np
import pytest
from diagnostics.evaluate_selector_generalization import leaf_box_bound,policy_metrics,experiment_budget


@pytest.mark.parametrize('scale',[1.,-2.])
def test_leaf_box_bound_encloses_all_routed_combinations(scale):
    values=np.array([-.2,.4,-1.,.3]);b=leaf_box_bound(values,[2,2],scale,.2)
    from scipy.special import expit
    truth=[float(expit(scale*(a+c)+.2)) for a in values[:2] for c in values[2:]]
    assert b['probability_lower']<=min(truth)<=max(truth)<=b['probability_upper']


def case(pid='f',good=True):
    def profile(t,ok):return dict(primary_accurate=ok,timing=dict(median=t))
    return dict(equation_id=pid,profiles=dict(R128=profile(10.,good),R256=profile(20.,good),MC1=profile(5.,False)))


def test_equal_method_selection_has_no_cost_saving_after_inference():
    a=policy_metrics([case()],['R256'],.1);b=policy_metrics([case()],['R256'],0.)
    assert a['correct']==b['correct']==1 and a['method_cost_seconds']==b['method_cost_seconds']
    assert a['warm_cost_seconds']>b['warm_cost_seconds'] and a['fastest_accurate_choices']==0


def test_infeasible_case_cannot_become_accurate_or_fastest_winner():
    z=policy_metrics([case(good=False)],['MC1'],0.)
    assert z['correct']==z['observed_feasible_cases']==z['fastest_accurate_choices']==0
    assert z['seconds_per_correct'] is None and z['violations']==1


def test_abstention_has_no_success_and_pays_inference():
    z=policy_metrics([case()],['abstain'],.1)
    assert z['coverage']==0 and z['seconds_per_correct'] is None and z['warm_cost_seconds']==.1


def test_prospective_budget_counts_all_three_profiles_and_is_not_permission():
    b=experiment_budget(8,12.,24.,[10.,30.])
    assert b['maximum_calls']==96 and b['central_scenario_seconds']==32*(12+24+20)
    assert not b['approved'] and not b['execution_authorized'] and b['not_upper_bound']


def test_evaluator_never_calls_fitting_solver_or_generator():
    p=Path(__file__).resolve().parents[1]/'diagnostics/evaluate_selector_generalization.py'
    tree=ast.parse(p.read_text())
    calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
    assert not calls&{'fit','fit_head','fit_sigmoid','solve','solve_monte_carlo_optimized','compile_backend','generate_dataset','load_inputs'}
