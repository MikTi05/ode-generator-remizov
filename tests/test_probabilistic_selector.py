"""Decision/count/leakage checks. Fixtures are never fitted as training data."""
import ast,json
from pathlib import Path
import numpy as np
import pytest
from ml.probabilistic_selector import choose,probability_scores,fit_head
ROOT=Path(__file__).resolve().parents[1]


def estimates():
    return dict(R256=dict(probability=dict(point=.98,low=.96,high=.99),cost=dict(point=20.,low=19.,high=21.,supported=True)),
        MC1=dict(probability=dict(point=.98,low=.96,high=.99),cost=dict(point=8.,low=7.,high=9.,supported=True)))


def test_MC_is_a_real_policy_action_without_winner_class_labels():
    e=estimates();assert choose(e,.95)['action']=='MC1' and choose(e,.95,robust=True)['action']=='MC1'


def test_low_MC_accuracy_overrides_speed():
    e=estimates();e['MC1']['probability']=dict(point=.2,low=.1,high=.3)
    assert choose(e,.8)['action']=='R256' and choose(e,.8,robust=True)['action']=='R256'


def test_probability_uncertainty_not_misreported_as_feasibility():
    e=estimates();e['MC1']['probability']=dict(point=.9,low=.6,high=.99)
    assert choose(e,.8)['action']=='MC1'
    assert choose(e,.8,robust=True)['action']=='abstain'


def test_uncertain_cost_and_sparse_R128_do_not_manufacture_MC_win():
    e=estimates();e['MC1']['cost']=dict(point=18.,low=8.,high=25.,supported=True)
    assert choose(e,.95,robust=True)['action']=='abstain'
    e=estimates();e['R128']=dict(probability=dict(point=.98,low=.96,high=.99),cost=dict(point=12.,low=10.,high=14.,supported=False))
    assert choose(e,.95,robust=True)['reason']=='potential_candidate_time_head_insufficient'


def test_point_10pct_tie_prefers_accurate_R():
    e=estimates();e['MC1']['cost']=dict(point=18.1,low=18.,high=18.2,supported=True)
    assert choose(e,.95)['action']=='R256'


@pytest.mark.parametrize('q',[0.,1.,float('nan')])
def test_caller_must_supply_valid_risk_threshold(q):
    with pytest.raises(ValueError):choose(estimates(),q)


def test_binomial_loss_matches_real_trial_loss_without_expanding_examples():
    score=probability_scores([2],[4],[.3])
    assert score['brier']==pytest.approx((2*(1-.3)**2+2*.3**2)/4)
    assert score['logloss']==pytest.approx(-.5*np.log(.3)-.5*np.log(.7))


@pytest.mark.parametrize('split',['validation','test'])
def test_head_fit_rejects_nontrain_before_any_fit(split):
    with pytest.raises(ValueError):fit_head(np.zeros((1,253)),[dict(split=split)],list(map(str,range(253))),'probability')


def test_new_ML_sources_never_call_numerical_solvers():
    for name in ['ml/probabilistic_evidence.py','ml/probabilistic_selector.py','diagnostics/train_probabilistic_selector.py']:
        tree=ast.parse((ROOT/name).read_text())
        calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
        assert not calls&{'solve','solve_monte_carlo','solve_monte_carlo_optimized','compile_backend'}
