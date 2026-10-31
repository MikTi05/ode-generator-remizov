"""Accounting/statistical checks only. Fixtures are never fitted as examples."""
import ast
from pathlib import Path
import pytest
from diagnostics.audit_probability_selector import (
    cp_interval, fixed_bins, policy_audit, probability_audit, zero_failure_required,
)


def row(id='fixture',k=1,n=1,p=.99,t=10.):
    return dict(equation_id=id,family_id=id,observed={'R256':dict(k=k,n=n)},observed_cost={'R256':t},
        baseline_predictions={'R256:probability':.8,'R256:cost':12.},
        estimates={'R256':dict(probability=dict(point=p,low=min(.9,p),high=max(.99,p)),
                               cost=dict(point=12.,low=11.,high=13.,supported=True))})


def test_cost_per_success_charges_failure_and_abstention():
    rows=[row('a'),row('b',k=0,t=20.),row('c')]
    z=policy_audit(rows,['R256','R256','abstain'],{'a':1.,'b':1.,'c':1.})
    assert z['accurate_deliveries']==1 and z['violations']==1
    assert z['seconds_per_accurate_delivery']==33. and z['abstention_rate']==pytest.approx(1/3)


def test_zero_delivery_cost_ratio_is_undefined():
    z=policy_audit([row()],['abstain'],{'fixture':.1})
    assert z['warm_cost']==.1 and z['seconds_per_accurate_delivery'] is None


def test_missing_cost_is_not_imputed_by_prediction():
    z=policy_audit([row(t=None)],['R256'])
    assert z['warm_cost'] is None and z['seconds_per_accurate_delivery'] is None


def test_binomial_brier_scores_trials_not_squared_fraction():
    z=probability_audit([row(k=2,n=4,p=.3)],'R256')
    assert z['scores']['brier']==pytest.approx(.5*.7**2+.5*.3**2)
    assert z['scores']['brier']!=pytest.approx((.5-.3)**2)


def test_fixed_bins_include_probability_one_once():
    bins=fixed_bins([row(p=1.)],'R256')
    assert len(bins)==1 and bins[0]['equations']==1 and bins[0]['bin']==[.95,1.]


def test_four_successes_cannot_certify_95pct_new_seed_probability():
    assert cp_interval(4,4)[0]==pytest.approx(.3976353643835253)
    assert zero_failure_required()==59


def test_tail_empty_is_unknown_not_zero_calibration_error():
    z=probability_audit([row(p=.1)],'R256')
    assert z['tail']['0.95']['equations']==0
    assert z['tail']['0.95']['observed_rate'] is None


@pytest.mark.parametrize('filename',['audit_probability_selector.py','report_probability_selector_audit.py'])
def test_auditor_never_calls_fit_solver_or_prediction_APIs(filename):
    path=Path(__file__).resolve().parents[1]/'diagnostics'/filename
    tree=ast.parse(path.read_text())
    calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id
           for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
    assert not calls & {'fit','fit_head','train','solve','solve_monte_carlo_optimized','select_method','predict','predict_proba','load_inputs'}


def test_ranking_ties_and_reversed_positive_not_hidden_by_mean_calibration():
    from diagnostics.report_probability_selector_audit import ranking_auc
    assert ranking_auc([.99,.99,.1],[True,False,True])==.25
    assert ranking_auc([.8,.8],[True,False])==.5
    assert ranking_auc([.8,.9],[False,False]) is None
