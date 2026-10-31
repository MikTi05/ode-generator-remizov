"""Math/accounting fixtures only; no ODE or fitting on fabricated research data."""
import ast,json
from pathlib import Path
import numpy as np
import pytest
from ml.nested_probability_calibration import (
    apply_sigmoid,calibration_objective,require_disjoint,family_weights,
    metric_summary,outer_blocks,baseline_default_choice,
)
ROOT=Path(__file__).resolve().parents[1]


def test_calibrator_identity_and_nonnegative_monotonicity():
    p=np.array([.01,.2,.8,.99])
    np.testing.assert_allclose(apply_sigmoid(dict(slope=1.,intercept=0.),p),p,rtol=1e-14)
    assert np.all(np.diff(apply_sigmoid(dict(slope=.4,intercept=-.1),p))>0)


def test_regularized_sigmoid_gradient_independent_finite_difference():
    z=np.array([-2.,.5,3.]);y=np.array([0.,1.,0.]);w=np.array([.5,.5,1.]);theta=np.array([.7,-.2])
    _,gradient=calibration_objective(theta,z,y,w)
    for i in range(2):
        step=np.eye(2)[i]*1e-6
        numeric=(calibration_objective(theta+step,z,y,w)[0]-calibration_objective(theta-step,z,y,w)[0])/2e-6
        assert gradient[i]==pytest.approx(numeric,rel=1e-8)


@pytest.mark.parametrize('split',['validation','test'])
def test_fit_scope_rejects_held_split_before_training(split):
    with pytest.raises(ValueError):require_disjoint([dict(split=split,family_id='f')],())


def test_outer_family_excluded_even_when_equation_id_differs():
    with pytest.raises(ValueError):require_disjoint([dict(split='train',family_id='held',equation_id='other_sample')],{'held'})


def test_family_scores_do_not_multiply_duplicate_sample_weight():
    a=dict(family_id='a',k=0,n=1);b=dict(family_id='b',k=1,n=1)
    one=metric_summary([a,b],[.8,.8])['family_weighted']
    repeated=metric_summary([a,a,b],[.8,.8,.8])['family_weighted']
    assert one==repeated
    np.testing.assert_array_equal(family_weights([a,a,b]),[.5,.5,1.])


def test_real_train_partition_covers103_and_preserves_old40():
    folder=ROOT/'results/ml_probability_selector_2026-10-26'
    rows=[r for r in json.loads((folder/'accuracy_targets.json').read_text()) if r['profile']=='R128']
    old=json.loads((folder/'family_cv.json').read_text())['folds'];blocks=outer_blocks(rows,old)
    assigned=[g for b in blocks for g in b['held_families']]
    assert len(assigned)==len(set(assigned))==90 and len(rows)==103
    assert all(blocks[i]['held_families']==old[i]['held_families'] for i in range(3))


def test_fallback_is_not_a_claim_of_probability_feasibility():
    e=dict(R256=dict(probability=dict(point=.1),cost=dict(point=20.)),
           R128=dict(probability=dict(point=.9),cost=dict(point=10.)),
           MC1=dict(probability=dict(point=.01),cost=dict(point=5.)))
    assert baseline_default_choice(e,.95)=='R256'
    assert baseline_default_choice(e,.8)=='R128'
    assert e['R256']['probability']['point']==.1


def test_MC_remains_real_switch_candidate_and_10pct_tie_is_preserved():
    e=dict(R256=dict(probability=dict(point=.1),cost=dict(point=20.)),
           MC1=dict(probability=dict(point=.98),cost=dict(point=8.)))
    assert baseline_default_choice(e,.95)=='MC1'
    e['MC1']['cost']['point']=19.
    assert baseline_default_choice(e,.95)=='R256'


def test_training_source_never_loads_mixed_split_inputs_or_calls_solver():
    tree=ast.parse((ROOT/'diagnostics/train_nested_probability_calibration.py').read_text())
    calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for n in ast.walk(tree)
           if isinstance(n,ast.Call) and isinstance(n.func,(ast.Attribute,ast.Name))}
    assert not calls&{'load_inputs','load_all_methods_equations','solve','solve_monte_carlo_optimized','compile_backend'}
