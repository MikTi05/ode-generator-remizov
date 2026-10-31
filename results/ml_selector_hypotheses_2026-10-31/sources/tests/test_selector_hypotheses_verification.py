"""Small independent-oracle checks for the study's artifact verifier."""
from __future__ import annotations

import pytest

from diagnostics.verify_selector_hypotheses import close, independent_metrics, verify


def test_independent_metrics_measures_actual_error_cost_of_false_mc():
    rows = [dict(E_R=.001,E_MC=.1,best_method='remizov'),
            dict(E_R=.2,E_MC=.01,best_method='monte_carlo'),
            dict(E_R=.003,E_MC=.003,best_method='remizov')]
    result = independent_metrics(rows,['monte_carlo','monte_carlo','remizov'])
    assert result['confusion_matrix'] == [[1,1],[0,1]]
    assert (result['mc_correct'],result['mc_false'],result['mc_missed']) == (1,1,0)
    assert result['mean_error'] == pytest.approx(.113/3)
    assert result['mean_regret'] == pytest.approx(.099/3)
    assert result['success_count'] == 2
    assert result['macro_f1'] == pytest.approx(2/3)


def test_verifier_comparison_rejects_changed_regret():
    with pytest.raises(AssertionError,match='regret'):
        close(dict(regret=.002),dict(regret=.001),'tampered')


def test_verifier_refuses_to_evaluate_before_final_stage(tmp_path):
    with pytest.raises(AssertionError,match='final test must complete'):
        verify(tmp_path)
