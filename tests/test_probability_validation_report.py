"""Pure report-accounting fixtures only; no numerical/ML calls or training data."""
import pytest
from diagnostics.report_probability_validation import offline_oracle,policy_metrics


def case():
    def profile(t,ok):
        return dict(primary_status='completed',primary_accurate=ok,primary_violation=not ok,
            timing=dict(median=t,min=t,max=t,MAD_fraction=0.))
    c=dict(equation_id='report_fixture',profiles=dict(R128=profile(10.,True),R256=profile(20.,True),MC1=profile(5.,False)))
    c['offline_oracle']=offline_oracle(c);return c


def test_inaccurate_fast_MC_not_oracle_or_negative_regret():
    c=case();assert c['offline_oracle']['operational10pct_profile']=='R128'
    m=policy_metrics([c],['MC1'],{'report_fixture':.1})
    assert m['finite_accuracy_violations']==1 and m['accurate_primary']==0
    assert m['accuracy_constrained_regret']==[]


def test_abstain_pays_inference_and_is_not_success():
    m=policy_metrics([case()],['abstain'],{'report_fixture':.1})
    assert m['warm_policy_seconds']==.1 and m['delivery_accuracy_rate']==0 and m['coverage']==0
    assert m['conditional_accuracy'] is None


def test_oracle_regret_includes_overhead_and_reports_common_accuracy_subset():
    m=policy_metrics([case()],['R128'],{'report_fixture':.1},baseline='R256')
    assert m['accuracy_constrained_regret'][0]['seconds']==pytest.approx(.1)
    assert m['common_accurate_saved_seconds_sum']==pytest.approx(9.9)


def test_missing_profile_does_not_become_none_or_known_oracle():
    c=case();c['profiles']['MC1']['primary_status']='not_run'
    o=offline_oracle(c);assert o['operational10pct_profile']=='unknown' and o['minimum_accurate_cost_seconds'] is None


def test_10pct_operational_tie_distinct_from_minimum_cost():
    c=case();c['profiles']['MC1'].update(primary_accurate=True,primary_violation=False,timing=dict(median=9.5,min=9.5,max=9.5,MAD_fraction=0.))
    o=offline_oracle(c);assert o['fastest_accurate_cost_profile']=='MC1' and o['operational10pct_profile']=='R128'
