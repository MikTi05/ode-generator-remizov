"""Fixed proposal and exact reliability rules; no numerical solver calls."""
import json
from pathlib import Path
import pytest
from diagnostics.final_grid5_stats import reliability
from diagnostics.final_grid5 import IDS


def test_four_observed_successes_do_not_prove_p95():
    r=reliability(4)
    assert not r['confirm_p95'] and not r['reject_p95']
    assert r['exact_family95_probability_interval'][0]==pytest.approx(.005**.25)


def test_predeclared_reliability_futility_threshold():
    assert reliability(0)['pvalue']==pytest.approx(.05**4)
    assert reliability(1)['pvalue']==pytest.approx(4*.95*.05**3+.05**4)
    assert reliability(0)['reject_p95'] and reliability(1)['reject_p95']
    assert not reliability(2)['reject_p95']


def test_invalid_probability_counts_rejected():
    with pytest.raises(ValueError):reliability(5)


def test_only_five_train_tasks_and_fresh_fixed_profiles():
    root=Path(__file__).resolve().parents[1];p=root/'results/final_grid5_2026-10-25/plan.json'
    if not p.exists():return
    plan=json.loads(p.read_text());assert tuple(t['problem_id'] for t in plan['tasks'])==IDS
    assert len(plan['jobs'])==25 and len(set(plan['seeds']))==4
    assert plan['resources']['series_seconds']==1200 and plan['resources']['rss_mib']==512
    assert all(t['split']=='train' for t in plan['tasks'])
    for j in plan['jobs']:
        if j['method']=='R':assert j['config']['n']==256 and j['config']['num_t']==513 and j['config']['boundary']=='zero'
        else:
            assert j['config']['M']==4096 and j['config']['chunk']==1024 and j['config']['dt']==1/128
            assert j['config']['T'] is None and j['config']['tail_eps']==1e-4
