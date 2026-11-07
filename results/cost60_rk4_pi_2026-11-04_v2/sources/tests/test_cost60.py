"""Selection, accuracy and noise-aware cost policy; no new ODE solves in tests."""
import copy
from collections import Counter
import json

import pytest

from diagnostics.cost60_common import (ROOT,DATA,SPLITS,OTHER_TEST,NS,METHODS,atomic_json,choose,
    select_metadata,read,robust_faster,saving,summarize_profile)
from diagnostics.report_cost60 import analyze


@pytest.fixture(scope='module')
def selection_inputs():
    meta=[json.loads(s) for s in (ROOT/DATA/'all_methods_equations_v1_metadata.jsonl').read_text().splitlines()]
    other=[json.loads(s) for s in (ROOT/OTHER_TEST).read_text().splitlines()]
    return meta,read(ROOT/SPLITS),other


def test_deterministic_development_selection_covers_20_cells_without_test(selection_inputs):
    a=select_metadata(*selection_inputs);b=select_metadata(*selection_inputs)
    assert [m['equation_id'] for m in a[0]]==[m['equation_id'] for m in b[0]]
    assert len(a[0])==len({m['family_id'] for m in a[0]})==len({m['structure_signature'] for m in a[0]})==60
    assert set(m['split'] for m in a[0])=={'train'}
    assert set(Counter((m['complexity_level'],m['regime']) for m in a[0]).values())=={3}
    assert a[3]['historical_test_selected']==0


def test_selection_ignores_exact_and_historical_solver_outcomes(selection_inputs):
    before=select_metadata(*selection_inputs)[1]
    metadata=copy.deepcopy(selection_inputs[0])
    for m in metadata:
        m['y0']=99999;m['v0']=-99999;m['y_exact_expr']='unavailable'
        m['serialized_ast']['y_shape']={'unread':'do not use exact AST'}
        m['validation']['max_abs']['y']=99999
        m['historical_error']=0 if m['split']=='test' else 1e99
    after=select_metadata(metadata,selection_inputs[1],selection_inputs[2])[1]
    assert before==after


@pytest.mark.parametrize('n',NS)
def test_33_target_nodes_are_exact_subset_without_interpolation(n):
    grid=[-1+2*i/n for i in range(n+1)]
    assert grid[::n//32]==[-1+i/16 for i in range(33)]


def profile_records(times,error=.01,usable=True):
    return [dict(equation_id='equation',family_id='family',method='rk4',N=32,
        solver_seconds=t,E33=error if usable else None,usable=usable,status='success' if usable else 'solver_failure',
        result=dict(trajectory='unchanged')) for t in times]


def test_median_not_minimum_and_equality_to_tolerance_is_accurate():
    p=summarize_profile(profile_records([10,100,12,11,13]))
    assert p['solver_median_seconds']==12 and p['solver_MAD_seconds']==1
    assert p['accurate'] and p['E33']==.01


def test_failure_or_changing_answer_cannot_be_a_feasible_configuration():
    p=summarize_profile(profile_records([1,1,1,1,1],usable=False))
    assert p['E33'] is None and not p['accurate']
    rr=profile_records([1,1,1,1,1]);rr[-1]['result']={'trajectory':'different'}
    assert not summarize_profile(rr)['accurate']


def test_choice_never_prefers_fast_inaccurate_result():
    fast=summarize_profile(profile_records([.1]*5,error=.010001))
    slow=summarize_profile(profile_records([1]*5,error=.01))
    assert choose([fast,slow])==slow and choose([fast]) is None


def test_small_timing_difference_or_overlapping_repeats_is_not_robust_win():
    slow=summarize_profile(profile_records([1]*5))
    tiny=summarize_profile(profile_records([.99]*5))
    overlap=summarize_profile(profile_records([.5,.5,.5,.5,2]))
    strong=summarize_profile(profile_records([.8]*5))
    assert not robust_faster(tiny,slow) and not robust_faster(overlap,slow)
    assert robust_faster(strong,slow)


def test_pooled_and_average_relative_gain_are_distinct():
    g=saving([1,100],[.5,90])
    assert g['mean_relative_saving_fraction']==pytest.approx(.3)
    assert g['pooled_saving_fraction']==pytest.approx(1-90.5/101)


def test_atomic_record_refuses_overwrite(tmp_path):
    p=tmp_path/'record.json';atomic_json(p,{'original':True})
    with pytest.raises(FileExistsError):atomic_json(p,{'original':False})
    assert read(p)=={'original':True}


def synthetic_records(kind):
    ids=[f'case{i:02}' for i in range(60)];records=[]
    for pid in ids:
        i=ids.index(pid)
        for method in METHODS:
            for n in NS:
                base=(.99 if i<30 else 1.01) if method=='product_integral' else 1.
                if kind=='strong':base=(.5 if i<30 else 1.) if method=='product_integral' else (1. if i<30 else .5)
                for episode in range(-1,5):
                    records.append(dict(sequence=len(records)+1,equation_id=pid,family_id=pid,method=method,N=n,
                        phase='warmup' if episode==-1 else 'measured',episode=episode,status='success',usable=True,
                        E33=.02 if kind=='none' and i==0 else .001,accuracy_met=not(kind=='none' and i==0),
                        solver_seconds=base*n/32,solver_nanoseconds=int(base*n/32*1e9),scoring_seconds=0,
                        result={'same':'deterministic'}))
    return records,ids


def test_raw_heterogeneity_from_one_percent_timing_differences_is_NO_GO():
    records,ids=synthetic_records('marginal');summary=analyze(records,ids)[-1]
    assert summary['raw_method_winners']=={'rk4':30,'product_integral':30}
    assert summary['recommendation']=='NO-GO'
    assert 'robust_each_method' in summary['failed_GO_gates']


def test_strong_diverse_accurate_cost_advantage_can_pass_preregistered_gate():
    records,ids=synthetic_records('strong');summary=analyze(records,ids)[-1]
    assert summary['recommendation']=='GO'
    assert summary['robust_method_winners']=={'product_integral':30,'rk4':30}


def test_no_accurate_configuration_is_explicit_and_not_a_winner():
    records,ids=synthetic_records('none');summary=analyze(records,ids)[-1]
    assert summary['no_configuration_meets_tolerance']==['case00']
    assert summary['best_constant'] is None and summary['recommendation']=='NO-GO'
