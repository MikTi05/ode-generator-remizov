"""Independent small arithmetic oracles and fail-closed release validation."""
import copy
import json
import math
from pathlib import Path
import pytest

from diagnostics.finalize_mc33 import (ROOT,strict_json,aggregate,validate_mc,
    Reader,stats)
from diagnostics.verify_mc33_final import scalar_stats


@pytest.mark.parametrize('value',['{"x":NaN}','{"x":Infinity}','{"x":-Infinity}',
                                 '{"x":1e400}','{"x":1,"x":2}'])
def test_strict_json_rejects_nonfinite_and_duplicate_keys(value):
    with pytest.raises(ValueError):strict_json(value)


def test_mean_function_error_is_not_mean_error_and_primary_is_preserved():
    exact=[0.]*33
    fs=[[sign]*33 for sign in (1.,-1.,2.,-2.)]
    mean,error,e0=aggregate(fs,exact)
    assert error==e0==0 and mean.tolist()==exact
    assert sum(max(map(abs,f)) for f in fs)/4==1.5
    assert max(map(abs,fs[0]))==1.  # the primary seed is still a distinct answer


def test_scalar_and_vector_statistics_agree():
    values=[0.,.01,.05,.051,.2,.9,2.]
    a,b=stats(values),scalar_stats(values)
    assert a['pass_005']==3 and a['pass_001']==2
    for key in b:assert math.isclose(a[key],b[key],rel_tol=2e-14,abs_tol=1e-15)


@pytest.fixture
def production_record():
    p=ROOT/'results/mc33_parallel_2026-11-05/production/records/AMV1_F0000_S00_MC1_episode0.json'
    return json.loads(p.read_text())['record']


def test_raw_record_validates_and_detects_identity_errors(production_record):
    r=production_record;j=r['job'];args=(j,r,r['y_exact'],j['forcing_G'],j['expected_T'])
    validate_mc(*args)
    for field,value in [('seed_entropy',1),('completed_paths',[4095]*33),('n_path_steps',0),
                         ('x',[0.]),('stderr',[-1.]*33),('dt_requested',1/64)]:
        bad=copy.deepcopy(r);bad['result'][field]=value
        with pytest.raises(ValueError):validate_mc(j,bad,r['y_exact'],j['forcing_G'],j['expected_T'])
    bad=copy.deepcopy(r);bad['E_grid']/=2
    with pytest.raises(ValueError,match='max error'):
        validate_mc(j,bad,r['y_exact'],j['forcing_G'],j['expected_T'])


def test_numpy_exact_keeps_summary_json_serializable(production_record):
    import numpy as np
    from collections import Counter
    r=production_record;j=r['job']
    err,zero=validate_mc(j,r,np.asarray(r['y_exact']),j['forcing_G'],j['expected_T'])
    histogram=Counter([sum(v<=.05 for v in [err]*4)])
    assert type(err) is float and type(zero) is float
    json.dumps({'repeat_histogram':histogram},allow_nan=False)


def test_reader_detects_source_mutation(tmp_path):
    path=tmp_path/'source.json';path.write_text('{"a":1}')
    reader=Reader(tmp_path);assert reader.json('source.json')=={'a':1}
    path.write_text('{"a":2}')
    with pytest.raises(ValueError,match='input changed'):reader.json('source.json')


def test_release_path_never_calls_solver(monkeypatch,production_record):
    import mc_solver,mc_optimized
    def forbidden(*args,**kwargs):pytest.fail('numerical solve called during validation')
    monkeypatch.setattr(mc_solver,'solve_monte_carlo',forbidden)
    monkeypatch.setattr(mc_optimized,'solve_monte_carlo_optimized',forbidden)
    r=production_record;j=r['job']
    validate_mc(j,r,r['y_exact'],j['forcing_G'],j['expected_T'])
