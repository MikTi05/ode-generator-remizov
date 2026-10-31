"""Inference/selection protocol tests, without numerical solver calls."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import norm
from diagnostics.grid100_statistics import aggregate_mc,adaptive_ids


def record(value=1.,se=.1):
    return dict(result=dict(M=1024,f=[value]*33,stderr=[se]*33,
        forcing_abs_upper_bound=3.,T=1.,dt_actual=1/128,tail_error_bound=1e-4),y_exact=[1.]*33)


def test_simultaneous_band_includes_grid_and_optional_look_adjustment():
    s=aggregate_mc([record()]);expected=norm.isf(.05/(2*33*2))
    assert s['normal_z']==expected and s['normal_z']>norm.isf(.05/(2*33))
    np.testing.assert_allclose(s['normal_simultaneous_half'],expected*.1)
    assert s['statistical_accuracy']=='uncertain' and not s['total_accuracy_certified']
    assert s['hoeffding_half']>s['normal_simultaneous_half'][0]


def test_four_seed_pool_retains_between_group_component():
    records=[record(v,.01) for v in (.8,.9,1.1,1.2)];s=aggregate_mc(records)
    assert s['paths_per_point']==4096 and s['seeds']==4
    M=1024;v=((M-1)*4*.01**2*M+M*sum((np.array([.8,.9,1.1,1.2])-1)**2))/(4*M-1)
    np.testing.assert_allclose(s['stderr'],np.sqrt(v/(4*M)))
    assert 'seed_t_simultaneous_half' in s


def test_repeat_selection_never_changes_h_M_T_and_excludes_resolved_or_failed():
    rows=[dict(problem_id=f'P{i}',both_completed=True,R_pass=i>5,
        mc=dict(statistical_accuracy='uncertain',error_mean_interval=[0.,i+1.])) for i in range(20)]
    rows += [dict(problem_id='failed',both_completed=False)]
    got=adaptive_ids(rows,12);assert len(got)==12 and got[:6]==[f'P{i}' for i in range(6)]
    assert 'failed' not in got


def test_frozen_selection_train_only_and_common_profile():
    root=Path(__file__).resolve().parents[1];path=root/'results/joint_grid100_2026-10-25/plan.json'
    if not path.exists():return
    p=json.loads(path.read_text());rows=p['selection'];rep=[r for r in rows if r['cohort']=='representative']
    assert len(rows)==len({r['problem_id'] for r in rows})==100
    assert len(rep)==80 and len({r['family_id'] for r in rep})==80
    assert set(Counter(tuple(r['cell']) for r in rep).values())=={4}
    assert all(r['split']=='train' for r in rows)
    tasks={r['problem_id']:r for r in rows}
    for j in p['jobs'][4:]:
        if j['method']=='R':
            assert j['config']['boundary']=='zero' and j['config']['interp']=='cubic'
            assert j['config']['T']==tasks[j['problem_id']]['T']
        else:
            assert j['config']['T'] is None and j['config']['M']==1024
            assert j['config']['dt']==1/128 and j['watchdog_seconds']>=180
    assert p['resources']['series_seconds']==14400 and p['resources']['rss_mib']==512
