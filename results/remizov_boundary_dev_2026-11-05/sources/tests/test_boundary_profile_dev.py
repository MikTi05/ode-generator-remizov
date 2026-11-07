"""Selection, boundary premises and saved-answer policy; no production solves."""
from dataclasses import replace
import copy
import json

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from chernov_solver import ChernovConfig,_Counts,_State,_apply,_geometry
from remizov_solver import RemizovProblem,compute_truncation_T
from diagnostics.boundary_profile_common import (ROOT,DATA,SPLITS,OTHER_TEST,GRID_RELEASE,BASE,OLD_DIAG,
    CONFIG,atomic_json,compare,read,select,sha,summary,table)
from diagnostics.run_boundary_profile_dev import check_saved_zero,numerical_inputs


@pytest.fixture(scope='module')
def selection_inputs():
    metadata=[json.loads(s) for s in (ROOT/DATA/'all_methods_equations_v1_metadata.jsonl').read_text().splitlines()]
    other=[json.loads(s) for s in (ROOT/OTHER_TEST).read_text().splitlines()]
    return table(ROOT/GRID_RELEASE/'grid33_all1000.csv'),metadata,read(ROOT/SPLITS),other


def test_all_bad_dev_cases_and_30_distinct_preselected_controls(selection_inputs):
    selected,audit=select(*selection_inputs)
    assert audit['case_count']==22 and audit['control_count']==30 and len(selected)==52
    cases=[r for r in selected if r['category']=='case'];controls=[r for r in selected if r['category']=='control']
    assert all(r['E_before']>.05 for r in cases) and all(r['E_before']<=.05 for r in controls)
    assert len({r['family_id'] for r in controls})==30
    assert not {r['family_id'] for r in cases}&{r['family_id'] for r in controls}
    assert {r['split'] for r in selected}=={'train','validation'}
    assert 'AMV1_F0180_S04' not in {r['equation_id'] for r in selected}
    assert len(audit['missing_R_in_development'])==3
    assert selected==select(*selection_inputs)[0]


def test_Test_errors_and_intervention_outcomes_cannot_change_selection(selection_inputs):
    rows,metadata,splits,other=selection_inputs;altered=copy.deepcopy(rows)
    test={m['equation_id'] for m in metadata if m['split']=='test'}
    for r in altered:
        if r['equation_id'] in test:r['R_E33']='nan'
        r['zero_error']='999999' if r['equation_id'].endswith('0') else '0'
    assert select(rows,metadata,splits,other)==select(altered,metadata,splits,other)


def fake_comparison(before_error,after_error,category='case'):
    selected=dict(equation_id='synthetic',category=category,E_before=before_error,family_id='f',expression_hash='h',structure_signature='s',complexity_level='L1',regime='constant',split='train')
    before=dict(result=dict(f=[before_error]*33,boundary_state_max_abs=999,T=1),y_exact=[0]*33)
    after=dict(result=dict(f=[after_error]*33,success=True,failure_reason='',boundary_state_max_abs=12),y_exact=[0]*33,
               before_source='old',zero_source='new',reused=False)
    return compare(selected,before,after)


def test_threshold_equality_passes_and_every_increase_is_exposed():
    row=fake_comparison(.05,.05+1e-12,'control')
    assert row['pass_before'] and not row['pass_after'] and row['pass_lost']
    assert row['increased_unrounded'] and not row['material_regression']
    assert summary([row])['recommendation']=='NO-GO'


def test_a_material_regression_is_not_hidden_by_larger_outlier_improvement():
    fixed=fake_comparison(100,.01);regression=fake_comparison(.01,.011,'control')
    result=summary([fixed,regression])
    assert result['recommendation']=='NO-GO'
    assert not result['GO_gates']['no_material_regressions']


def test_new_catastrophe_and_failed_zero_cannot_pass_gate():
    row=fake_comparison(.1,2)
    assert row['new_catastrophe'] and summary([row])['recommendation']=='NO-GO'
    selected={k:row[k] for k in ('equation_id','category','E_before','family_id','expression_hash','structure_signature','complexity_level','regime','split')}
    before=dict(result=dict(f=[.1]*33,boundary_state_max_abs=1,T=1),y_exact=[0]*33)
    after=dict(result=dict(success=False,failure_reason='nonfinite_state',f=[]),y_exact=[0]*33,before_source='old',zero_source='new',reused=False)
    failed=compare(selected,before,after)
    assert failed['E_after'] is failed['pass_after'] is None
    assert not failed['usable_zero'] and summary([failed])['recommendation']=='NO-GO'


def test_experimental_gate_can_pass_without_claiming_production_promotion():
    result=summary([fake_comparison(100,.04),fake_comparison(.01,.01,'control')])
    assert result['recommendation']=='GO' and not result['production_profile_changed']
    assert 'not production promotion' in result['scope']


def small_state():
    x=np.arange(4,dtype=float)
    return _State(x,np.full(4,.25),np.full(4,4.),np.zeros(4),np.zeros(4),1.,128,'cubic','zero',.3)


def test_zero_extension_is_discontinuous_if_endpoint_state_is_nonzero():
    state=small_state();u=np.ones(4)
    q=np.array([-1e-6,0,3,3+1e-6])
    out=_apply(state,u,1/16,(q,q,q),_Counts())
    np.testing.assert_array_equal(out,[0,1,1,0])


def test_zero_endpoint_values_do_not_make_extension_globally_C1():
    spline=CubicSpline(np.arange(4),[0,1,1,0],extrapolate=False)
    assert spline(0)==spline(3)==0
    assert spline(0,1)==1.5 and spline(3,1)==-1.5


def test_cubic_zero_grid_operator_need_not_be_contractive_even_with_zero_endpoints():
    state=small_state();u=np.array([0.,1.,1.,0.]);tau=1/16
    out=_apply(state,u,tau,_geometry(state,tau),_Counts())
    assert out[1]==67/64 and max(abs(out))>max(abs(u))
    assert state.c.max()==0  # formal bound would be 1, not 67/64


def test_cubic_zero_does_not_preserve_positivity():
    state=small_state();u=np.array([1.,0.,0.,0.]);tau=1/16
    out=_apply(state,u,tau,_geometry(state,tau),_Counts())
    assert out[1]==pytest.approx(-1/64,abs=1e-16)
    assert min(out)<0 and min(u)==0


def test_inside_queries_unchanged_and_lambda_absent_from_same_step():
    state=small_state();u=np.array([0.,1.,1.,0.]);tau=1/16;q=_geometry(state,tau)
    zero=_apply(state,u,tau,q,_Counts());taper=_apply(replace(state,boundary='taper'),u,tau,q,_Counts())
    assert zero[1]==taper[1]
    np.testing.assert_array_equal(zero,_apply(replace(state,lam=20),u,tau,q,_Counts()))


def test_input_fingerprints_match_without_any_operator_application():
    problem=RemizovProblem('synthetic',lambda x:1.,lambda x:.2,lambda x:.1,lambda x:np.exp(-x*x),
        1.,1.,1.,.2,.1,'analytic','all_real_x',-1.,1.)
    config=ChernovConfig(**CONFIG);T=compute_truncation_T(problem,config.tail_eps)
    before=dict(target_x=np.linspace(-1,1,33).tolist(),y_exact=[0.]*33,
                result=dict(T=T,work_domain=[-13.,13.],work_grid_points=2081))
    state,t,w,fp=numerical_inputs(problem,config,before)
    z,zt,zw,zfp=numerical_inputs(problem,replace(config,boundary='zero'),before)
    assert fp==zfp and state.boundary=='taper' and z.boundary=='zero'
    np.testing.assert_array_equal(t,zt);np.testing.assert_array_equal(w,zw)
    with pytest.raises(ValueError,match='effective T'):
        numerical_inputs(problem,config,dict(before,result=dict(before['result'],T=T+1)))


def test_saved_zero_reuse_checks_exact_config_and_baseline_identity():
    pid='AMV1_F0141_S00';path=ROOT/BASE/(pid+'.json');before=read(path);before['_source_sha256']=sha(path)
    zero=read(ROOT/OLD_DIAG/'runs'/(pid+'_cubic_zero.json'))
    check_saved_zero(zero,before,zero['fingerprints'])
    changed=copy.deepcopy(zero);changed['config']['T']=zero['result']['T']
    with pytest.raises(ValueError,match='config'):check_saved_zero(changed,before,zero['fingerprints'])
    changed=copy.deepcopy(zero);changed['production_result_sha256']='wrong'
    with pytest.raises(ValueError,match='identity'):check_saved_zero(changed,before,zero['fingerprints'])


def test_saved_results_cannot_be_overwritten(tmp_path):
    path=tmp_path/'result.json';atomic_json(path,{'old':True})
    with pytest.raises(FileExistsError):atomic_json(path,{'old':False})
    assert read(path)=={'old':True}
