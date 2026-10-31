"""Research release and final-test guards. No numerical solver is called."""
import copy,json,subprocess,sys
from collections import Counter
from pathlib import Path
import numpy as np
import pytest
from ml.final_selector import FinalResearchSelector
from ml.revision_evidence import ROOT,read,sha

RELEASE=ROOT/'results/ml_selector_final_2026-10-27'
EXPERIMENT=RELEASE/'final_experiment'


@pytest.fixture(scope='module')
def model():return FinalResearchSelector(RELEASE/'model')


def payload():return read(RELEASE/'example_input.json')


def test_working_input_to_profile(model):
    p=payload();r=model.predict(p['coefficients'],p['x_grid'],p['coefficient_metadata'])
    assert r['selected_profile'] in ('R128','R256','MC1') and r['action']=='select_profile'
    assert r['selected_profile'] in r['predicted_admissible_profiles']
    assert not r['accuracy_certificate'] and not r['production_allowed']
    assert set(r['estimates'])=={'R128','R256','MC1'} and r['accuracy_event']['threshold']==.01


def test_extra_solution_labels_cannot_change_inference(model):
    p=payload();first=model.predict(p['coefficients'],p['x_grid'],p['coefficient_metadata'])
    meta=copy.deepcopy(p['coefficient_metadata']);meta.update(y_exact=[1e90]*256,E_grid=0.,equation_id='injected',family_id='injected')
    meta['serialized_ast']['y_shape']={'op':'execute_unsafe','children':[]}
    second=model.predict(p['coefficients'],p['x_grid'],meta)
    assert first['estimates']==second['estimates'] and first['selected_profile']==second['selected_profile']


@pytest.mark.parametrize('bad',['shape','nan','grid','cert'])
def test_invalid_input_rejected(model,bad):
    p=payload()
    if bad=='shape':p['coefficients']=p['coefficients'][:3]
    if bad=='nan':p['coefficients'][0][0]=float('nan')
    if bad=='grid':p['x_grid'][0]+=.0001
    if bad=='cert':p['coefficient_metadata']['a_lower_cert']=-1
    with pytest.raises(ValueError):model.predict(p['coefficients'],p['x_grid'],p['coefficient_metadata'])


def test_context_and_production_guards(model):
    p=payload()
    assert model.predict(p['coefficients'],p['x_grid'],p['coefficient_metadata'],domain='IVP')['selected_profile'] is None
    assert model.predict(p['coefficients'],p['x_grid'],p['coefficient_metadata'],profile_context={})['status']=='unsupported_profiles'
    assert model.production_select_method()['action']=='abstain'


def test_MC_can_be_selected_by_research_rule_without_fake_labels(model,monkeypatch):
    p=payload();est={k:dict(p=.99,seconds=t,time_low=t*.9,time_high=t*1.1) for k,t in [('R128',12),('R256',24),('MC1',5)]}
    monkeypatch.setattr(model.model,'predict_features',lambda X:[dict(binary_sigmoid=est)])
    assert model.predict(p['coefficients'],p['x_grid'],p['coefficient_metadata'])['selected_profile']=='MC1'


def test_model_tampering_rejected(tmp_path):
    import shutil
    dest=tmp_path/'model';shutil.copytree(RELEASE/'model',dest)
    file=next(dest.rglob('*.cbm'));file.write_bytes(file.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='integrity'):FinalResearchSelector(dest)


def test_actual_CLI(tmp_path):
    output=tmp_path/'prediction.json'
    r=subprocess.run([sys.executable,'-B',str(ROOT/'diagnostics/predict_final_selector.py'),'--model',str(RELEASE/'model'),'--input',str(RELEASE/'example_input.json'),'--output',str(output)],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    assert read(output)['selected_profile']==read(RELEASE/'example_prediction.json')['selected_profile']


def test_manifest_freshness_and_fixed_selection():
    from diagnostics.prepare_final_selector_experiment import select
    plan=read(EXPERIMENT/'plan.json');audit=read(EXPERIMENT/'independence.json')
    assert not set(audit['excluded_topologies'])&{c['structure_signature'] for c in plan['cases']}
    assert Counter((c['complexity_level'],c['regime']) for c in plan['cases'])=={(l,r):3 for l in ('L1','L2','L3','L4') for r in ('constant','slow_varying','moderate_varying','oscillatory','high_contrast')}
    rows=[json.loads(l) for l in (EXPERIMENT/'input_pool/all_methods_equations_v1_metadata.jsonl').read_text().splitlines()]
    selected=select(list(reversed(rows)),set(audit['excluded_topologies']),plan['cases'][0]['dataset_sha256'])
    assert [r['equation_id'] for r in selected]==[c['equation_id'] for c in plan['cases']]
    assert all(c['split']=='test' for c in plan['cases'])


def test_exact_call_budget_and_seed_scope():
    from diagnostics.run_final_selector_experiment import validate_scope
    p=read(EXPERIMENT/'plan.json');validate_scope(p)
    assert len(p['jobs'])==720 and not p['execution_authorized']
    bad=copy.deepcopy(p);bad['jobs'][0]['config']['n']=999
    with pytest.raises(AssertionError):validate_scope(bad)
    assert {j['config']['seed'] for j in p['jobs'] if j['profile']=='MC1'}==set(p['MC_seed_entropy'])


def test_execution_impossible_without_separate_approval(monkeypatch):
    from diagnostics import run_final_selector_experiment as run
    import chernov_solver,mc_optimized
    def forbidden(*a,**kw):pytest.fail('no solver or worker allowed before approval')
    monkeypatch.setattr(run,'GuardedProcess',forbidden);monkeypatch.setattr(chernov_solver.ChernovSolver,'solve',forbidden);monkeypatch.setattr(mc_optimized,'solve_monte_carlo_optimized',forbidden)
    with pytest.raises(PermissionError,match='No separate user approval'):run.execute(EXPERIMENT)
    assert not (EXPERIMENT/'execution_authorization.json').exists()
    assert not (EXPERIMENT/'budget.json').exists()
    assert not list((EXPERIMENT/'records').iterdir()) and not list((EXPERIMENT/'events').iterdir())


def test_final_evaluator_does_not_invent_results():
    from diagnostics.evaluate_final_selector_experiment import analyze
    r=analyze(EXPERIMENT)
    assert r['status']=='not_executed' and not r['final_evaluation_complete']


def test_weights_frozen_before_new_inputs():
    freeze=read(RELEASE/'release_freeze.json');protocol=read(EXPERIMENT/'preregistration.json')
    assert freeze['before_final_holdout_inputs'] and protocol['model_frozen_before_inputs']
    assert protocol['release_freeze_sha256']==sha(RELEASE/'release_freeze.json')
    assert freeze['model_registry_sha256']==sha(RELEASE/'model/registry.json')
