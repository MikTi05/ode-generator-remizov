"""Label/identity/leakage/serving tests; no numerical methods or fake ML data fits."""
import copy,json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from ml.selector_labels import Evidence,GRID,TIMING_SCOPE,label_pair,training_gate
from ml.selector_evidence import validate_record,sha
from ml.method_selector import classification_metrics,fit_catboost,train_only_diagnostic,retrospective_policy,MethodSelector
ROOT=Path(__file__).resolve().parents[1]


def pair(er=.001,em=.002):
    common=dict(equation_id='fixture_ID',expression_hash='hash',family_id='family',split='train',status='completed',T=2.,timing_scope=TIMING_SCOPE,timing_cohort='same_hardware_cohort')
    return (Evidence('remizov',profile='R128',error=er,times=[8.,8.,8.,8.],deterministic_repeats_equal=True,**common),
        Evidence('monte_carlo',profile='MC1',error=em,times=[10.,10.,10.,10.],**common))


@pytest.mark.parametrize('r,mc,want',[(.01,.02,'remizov'),(.02,.01,'monte_carlo'),(.02,.02,'none'),(.01,.01,'remizov'),(None,.02,'unknown'),(.01,None,'unknown'),(None,None,'unknown')])
def test_label_truth_table_and_inclusive_threshold(r,mc,want):
    a,b=pair(r,mc);assert label_pair(a,b,'pair')['label']==want


def test_seed_or_mean_not_chosen_after_viewing_results():
    a,b=pair(.02,.03);b.records=[{'episode':0,'error':.03},{'episode':1,'error':.001}]
    assert label_pair(a,b,'pair')['label']=='none'


def test_sole_accuracy_and_none_do_not_need_time_comparison():
    a,b=pair(.002,.02);a.times=[];a.timing_scope='old';b.timing_cohort='other'
    assert label_pair(a,b,'pair')['label']=='remizov'
    a.error=.02;assert label_pair(a,b,'pair')['label']=='none'


@pytest.mark.parametrize('change',[dict(times=[1.]),dict(times=[float('nan')]*4),dict(times=[1.,2.,9.,10.]),dict(timing_scope='legacy'),dict(timing_cohort='other'),dict(deterministic_repeats_equal=False)])
def test_two_accurate_requires_valid_uniform_stable_timing(change):
    a,b=pair();a=replace(a,**change);r=label_pair(a,b,'pair');assert r['label']=='unknown' and r['reason']=='timing_unknown'


def test_tie_requires_both_accurate_and_prefers_R():
    a,b=pair();a.times=[10.,11.,12.,13.];b.times=[11.,12.,13.,14.]
    r=label_pair(a,b,'pair');assert r['label']=='remizov' and r['tie']
    a.times=[20.]*4;b.times=[18.]*4
    assert label_pair(a,b,'pair')['label']=='monte_carlo'
    b.times=[18.01]*4;assert label_pair(a,b,'pair')['tie']


def test_cross_dataset_or_horizon_not_pooled():
    a,b=pair(.02,.02)
    assert label_pair(a,replace(b,expression_hash='other'),'pair')['label']=='unknown'
    assert label_pair(a,replace(b,T=3.),'pair')['reason']=='common_horizon_mismatch'
    assert label_pair(a,replace(b,status='worker_exception'),'pair')['label']=='unknown'


def real_record():
    path=ROOT/'results/joint_grid100_2026-10-25/records/AMV1_F0047_S00_R.json';r=json.loads(path.read_text())
    metas=[json.loads(s) for s in (ROOT/'data/all_methods_v1_1000_2026-10-18/all_methods_equations_v1_metadata.jsonl').read_text().splitlines()]
    m=next(x for x in metas if x['equation_id']=='AMV1_F0047_S00')
    return r,m


@pytest.mark.parametrize('kind',['hash','grid','error','profile','nan'])
def test_saved_record_comparability_rejects_corruption(kind):
    r,m=real_record();job=copy.deepcopy(r['job']);ph=r['plan_sha256'];exact=np.asarray(r['y_exact'])
    validate_record(r,job,ph,m,exact,'remizov')
    if kind=='hash':r['expression_hash']='corrupt'
    if kind=='grid':r['result']['x']=[0.]
    if kind=='error':r['E_inf']=0.
    if kind=='profile':r['job']['config']['boundary']='taper';job=copy.deepcopy(r['job'])
    if kind=='nan':r['result']['f'][0]=float('nan')
    with pytest.raises(ValueError):validate_record(r,job,ph,m,exact,'remizov')


def test_execution_failure_has_unknown_error():
    r,m=real_record();r['status']='worker_exception';assert validate_record(r,r['job'],r['plan_sha256'],m,np.array(r['y_exact']),'remizov') is None


def test_unknown_never_trained_as_none_and_missing_classes_block_fit():
    rows=[dict(split='train',label='remizov',family_id='R'),dict(split='train',label='none',family_id='N'),dict(split='validation',label='unknown',family_id='V')]
    gate=training_gate(rows);assert not gate['allowed'] and gate['support']['train']['known']==2 and gate['support']['validation']['known']==0
    with pytest.raises(ValueError):fit_catboost(np.zeros((2,253)),['remizov','unknown'],list(map(str,range(253))))
    with pytest.raises(ValueError):fit_catboost(np.zeros((2,253)),['remizov','remizov'],list(map(str,range(253))))


def test_no_support_is_not_successful_MC_recognition():
    m=classification_metrics(['remizov','none'],['remizov','remizov'])
    assert m['accuracy']==.5 and m['recall']['monte_carlo'] is None and m['support']['monte_carlo']==0


def test_missing_baseline_output_is_unobserved_not_fake_failure():
    a,b=pair();a.status='not_run';a.error=None
    e={a.equation_id+':remizov':a.__dict__};rows=[dict(equation_id=a.equation_id,label='unknown')]
    m=retrospective_policy(rows,e,['remizov']);assert m['delivery_accuracy_rate'] is None and m['observed_execution_failures']==0 and m['unobserved_selected_outputs']==1


def test_training_sources_do_not_import_or_call_solvers():
    import ast
    for name in ['ml/selector_labels.py','ml/selector_evidence.py','ml/method_selector.py','diagnostics/train_method_selector.py']:
        tree=ast.parse((ROOT/name).read_text())
        calls=[n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))]
        assert not {'solve','solve_monte_carlo','solve_monte_carlo_optimized','compile_backend'}&set(calls)


def test_saved_run_leakage_and_serving_if_available():
    out=ROOT/'results/ml_selector_observed_2026-10-26'
    if not (out/'data_audit.json').exists():pytest.skip('run artifact integration check after pipeline')
    rows=[json.loads(s) for s in (out/'labels.jsonl').read_text().splitlines()];d=json.loads((out/'diagnostic_cv.json').read_text());audit=json.loads((out/'data_audit.json').read_text())
    assert audit['support']['train']['counts']=={'remizov':22,'none':14,'monte_carlo':0,'unknown':664}
    assert len(d['oof'])==len(d['fit_indices'])==36
    with np.load(out/'features.npz',allow_pickle=False) as z:
        X=z['X'];assert X.shape==(1000,253) and set(z.files)=={'X','feature_names'}
    for fold in d['folds']:
        fit=fold['fit_indices'];held=fold['held_out_indices'];assert not set(fold['fit_families'])&set(fold['held_out_families'])
        assert all(rows[i]['split']=='train' and rows[i]['label']!='unknown' for i in fit+held)
        np.testing.assert_allclose(fold['scaler_mean'],X[fit].mean(axis=0),rtol=0,atol=1e-12)
    assert not d['original_validation_or_test_used'] and not d['MC_recognition_learned']
    api=MethodSelector(out/'registry.json');assert api.model is None
    check=json.loads((out/'serving_check.json').read_text());assert check['action']=='abstain' and check['status']=='insufficient_evidence'
