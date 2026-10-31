"""Saved research artifacts only; never train or solve in these checks."""
import ast,json
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/ml_nested_calibration_2026-10-26'


def read(name):
    if not (OUT/name).exists():pytest.skip('nested artifacts not generated')
    return json.loads((OUT/name).read_text())


def test_full_OOF_covers_each_real_target_once_and_family_held_together():
    targets=read('accuracy_targets.json');oof=read('oof.json')
    for p in ('R128','R256'):
        expected={r['equation_id'] for r in targets if r['profile']==p}
        assert len(oof[p])==len(expected) and {r['equation_id'] for r in oof[p]}==expected
        mapping={}
        for r in oof[p]:assert mapping.setdefault(r['family_id'],r['outer_fold'])==r['outer_fold']


def test_outer_evaluation_never_uses_final_or_held_family_calibrator():
    nodes=read('model_index.json')
    for rs in read('oof.json').values():
        for r in rs:
            assert r['calibrator_key'].startswith('outer')
            cal=read('calibrators/'+r['calibrator_key']+'.json')
            assert r['family_id'] not in cal['families']
            for block in cal['inner_blocks']:
                assert r['family_id'] not in nodes[block['node']]['train_families']
                assert not set(block['held_families'])&set(nodes[block['node']]['train_families'])


def test_new_research_artifacts_preserve_production_block():
    from diagnostics.audit_probability_selector import sha
    registry=read('registry.json')
    assert registry['status']=='research_only' and not registry['deployment_ready']
    assert registry['no_production_dispatch']
    assert sha(ROOT/'ml/probabilistic_selector.py')==registry['original_production_code_sha256']


def test_missing_cost_is_never_replaced_by_model_prediction():
    policies=read('metrics.json')['policies']
    for mode in ('raw','calibrated'):
        for variants in policies[mode].values():
            for m in variants.values():
                if m['missing_method_cost_cases']:
                    assert m['observed_method_cost_sum'] is None and m['observed_seconds_per_accurate'] is None


def test_blind_pilot_input_forecasts_have_no_accuracy_labels():
    data=read('prospective_train20_predictions.json');nodes=read('model_index.json')
    fitted=set().union(*(set(n['train_families']) for n in nodes.values()))
    assert len(data['rows'])==20 and not data['model_fit_after_pilot_features']
    for r in data['rows']:
        assert r['split']=='train' and r['family_id'] not in fitted and r['accuracy_unknown']
        assert not {'k','n','E_grid','y_exact','accuracy_met'}&set(r)


def test_saved_only_report_never_fits_models_or_calibrators_or_ODEs():
    tree=ast.parse((ROOT/'diagnostics/report_nested_probability_calibration.py').read_text())
    calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for n in ast.walk(tree)
           if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
    assert not calls&{'fit','fit_head','fit_sigmoid','solve','solve_monte_carlo_optimized','load_inputs'}
