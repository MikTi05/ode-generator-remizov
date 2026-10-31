"""Frozen inference scopes, input isolation, and artifact integrity; no solve/fit."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from ml.quality_prototype import DEFAULT_REGISTRY, QualityPrototype

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/all_methods_v1_1000_2026-10-18'


@pytest.fixture(scope='module')
def sample():
    with np.load(DATA/'all_methods_equations_v1_ml.npz', allow_pickle=False) as z:
        values, grid = z['X'][0], z['x_grid']
    meta = json.loads((DATA/'all_methods_equations_v1_metadata.jsonl').read_text().splitlines()[0])
    return values, grid, meta


@pytest.fixture(scope='module')
def prototype():
    return QualityPrototype()


def test_profiles_never_fabricate_winner_or_transfer_point_grid(prototype, sample):
    answer = prototype.predict(*sample)
    assert answer['method_selection'] == dict(status='uncertain', winner=None,
        reason='No validated model of fastest reliable solver on a common spatial/profile budget')
    rows = {r['profile_id']:r for r in answer['results']}
    for name in ('remizov_zero_grid33_n128', 'remizov_zero_grid33_n256',
                 'mc_grid33_single1024', 'mc_grid33_single4096', 'mc_grid33_mean4x4096'):
        assert rows[name]['status'] == 'unsupported_no_trained_model'
        assert rows[name]['predictions'] == {}
    assert 'p_accuracy_met' not in rows['mc_x0_mean4x1024']['predictions']
    assert 'log10_absolute_error' not in rows['mc_x0_single1024']['predictions']
    assert rows['mc_x0_mean4x1024']['parameters']['total_paths_per_point'] == 4096
    for row in rows.values():
        assert row['prediction_interval'] is None and not row['accuracy_certificate']


def test_runtime_requires_explicit_historical_context(prototype, sample):
    row = prototype.predict(*sample, profiles=['remizov_taper_grid33'])['results'][0]
    assert row['runtime_status'] == 'context_required'
    assert 'historical_solver_seconds' not in row['predictions']
    registry = json.loads(DEFAULT_REGISTRY.read_text())
    medians = json.loads((ROOT/registry['artifacts']['R_grid_time']['path']).read_text())
    for workers in (1, 8):
        row = prototype.predict(*sample, profiles=['remizov_taper_grid33'], planned_workers=workers)['results'][0]
        assert row['predictions']['historical_solver_seconds'] == medians[str(workers)]
        assert not row['timing_context']['counterfactual_single_worker_cost']
    with pytest.raises(ValueError):
        prototype.predict(*sample, planned_workers=True)


def test_input_only_api_ignores_exact_identity_and_labels(prototype, sample):
    values, grid, meta = sample
    clean = {key:meta[key] for key in ('a_lower_cert', 'a_upper_cert', 'b_abs_cert',
                                     'remizov_c_abs_cert', 'remizov_lambda')}
    clean['serialized_ast'] = {key:meta['serialized_ast'][key] for key in ('a_raw', 'b_raw', 'c_raw')}
    dirty = copy.deepcopy(clean)
    for key in ('equation_id', 'family_id', 'split', 'exact', 'max_abs_error',
                'solver_seconds', 'y0', 'v0', 'target_amplitude', 'support_left'):
        dirty[key] = 'FORBIDDEN'
    dirty['serialized_ast']['y_shape'] = 'FORBIDDEN'
    assert prototype.predict(values, grid, clean) == prototype.predict(values, grid, dirty)


@pytest.mark.parametrize('fault', ['grid', 'shape', 'nonfinite', 'bounds', 'domain', 'profile'])
def test_invalid_or_unvalidated_requests_fail_closed(prototype, sample, fault):
    values, grid, meta = copy.deepcopy(sample)
    kwargs = {}
    if fault == 'grid':grid[10] += 1e-10
    if fault == 'shape':values = values[:, :33]
    if fault == 'nonfinite':values[0, 0] = np.nan
    if fault == 'bounds':meta['a_upper_cert'] = meta['a_lower_cert']/2
    if fault == 'domain':kwargs['domain'] = 'arbitrary_ode'
    if fault == 'profile':kwargs['profiles'] = ['mc_optimal']
    with pytest.raises(ValueError):
        prototype.predict(values, grid, meta, **kwargs)


def test_model_hash_and_feature_schema_are_enforced(tmp_path, sample):
    registry = json.loads(DEFAULT_REGISTRY.read_text())
    registry['artifacts']['MC_mean_error']['sha256'] = '0'*64
    target = tmp_path/'registry.json';target.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match='integrity'):
        QualityPrototype(target)
    registry = json.loads(DEFAULT_REGISTRY.read_text())
    registry['feature_names'] = registry['feature_names'][::-1]
    target.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match='order'):
        QualityPrototype(target)


def test_serving_does_not_call_fit(monkeypatch, sample):
    from catboost import CatBoostClassifier, CatBoostRegressor
    def forbidden(*args, **kwargs):
        raise AssertionError('fit is forbidden during inference')
    monkeypatch.setattr(CatBoostClassifier, 'fit', forbidden)
    monkeypatch.setattr(CatBoostRegressor, 'fit', forbidden)
    assert QualityPrototype().predict(*sample)['method_selection']['winner'] is None


def test_new_serving_and_report_code_has_no_solver_or_fit_calls():
    for source in ('ml/quality_prototype.py', 'diagnostics/predict_quality.py',
                   'diagnostics/close_ml_research.py'):
        tree = ast.parse((ROOT/source).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in ('remizov_solver', 'chernov_solver', 'mc_solver', 'mc_optimized')
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    assert node.func.attr not in ('fit', 'solve', 'solve_monte_carlo')
                if isinstance(node.func, ast.Name):
                    assert node.func.id not in ('eval', 'exec', 'ChernovSolver')


def test_cli_accepts_standalone_coefficients_without_exact_or_dataset(tmp_path, sample):
    values, grid, meta = sample
    metadata = {key:meta[key] for key in ('a_lower_cert', 'a_upper_cert', 'b_abs_cert',
                                        'remizov_c_abs_cert', 'remizov_lambda')}
    metadata['serialized_ast'] = {key:meta['serialized_ast'][key] for key in ('a_raw', 'b_raw', 'c_raw')}
    path = tmp_path/'coefficients.json'
    path.write_text(json.dumps(dict(coefficients=values.tolist(), x_grid=grid.tolist(), metadata=metadata)))
    result = subprocess.run([sys.executable, '-B', str(ROOT/'diagnostics/predict_quality.py'),
        '--input-json', str(path), '--dataset', str(tmp_path/'nonexistent'),
        '--profiles', 'mc_x0_single1024', 'mc_grid33_single4096'],
        check=True, text=True, capture_output=True)
    answer = json.loads(result.stdout)[0]
    assert answer['method_selection']['status'] == 'uncertain'
    assert answer['results'][1]['predictions'] == {}


def test_new_report_refuses_historical_directory_without_writing():
    target = ROOT/'results/remizov_ml_v1_2026-10-25'
    result = subprocess.run([sys.executable, '-B', str(ROOT/'diagnostics/close_ml_research.py'),
        '--out', str(target)], text=True, capture_output=True)
    assert result.returncode != 0
    assert 'refusing to overwrite' in result.stderr
    assert not (target/'registry.json').exists()
