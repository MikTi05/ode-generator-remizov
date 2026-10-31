"""Import isolation and one bounded generated-equation calculation."""

import ast
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.all_methods_dataset_v1 import AllMethodsConfig, generate_dataset, write_dataset

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {
    'dataset_loader', 'screening_solver', 'rk4_solver', 'product_integral_solver',
    'generate_dataset', 'generate_production_dataset', 'pilot_benchmark_v0',
    'calibration_benchmark_v1', 'production_benchmark_v1',
    'runtime_stability_audit_v1', 'ml_baseline_v1', 'ml_cnn_v1',
    'ml_feature_ablation_v1', 'ml_final_test_v1',
}


def test_working_components_import_only_from_this_project():
    names = [
        'generate_all_methods_dataset_v1', 'validate_all_methods_dataset_v1',
        'all_methods_dataset_loader', 'remizov_solver',
        'src.all_methods_dataset_v1', 'src.all_methods_expressions', 'src.problem_types',
    ]
    for name in names:
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve().is_relative_to(ROOT)
    assert FORBIDDEN.isdisjoint(sys.modules)
    for path in list(ROOT.glob('*.py')) + list((ROOT / 'src').glob('*.py')):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or '']
            else:
                continue
            assert FORBIDDEN.isdisjoint(names), str(path)


def test_mathematical_core_matches_import_manifest():
    manifest = json.loads((ROOT / 'MIGRATION_MANIFEST.json').read_text())
    for relative in manifest['unchanged_mathematical_core']:
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == manifest['source_sha256'][relative]


@pytest.mark.parametrize('equation_id', ['AMV1_F0000_S00', 'AMV1_F0005_S00'])
def test_generated_equation_cli_reports_status_and_exact_error(tmp_path, equation_id):
    dataset = generate_dataset(AllMethodsConfig(
        seed=2488, n_families=20, samples_per_family=1,
        ml_grid_size=32, validation_grid_size=257,
    ))
    write_dataset(dataset, tmp_path)
    process = subprocess.run([
        sys.executable, str(ROOT / 'examples/solve_generated_equation.py'),
        '--input-dir', str(tmp_path), '--equation-id', equation_id,
        '--n', '1', '--points', '33',
        '--verification-budget',
    ], cwd=ROOT, text=True, capture_output=True, timeout=60)
    assert process.returncode in (0, 1), process.stdout + process.stderr
    summary = json.loads(process.stdout)
    # Analytic admissibility does not guarantee convergence at a fixed budget.
    # Exercise both public result contracts without changing solver controls.
    assert process.returncode == (0 if summary['success'] else 1)
    if summary['success']:
        assert summary['failure_reason'] == ''
        assert summary['max_abs_error'] >= 0
        assert summary['solution_points'] == summary['target_points']
    else:
        assert summary['failure_reason'] in {
            'quadrature_not_converged', 'spatial_not_converged', 'work_domain_too_large',
        }
        assert summary['max_abs_error'] is None
        assert summary['solution_points'] == 0
    assert summary['equation_id'] == equation_id
    assert summary['success'] is (equation_id == 'AMV1_F0005_S00')
    assert summary['target_points'] == 33
    assert summary['total_error_certified'] is False
    assert summary['config']['n'] == 1
    assert summary['config']['max_quadrature_refinements'] == 8
    assert summary['config']['max_spatial_refinements'] == 5
