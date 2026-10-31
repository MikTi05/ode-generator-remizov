"""Persistence and rejection tests for the standalone safe loader."""

import copy
import csv
import json

import numpy as np
import pytest

from all_methods_dataset_loader import (
    AllMethodsDatasetFormatError, load_all_methods_equations,
)
from src.all_methods_dataset_v1 import (
    AllMethodsConfig, CSV_FIELDS, generate_dataset,
    validate_dataset_artifacts, write_dataset, GenerationError,
)
from src.problem_types import CauchyProblem


@pytest.fixture(scope='module')
def generated():
    return generate_dataset(AllMethodsConfig(
        seed=2488, n_families=20, samples_per_family=1,
        ml_grid_size=32, validation_grid_size=257,
    ))


def save_pair(directory, rows, metadata):
    csv_path = directory / 'equations.csv'
    metadata_path = directory / 'metadata.jsonl'
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    metadata_path.write_text(''.join(
        json.dumps(row, allow_nan=False) + '\n' for row in metadata
    ), encoding='utf-8')
    return csv_path, metadata_path


def test_round_trip_preserves_initial_conditions_and_signs(generated, tmp_path):
    paths = save_pair(tmp_path, generated.equations, generated.metadata)
    loaded = load_all_methods_equations(*paths)
    assert len(loaded) == 20
    x = np.linspace(-1, 1, 17)
    for equation, row in zip(loaded, generated.equations):
        ordinary = equation.cauchy_problem
        remizov = equation.remizov_problem
        assert isinstance(ordinary, CauchyProblem)
        assert ordinary.y0 == float(row['y0'])
        assert ordinary.v0 == float(row['v0'])
        assert ordinary.y0 == pytest.approx(equation.y_exact(-1))
        assert ordinary.v0 == pytest.approx(equation.dy_exact(-1))
        np.testing.assert_allclose(
            [ordinary.c(float(t)) for t in x],
            remizov.c_remizov(x) - remizov.lambda_value,
        )
        np.testing.assert_allclose(
            [ordinary.g(float(t)) for t in x], -remizov.g_remizov(x),
        )


def test_csv_without_metadata_is_rejected(generated, tmp_path):
    csv_path, _ = save_pair(tmp_path, generated.equations, generated.metadata)
    with pytest.raises(AllMethodsDatasetFormatError):
        load_all_methods_equations(csv_path, tmp_path / 'missing.jsonl')


@pytest.mark.parametrize('mutation', [
    'id_set', 'duplicate_csv_id', 'duplicate_metadata_id',
    'formula', 'unsafe_ast', 'certificate', 'initial_condition',
])
def test_inconsistent_pair_is_rejected(generated, tmp_path, mutation):
    rows = copy.deepcopy(generated.equations)
    metadata = copy.deepcopy(generated.metadata)
    if mutation == 'id_set':
        metadata.pop()
    elif mutation == 'duplicate_csv_id':
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == 'duplicate_metadata_id':
        metadata.append(copy.deepcopy(metadata[0]))
    elif mutation == 'formula':
        rows[0]['a_expr'] = '1'
    elif mutation == 'unsafe_ast':
        metadata[0]['serialized_ast']['a_raw']['op'] = 'log'
    elif mutation == 'certificate':
        metadata[0]['remizov_lambda'] = metadata[0]['remizov_c_abs_cert']
    elif mutation == 'initial_condition':
        value = float(rows[0]['v0']) + 1
        rows[0]['v0'] = str(value)
        metadata[0]['v0'] = value
    paths = save_pair(tmp_path, rows, metadata)
    with pytest.raises(AllMethodsDatasetFormatError):
        load_all_methods_equations(*paths)


def test_duplicate_json_key_is_rejected(generated, tmp_path):
    csv_path, metadata_path = save_pair(
        tmp_path, generated.equations, generated.metadata,
    )
    lines = metadata_path.read_text().splitlines()
    lines[0] = '{"equation_id":"duplicate",' + lines[0][1:]
    metadata_path.write_text('\n'.join(lines) + '\n')
    with pytest.raises(AllMethodsDatasetFormatError):
        load_all_methods_equations(csv_path, metadata_path)


def test_full_validator_rejects_modified_npz(generated, tmp_path):
    paths = write_dataset(generated, tmp_path)['paths']
    with np.load(paths['ml'], allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays['X'][0, 0, 0] += 0.125
    np.savez(paths['ml'], **arrays)
    with pytest.raises(GenerationError):
        validate_dataset_artifacts(tmp_path)
