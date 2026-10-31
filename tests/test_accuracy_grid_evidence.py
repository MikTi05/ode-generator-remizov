"""Verify saved evidence without numerical solves or model fitting."""
import json
from pathlib import Path

import pytest

from ml.accuracy_grid_evidence import build_grid_evidence

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def evidence():
    return build_grid_evidence(ROOT)


def test_primary_grid_uses_historical_mass_values_and_primary_mc_seed(evidence):
    rows = [r for r in evidence['rows'] if r['group'] == 'mass_R128_MC1024_primary_grid33']
    assert len(rows) == 100
    different_from_x0 = 0
    for row in rows:
        assert row['epsilon'] == .05
        assert row['remizov_source'].startswith('results/chernov_solve1000_2026-10-18/solutions/')
        assert row['remizov_config']['boundary'] == 'taper'
        assert row['remizov_n'] == 128
        assert row['mc_M'] == 1024
        assert row['mc_seeds'] == [202610250330]
        assert row['mc_seed_count'] == 1
        r = json.loads((ROOT / row['remizov_source']).read_text())
        m = json.loads((ROOT / row['mc_sources'][0]).read_text())
        assert row['E_R'] == max(abs(a - b) for a, b in zip(r['result']['f'], r['y_exact']))
        assert row['E_MC'] == max(abs(a - b) for a, b in zip(m['result']['f'], m['y_exact']))
        if row['E_MC'] != abs(m['result']['f'][16] - m['y_exact'][16]):
            different_from_x0 += 1
    assert different_from_x0 > 0


def test_grid_missingness_and_separate_method_profiles(evidence):
    groups = {g['name']: g for g in evidence['audit']['grid_groups']}
    primary = groups['mass_R128_MC1024_primary_grid33']
    assert primary['pairs_n'] == 100
    assert primary['missing_pairs_n'] == 900
    assert {s: r['missing_pairs_n'] for s, r in primary['by_split'].items()} == {
        'train': 600, 'validation': 150, 'test': 150}
    assert primary['winners'] == {'remizov': 95, 'monte_carlo': 5}
    assert primary['success_counts'] == {'remizov': 87, 'monte_carlo': 29, 'oracle': 87}
    assert groups['secondary_R256_MC1024_primary_grid33']['pairs_n'] == 37
    assert groups['secondary_R256_MC4096_grid33']['pairs_n'] == 48
    assert groups['external_R256_MC4096_grid33']['pairs_n'] == 20
    assert groups['external_R256_MC4096_grid33']['missing_pairs_n'] == 40
    for row in evidence['rows']:
        expected_m = 1024 if 'MC1024' in row['group'] else 4096
        assert row['mc_M'] == expected_m
        assert all(config['M'] == expected_m for config in row['mc_configs'])


def test_mc_timing_replays_are_not_independent_seed_trials(evidence):
    original = [r for r in evidence['rows'] if r['group'] == 'secondary_R256_MC4096_grid33']
    f0047 = next(r for r in original if r['equation_id'] == 'AMV1_F0047_S00')
    assert f0047['mc_seed_count'] == 12
    assert len(set(f0047['mc_seeds'])) == 12
    assert len(evidence['audit']['mc4096_replays_excluded']) == 4
    group = next(g for g in evidence['audit']['grid_groups']
                 if g['name'] == 'secondary_R256_MC4096_grid33')
    assert group['seed_count_distribution'] == {4: 43, 8: 4, 12: 1}
    # Raw fixed-profile files: 20 final5 + 20 sentinel + 140 pilot +
    # 4 timing + 4 later seeds + 32 validation; timing replays repeat 4 seeds.
    assert sum(len(r['mc_seed_errors']) for r in original) == 220 - 4


def test_external_rows_keep_parent_dataset_identity(evidence):
    rows = [r for r in evidence['rows'] if r['cohort'] == 'independent_60']
    expected_sha = '2069976b7772c13b17a73e4e995f346fa0b3db2f7c6d425458aeea079971f130'
    assert len(rows) == 20
    assert {r['dataset_sha256'] for r in rows} == {expected_sha}
    assert {r['mc_seed_count'] for r in rows} == {1}
    assert {tuple(r['mc_seeds']) for r in rows} == {(248800100100,)}
    assert all(r['best_method'] == 'remizov' for r in rows)


def test_wrong_point_stream_is_rejected(monkeypatch):
    target = ROOT / 'results/joint_grid100_2026-10-25/records/AMV1_F0001_S02_MC0.json'
    read_text = Path.read_text

    def corrupt_saved_stream(path, *args, **kwargs):
        text = read_text(path, *args, **kwargs)
        if path == target:
            raw = json.loads(text)
            raw['result']['point_spawn_keys'][16][0] += 1
            return json.dumps(raw)
        return text

    monkeypatch.setattr(Path, 'read_text', corrupt_saved_stream)
    with pytest.raises(ValueError, match='correct semantic point streams'):
        build_grid_evidence(ROOT)
