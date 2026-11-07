"""Scientific semantics of saved-only presentation; no numerical solver calls."""
import copy
import csv
import shutil
from pathlib import Path

import pytest

from diagnostics.present_saved_grid33 import ROOT, SOURCE, build, build_data, load_rows, requirement_rows
from diagnostics.verify_grid33_presentation import check_tex, verify


@pytest.fixture(scope='module')
def saved():
    return load_rows(ROOT/SOURCE/'grid33_all1000.csv')


def test_different_denominators_and_MC_unknown_are_preserved(saved):
    data=build_data(saved)
    rows={(r['scope'],r['method']):r for r in data['comparison']}
    r=rows['all_available','R']
    assert r['E33_pass']==970 and r['n_values']==997 and r['n_scope']==1000
    assert r['pass_fraction_available']==970/997 and r['pass_fraction_scope']==.97
    assert rows['all_available','PI']['E33_pass']==404
    assert rows['common997','PI']['E33_pass']==402
    mc=rows['all_available','MC']
    assert mc['metric']=='E0_lower_bound'
    assert mc['E33_pass'] is mc['E33_fail'] is mc['pass_fraction_available'] is None
    assert mc['E33_fail_lower_bound']==62 and mc['E33_unknown']==938 and mc['missing_E33']==1000


def test_exceedances_are_not_trimmed_and_missing_are_not_failures(saved):
    data=build_data(saved)
    assert len(data['exceedances'])==27 and data['R_above_one']==4
    assert data['exceedances'][0]['equation_id']=='AMV1_F0141_S00'
    assert data['R_largest_two_sum_share']==pytest.approx(.9884262516870093)
    assert len(data['missing_ids'])==3
    r=next(r for r in data['shares'] if r['method']=='R')
    assert (r['measured_pass'],r['measured_fail'],r['missing_answer'])==(970,27,3)
    assert r['denominator']==1000


def test_strict_winner_claim_rejected_when_MC_bound_is_too_small(saved):
    rows=copy.deepcopy(saved)
    rows[0]['MC_E0_lower_bound']=0
    with pytest.raises(ValueError,match='unique RK4 winner'):build_data(rows)


@pytest.mark.parametrize('mutation',['MC_full','duplicate_ID','false_accuracy','nonfinite'])
def test_false_scientific_input_is_rejected(tmp_path,mutation):
    with (ROOT/SOURCE/'grid33_all1000.csv').open(newline='') as handle:
        raw=list(csv.DictReader(handle))
    if mutation=='MC_full':raw[0]['MC_E33']='0.0'
    elif mutation=='duplicate_ID':raw[1]['equation_id']=raw[0]['equation_id']
    elif mutation=='false_accuracy':raw[0]['RK4_accuracy_met']='False'
    else:raw[0]['R_E33']='nan'
    path=tmp_path/'input.csv'
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(raw[0]));writer.writeheader();writer.writerows(raw)
    with pytest.raises(ValueError):load_rows(path)


def test_original_tolerance_and_speed_requirement_are_not_declared_closed():
    rows={r['id']:r for r in requirement_rows()}
    assert '0,01' in rows['T08']['requirement'] and 'не закрыто' in rows['T08']['status']
    assert 'не доказано' in rows['T06']['status']
    assert 'не сертифицировано' in rows['T07']['status']


def test_existing_output_rejected_before_source_read(tmp_path):
    with pytest.raises(ValueError,match='new output'):build(tmp_path/'absent',tmp_path)


def test_tex_validation_recognizes_escaped_percent_and_detects_bad_braces():
    assert check_tex(r'\caption{98.8\% $E_{33}$}')==[]
    with pytest.raises(ValueError):check_tex(r'\caption{missing brace')


def test_independent_verifier_rejects_fabricated_table(tmp_path):
    folder=ROOT/'results/grid33_scientific_presentation_2026-11-04'
    # Verification fails at the numerical table before images are accessed.
    for name in ('input_sha256.json','comparison.csv'):
        shutil.copyfile(folder/name,tmp_path/name)
    rows=list(csv.DictReader((tmp_path/'comparison.csv').open()))
    rows[0]['mean']='0'
    with (tmp_path/'comparison.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with pytest.raises(ValueError,match='comparison cell'):verify(ROOT,tmp_path)
