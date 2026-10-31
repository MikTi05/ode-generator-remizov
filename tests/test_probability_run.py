"""Actual fitted artifacts: no new fitting or numerical computation."""
import json
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/ml_probability_selector_2026-10-26'


def read(name):
    if not (OUT/name).exists():pytest.skip('saved diagnostic run not present')
    return json.loads((OUT/name).read_text())


def test_real_MC_counts_and_both_learned_heads_without_MC_winner_gate():
    a=read('data_audit.json');r=read('registry.json')
    assert a['counts']['MC1']==dict(equations=40,successes=7,trials=160,cost_equations=40)
    assert 'path' in r['heads']['MC1:probability']['models'][0] and 'path' in r['heads']['MC1:cost']['models'][0]
    assert not r['old_MC_winner_gate_applied'] and not r['deployment_ready']
    assert r['heads']['R128:cost']['metadata']['constant_replicas']==11


def test_family_cv_no_leakage_all_heads_and_train_scalers():
    cv=read('family_cv.json')
    for fold in cv['folds']:
        for h in fold['heads'].values():assert not set(h['families'])&set(fold['held_families'])
    assert len(cv['oof'])==len({r['family_id'] for r in cv['oof']})==40
    assert all(r['split']=='train' for r in cv['oof'])


def test_fresh_F0047_not_fitted_and_not_treated_as_more_families():
    a=read('seen_family_fresh_seed_audit.json');targets=read('accuracy_targets.json')
    assert not a['used_for_fit_or_calibration'] and a['fresh_k']==1 and a['fresh_n']==4
    row=next(r for r in targets if r['profile']=='MC1' and r['equation_id']==a['equation_id'])
    assert row['k']==row['n']==4 and len(row['sources'])==4
    assert all('stage1_episode' in p for p in row['sources'])


def test_serving_exposes_MC_and_blocks_uncalibrated_deployment():
    r=read('serving_check.json')['result']
    assert r['action']=='abstain' and r['status']=='research_only' and not r['accuracy_certificate']
    assert set(r['estimates'])=={'R128','R256','MC1'}
    assert 0<=r['estimates']['MC1']['probability']['point']<=1 and r['estimates']['MC1']['cost']['point']>0
    assert r['uncertainty_status']=='uncalibrated_family_bootstrap_model_spread'


def test_validation_proposal_uses_original_validation_only():
    r=read('NEXT_VALIDATION.json')
    assert len(r['cases'])==len({x['family_id'] for x in r['cases']})==8
    assert all(x['split']=='validation' for x in r['cases']) and not r['execution_authorized']
    assert r['total_calls']==64
