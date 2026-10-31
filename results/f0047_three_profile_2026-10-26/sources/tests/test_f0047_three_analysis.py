"""Saved evidence regression: independent failure cannot become a lucky-seed win."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/f0047_three_profile_2026-10-26'


def read(name):return json.loads((OUT/name).read_text())


def test_lucky_later_seed_does_not_replace_prespecified_primary():
    a=read('analysis.json');errors=a['errors']['MC1']
    assert sum(e<=.01 for e in errors)==a['independent_accuracy']['k']
    assert errors[0]>.01 and errors[3]<=.01
    assert all(r['label']=='remizov' and r['mc_error']==errors[0] for r in a['pair_labels_new_independent_block'].values())


def test_speed_and_accuracy_are_distinct():
    a=read('analysis.json')
    assert all(a['MC_disjoint_faster_than'].values())
    assert a['fastest_observed_profile']=='R128' and a['best_accurate_R']=='R128'


def test_old_observed_label_not_overwritten_or_pooled_with_fresh_group():
    a=read('analysis.json');old=json.loads((ROOT/'results/f0047_observed_timing_2026-10-26/label.json').read_text())
    assert old['label']=='monte_carlo' and a['old_replay_block']['label']=='monte_carlo'
    assert a['independent_accuracy']['n']==4 and a['old_replay_block']['new_accuracy_trials_in_old_replay']==0
    assert not (OUT/'selected_labels_1000.json').exists()


def test_saved_screen_does_not_promote_nonprimary_hits_or_old_times():
    rows={r['equation_id']:r for r in read('train_candidate_rows.json')}
    for pid in ('AMV1_F0005_S02','AMV1_F0136_S01'):
        r=rows[pid];assert r['MC1_primary_error']>.01 and r['MC1_successes4']>0
        assert not r['cost_scope_comparable'] and not r['saved_cost_potential']
    assert len(rows)==40 and sum(r['MC1_successes4'] for r in rows.values())==7
