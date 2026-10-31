import numpy as np
import pytest

from ml.selector_hypotheses import (choices, metrics, selection_key,
                                    stratified_family_split, threshold_scan)


def row(pid,er,em,family='F0',split='train'):
    return dict(equation_id=pid,E_R=er,E_MC=em,
                best_method='remizov' if er<=em else 'monte_carlo',
                family_id=family,structure_signature=family,split=split)


def test_threshold_optimizes_actual_error_rather_than_label_accuracy():
    rows=[row('a',100,1),row('b',0,1),row('c',0,1)]
    scores=np.array([.1,.2,.3])
    table,best=threshold_scan(rows,scores)
    assert best['mc_predictions']==3
    assert best['mean_regret']==pytest.approx(2/3)
    assert {round(r['threshold'],2) for r in table if r['fixed_grid']}==set(np.arange(1,11)/20)
    assert selection_key(best)<selection_key(metrics(rows,['remizov']*3))


def test_threshold_enumerates_all_partitions_and_ties_choose_r():
    rows=[row('a',1,0),row('b',0,1),row('c',0,1)]
    scores=np.array([.2,.2,.8])
    table,best=threshold_scan(rows,scores,False)
    independent=min(metrics(rows,choices(scores,t))['mean_regret'] for t in [-1,.2,.5,.8,2])
    assert best['mean_regret']==independent
    assert choices([0],0)==['remizov']


def test_metrics_independent_confusion_and_regret():
    rows=[row('a',1,0),row('b',0,1),row('c',0,1)]
    result=metrics(rows,['monte_carlo','monte_carlo','remizov'])
    assert result['confusion_matrix']==[[1,1],[0,1]]
    assert result['mc_precision']==.5 and result['mc_recall']==1
    assert result['mc_f1']==pytest.approx(2/3)
    assert result['mean_regret']==pytest.approx(1/3)


def test_group_split_exact_reproducible_and_no_leakage():
    rows=[]
    for f in range(30):
        for s in range(5 if f!=29 else 4):
            rows.append(row(f'{f}_{s}',int(s==0),.5,f'F{f:02}',
                            'test' if f<3 else 'train'))
    a,info=stratified_family_split(rows,validation_size=20)
    b,_=stratified_family_split(rows,validation_size=20)
    assert a==b and sum(v=='validation' for v in a.values())==20
    assert all(a[r['equation_id']]=='test' for r in rows if r['split']=='test')
    for family in {r['family_id'] for r in rows}:
        assert len({a[r['equation_id']] for r in rows if r['family_id']==family})==1
    assert a['29_0']=='train'


def test_reject_duplicate_ids_and_related_topologies():
    r=row('a',0,1)
    with pytest.raises(ValueError,match='duplicate'):
        stratified_family_split([r,r])
    other=row('b',0,1,'F1');other['structure_signature']='F0'
    with pytest.raises(ValueError,match='topology'):
        stratified_family_split([r,other])
