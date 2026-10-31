import copy
import numpy as np
import pytest
from ml.revision_evidence import equation_key,collapse_accuracy,semantic_MC_key,profile_of,read,CONTRACT
from ml.revision_selector import action,candidate_grid,check_fit,fit_binomial_sigmoid,sigmoid,weighted_cdf,family_blocks


def trial(seed=1,source='one'):
    return dict(equation_key='key',dataset_sha256='a'*64,equation_id='same',expression_hash='b'*64,family_id='topology',canonical_family_id='F0001',
        split='train',profile='MC1',T=10.,N=1280,seed=seed,stream_id=0,rng_scheme='PCG',point_spawn_keys=[[1,2]],f=[1.],error=.005,accurate=True,source=source)


def test_dataset_qualified_identity():
    assert equation_key('a'*64,'same','b'*64)!=equation_key('c'*64,'same','b'*64)
    assert equation_key('a'*64,'same','b'*64)!=equation_key('a'*64,'same','c'*64)
    with pytest.raises(ValueError):equation_key('short','same','b'*64)


def test_MC_dedup_counts_real_seeds_not_rows():
    a=trial();b=trial(source='replay');c=trial(seed=2,source='independent');c.update(f=[2.],error=.02,accurate=False)
    rows,duplicates=collapse_accuracy([a,b,c])
    assert len(rows)==1 and rows[0]['n']==2 and rows[0]['k']==1 and len(duplicates)==1
    assert semantic_MC_key(a)==semantic_MC_key(b)!=semantic_MC_key(c)


def test_conflicting_replay_rejected():
    b=trial(source='bad');b['f']=[3.]
    with pytest.raises(ValueError,match='conflicting MC'):collapse_accuracy([trial(),b])


def test_deterministic_repeats_not_trials():
    a=trial();a['profile']='R128';b=copy.deepcopy(a);b['source']='another'
    rows,_=collapse_accuracy([a,b]);assert rows[0]['n']==1 and rows[0]['k']==1


def test_profiles_resources_are_not_numerics():
    contract=read(CONTRACT/'plan.json');cfg=copy.deepcopy(contract['profiles']['MC1']['config']);cfg.update(seed=123,max_path_steps=999999999,max_runtime_seconds=200)
    assert profile_of(cfg,contract)=='MC1'
    cfg['M']=1024;assert profile_of(cfg,contract) is None
    cfg=copy.deepcopy(contract['profiles']['R128']['config']);cfg['T']=50
    assert profile_of(cfg,contract)=='R128'
    cfg['boundary']='taper';assert profile_of(cfg,contract) is None
    cfg=copy.deepcopy(contract['profiles']['R128']['config']);cfg['tail_eps']=1e-8
    assert profile_of(cfg,contract) is None


def test_held_family_cannot_fit_or_calibrate():
    rows=[dict(split='train',family_id='held',k=0,n=4,equation_key='one')]
    with pytest.raises(ValueError):check_fit(rows,{'held'})
    with pytest.raises(ValueError):fit_binomial_sigmoid(rows,[.1],{'held'})
    with pytest.raises(ValueError):check_fit([dict(rows[0],split='validation')],set())


def test_binomial_calibration_replicate_equivalence():
    rows=[dict(split='train',family_id='a',equation_key='one',k=1,n=4),dict(split='train',family_id='b',equation_key='two',k=3,n=4)]
    c=fit_binomial_sigmoid(rows,[.25,.75],set())
    assert c['n_trials']==8 and c['equation_keys']==['one','two']
    assert abs(c['slope']-1)<1e-5 and abs(c['intercept'])<1e-5


def test_MC_remains_reachable_without_forcing():
    e={p:dict(p=.99,seconds=t) for p,t in [('R128',12),('R256',24),('MC1',5)]}
    assert action(e,dict(rule='parity',delta=.02))=='MC1'
    e['MC1']['p']=.1
    assert action(e,dict(rule='parity',delta=.02))=='R128'
    e['MC1'].update(p=.99,seconds=11.5)
    assert action(e,dict(rule='parity',delta=.02))=='R128'


def test_CDF_and_family_split():
    assert weighted_cdf(dict(residuals=[-1,0,1],weights=[1,1,2]),[0])[0]==.5
    blocks=family_blocks(list('abcdefghi'));assert len(set.union(*blocks))==9 and sum(map(len,blocks))==9
    assert len(candidate_grid())==20


def test_saved_training_targets_and_quarantine():
    from ml.revision_evidence import ROOT
    out=ROOT/'results/ml_train_revision_2026-10-27';rows=read(out/'accuracy.json');quarantine=read(out/'quarantine.json')
    mc=[r for r in rows if r['profile']=='MC1']
    assert len(mc)==40 and sum(r['n'] for r in mc)==180 and sum(r['k'] for r in mc)==12
    assert len({r['family_id'] for r in mc if r['k']})==3
    assert not {s for r in rows for s in r['sources']}&{r['source'] for r in quarantine}
    assert all(r['dataset_sha256']==read(out/'audit.json')['original_dataset_sha256'] for r in rows)


def test_nested_policy_never_sees_outer_labels():
    from ml.revision_evidence import ROOT
    out=ROOT/'results/ml_train_revision_2026-10-27'
    for fold in range(3):
        r=read(out/f'outer{fold}_selection.json')
        assert not set(r['outer_held_families'])&{z['family_id'] for z in r['inner_predictions']}
        assert r['selected_before_outer_evaluation']
    index=read(out/'bundle_index.json')
    for name,b in index.items():
        held=set(b['excluded_families'])
        assert not held&set(b['train_families'])
        for block in b['calibration']['subfolds']:
            for path in block['paths'].values():
                meta=read(out/path/'metadata.json')
                assert not (held|set(block['held_families']))&set(meta['families'])


def test_missing_times_not_converted_to_scored_costs():
    from ml.revision_evidence import ROOT
    m=read(ROOT/'results/ml_train_revision_2026-10-27/metrics.json')
    assert m['common_cost_tasks']==9
    for key in ('nested_selected','constant_R128','constant_R256'):
        row=m['policies'][key]
        assert row['observed_total_method_seconds'] is None and row['full40_cost_per_accurate'] is None
        assert row['missing_selected_times']>0


def test_research_dispatch_is_blocked():
    from ml.revision_evidence import ROOT
    from ml.revision_selector import ResearchRevision
    out=ROOT/'results/ml_train_revision_2026-10-27';model=ResearchRevision(out)
    with np.load(out/'features_train.npz',allow_pickle=False) as z:X=z['X'][:1]
    result=model.select_features(X)[0]
    assert result['action']=='abstain' and result['status']=='research_only' and not result['accuracy_certificate']


def test_legacy_small_dataset_never_merges_by_string_ID():
    from ml.revision_evidence import ROOT
    out=ROOT/'results/ml_train_revision_2026-10-27'
    entries=read(out/'legacy_scope_audit.json')['entries']
    legacy=next(r for r in entries if r['folder']=='results/joint_pilot_2026-10-25')
    assert not legacy['permitted_training_attempts']
    assert read(out/'audit.json')['original_dataset_sha256'] not in legacy['datasets']


def test_binomial_proper_score_is_not_MSE_of_empirical_fraction():
    from diagnostics.report_train_revision import probability_scores
    rows=[dict(family_id=g,k=1,n=2) for g in ('a','b')]
    scores=probability_scores(rows,[.5,.5])
    assert scores['brier']==.25
    assert scores['logloss']==pytest.approx(np.log(2))
