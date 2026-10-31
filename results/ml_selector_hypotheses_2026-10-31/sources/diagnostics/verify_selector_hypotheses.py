"""Independently verify the completed selector study without fitting or tuning.

Uses NumPy/sklearn rather than the experiment's metrics, split, or threshold
helpers. Run only after the locked final test stage has completed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (brier_score_loss, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score)

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('remizov', 'monte_carlo')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_read(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def decoded_records(path):
    records=[]
    for row in csv_read(path):
        parsed={}
        for key,value in row.items():
            if value in ('True','False'):
                parsed[key]=value=='True'
            else:
                try:
                    parsed[key]=json.loads(value)
                except json.JSONDecodeError:
                    parsed[key]=value
        records.append(parsed)
    return records


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, label):
    if isinstance(expected, dict):
        check(isinstance(actual, dict), label + ': dictionary required')
        for key, value in expected.items():
            check(key in actual, label + ': missing ' + key)
            close(actual[key], value, label + '.' + key)
    elif isinstance(expected, (list, tuple)):
        check(len(actual) == len(expected), label + ': length mismatch')
        for i, (a, b) in enumerate(zip(actual, expected)):
            close(a, b, label + '.' + str(i))
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        check(np.isclose(actual, expected, rtol=2e-12, atol=2e-12),
              label + ': ' + repr(actual) + ' != ' + repr(expected))
    else:
        check(actual == expected, label + ': value mismatch')


def independent_metrics(rows, predictions):
    y = [r['best_method'] for r in rows]
    er = np.asarray([r['E_R'] for r in rows], dtype=float)
    em = np.asarray([r['E_MC'] for r in rows], dtype=float)
    selected = np.asarray(predictions) == METHODS[1]
    errors = np.where(selected, em, er)
    regrets = errors - np.minimum(er, em)
    cm = confusion_matrix(y, predictions, labels=list(METHODS))
    tn, fp, fn, tp = map(int, cm.ravel())
    recalls = {m:float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None
               for i, m in enumerate(METHODS)}
    return dict(n=len(rows), class_order=list(METHODS), confusion_matrix=cm.tolist(),
        accuracy=float(np.mean(np.asarray(y) == np.asarray(predictions))),
        balanced_accuracy=float(np.mean([v for v in recalls.values() if v is not None])),
        macro_f1=float(f1_score(y, predictions, labels=list(METHODS),
                                average='macro', zero_division=0)),
        success_rate=float(np.mean(errors <= .05)), success_count=int(np.sum(errors <= .05)),
        mean_error=float(errors.mean()), median_error=float(np.median(errors)),
        mean_regret=float(regrets.mean()), median_regret=float(np.median(regrets)),
        max_regret=float(regrets.max()),
        true_class_counts={m:y.count(m) for m in METHODS},
        selection_counts={m:predictions.count(m) for m in METHODS},
        selection_fractions={m:predictions.count(m)/len(rows) for m in METHODS},
        recall=recalls, status='evaluated',
        mc_precision=float(precision_score(y, predictions, pos_label=METHODS[1], zero_division=0)),
        mc_recall=float(recall_score(y, predictions, pos_label=METHODS[1], zero_division=0)),
        mc_f1=float(f1_score(y, predictions, pos_label=METHODS[1], zero_division=0)),
        mc_correct=tp, mc_false=fp, mc_missed=fn, mc_predictions=tp+fp,
        selection_errors=fp+fn)


def predicted(scores, threshold):
    return [METHODS[int(s > threshold)] for s in scores]


def rank(result):
    return (result['mean_regret'], -result['success_rate'], -result['macro_f1'],
            result['mc_predictions'])


def independent_split(rows, seed):
    families = defaultdict(list)
    for r in rows:
        families[r['family_id']].append(r)
    strata = defaultdict(list)
    for fid, own in sorted(families.items()):
        if own[0]['split'] != 'test' and len(own) == 5:
            strata[sum(r['best_method'] == METHODS[1] for r in own)].append(fid)
    n = sum(map(len, strata.values()))
    quotas = {k:20*len(v)//n for k, v in strata.items()}
    remainder_order = sorted(strata, key=lambda k:(-(20*len(strata[k]) % n), k))
    for k in remainder_order[:20-sum(quotas.values())]:
        quotas[k] += 1
    rng = np.random.default_rng(seed)
    chosen = set()
    for k in sorted(strata):
        chosen.update(rng.permutation(strata[k])[:quotas[k]].tolist())
    return {r['equation_id']:('test' if r['split'] == 'test' else
            'validation' if r['family_id'] in chosen else 'train') for r in rows}


def verify_cv(out, by_id, split):
    manifests=read(out/'cv_splits.json')['folds']
    folds=decoded_records(out/'cv_folds.csv')
    predictions=decoded_records(out/'cv_predictions.csv')
    inner=decoded_records(out/'cv_inner_selection.csv')
    summary=read(out/'cv_summary.json')
    check(len(manifests)==10 and len(folds)==110 and len(inner)==100,'CV grid coverage')
    check(summary['fit_count']==summary['expected_fit_count']==160,'CV fit budget')
    check(summary['test_labels_accessed'] is False,'CV accessed test labels')
    for manifest in manifests:
        scenario=manifest['scenario']; fold=manifest['fold']
        ot=set(manifest['outer_train_ids']); ov=set(manifest['outer_validation_ids'])
        it=set(manifest['inner_train_ids']); iv=set(manifest['inner_validation_ids'])
        check(ot|ov==set(split[scenario]['train']['ids']) and not ot&ov,'outer CV partition')
        check(it|iv==ot and not it&iv,'inner CV partition')
        for left,right in ((ot,ov),(it,iv)):
            for key in ('family_id','structure_signature'):
                check(not {by_id[p][key] for p in left}&{by_id[p][key] for p in right},
                      'CV '+key+' leakage')
        for key, ids in (('outer_train',ot),('outer_validation',ov),('inner_train',it),('inner_validation',iv)):
            own=[by_id[p] for p in ids]
            close(manifest[key],dict(n=len(ids),classes=dict(Counter(r['best_method'] for r in own)),
                families=len({r['family_id'] for r in own})),'CV '+key+' summary')
        candidates=[r for r in inner if r['scenario']==scenario and r['fold']==fold]
        check(len(candidates)==10 and sum(r['chosen'] for r in candidates)==1,'inner candidate coverage')
        selected=min(candidates,key=lambda r:(rank(r),r['spec'],-r['threshold']))
        check(selected['chosen'],'inner selection criterion')
        nested=next(r for r in folds if r['scenario']==scenario and r['fold']==fold and r['policy']=='nested_selection')
        check(nested['chosen_spec']==selected['spec'],'nested chosen specification')
        close(nested['threshold'],selected['threshold'],'nested inner threshold')
        oracle_mean=np.mean([min(by_id[p]['E_R'],by_id[p]['E_MC']) for p in iv])
        for candidate in candidates:
            check(candidate['n']==len(iv),'inner metric support')
            close(candidate['mean_error']-candidate['mean_regret'],oracle_mean,'inner error/regret contract')

    for metric in folds:
        own=[p for p in predictions if all(p[k]==metric[k] for k in ('scenario','fold','policy'))]
        manifest=next(m for m in manifests if m['scenario']==metric['scenario'] and m['fold']==metric['fold'])
        check([p['equation_id'] for p in own]==manifest['outer_validation_ids'],'OOF prediction identity order')
        for p in own:
            check(p['selected_method']==METHODS[int(p['score']>p['threshold'])],'OOF decision threshold')
            close(p['threshold'],metric['threshold'],'OOF threshold')
            close(p['E_R'],by_id[p['equation_id']]['E_R'],'OOF R error')
            close(p['E_MC'],by_id[p['equation_id']]['E_MC'],'OOF MC error')
        close(metric,independent_metrics([by_id[p['equation_id']] for p in own],
                                         [p['selected_method'] for p in own]),'CV fold metrics')
    for policy in summary['policy_summary']:
        scenario=policy['scenario']; name=policy['policy']
        own=[r for r in predictions if r['scenario']==scenario and r['policy']==name]
        ids=[r['equation_id'] for r in own]
        check(len(ids)==len(set(ids)) and set(ids)==set(split[scenario]['train']['ids']),
              'OOF complete train coverage')
        close(policy['pooled'],independent_metrics([by_id[p] for p in ids],
              [p['selected_method'] for p in own]),'CV pooled metrics')
        records=[r for r in folds if r['scenario']==scenario and r['policy']==name]
        check(len(records)==policy['folds']==5,'CV summary fold count')
        for key,value in policy['fold_variability'].items():
            values=np.array([r[key] for r in records],float)
            close(value,dict(mean=float(values.mean()),sd=float(values.std(ddof=1)),
                   minimum=float(values.min()),maximum=float(values.max())),'CV fold variability '+key)
    return dict(folds=len(manifests),fold_policies=len(folds),prediction_rows=len(predictions),
                inner_candidates=len(inner),fit_count=160,
                inner_threshold_optimality='not independently checked: inner scores not retained')


def verify(out):
    out = Path(out).resolve()
    check((out/'test_metrics.json').is_file(), 'final test must complete before verification')
    plan = read(out/'plan.json')
    source = ROOT/plan['source']
    evidence = read(source/'evidence.json')
    rows = evidence['main_rows']
    by_id = {r['equation_id']:r for r in rows}
    check(len(rows) == len(by_id) == len({r['expression_hash'] for r in rows}) == 997,
          '997 unique comparable equations and expressions required')
    for r in rows:
        close(r['E_R'], abs(r['remizov_f0']-r['y_exact0']), 'saved Remizov error')
        close(r['E_MC'], abs(float(np.mean(r['mc_seed_f0']))-r['y_exact0']), 'saved MC mean error')
        check(r['best_method'] == METHODS[int(r['E_MC'] < r['E_R'])], 'winner contract')
    pins = read(out/'preserved_sha256.json')
    for name, digest in pins.items():
        check((ROOT/name).is_file() and sha(ROOT/name) == digest, 'changed source: '+name)
    for name, digest in read(source/'artifact_sha256.json').items():
        check(sha(source/name) == digest, 'changed previous ML artifact: '+name)

    manifest = read(out/'splits.json')
    split = manifest['scenarios']
    test_ids = [r['equation_id'] for r in rows if r['split'] == 'test']
    new = independent_split(rows, manifest['seed'])
    for scenario, sizes in (('old', (697,150,150)), ('new', (747,100,150))):
        seen = set()
        families = {}; topologies = {}
        for s, size in zip(('train','validation','test'), sizes):
            own_ids = split[scenario][s]['ids']
            check(len(own_ids) == len(set(own_ids)) == size, scenario+' split size or duplicate')
            check(not seen.intersection(own_ids), scenario+' ID intersection')
            seen.update(own_ids)
            own = [by_id[p] for p in own_ids]
            close(split[scenario][s], dict(n=size, classes=dict(Counter(r['best_method'] for r in own)),
                  families=len({r['family_id'] for r in own})), scenario+'.'+s)
            for r in own:
                check(families.setdefault(r['family_id'],s) == s, scenario+' family leakage')
                check(topologies.setdefault(r['structure_signature'],s) == s, scenario+' topology leakage')
                check((r['split'] if scenario == 'old' else new[r['equation_id']]) == s,
                      scenario+' deterministic split mismatch')
        check(seen == set(by_id), scenario+' incomplete split')
        check(split[scenario]['test']['ids'] == test_ids, scenario+' fixed test changed')
    development = read(out/'development_rows.json')
    check({r['equation_id'] for r in development} == set(by_id)-set(test_ids), 'development includes test')
    for r in development:
        source_row = by_id[r['equation_id']]
        close(r['delta'], source_row['E_R']-source_row['E_MC'], 'regression delta')
        check(r['old_split'] == source_row['split'] and r['new_split'] == new[r['equation_id']],
              'development split mismatch')

    candidates = read(out/'candidates.json')
    by_candidate = {c['id']:c for c in candidates}
    check(len(candidates) == len(by_candidate) == 100, 'candidate grid coverage')
    check(sum(c['kind'] != 'constant' for c in candidates) == 96, 'fit grid coverage')
    for c in candidates:
        if c['kind'] != 'constant':
            check(sha(out/c['model']) == c['model_sha256'], 'changed model: '+c['id'])
    with np.load(out/'features.npz',allow_pickle=False) as z:
        feature_ids = z['equation_id'].tolist()
        feature_index = {p:i for i,p in enumerate(feature_ids)}
        feature_values = {k:z[k].copy() for k in ('X','augmented')}
        feature_names = {'X':z['feature_names'].tolist(),'augmented':z['augmented_names'].tolist()}
    with np.load(source/'input_features.npz',allow_pickle=False) as z:
        original_index = {p:i for i,p in enumerate(z['equation_id'].tolist())}
        check(np.array_equal(feature_values['X'],z['X'][[original_index[p] for p in feature_ids]]),
              'original feature values changed')
        check(feature_names['X']==z['feature_names'].tolist(),'original feature names changed')
    check(feature_values['X'].shape==(997,253) and feature_values['augmented'].shape==(997,258),
          'feature matrix shape')
    check(np.array_equal(feature_values['augmented'][:,:253],feature_values['X']),
          'ablation changed original feature channels')

    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    direct_scores = {}
    for c in candidates:
        if c['kind']=='constant':
            continue
        tr_ids = split[c['scenario']]['train']['ids']
        if c['kind']=='asinh':
            delta = [by_id[p]['E_R']-by_id[p]['E_MC'] for p in tr_ids]
            close(c['scale'],max(float(np.median(np.abs(delta))),1e-12),'train-only asinh scale')
        model = CatBoostClassifier() if c['kind']=='classifier' else CatBoostRegressor()
        model.load_model(str(out/c['model']))
        names = feature_names[c['feature_set']]
        check(model.feature_names_==names,'CBM feature names')
        indices = [feature_index[p] for p in feature_ids]
        pool = Pool(feature_values[c['feature_set']][indices],feature_names=names)
        if c['kind']=='classifier':
            check(list(model.classes_)==list(METHODS),'CBM class order')
            scores = model.predict_proba(pool,thread_count=1)[:,1]
        else:
            scores = model.predict(pool,thread_count=1)
            if c['kind']=='asinh':
                scores = c['scale']*np.sinh(scores)
        direct_scores[c['id']] = dict(zip(feature_ids,map(float,scores)))
    val_scores = defaultdict(dict)
    for r in csv_read(out/'validation_predictions.csv'):
        check(r['equation_id'] not in val_scores[r['candidate_id']], 'duplicate validation score')
        val_scores[r['candidate_id']][r['equation_id']] = float(r['score'])
        close(float(r['E_R']), by_id[r['equation_id']]['E_R'], 'validation R error')
        close(float(r['E_MC']), by_id[r['equation_id']]['E_MC'], 'validation MC error')
    val_metrics = read(out/'validation_metrics.json')
    policy_count = sum(1 if c['kind']=='constant' else 11 if c['kind']=='classifier' else 2
                       for c in candidates)
    check(len(val_metrics) == policy_count, 'fixed/default/optimized validation policy count')
    check(len({(p['candidate_id'],p['policy']) for p in val_metrics})==policy_count,
          'duplicate validation policy')
    scan_count = 0
    for c in candidates:
        ids = split[c['scenario']]['validation']['ids']
        own = [by_id[p] for p in ids]
        policies = [p for p in val_metrics if p['candidate_id'] == c['id']]
        if c['kind'] == 'constant':
            check(len(policies) == 1, 'constant policy coverage')
            predictions = [c['constant']]*len(own)
            close(policies[0], independent_metrics(own, predictions), c['id']+' validation')
            continue
        check(set(val_scores[c['id']]) == set(ids), 'validation prediction identity set')
        scores = np.array([val_scores[c['id']][p] for p in ids])
        close(scores.tolist(),[direct_scores[c['id']][p] for p in ids],c['id']+' direct CBM validation scores')
        fixed = np.arange(1,11)/20 if c['kind'] == 'classifier' else np.array([0.])
        thresholds = sorted(set(map(float, fixed)) |
            {float(np.nextafter(scores.min(),-np.inf))} | set(map(float,scores)))
        scan = csv_read(out/'thresholds'/f"{c['id']}.csv")
        check([float(r['threshold']) for r in scan] == thresholds, 'incomplete threshold scan '+c['id'])
        independent = []
        for entry, t in zip(scan, thresholds):
            result = independent_metrics(own, predicted(scores,t))
            parsed = {}
            for key, value in entry.items():
                try:
                    parsed[key] = json.loads(value)
                except json.JSONDecodeError:
                    parsed[key] = value
            close(parsed, result, c['id']+' threshold '+str(t))
            independent.append(dict(threshold=t, **result))
            scan_count += 1
        best = min(independent, key=lambda r:(rank(r),-r['threshold']))
        close(c['threshold'], best['threshold'], c['id']+' optimal threshold')
        check(len(policies) == (11 if c['kind']=='classifier' else 2), 'nonconstant policy coverage')
        for p in policies:
            if p['policy']=='default':
                expected_t = .5 if c['kind']=='classifier' else 0.
            elif p['policy']=='validation_optimal':
                expected_t = best['threshold']
            else:
                check(p['policy'].startswith('grid_'),'unknown policy')
                expected_t = float(p['policy'][5:])
            close(p['threshold'], expected_t, c['id']+' policy threshold')
            close(p, independent_metrics(own,predicted(scores,expected_t)), c['id']+' '+p['policy'])

    calibration = read(out/'calibration.json')
    check(len(calibration)==sum(c['kind']=='classifier' for c in candidates),'calibration model count')
    for result in calibration:
        c = by_candidate[result['candidate_id']]
        ids = split[c['scenario']]['validation']['ids']
        y = np.array([by_id[p]['best_method']==METHODS[1] for p in ids],int)
        scores = np.array([val_scores[c['id']][p] for p in ids])
        close(result['brier'],float(brier_score_loss(y,scores)),'Brier')
        close(result['logloss'],float(log_loss(y,scores,labels=[0,1])),'logloss')
        check(sum(b['n'] for b in result['bins'])==len(ids),'calibration bin coverage')
        close(result['ece'],sum(b['n']*abs(b['mean_score']-b['MC_fraction'])
                                for b in result['bins'])/len(ids),'ECE')

    lock = read(out/'selection_lock.json')
    for name, key in (('plan.json','plan_sha256'),('candidates.json','candidates_sha256'),
                      ('splits.json','splits_sha256'),('validation_metrics.json','validation_metrics_sha256')):
        check(sha(out/name) == lock[key], 'selection lock changed '+name)
    for name, digest in lock.get('source_sha256',{}).items():
        check(sha(ROOT/name)==sha(out/'sources'/name)==digest,'locked implementation changed '+name)
    check(lock['test_evaluated'] is False and lock['test_used_for_selection'] is False,
          'test was used before selection lock')
    for scenario in ('new','old'):
        # Metrics were independently verified above; preserve the predeclared exact tie order.
        selected = min((p for p in val_metrics if p['scenario']==scenario),
                       key=lambda p:(rank(p),p['candidate_id'],p['policy']))
        close(lock['selected'][scenario], selected, 'locked selected '+scenario)
    check(read(out/'FINAL_STARTED.json')['selection_lock_sha256'] == sha(out/'selection_lock.json'),
          'final stage did not use current lock')

    final = read(out/'test_metrics.json')
    check(final['test_evaluation_passes'] == 1 and final['exploratory'] and not final['independent_test'],
          'exploratory one-pass test interpretation')
    test_rows = [by_id[p] for p in test_ids]
    test_predictions = defaultdict(list)
    for r in csv_read(out/'test_predictions.csv'):
        test_predictions[(r['candidate_id'],r['policy'])].append(r)
    check(len(final['policies']) == len(test_predictions) == policy_count, 'test policy coverage')
    for p in final['policies']:
        key = (p['candidate_id'],p['policy'])
        table = test_predictions[key]
        check([r['equation_id'] for r in table] == test_ids, 'test identity order '+str(key))
        predictions = []
        for r, original in zip(table,test_rows):
            t = float(r['threshold']); score = float(r['score'])
            candidate = by_candidate[p['candidate_id']]
            expected_score = (1. if candidate['constant']==METHODS[1] else -1.) if candidate['kind']=='constant' else direct_scores[p['candidate_id']][r['equation_id']]
            close(score,expected_score,'direct CBM test score')
            check(r['best_method'] == original['best_method'], 'test winner')
            check(r['selected_method'] == METHODS[int(score > t)], 'test decision threshold')
            close(t,p['threshold'],'test policy threshold')
            for field in ('E_R','E_MC'):
                close(float(r[field]),original[field],'test '+field)
            error = original['E_R'] if r['selected_method']==METHODS[0] else original['E_MC']
            close(float(r['E_selected']),error,'selected error')
            close(float(r['regret']),error-min(original['E_R'],original['E_MC']),'test regret')
            predictions.append(r['selected_method'])
        close(p,independent_metrics(test_rows,predictions),'test '+str(key))
        locked = lock['selected'][p['scenario']]
        selected = all(p[k] == locked[k] for k in ('candidate_id','policy','threshold'))
        check(p['selected_on_validation'] == selected, 'test selected flag')
    for scenario in ('new','old'):
        selected = [p for p in final['policies'] if p['scenario']==scenario and p['selected_on_validation']]
        check(len(selected)==1 and final['selected'][scenario]==selected[0], 'final selected '+scenario)
    close(final['oracle'], independent_metrics(test_rows,[r['best_method'] for r in test_rows]), 'oracle')
    cv_result=verify_cv(out,by_id,split)
    result = dict(status='verified', verifier='independent_numpy_sklearn_no_experiment_helpers',
        unique_comparable_ids=997, preserved_files=len(pins), source_artifacts=31,
        exact_split_sizes=True, fixed_test_ids=150, no_ID_family_topology_leakage=True,
        development_only_selection=True, candidate_models=96, validation_policies=policy_count,
        threshold_rows=scan_count, test_policies=policy_count, test_prediction_rows=policy_count*150,
        numerical_solver_calls=0, model_fits=0, cv=cv_result,
        selection_lock_sha256=sha(out/'selection_lock.json'))
    (out/'independent_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.out),ensure_ascii=False,indent=2))
