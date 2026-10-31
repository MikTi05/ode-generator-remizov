"""Development-only feature descriptions and nested family-fold variability.

No historical test labels, test metrics or numerical methods are accessed.
Nested selection uses one reproducible approximately quarter-sized family
holdout inside each outer training fold. It reuses the outer fitted models,
so two scenarios require exactly 160 fits, including fixed-default controls.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diagnostics.run_selector_hypotheses import (
    CONFIGS, METHODS, SEED, WEIGHTS, check_pins, csv_write, fit_model,
    load_dev, read, score_model, summary, write,
)
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from ml.selector_hypotheses import choices, metrics, selection_key, threshold_scan
from ml.selector_research_features import FEATURE_NAMES


def _subset(rows, indices):
    return [rows[int(i)] for i in indices]


def _assert_development(rows):
    if (len(rows) != 847 or len({r['equation_id'] for r in rows}) != 847
            or any(r['old_split'] == 'test' or r['new_split'] == 'test' for r in rows)):
        raise ValueError('only the 847 unique development rows are permitted')


def features(out):
    rows, matrices, names = load_dev(out)
    _assert_development(rows)
    X = matrices['augmented']
    own_names = names['augmented']
    records = []
    partitions = []
    for scenario in ('old', 'new'):
        for split in ('train', 'validation'):
            ix = [i for i, r in enumerate(rows) if r[scenario + '_split'] == split]
            own = _subset(rows, ix)
            y = np.array([r['best_method'] == METHODS[1] for r in own], dtype=int)
            partitions.append(dict(scenario=scenario, split=split, **summary(own),
                mc_families=len({r['family_id'] for r in own if r['best_method'] == METHODS[1]})))
            for j, name in enumerate(own_names):
                values = X[ix, j]
                r = dict(scenario=scenario, split=split, feature=name,
                         feature_set='additional5' if name in FEATURE_NAMES else 'original253',
                         AUC=float(roc_auc_score(y, values)) if len(np.unique(y)) == 2 else None)
                for label, selected in (('R', y == 0), ('MC', y == 1)):
                    values_class = values[selected]
                    r[label + '_n'] = int(selected.sum())
                    for quantile, q in (('q25', .25), ('median', .5), ('q75', .75)):
                        r[label + '_' + quantile] = float(np.quantile(values_class, q)) if len(values_class) else None
                records.append(r)
    csv_write(out / 'feature_distributions.csv', records)
    winners = []
    for i, r in enumerate(rows):
        if r['best_method'] == METHODS[1]:
            winners.append({**r, **{name:float(X[i, j]) for j, name in enumerate(own_names)}})
    csv_write(out / 'development_MC_wins.csv', winners)
    top = {}
    for scenario in ('old', 'new'):
        for split in ('train', 'validation'):
            own = [r for r in records if r['scenario'] == scenario and r['split'] == split]
            top[scenario + '_' + split] = sorted(own, key=lambda r:abs(r['AUC'] - .5), reverse=True)[:20]
    write(out / 'feature_analysis.json', dict(
        development_rows=847, test_labels_accessed=False, partitions=partitions,
        feature_count=len(own_names), original_feature_count=253,
        additional_feature_names=list(FEATURE_NAMES),
        constant_on_development=[name for j, name in enumerate(own_names) if np.ptp(X[:, j]) == 0],
        input_audit=dict(
            source='ml/remizov_features.py: coefficient_features and load_inputs',
            allowed=['given coefficient channels a,b,c,g', 'input coefficient grid',
                     'five analytic coefficient certificates', 'a_raw/b_raw/c_raw coefficient ASTs'],
            forbidden_not_used=['exact/y/y_AST', 'method outputs/errors/success/time',
                                'winner', 'equation_id/family_id/split as model features'],
            forcing='g is an input of the equation; its manufactured origin does not make it a result channel',
            additional='ml/selector_research_features.py: five finite-grid interactions on [-1,1], not certificates'),
        distributions='feature_distributions.csv', real_MC_wins='development_MC_wins.csv',
        strongest_descriptive_differences=top,
        interpretation='Descriptive overlapping distributions only. Family dependence, 258 comparisons, and selection on development prevent causal or confirmatory significance claims.',
        additional_features='Re-express available input information; no independent new evidence or solver calls.',
        numerical_calls=0))
    print('FEATURES', len(records), 'distributions;', len(winners), 'non-test MC winners', flush=True)


def specs():
    return ([dict(id=f'classifier_w{w}', kind='classifier', weight=w) for w in WEIGHTS]
            + [dict(id=k, kind=k, weight=1) for k in ('raw', 'asinh')])


def _group_split(rows, n_splits):
    y = np.array([r['best_method'] == METHODS[1] for r in rows], dtype=int)
    groups = np.array([r['family_id'] for r in rows])
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return splitter.split(np.zeros((len(rows), 1)), y, groups)


def _separated(left, right):
    for key in ('equation_id', 'family_id', 'structure_signature'):
        if {r[key] for r in left} & {r[key] for r in right}:
            raise ValueError('CV leakage in ' + key)


def _policy_summary(records, predictions, dev_rows):
    by_id = {r['equation_id']:r for r in dev_rows}
    output = []
    keys = ('mean_regret', 'mean_error', 'mc_precision', 'mc_recall', 'mc_f1',
            'macro_f1', 'accuracy', 'success_rate', 'mc_correct', 'mc_false', 'mc_predictions')
    for scenario, policy in sorted({(r['scenario'], r['policy']) for r in records}):
        own = [r for r in records if r['scenario'] == scenario and r['policy'] == policy]
        preds = [r for r in predictions if r['scenario'] == scenario and r['policy'] == policy]
        ids = [r['equation_id'] for r in preds]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate outer OOF prediction')
        pooled = metrics([by_id[pid] for pid in ids], [r['selected_method'] for r in preds])
        variability = {key:dict(mean=float(np.mean([r[key] for r in own])),
                               sd=float(np.std([r[key] for r in own], ddof=1)),
                               minimum=float(min(r[key] for r in own)),
                               maximum=float(max(r[key] for r in own))) for key in keys}
        output.append(dict(scenario=scenario, policy=policy, folds=len(own), pooled=pooled,
                           fold_variability=variability))
    return output


def cv(out):
    if (out / 'cv_summary.json').exists() or (out / 'CV_STARTED.json').exists():
        raise ValueError('CV already started; preserve this run')
    if (out / 'FINAL_STARTED.json').exists():
        raise ValueError('development analysis must precede the final test stage')
    plan = read(out / 'plan.json')
    if plan['fit_budget']['cv_default_and_nested_max'] < 160:
        raise ValueError('plan must predeclare the 160-fit inner-holdout protocol')
    check_pins(out)
    write(out / 'CV_STARTED.json', dict(seed=SEED, expected_fits=160,
          protocol='5 outer family folds; one 4-fold SGKF inner holdout; 8 baseline original253 specifications'))
    rows, matrices, names = load_dev(out)
    _assert_development(rows)
    X, feature_names = matrices['X'], names['X']
    records, predictions, manifests, inner_records = [], [], [], []
    fit_count = 0
    started = time.perf_counter()
    for scenario in ('old', 'new'):
        indices = np.array([i for i, r in enumerate(rows) if r[scenario + '_split'] == 'train'])
        scenario_rows = _subset(rows, indices)
        for fold, (train_local, valid_local) in enumerate(_group_split(scenario_rows, 5), 1):
            ti, vi = indices[train_local], indices[valid_local]
            tr, va = _subset(rows, ti), _subset(rows, vi)
            _separated(tr, va)
            inner_t, inner_v = next(iter(_group_split(tr, 4)))
            it, iv = ti[inner_t], ti[inner_v]
            itr, iva = _subset(rows, it), _subset(rows, iv)
            _separated(itr, iva)
            manifest = dict(scenario=scenario, fold=fold, seed=SEED,
                outer_train_ids=[r['equation_id'] for r in tr],
                outer_validation_ids=[r['equation_id'] for r in va],
                inner_train_ids=[r['equation_id'] for r in itr],
                inner_validation_ids=[r['equation_id'] for r in iva],
                outer_train=summary(tr), outer_validation=summary(va),
                inner_train=summary(itr), inner_validation=summary(iva))
            manifests.append(manifest)
            outer_scores, inner_candidates = {}, []
            for spec in specs():
                sid, kind, weight = spec['id'], spec['kind'], spec['weight']
                model, scale = fit_model(kind, CONFIGS[0], weight, X[ti], tr, feature_names)
                fit_count += 1
                scores = score_model(model, kind, scale, X[vi], feature_names)
                outer_scores[sid] = scores
                default = .5 if kind == 'classifier' else 0.
                selected = choices(scores, default)
                result = metrics(va, selected)
                records.append(dict(scenario=scenario, fold=fold, policy=sid,
                    threshold=default, chosen_spec=sid, outer_train_MC=summary(tr)['classes'].get(METHODS[1], 0),
                    outer_validation_MC=summary(va)['classes'].get(METHODS[1], 0), **result))
                predictions.extend(dict(scenario=scenario, fold=fold, policy=sid,
                    equation_id=r['equation_id'], family_id=r['family_id'],
                    score=float(score), threshold=default, selected_method=prediction,
                    best_method=r['best_method'], E_R=r['E_R'], E_MC=r['E_MC'])
                    for r, score, prediction in zip(va, scores, selected))
                inner_model, inner_scale = fit_model(kind, CONFIGS[0], weight, X[it], itr, feature_names)
                fit_count += 1
                inner_scores = score_model(inner_model, kind, inner_scale, X[iv], feature_names)
                _, best = threshold_scan(iva, inner_scores, kind == 'classifier')
                inner_candidates.append(dict(spec=sid, **best))
            for constant in METHODS:
                sid = 'always_' + constant
                inner_candidates.append(dict(spec=sid, threshold=0., **metrics(iva, [constant] * len(iva))))
                scores = np.full(len(va), 1. if constant == METHODS[1] else -1.)
                outer_scores[sid] = scores
                selected = [constant] * len(va)
                records.append(dict(scenario=scenario, fold=fold, policy=sid, threshold=0.,
                    chosen_spec=sid, outer_train_MC=summary(tr)['classes'].get(METHODS[1], 0),
                    outer_validation_MC=summary(va)['classes'].get(METHODS[1], 0), **metrics(va, selected)))
                predictions.extend(dict(scenario=scenario, fold=fold, policy=sid,
                    equation_id=r['equation_id'], family_id=r['family_id'], score=float(score),
                    threshold=0., selected_method=prediction, best_method=r['best_method'],
                    E_R=r['E_R'], E_MC=r['E_MC']) for r, score, prediction in zip(va, scores, selected))
            best = min(inner_candidates, key=lambda r:(selection_key(r), r['spec'], -r['threshold']))
            for candidate in inner_candidates:
                inner_records.append(dict(scenario=scenario, fold=fold,
                    chosen=candidate is best, inner_train_n=len(itr), inner_validation_n=len(iva),
                    **{key:value for key, value in candidate.items() if key != 'fixed_grid'}))
            scores = outer_scores[best['spec']]
            selected = choices(scores, best['threshold'])
            records.append(dict(scenario=scenario, fold=fold, policy='nested_selection',
                threshold=best['threshold'], chosen_spec=best['spec'],
                outer_train_MC=summary(tr)['classes'].get(METHODS[1], 0),
                outer_validation_MC=summary(va)['classes'].get(METHODS[1], 0), **metrics(va, selected)))
            predictions.extend(dict(scenario=scenario, fold=fold, policy='nested_selection',
                equation_id=r['equation_id'], family_id=r['family_id'], score=float(score),
                threshold=best['threshold'], selected_method=prediction, best_method=r['best_method'],
                E_R=r['E_R'], E_MC=r['E_MC']) for r, score, prediction in zip(va, scores, selected))
            write(out / 'cv_progress.json', dict(completed_folds=len(manifests), fit_count=fit_count,
                  elapsed_seconds=time.perf_counter() - started))
            print('CV', scenario, fold, 'fits', fit_count, 'nested', best['spec'],
                  'regret', records[-1]['mean_regret'], flush=True)
    if fit_count != 160:
        raise ValueError('unexpected CV fit count')
    csv_write(out / 'cv_folds.csv', records)
    csv_write(out / 'cv_predictions.csv', predictions)
    csv_write(out / 'cv_inner_selection.csv', inner_records)
    write(out / 'cv_splits.json', dict(seed=SEED, folds=manifests,
          group_keys=['family_id', 'structure_signature'],
          inner_rule='first validation fold of SGKF(n_splits=4,shuffle=True,random_state=2488)'))
    write(out / 'cv_summary.json', dict(
        fit_count=fit_count, expected_fit_count=160, seed=SEED,
        test_labels_accessed=False, numerical_calls=0,
        classifier_configs=CONFIGS[:1], classifier_weights=WEIGHTS,
        feature_set='original253', regression_targets=['raw', 'asinh'],
        outer_folds=5, inner_holdout='one approximately quarter-sized family fold; threshold/spec selected inside each outer train',
        global_selected_hyperparameters_used=False,
        policy_summary=_policy_summary(records, predictions, rows),
        nested_selection_counts={s:dict(Counter(r['chosen_spec'] for r in records
            if r['scenario'] == s and r['policy'] == 'nested_selection')) for s in ('old', 'new')},
        outer_MC_supports={s:[r['outer_validation']['classes'].get(METHODS[1], 0) for r in manifests
                             if r['scenario'] == s] for s in ('old', 'new')},
        limitations='Five correlated training-fold estimates; inner validation has few MC wins. Variability does not establish a prospective test improvement.',
        elapsed_seconds=time.perf_counter() - started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['features', 'cv', 'all'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    folder = args.out.resolve()
    if args.stage in ('features', 'all'):
        features(folder)
    if args.stage in ('cv', 'all'):
        cv(folder)
