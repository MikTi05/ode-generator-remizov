"""Independently verify saved mean-four MC labels and classifier evaluation.

Reads immutable numerical results and performs saved-model inference only.
Does not import evidence builders, fit a model, or run a numerical method.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PREVIOUS = ROOT / 'results/ml_accuracy_selector_2026-10-31/run'
DATA = ROOT / 'data/all_methods_v1_1000_2026-10-18'
METHODS = ['remizov', 'monte_carlo']
SEEDS = [202610250101, 202610250102, 202610250103, 202610250104]
EPSILON = .05


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def csvread(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same_float(actual, expected, message):
    require(actual is not None and math.isclose(float(actual), float(expected),
                                              rel_tol=1e-14, abs_tol=1e-15), message)


def verify(out, previous=PREVIOUS):
    import numpy as np
    from catboost import CatBoostClassifier, Pool
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
    from ml.mean_mc_selector import MeanMCMethodSelector

    out, previous = Path(out).resolve(), Path(previous).resolve()
    require(out != previous, 'new and previous runs must differ')
    evidence = read(out / 'evidence.json')
    metrics = read(out / 'metrics.json')
    audit = read(out / 'data_audit.json')
    registry = read(out / 'model/registry.json')
    previous_registry = read(previous / 'model/registry.json')
    require(registry['schema'] == 'two_method_x0_mc_mean4_accuracy_v1', 'wrong mean-four schema')
    require(registry['epsilon'] == EPSILON and registry['methods'] == METHODS,
            'wrong two-method accuracy contract')
    require(registry['fit_count'] == 1 and audit['fit_count'] == 1,
            'one declared fit required')
    require(registry['catboost'] == previous_registry['catboost'],
            'the fixed CatBoost configuration changed')
    require(registry['class_balancing'] is False and
            registry['validation_test_used_for_fit_or_tuning'] is False,
            'fit or balancing contract changed')
    context = registry['comparison_context']
    require(context['remizov']['n'] == 128 and context['remizov']['boundary'] == 'taper',
            'primary Remizov profile changed')
    require(context['monte_carlo']['estimator'] == 'abs(mean(f_seed(0))-exact(0))',
            'mean of predictions must precede absolute error')
    require(context['monte_carlo']['seeds'] == SEEDS and context['monte_carlo']['M'] == 1024,
            'four frozen M1024 seeds required')

    # Verify both output sets; the preceding run is not rewritten by this task.
    old_artifacts = read(previous / 'artifact_sha256.json')
    for name, expected in old_artifacts.items():
        require(sha(previous / name) == expected, 'previous artifact changed: ' + name)
    artifacts = read(out / 'artifact_sha256.json')
    for name, expected in artifacts.items():
        require(sha(out / name) == expected, 'new artifact changed: ' + name)
    source_pins = read(out / 'source_sha256.json')
    for name, expected in source_pins.items():
        require(sha(ROOT / name) == expected, 'source changed: ' + name)

    # Features are copied from the preceding 253-input artifact, byte for byte.
    require(sha(out / 'input_features.npz') == sha(previous / 'input_features.npz'),
            'previous coefficient feature artifact was not preserved byte-for-byte')
    with np.load(out / 'input_features.npz', allow_pickle=False) as z:
        features = z['X'].copy()
        names = z['feature_names'].tolist()
        ids = z['equation_id'].tolist()
        splits = z['split'].tolist()
        families = z['family_id'].tolist()
    require(features.shape == (1000, 253) and np.isfinite(features).all(),
            'finite unchanged 1000-by-253 coefficient features required')
    require(names == registry['feature_names'] and len(set(names)) == 253,
            'model/feature name mismatch')
    require(Counter(splits) == {'train': 700, 'validation': 150, 'test': 150},
            'original split changed')
    require(len(set(ids)) == 1000, 'duplicate feature equation ID')
    meta = {m['equation_id']: m for line in
            (DATA / 'all_methods_equations_v1_metadata.jsonl').read_text().splitlines()
            for m in [json.loads(line)]}
    require(set(ids) == set(meta), 'feature IDs differ from canonical dataset')
    lookup = {pid: i for i, pid in enumerate(ids)}
    family_split, topology_split = {}, {}
    for pid, split, family in zip(ids, splits, families):
        m = meta[pid]
        require(m['split'] == split and m['family_id'] == family,
                'feature/canonical split identity mismatch')
        require(family_split.setdefault(family, split) == split, 'family leakage')
        require(topology_split.setdefault(m['structure_signature'], split) == split,
                'topology leakage')

    rows = evidence['main_rows']
    coverage = evidence['coverage_rows']
    require(len(coverage) == 1000 and len(rows) == 997, 'wrong comparable/missing coverage')
    by_id = {r['equation_id']: r for r in rows}
    require(len(by_id) == 997 and set(by_id).issubset(meta), 'duplicate or unknown pair')
    old_coverage = {r['equation_id']: r for r in read(previous / 'evidence.json')['coverage_rows']}
    require(set(r['equation_id'] for r in coverage) == set(meta), 'coverage lost an equation')
    mc_records_checked = 0
    changed_labels = 0
    raw_truth = {}
    missing = []
    for row in coverage:
        pid = row['equation_id']
        m, old = meta[pid], old_coverage[pid]
        for key in ('expression_hash', 'family_id', 'split', 'structure_signature'):
            require(row[key] == m[key] == old[key], 'equation identity changed: ' + pid)
        require(row['epsilon'] == EPSILON, 'epsilon mismatch')
        sources = row['mc_sources']
        require(len(sources) == 4, 'four raw MC sources required')
        values, errors, exacts, seeds = [], [], [], []
        for source in sources:
            raw = read(ROOT / source)
            cfg, z = raw['config'], raw['result']
            require(raw['metadata_identity']['expression_hash'] == m['expression_hash'],
                    'MC expression mismatch')
            require(z['x'] == [0.] and cfg['M'] == 1024 and cfg['dt'] == 1 / 128 and
                    cfg['tail_eps'] == 1e-4 and z['success'], 'wrong MC profile or point')
            require(z['completed_paths'] == [1024] and z['seed_entropy'] == cfg['seed'],
                    'MC seed/completed paths mismatch')
            exact = float(np.asarray(raw['y_exact']).reshape(-1)[0])
            value = float(z['f'][0])
            values.append(value)
            errors.append(abs(value - exact))
            exacts.append(exact)
            seeds.append(cfg['seed'])
            mc_records_checked += 1
        require(seeds == SEEDS and len(set(seeds)) == 4 and len(set(exacts)) == 1,
                'independent frozen seeds/exact required')
        require(values == row['mc_seed_f0'] == old['mc_seed_f0'], 'saved MC estimates changed')
        require(errors == row['mc_seed_errors'], 'saved MC continuous seed errors changed')
        fmean = float(np.mean(values))
        em = abs(fmean - exacts[0])
        require(row['mc_mean_f0'] == fmean and row['E_MC'] == em,
                'error must be abs(mean(f_seed)-exact): ' + pid)
        require(row['E_MC_median_seed_error'] == float(np.median(errors)),
                'historical median error diagnostic changed')
        require(row['mc_success'] == (em <= EPSILON), 'mean MC success mismatch')
        if not row['pair_available']:
            require(row['E_R'] is None and row['best_method'] is None and pid not in by_id,
                    'missing Remizov answer received a synthetic label')
            missing.append(pid)
            continue
        require(pid in by_id, 'comparable coverage row lost from ML sample')
        rraw = read(ROOT / row['remizov_sources'][0])
        rz = rraw['result']
        require(rz['x'][16] == 0. and rz['n'] == 128 and
                rraw['config']['boundary'] == 'taper' and rz['success'],
                'wrong historical Remizov profile')
        require(rraw['metadata']['expression_hash'] == m['expression_hash'], 'Remizov expression mismatch')
        require(rraw['y_exact'][16] == exacts[0], 'R/MC exact mismatch')
        er = abs(rz['f'][16] - rraw['y_exact'][16])
        winner = METHODS[0] if er <= em else METHODS[1]
        require(er == row['E_R'] and er == old['E_R'], 'Remizov raw error changed')
        require(winner == row['best_method'] and row['remizov_success'] == (er <= EPSILON) and
                row['chosen_success'] == (min(er, em) <= EPSILON), 'winner/success mismatch')
        require(by_id[pid] == row, 'main/coverage row mismatch')
        changed_labels += winner != old['best_method']
        raw_truth[pid] = (er, em, winner)
    require(sorted(missing) == ['AMV1_F0134_S03', 'AMV1_F0136_S01', 'AMV1_F0139_S00'],
            'wrong missing Remizov pairs')

    # Declared fit population is checked against the original split and topology.
    train_rows = [r for r in rows if r['split'] == 'train']
    train_ids = [r['equation_id'] for r in train_rows]
    require(registry['train_equations'] == train_ids, 'declared fit IDs mismatch')
    require(all(splits[lookup[pid]] == 'train' for pid in train_ids), 'holdout in fit population')
    require(registry['train_families'] == sorted({meta[pid]['family_id'] for pid in train_ids}),
            'declared fit families mismatch')
    require(registry['train_topologies'] == sorted({meta[pid]['structure_signature'] for pid in train_ids}),
            'declared fit topologies mismatch')
    require(registry['observed_training_classes'] == dict(Counter(raw_truth[pid][2] for pid in train_ids)),
            'fit class counts do not match new mean labels')

    raw_model = CatBoostClassifier()
    raw_model.load_model(str(out / 'model/model.cbm'))
    actual_params = raw_model.get_all_params()
    for key in ('iterations', 'depth', 'l2_leaf_reg', 'loss_function', 'random_seed',
                'bootstrap_type', 'random_strength', 'class_names'):
        require(actual_params[key] == registry['catboost'][key],
                'saved CatBoost parameter mismatch: ' + key)
    require(math.isclose(actual_params['learning_rate'], registry['catboost']['learning_rate'],
                         rel_tol=1e-7, abs_tol=1e-12), 'saved learning rate changed')
    require(actual_params.get('class_weights') in (None, [1., 1.]) and
            actual_params.get('auto_class_weights') == 'None' and
            actual_params.get('use_best_model') is False,
            'class balancing or holdout-based best model selection present')
    require(raw_model.get_probability_threshold() == .5, 'post-fit decision threshold changed')
    require(list(raw_model.classes_) == METHODS and list(raw_model.feature_names_) == names,
            'saved CatBoost class or feature order mismatch')
    require(sha(out / 'model/model.cbm') == registry['model_sha256'], 'model hash mismatch')
    wrapper = MeanMCMethodSelector(out / 'model')
    saved_predictions = csvread(out / 'predictions.csv')
    prediction_by_id = {r['equation_id']: r for r in saved_predictions}
    require(len(prediction_by_id) == len(saved_predictions) == len(rows) and
            set(prediction_by_id) == set(by_id), 'prediction coverage mismatch')
    metric_groups_checked = 0
    all_selected = []
    outliers = []
    for split in ('train', 'validation', 'test'):
        own = [r for r in rows if r['split'] == split]
        values = features[[lookup[r['equation_id']] for r in own]]
        pool = Pool(values, feature_names=names)
        selected = raw_model.predict(pool, thread_count=1).reshape(-1).tolist()
        probabilities = raw_model.predict_proba(pool, thread_count=1)
        wrapped, wrapped_probabilities = wrapper.predict_features(values, names)
        require(selected == wrapped and np.array_equal(probabilities, wrapped_probabilities),
                'wrapper/direct CatBoost mismatch')
        all_selected.extend(selected)
        for row, choice, probability in zip(own, selected, probabilities):
            pid = row['equation_id']
            saved = prediction_by_id[pid]
            require(saved['split'] == split and saved['best_method'] == row['best_method'] and
                    saved['selected_method'] == choice, 'saved model decision mismatch')
            require(float(saved['p_remizov']) == probability[0] and
                    float(saved['p_monte_carlo']) == probability[1], 'saved probability mismatch')
            er, em, winner = raw_truth[pid]
            error = er if choice == 'remizov' else em
            require(float(saved['E_R']) == er and float(saved['E_MC']) == em and
                    float(saved['E_selected']) == error and float(saved['regret']) == error - min(er, em),
                    'saved selected error/regret mismatch')
            require(saved['selected_success'] == str(error <= EPSILON), 'selected success mismatch')
            if split == 'test':
                outliers.append(dict(equation_id=pid, E_R=er, E_MC=em, winner=winner,
                                     selected_method=choice, regret=error-min(er, em),
                                     p_remizov=float(probability[0]), p_monte_carlo=float(probability[1])))
        truth = [r['best_method'] for r in own]
        for strategy, choice in (('ML', selected), ('always_remizov', ['remizov'] * len(own)),
                                 ('always_monte_carlo', ['monte_carlo'] * len(own)), ('oracle', truth)):
            actual = metrics[split][strategy]
            require(actual['n'] == len(own) and actual['class_order'] == METHODS,
                    'metric population or class order mismatch')
            matrix = confusion_matrix(truth, choice, labels=METHODS).tolist()
            require(actual['confusion_matrix'] == matrix, 'confusion matrix mismatch')
            expected = dict(accuracy=accuracy_score(truth, choice),
                            balanced_accuracy=balanced_accuracy_score(truth, choice),
                            macro_f1=f1_score(truth, choice, labels=METHODS,
                                              average='macro', zero_division=0))
            errors = np.asarray([r['E_R'] if p == 'remizov' else r['E_MC']
                                 for r, p in zip(own, choice)])
            oracle = np.asarray([min(r['E_R'], r['E_MC']) for r in own])
            regret = errors - oracle
            expected.update(success_rate=np.mean(errors <= EPSILON), mean_error=np.mean(errors),
                            median_error=np.median(errors), mean_regret=np.mean(regret),
                            median_regret=np.median(regret), max_regret=np.max(regret))
            for key, value in expected.items():
                same_float(actual[key], value, split + '/' + strategy + '/' + key + ' mismatch')
            counts = Counter(choice)
            require(actual['selection_counts'] == {m: counts[m] for m in METHODS},
                    'selection counts mismatch')
            require(actual['success_count'] == int(np.sum(errors <= EPSILON)), 'success count mismatch')
            for method in METHODS:
                same_float(actual['selection_fractions'][method], counts[method] / len(own),
                           'selection fraction mismatch')
            metric_groups_checked += 1

    test_matrix = read(out / 'confusion_matrix_test.json')
    require(test_matrix['matrix'] == metrics['test']['ML']['confusion_matrix'] and
            test_matrix['class_order'] == METHODS and test_matrix['n'] == 150,
            'standalone test confusion matrix mismatch')
    example = read(out / 'example_input.json')
    normal_prediction = wrapper.predict(**example)
    contaminated = json.loads(json.dumps(example))
    contaminated['coefficient_metadata'].update(y_exact=1e300, E_R=0., E_MC=1e300,
                                               success=False, runtime_seconds=1e300)
    require(wrapper.predict(**contaminated) == normal_prediction,
            'forbidden result metadata changed inference')
    require(read(out / 'example_prediction.json') == normal_prediction, 'saved API example mismatch')
    outliers.sort(key=lambda row: row['regret'], reverse=True)
    return dict(status='passed', epsilon=EPSILON, aggregation='arithmetic_mean_of_four_saved_MC1024_estimates',
                raw_MC_records_checked=mc_records_checked, raw_pairs_checked=len(rows),
                absent_pairs=missing, changed_winner_labels_vs_median=changed_labels,
                source_hashes_checked=len(source_pins), artifact_hashes_checked=len(artifacts),
                previous_artifact_hashes_checked=len(old_artifacts),
                unchanged_input_feature_sha256=sha(out / 'input_features.npz'),
                features_n=253, original_split=dict(Counter(splits)),
                fixed_CatBoost_configuration_verified=True, binary_decision_threshold=.5,
                declared_fit_n=len(train_ids), family_and_topology_leakage=False,
                direct_CatBoost_predictions_checked=len(saved_predictions),
                strategy_split_metrics_checked=metric_groups_checked,
                classifier_choices=dict(Counter(all_selected)),
                largest_test_regret=outliers[0], test_confusion_matrix=test_matrix['matrix'],
                forbidden_result_metadata_ignored=True, numerical_calls=0, fit_calls=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--previous', type=Path, default=PREVIOUS)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify(args.out, args.previous)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(text)
    print(text, end='')
