"""Protocol helpers for retrospective, family-separated selector research.

The numerical contract is unchanged: lower saved absolute error at x=0,
with ties assigned to Remizov. Scores need not be calibrated probabilities.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import numpy as np

from ml.accuracy_selector import METHODS, selection_metrics


def stratified_family_split(rows, seed=2488, validation_size=100):
    """Keep the historical test; stratify full five-row development families.

    Three four-row families must stay in train to obtain exactly 747/100.
    Largest-remainder quotas stratify families by their number of MC winners.
    The split never uses model predictions or test outcomes.
    """
    if len({r['equation_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate equation ID')
    groups = defaultdict(list)
    topology_family = {}
    for row in rows:
        groups[row['family_id']].append(row)
        key = row['structure_signature']
        if topology_family.setdefault(key, row['family_id']) != row['family_id']:
            raise ValueError('topology links families: merge groups before splitting')
    test = {r['equation_id'] for r in rows if r['split'] == 'test'}
    strata = defaultdict(list)
    for family, own in sorted(groups.items()):
        splits = {r['split'] for r in own}
        if 'test' in splits:
            if splits != {'test'}:
                raise ValueError('historical test family leakage')
            continue
        if len(own) == 5:
            strata[sum(r['best_method'] == METHODS[1] for r in own)].append(family)
    if validation_size % 5:
        raise ValueError('validation size must preserve five-row groups')
    n_groups = validation_size // 5
    total = sum(map(len, strata.values()))
    quotas = {k:len(v)*n_groups//total for k,v in strata.items()}
    remaining = n_groups - sum(quotas.values())
    order = sorted(strata, key=lambda k:(-(len(strata[k])*n_groups % total), k))
    for key in order[:remaining]:
        quotas[key] += 1
    rng = np.random.default_rng(seed)
    selected = set()
    for key in sorted(strata):
        selected.update(rng.permutation(sorted(strata[key]))[:quotas[key]].tolist())
    split = {r['equation_id']:('test' if r['equation_id'] in test else
             'validation' if r['family_id'] in selected else 'train') for r in rows}
    counts = Counter(split.values())
    if counts['validation'] != validation_size:
        raise ValueError('group-preserving target size unavailable')
    return split, dict(seed=seed, strata_sizes={str(k):len(v) for k,v in strata.items()},
                      validation_family_quotas={str(k):v for k,v in quotas.items()},
                      validation_families=sorted(selected),
                      rule='largest_remainder_family_MC_count_then_seeded_permutation')


def choices(scores, threshold):
    return [METHODS[int(v > threshold)] for v in np.asarray(scores)]


def metrics(rows, predictions):
    result = selection_metrics(rows, predictions)
    cm = result['confusion_matrix']
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    result.update(mc_precision=tp/(tp+fp) if tp+fp else 0.,
                  mc_recall=tp/(tp+fn) if tp+fn else 0.,
                  mc_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                  mc_correct=tp, mc_false=fp, mc_missed=fn,
                  mc_predictions=tp+fp, selection_errors=fp+fn)
    return result


def selection_key(result):
    """Predeclared order: actual error, success, macro-F1, fewer MC choices."""
    return (result['mean_regret'], -result['success_rate'],
            -result['macro_f1'], result['mc_predictions'])


def threshold_scan(rows, scores, classifier=True):
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(rows),) or not np.isfinite(scores).all() or not len(rows):
        raise ValueError('finite nonempty validation scores required')
    fixed = np.arange(1,11)/20 if classifier else [0.]
    # With strict score > threshold, each distinct score represents one partition.
    exact = np.r_[np.nextafter(scores.min(), -np.inf), np.unique(scores)]
    candidates = sorted(set(map(float, fixed)) | set(map(float, exact)))
    table = [dict(threshold=t, fixed_grid=bool(t in fixed),
                  **metrics(rows, choices(scores,t))) for t in candidates]
    best = min(table, key=lambda r:(selection_key(r), -r['threshold']))
    return table, best
