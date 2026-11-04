"""Input-only R128 / MC mean4 / PI selection on saved x=0 evidence.

The existing feature builders and CatBoost configurations are reused. No ODE
solver, generator, exact evaluator, or result reader is imported here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ml.accuracy_selector import CATBOOST_CONFIG, EPSILON
from ml.remizov_features import coefficient_features
from ml.selector_research_features import additional_coefficient_features

METHODS = ("remizov", "monte_carlo", "product_integral")
SEED = 2488
CONFIGS = [dict(name="baseline", iterations=120, depth=2, l2_leaf_reg=10),
           dict(name="medium", iterations=240, depth=3, l2_leaf_reg=10),
           dict(name="deeper", iterations=300, depth=4, l2_leaf_reg=20)]
WEIGHTS = [1, 2, 3, 5, 8, 12]


def winner(errors):
    values = np.asarray(errors, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("three finite nonnegative errors required")
    return METHODS[int(np.argmin(values))]


def error_matrix(rows):
    try:
        values = np.asarray([[r["E_" + m] for m in METHODS] for r in rows], dtype=float).reshape(-1, 3)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("three measured errors required") from exc
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("finite nonnegative measured errors required")
    for r, v in zip(rows, values):
        if r["best_method"] != winner(v):
            raise ValueError("label disagrees with measured errors")
    return values


def decide(scores, biases=(0., 0.)):
    scores = np.asarray(scores, dtype=float)
    biases = np.asarray(biases, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != 3 or biases.shape != (2,) or not np.isfinite(scores).all() or not np.isfinite(biases).all():
        raise ValueError("finite three-method scores and two biases required")
    indices = np.argmax(scores + np.r_[0., biases], axis=1)
    return [METHODS[int(i)] for i in indices]


def metrics(rows, predictions):
    predictions = list(predictions)
    if len(predictions) != len(rows) or any(p not in METHODS for p in predictions):
        raise ValueError("one supported method prediction per equation required")
    errors = error_matrix(rows)
    cm = np.zeros((3, 3), dtype=int)
    truth = np.argmin(errors, axis=1) if len(rows) else np.empty(0, dtype=int)
    pred = np.asarray([METHODS.index(p) for p in predictions], dtype=int)
    np.add.at(cm, (truth, pred), 1)
    support, count = cm.sum(axis=1), cm.sum(axis=0)
    precision = np.divide(cm.diagonal(), count, out=np.zeros(3), where=count != 0)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(3), where=support != 0)
    f1 = np.divide(2 * cm.diagonal(), support + count, out=np.zeros(3), where=support + count != 0)
    selected = errors[np.arange(len(rows)), pred]
    regrets = selected - errors.min(axis=1) if len(rows) else selected
    return dict(n=len(rows), status="evaluated" if rows else "no_comparable_tasks",
                class_order=list(METHODS), confusion_matrix=cm.tolist(),
                accuracy=float(np.trace(cm) / len(rows)) if rows else None,
                balanced_accuracy=float(recall[support > 0].mean()) if rows else None,
                macro_f1=float(f1.mean()) if rows else None,
                precision=dict(zip(METHODS, precision.tolist())),
                recall={m: float(recall[i]) if support[i] else None for i, m in enumerate(METHODS)},
                f1=dict(zip(METHODS, f1.tolist())),
                true_class_counts=dict(zip(METHODS, support.tolist())),
                selection_counts=dict(zip(METHODS, count.tolist())),
                selection_fractions={m: int(count[i]) / len(rows) if rows else None for i, m in enumerate(METHODS)},
                non_R_predictions=int(count[1:].sum()), selection_errors=int(len(rows) - np.trace(cm)),
                success_count=int(np.sum(selected <= EPSILON)),
                success_rate=float(np.mean(selected <= EPSILON)) if rows else None,
                mean_error=float(selected.mean()) if rows else None,
                median_error=float(np.median(selected)) if rows else None,
                max_error=float(selected.max()) if rows else None,
                mean_regret=float(regrets.mean()) if rows else None,
                median_regret=float(np.median(regrets)) if rows else None,
                max_regret=float(regrets.max()) if rows else None)


def selection_key(result):
    return (result["mean_regret"], -result["success_rate"],
            -result["macro_f1"], result["non_R_predictions"])


def policy_scan(rows, scores, kind):
    """Predeclared 2D coarse grid, then <=3 coordinate partition sweeps.

    Each coordinate enumerates boundaries and both neighbouring float values.
    Two deterministic starts are used. This is a coordinate-local search;
    no global optimality of the two-dimensional search is asserted.
    """
    scores = np.asarray(scores, dtype=float)
    if not rows or scores.shape != (len(rows), 3) or not np.isfinite(scores).all():
        raise ValueError("finite nonempty development scores required")
    errors = error_matrix(rows)
    truth = np.argmin(errors, axis=1)
    oracle = errors.min(axis=1)
    support = np.bincount(truth, minlength=3)
    visited = {}

    def evaluate(pairs, phase):
        pairs = sorted(set(tuple(map(float, pair)) for pair in pairs) - set(visited))
        if not pairs:
            return
        # All policies are evaluated on exactly the same unrounded saved errors.
        for begin in range(0, len(pairs), 512):
            own = pairs[begin:begin + 512]
            offset = np.c_[np.zeros(len(own)), np.asarray(own)]
            pred = np.argmax(scores[None, :, :] + offset[:, None, :], axis=2)
            losses = errors[np.arange(len(rows))[None, :], pred]
            counts = np.stack([(pred == j).sum(axis=1) for j in range(3)], axis=1)
            tp = np.stack([((pred == j) & (truth[None, :] == j)).sum(axis=1) for j in range(3)], axis=1)
            denominator = counts + support
            f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0)
            for i, biases in enumerate(own):
                visited[biases] = dict(biases=list(biases), phase=phase,
                    mean_regret=float(np.mean(losses[i] - oracle)),
                    success_rate=float(np.mean(losses[i] <= EPSILON)),
                    macro_f1=float(f1[i].mean()), non_R_predictions=int(counts[i, 1:].sum()))

    def rank(row):
        return selection_key(row) + (sum(abs(v) for v in row["biases"]), *row["biases"])

    if kind == "classifier":
        values = np.log((1 - np.arange(1, 11) / 20) / (np.arange(1, 11) / 20))
        axes = [values, values]
    elif kind in ("raw", "asinh"):
        axes = [np.unique(np.r_[0., np.quantile(np.max(np.delete(scores, j, axis=1), axis=1) - scores[:, j], np.arange(1, 10) / 10)]) for j in (1, 2)]
    else:
        raise ValueError("unknown score kind")
    evaluate([(a, b) for a in axes[0] for b in axes[1]] + [(0., 0.)], "coarse")
    coarse = min(visited.values(), key=rank)["biases"]
    for start in ([0., 0.], coarse):
        current = list(start)
        for sweep in range(3):
            before = tuple(current)
            for j in (1, 2):
                adjusted = scores + np.r_[0., current]
                boundary = np.max(np.delete(adjusted, j, axis=1), axis=1) - scores[:, j]
                levels = np.unique(np.r_[boundary, np.nextafter(boundary, -np.inf),
                                         np.nextafter(boundary, np.inf), current[j - 1], 0.])
                pairs = [tuple(v if k == j - 1 else current[k] for k in range(2)) for v in levels]
                evaluate(pairs, "coordinate")
                choices = [visited[p] for p in set(pairs) | {tuple(current)}]
                current = min(choices, key=rank)["biases"]
            if tuple(current) == before:
                break
    best = min(visited.values(), key=rank)
    exact = metrics(rows, decide(scores, best["biases"]))
    return list(visited.values()), dict(biases=best["biases"], **exact)


def assert_separated(left, right):
    for key in ("equation_id", "family_id", "structure_signature"):
        if {r[key] for r in left} & {r[key] for r in right}:
            raise ValueError("split leakage: " + key)


def fit_bundle(kind, config, weight, X, rows, names):
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    params = dict(CATBOOST_CONFIG)
    params.update({k: v for k, v in config.items() if k != "name"})
    errors = error_matrix(rows)
    if kind == "classifier":
        present = [m for m in METHODS if any(r["best_method"] == m for r in rows)]
        if len(present) < 2:
            raise ValueError("classification requires at least two observed training classes")
        params.update(loss_function="MultiClass" if len(present) > 2 else "Logloss",
                      class_names=present, class_weights=[1 if m == METHODS[0] else weight for m in present])
        model = CatBoostClassifier(**params)
        model.fit(Pool(X, label=[r["best_method"] for r in rows], feature_names=names))
        return [model], []
    if kind not in ("raw", "asinh"):
        raise ValueError("unknown fit kind")
    params.pop("class_names")
    params.pop("allow_const_label")
    params["loss_function"] = "RMSE"
    models, scales = [], []
    for j in (1, 2):
        y = errors[:, 0] - errors[:, j]
        scale = max(float(np.median(np.abs(y))), 1e-12) if kind == "asinh" else 1.
        target = np.arcsinh(y / scale) if kind == "asinh" else y
        model = CatBoostRegressor(**params)
        model.fit(Pool(X, label=target, feature_names=names))
        models.append(model)
        scales.append(scale)
    return models, scales


def score_bundle(models, kind, scales, X, names):
    from catboost import Pool
    pool = Pool(X, feature_names=names)
    if kind == "classifier":
        raw = models[0].predict_proba(pool, thread_count=1)
        probabilities = np.zeros((len(X), 3))
        for i, label in enumerate(models[0].classes_):
            probabilities[:, METHODS.index(str(label))] = raw[:, i]
        return np.log(np.maximum(probabilities, np.finfo(float).tiny)), probabilities
    predictions = [np.asarray(model.predict(pool, thread_count=1), dtype=float) for model in models]
    if kind == "asinh":
        predictions = [scale * np.sinh(p) for scale, p in zip(scales, predictions)]
    scores = np.column_stack([np.zeros(len(X))] + predictions)
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite regression scores")
    return scores, None


def load_bundle(candidate, out):
    from catboost import CatBoostClassifier, CatBoostRegressor
    models = []
    for path in candidate["models"]:
        model = CatBoostClassifier() if candidate["kind"] == "classifier" else CatBoostRegressor()
        model.load_model(str(Path(out) / path))
        models.append(model)
    return models


def calibration(rows, probabilities):
    if probabilities is None:
        return None
    from sklearn.metrics import log_loss, roc_auc_score
    truth = np.asarray([METHODS.index(r["best_method"]) for r in rows])
    y = np.eye(3)[truth]
    confidence, pred = probabilities.max(axis=1), probabilities.argmax(axis=1)
    bins = []
    for i in range(10):
        mask = (confidence >= i / 10) & ((confidence < (i + 1) / 10) if i < 9 else confidence <= 1)
        if mask.any():
            bins.append(dict(lower=i / 10, n=int(mask.sum()), confidence=float(confidence[mask].mean()),
                             accuracy=float(np.mean(pred[mask] == truth[mask]))))
    auc = {m: float(roc_auc_score(y[:, j], probabilities[:, j])) if len(np.unique(y[:, j])) == 2 else None for j, m in enumerate(METHODS)}
    return dict(brier_multiclass=float(np.mean(np.sum((probabilities - y) ** 2, axis=1))),
                logloss=float(log_loss(truth, probabilities, labels=[0, 1, 2])),
                ece=float(sum(b["n"] * abs(b["confidence"] - b["accuracy"]) for b in bins) / len(rows)),
                roc_auc_ovr=auc, bins=bins,
                interpretation="raw model scores only; descriptive, not a probability calibration guarantee")


class ThreeMethodSelector:
    """Reload the frozen Validation-selected bundle; inputs contain no results."""
    def __init__(self, folder):
        self.folder = Path(folder)
        self.registry = json.loads((self.folder / "registry.json").read_text())
        if self.registry["methods"] != list(METHODS):
            raise ValueError("method contract mismatch")
        for name, expected in self.registry["model_sha256"].items():
            if hashlib.sha256((self.folder / name).read_bytes()).hexdigest() != expected:
                raise ValueError("model checksum mismatch")
        self.models = load_bundle(self.registry["candidate"], self.folder)

    def predict_features(self, X, names):
        values = np.asarray(X, dtype=float)
        if list(names) != self.registry["feature_names"] or values.ndim != 2 or values.shape[1] != len(names) or not np.isfinite(values).all():
            raise ValueError("feature schema mismatch")
        c = self.registry["candidate"]
        if c["kind"] == "constant":
            return [c["constant"]] * len(values), None
        scores, probabilities = score_bundle(self.models, c["kind"], c["scales"], values, names)
        return decide(scores, self.registry["biases"]), probabilities

    def predict(self, coefficients, x_grid, metadata):
        x = np.asarray(x_grid, dtype=float)
        if hashlib.sha256(x.tobytes()).hexdigest() != self.registry["input_grid_sha256"]:
            raise ValueError("original coefficient grid required")
        features = coefficient_features(coefficients, x, metadata)
        if self.registry["candidate"]["feature_set"] == "augmented":
            features.update(additional_coefficient_features(coefficients, x))
        names = self.registry["feature_names"]
        selected, probabilities = self.predict_features([[features[n] for n in names]], names)
        return dict(selected_method=selected[0], raw_probabilities=None if probabilities is None else dict(zip(METHODS, probabilities[0].tolist())),
                    scope="saved-profile comparison at x=0 on AMV1; no accuracy certificate")
