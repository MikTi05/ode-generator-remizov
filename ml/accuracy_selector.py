"""Choose the lower-error of two frozen methods at x=0 from inputs only.

Numerical answers belong only to retrospective labels and evaluation. Inference
uses the existing 253 coefficient features and never executes either method.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from ml.remizov_features import CERT_FIELDS, VERSION, coefficient_features

EPSILON = 0.05
METHODS = ("remizov", "monte_carlo")
SCHEMA = "two_method_x0_accuracy_v1"
CATBOOST_CONFIG = dict(
    iterations=120, depth=2, learning_rate=0.05, l2_leaf_reg=10,
    loss_function="Logloss", random_seed=2488, thread_count=1,
    bootstrap_type="No", random_strength=0, allow_writing_files=False,
    verbose=False, allow_const_label=True, class_names=list(METHODS),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def accuracy_target(error_r, mc_seed_errors):
    """Median of absolute seed errors, never error of the mean prediction."""
    errors = np.asarray(mc_seed_errors, dtype=float)
    if not np.isfinite(error_r) or error_r < 0 or errors.ndim != 1 or not len(errors):
        raise ValueError("real nonnegative errors for both methods required")
    if not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError("invalid MC seed errors")
    error_mc = float(np.median(errors))
    return dict(E_R=float(error_r), E_MC=error_mc,
                best_method=METHODS[0] if error_r <= error_mc else METHODS[1],
                remizov_success=bool(error_r <= EPSILON),
                mc_success=bool(error_mc <= EPSILON),
                chosen_success=bool(min(error_r, error_mc) <= EPSILON))


def selection_metrics(rows, predictions):
    """Fixed two-class F1; balanced accuracy averages supported true classes.

    The empty split is unavailable, with null metrics and a zero-count matrix.
    It must never be represented as a zero-error or perfect evaluation.
    """
    predictions = list(predictions)
    if len(rows) != len(predictions) or any(p not in METHODS for p in predictions):
        raise ValueError("one valid method prediction per comparable equation required")
    matrix = [[0, 0], [0, 0]]
    counts = Counter(predictions)
    base = dict(n=len(rows), class_order=list(METHODS), confusion_matrix=matrix,
                accuracy=None, balanced_accuracy=None, macro_f1=None,
                success_rate=None, success_count=0, mean_error=None, median_error=None,
                mean_regret=None, median_regret=None, max_regret=None,
                true_class_counts=dict.fromkeys(METHODS, 0),
                selection_counts={m:counts[m] for m in METHODS},
                selection_fractions=dict.fromkeys(METHODS, None),
                recall=dict.fromkeys(METHODS, None))
    if not rows:
        return dict(base, status="no_comparable_pairs")
    errors = []
    regrets = []
    for row, prediction in zip(rows, predictions):
        er, em = float(row["E_R"]), float(row["E_MC"])
        if not np.isfinite([er, em]).all() or min(er, em) < 0:
            raise ValueError("missing or invalid pair cannot receive a metric")
        truth = METHODS[0] if er <= em else METHODS[1]
        if row["best_method"] != truth:
            raise ValueError("label disagrees with continuous errors")
        matrix[METHODS.index(truth)][METHODS.index(prediction)] += 1
        error = er if prediction == METHODS[0] else em
        errors.append(error)
        regrets.append(error - min(er, em))
    matrix = np.asarray(matrix)
    support = matrix.sum(axis=1)
    recalls = {m:float(matrix[i, i] / support[i]) if support[i] else None
               for i, m in enumerate(METHODS)}
    f1 = []
    for i in range(2):
        denominator = support[i] + matrix[:, i].sum()
        f1.append(float(2 * matrix[i, i] / denominator) if denominator else 0.)
    return dict(base, status="evaluated", confusion_matrix=matrix.tolist(),
                accuracy=float(np.trace(matrix) / len(rows)),
                balanced_accuracy=float(np.mean([v for v in recalls.values() if v is not None])),
                macro_f1=float(np.mean(f1)), recall=recalls,
                success_rate=float(np.mean(np.asarray(errors) <= EPSILON)),
                success_count=int(np.sum(np.asarray(errors) <= EPSILON)),
                mean_error=float(np.mean(errors)), median_error=float(np.median(errors)),
                mean_regret=float(np.mean(regrets)), median_regret=float(np.median(regrets)),
                max_regret=float(np.max(regrets)),
                true_class_counts={m:int(support[i]) for i, m in enumerate(METHODS)},
                selection_fractions={m:counts[m] / len(rows) for m in METHODS})


class AccuracyMethodSelector:
    """Frozen CatBoost binary classifier for the historical R128/MC1024 pair."""

    def __init__(self, model_dir):
        from catboost import CatBoostClassifier
        folder = Path(model_dir)
        self.registry = json.loads((folder / "registry.json").read_text())
        r = self.registry
        if r["schema"] != SCHEMA or r["epsilon"] != EPSILON or r["methods"] != list(METHODS):
            raise ValueError("unsupported accuracy selector contract")
        if r["feature_version"] != VERSION or r["feature_code_sha256"] != sha(Path(__file__).with_name("remizov_features.py")):
            raise ValueError("feature implementation changed")
        if sha(folder / "model.cbm") != r["model_sha256"]:
            raise ValueError("model artifact integrity failure")
        self.model = CatBoostClassifier()
        self.model.load_model(str(folder / "model.cbm"))
        if list(self.model.classes_) != list(METHODS):
            raise ValueError("unexpected classifier classes")
        if (len(r["feature_names"]) != 253 or len(set(r["feature_names"])) != 253
                or list(self.model.feature_names_) != r["feature_names"]):
            raise ValueError("unexpected model feature schema")

    def predict_features(self, values, feature_names):
        from catboost import Pool
        values = np.asarray(values, dtype=float)
        if list(feature_names) != self.registry["feature_names"]:
            raise ValueError("feature schema mismatch")
        if values.ndim != 2 or values.shape[1] != len(feature_names) or not np.isfinite(values).all():
            raise ValueError("finite feature matrix required")
        if not len(values):
            return [], np.empty((0, 2))
        pool = Pool(values, feature_names=feature_names)
        probabilities = self.model.predict_proba(pool, thread_count=1)
        predictions = self.model.predict(pool, thread_count=1).reshape(-1).tolist()
        return predictions, probabilities

    def predict(self, coefficients, x_grid, coefficient_metadata):
        x = np.asarray(x_grid, dtype=float)
        values = np.asarray(coefficients, dtype=float)
        if x.shape != (256,) or not np.isfinite(x).all():
            raise ValueError("original finite input grid required")
        if hashlib.sha256(x.tobytes()).hexdigest() != self.registry["input_grid_sha256"]:
            raise ValueError("original input grid required")
        if values.shape != (4, 256) or not np.isfinite(values).all():
            raise ValueError("finite a,b,c,g coefficient channels required")
        meta = {k:float(coefficient_metadata[k]) for k in CERT_FIELDS}
        if (not np.isfinite(list(meta.values())).all()
                or not 0 < meta["a_lower_cert"] <= meta["a_upper_cert"]
                or min(meta["b_abs_cert"], meta["remizov_c_abs_cert"]) < 0
                or meta["remizov_lambda"] <= meta["remizov_c_abs_cert"]):
            raise ValueError("invalid analytic coefficient certificates")
        meta["serialized_ast"] = {k:coefficient_metadata["serialized_ast"][k]
                                  for k in ("a_raw", "b_raw", "c_raw")}
        features = coefficient_features(values, x, meta)
        names = list(features)
        prediction, probabilities = self.predict_features([[features[n] for n in names]], names)
        return dict(selected_method=prediction[0],
                    probabilities={m:float(probabilities[0, i]) for i, m in enumerate(METHODS)},
                    criterion="lower_absolute_error_at_x0", epsilon=EPSILON,
                    comparison_context=self.registry["comparison_context"],
                    accuracy_certificate=False,
                    scope="x=0; no prediction of interval-wide accuracy or guaranteed success")
