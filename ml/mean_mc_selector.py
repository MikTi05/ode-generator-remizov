"""Binary input-only selector: R128 versus mean of four saved MC1024 estimates."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from ml.accuracy_selector import (AccuracyMethodSelector, CATBOOST_CONFIG,
                                  EPSILON, METHODS, selection_metrics, sha)
from ml.remizov_features import VERSION

SCHEMA = "two_method_x0_mc_mean4_accuracy_v1"
AGGREGATION = "arithmetic_mean_of_four_saved_MC1024_estimates"


def mean_accuracy_target(error_r, mc_seed_f0, y_exact0):
    values = np.asarray(mc_seed_f0, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all() or not np.isfinite(y_exact0):
        raise ValueError("four finite saved MC values and exact(0) required")
    if not np.isfinite(error_r) or error_r < 0:
        raise ValueError("real nonnegative Remizov error required")
    mean = float(np.mean(values, dtype=np.float64))
    error_mc = abs(mean - float(y_exact0))
    if not np.isfinite(error_mc):
        raise ValueError("nonfinite aggregate error")
    return dict(mc_mean_f0=mean, E_R=float(error_r), E_MC=error_mc,
                best_method=METHODS[0] if error_r <= error_mc else METHODS[1],
                remizov_success=bool(error_r <= EPSILON), mc_success=bool(error_mc <= EPSILON),
                chosen_success=bool(min(error_r, error_mc) <= EPSILON))


class MeanMCMethodSelector(AccuracyMethodSelector):
    """Reuse the same 253 input features; keep the former selector immutable."""

    def __init__(self, model_dir):
        from catboost import CatBoostClassifier
        folder = Path(model_dir)
        self.registry = r = json.loads((folder / "registry.json").read_text())
        if r["schema"] != SCHEMA or r["epsilon"] != EPSILON or r["methods"] != list(METHODS):
            raise ValueError("unsupported MC mean4 selector contract")
        if r["comparison_context"]["monte_carlo"]["aggregation"] != AGGREGATION:
            raise ValueError("wrong MC aggregation")
        if r["comparison_context"]["remizov"]["n"] != 128:
            raise ValueError("frozen mass R128 required")
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
