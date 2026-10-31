"""Independently check saved x0 labels, predictions, metrics and source hashes.

Only reads saved outputs and runs classifier inference. No fitting or numerical
method execution. Metrics are recomputed with sklearn, independently of the
training module's metric implementation.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from ml.accuracy_selector import AccuracyMethodSelector, EPSILON, METHODS, sha


def read(path):
    return json.loads(Path(path).read_text())


def verify(out):
    out = Path(out).resolve()
    evidence = read(out / "evidence.json")
    metrics = read(out / "metrics.json")
    model = AccuracyMethodSelector(out / "model")
    pins = read(out / "source_sha256.json")
    for name, expected in pins.items():
        if sha(ROOT / name) != expected:
            raise ValueError("changed source: " + name)
    rows = evidence["main_rows"]
    for row in rows:
        raw = read(ROOT / row["remizov_sources"][0])
        z = raw["result"]
        grid = z.get("x", raw.get("target_x"))
        index = grid.index(0.)
        if z["n"] != 128 or raw["config"]["boundary"] != "taper":
            raise ValueError("wrong primary Remizov profile")
        er = abs(z["f"][index] - raw["y_exact"][index])
        errors = []
        seeds = []
        for source in row["mc_sources"]:
            mc = read(ROOT / source)
            if mc["result"]["x"] != [0.] or mc["config"]["M"] != 1024:
                raise ValueError("wrong primary MC profile or point")
            exact = float(np.asarray(mc["y_exact"]).reshape(-1)[0])
            if exact != raw["y_exact"][index]:
                raise ValueError("exact mismatch")
            errors.append(abs(mc["result"]["f"][0] - exact))
            seeds.append(mc["config"]["seed"])
        if len(set(seeds)) != 4:
            raise ValueError("independent four seeds required")
        em = float(np.median(errors))
        winner = METHODS[0] if er <= em else METHODS[1]
        if er != row["E_R"] or em != row["E_MC"] or winner != row["best_method"]:
            raise ValueError("raw x0 label mismatch")
        if row["chosen_success"] != (min(er, em) <= EPSILON):
            raise ValueError("oracle success mismatch")
    with np.load(out / "input_features.npz", allow_pickle=False) as archive:
        features = archive["X"].copy()
        names = archive["feature_names"].tolist()
        ids = archive["equation_id"].tolist()
        splits = archive["split"].tolist()
    lookup = {pid:i for i,pid in enumerate(ids)}
    train_ids = [r["equation_id"] for r in rows if r["split"] == "train"]
    if model.registry["train_equations"] != train_ids:
        raise ValueError("fit IDs mismatch")
    if any(splits[lookup[pid]] != "train" for pid in model.registry["train_equations"]):
        raise ValueError("holdout reached training")
    checked = 0
    predictions = []
    for split in ("train", "validation", "test"):
        own = [r for r in rows if r["split"] == split]
        selected, _ = model.predict_features(features[[lookup[r["equation_id"]] for r in own]], names)
        truth = [r["best_method"] for r in own]
        predictions.extend(selected)
        for strategy, choice in (("ML", selected), ("always_remizov", [METHODS[0]]*len(own)),
                                 ("always_monte_carlo", [METHODS[1]]*len(own)), ("oracle", truth)):
            actual = metrics[split][strategy]
            classification = dict(accuracy=accuracy_score(truth, choice),
                                  balanced_accuracy=balanced_accuracy_score(truth, choice),
                                  macro_f1=f1_score(truth, choice, labels=list(METHODS), average="macro", zero_division=0))
            matrix = confusion_matrix(truth, choice, labels=list(METHODS)).tolist()
            if matrix != actual["confusion_matrix"]:
                raise ValueError("confusion matrix mismatch")
            errors = np.array([r["E_R"] if method == METHODS[0] else r["E_MC"] for r,method in zip(own,choice)])
            oracle = np.array([min(r["E_R"], r["E_MC"]) for r in own])
            regression = dict(success_rate=np.mean(errors <= EPSILON), mean_error=np.mean(errors),
                              median_error=np.median(errors), mean_regret=np.mean(errors-oracle),
                              median_regret=np.median(errors-oracle), max_regret=np.max(errors-oracle))
            for key, expected in {**classification, **regression}.items():
                if not np.isclose(expected, actual[key], rtol=1e-14, atol=1e-15):
                    raise ValueError(f"{split}/{strategy}/{key} mismatch")
            checked += 1
    result = dict(status="passed", raw_pairs_checked=len(rows), source_hashes_checked=len(pins),
                  strategy_split_metrics_checked=checked, classifier_choices=dict(Counter(predictions)),
                  numerical_calls=0, fit_calls=0, epsilon=EPSILON,
                  test_confusion_matrix=metrics["test"]["ML"]["confusion_matrix"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.out)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(encoded)
    print(encoded, end="")
