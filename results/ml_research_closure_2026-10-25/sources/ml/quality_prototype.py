"""Frozen input-only quality surrogates. No solver, fitting, or winner ranking.

The registry pins both model bytes and feature semantics. Grid33 zero profiles
remain explicitly unsupported; point predictions never stand in for grid ones.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ml.remizov_features import CERT_FIELDS, VERSION, coefficient_features

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / 'results/ml_research_closure_2026-10-25/registry.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class QualityPrototype:
    """Serve only validation-selected, byte-pinned historical models.

    values: the given a,b,c,g on the ORIGINAL 256-node input grid, not a
    numerical solution. metadata: global coefficient certificates and a/b/c
    ASTs. IDs, exact, solution trees and result files are never consulted here.
    Supplying arbitrary coefficients does not establish an in-distribution input.
    """

    def __init__(self, registry=DEFAULT_REGISTRY):
        self.registry = json.loads(Path(registry).read_text())
        if self.registry['feature_version'] != VERSION:
            raise ValueError('feature version mismatch')
        if sha(ROOT / 'ml/remizov_features.py') != self.registry['feature_code_sha256']:
            raise ValueError('feature implementation mismatch')
        self._models = {}
        for name, spec in self.registry['artifacts'].items():
            path = ROOT / spec['path']
            if sha(path) != spec['sha256']:
                raise ValueError(f'model integrity mismatch: {name}')
            if spec['kind'] == 'context_median':
                model = json.loads(path.read_text())
            elif spec['kind'] == 'joblib':
                import joblib
                model = joblib.load(path)
            else:
                from catboost import CatBoostClassifier, CatBoostRegressor
                model = CatBoostClassifier() if spec['kind'] == 'classifier' else CatBoostRegressor()
                model.load_model(str(path))
                if list(model.feature_names_) != self.registry['feature_names']:
                    raise ValueError('model feature order mismatch')
            self._models[name] = model

    def predict(self, values, x_grid, metadata, *, profiles=None, planned_workers=None,
                domain='all_methods_v1'):
        x = np.asarray(x_grid, dtype=float)
        values = np.asarray(values, dtype=float)
        if domain != 'all_methods_v1':
            raise ValueError('unvalidated input domain; only all_methods_v1 is supported')
        if x.shape != (256,) or not np.isfinite(x).all() or sha_bytes(x) != self.registry['input_grid_sha256']:
            raise ValueError('original 256-node coefficient grid required')
        if values.shape != (4, 256) or not np.isfinite(values).all():
            raise ValueError('finite a,b,c,g coefficient channels required')
        cert = {key: float(metadata[key]) for key in CERT_FIELDS}
        if (not all(np.isfinite(v) for v in cert.values()) or
            not 0 < cert['a_lower_cert'] <= cert['a_upper_cert'] or
            cert['b_abs_cert'] < 0 or cert['remizov_c_abs_cert'] < 0 or
            cert['remizov_lambda'] <= cert['remizov_c_abs_cert']):
            raise ValueError('invalid global coefficient certificates')
        if planned_workers is not None and (type(planned_workers) is not int or planned_workers not in (1, 8)):
            raise ValueError('historical timing context supports only 1 or 8 workers')
        features = coefficient_features(values, x, metadata)
        if list(features) != self.registry['feature_names']:
            raise ValueError('feature schema mismatch')
        import pandas as pd
        frame = pd.DataFrame([features], columns=self.registry['feature_names'])
        names = list(self.registry['profiles']) if profiles is None else list(profiles)
        if not names or len(names) != len(set(names)):
            raise ValueError('nonempty unique profile list required')
        output = []
        for name in names:
            if name not in self.registry['profiles']:
                raise ValueError(f'unknown profile: {name}')
            spec = self.registry['profiles'][name]
            row = dict(profile_id=name, scope=spec['scope'], parameters=spec['parameters'],
                       status=spec['status'], predictions={}, warnings=spec['warnings'])
            for target, key in spec.get('artifacts', {}).items():
                artifact = self.registry['artifacts'][key]
                model = self._models[key]
                if artifact['kind'] == 'context_median':
                    if planned_workers is None:
                        row['runtime_status'] = 'context_required'
                        continue
                    result = float(model[str(planned_workers)])
                    row['timing_context'] = dict(planned_external_workers=planned_workers,
                        counterfactual_single_worker_cost=False)
                elif artifact['kind'] == 'classifier':
                    result = float(model.predict_proba(frame, thread_count=1)[0, 1])
                else:
                    result = float(model.predict(frame, thread_count=1)[0])
                if not np.isfinite(result):
                    raise ValueError('nonfinite prediction')
                row['predictions'][target] = result
                if artifact.get('transform') == 'pow10':
                    row['predictions']['absolute_error_estimate'] = float(10. ** result)
            if name == 'remizov_taper_grid33':
                p = row['predictions']['p_accuracy_met']
                row['validation_threshold_decision'] = bool(1-p < spec['failure_threshold'])
            row['accuracy_certificate'] = False
            row['prediction_interval'] = None
            output.append(row)
        return dict(schema='quality_prototype_v1', results=output,
                    method_selection=dict(status='uncertain', winner=None,
                        reason='No validated model of fastest reliable solver on a common spatial/profile budget'),
                    input_scope=dict(domain=domain, distribution_membership_verified=False,
                        channels=['a', 'b', 'c', 'g'], input_grid_points=256),
                    total_accuracy_certified=False)


def sha_bytes(array):
    return hashlib.sha256(array.tobytes()).hexdigest()
