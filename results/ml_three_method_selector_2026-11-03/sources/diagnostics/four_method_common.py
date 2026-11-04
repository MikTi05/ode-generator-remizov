"""Data contracts for the fixed four-method accuracy release; no solvers or ML."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/all_methods_v1_1000_2026-10-18"
METHODS = ("remizov", "monte_carlo", "rk4", "product_integral")
CONFIG = dict(x0=-1.0, x1=1.0, n_steps=128, h=1 / 64, a_min=0.001,
              target_x=0.0, target_index=64, epsilon=0.05)
RECORD_SCHEMA = "fixed_ivp_mass_record_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def sealed(record):
    if "record_sha256" in record:
        raise ValueError("record already sealed")
    return dict(record, record_sha256=hashlib.sha256(canonical(record)).hexdigest())


def verify_seal(record):
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    if record.get("record_sha256") != hashlib.sha256(canonical(body)).hexdigest():
        raise ValueError("record checksum mismatch")


def atomic_json(path, value):
    """Commit a fully fsynced file atomically, refusing to replace old data."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="wb", prefix="." + path.name + ".",
                                     suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False).encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, path)  # atomic exclusive publication on this filesystem
    finally:
        temporary.unlink(missing_ok=True)


def winner(errors, order=METHODS):
    """Use exact unrounded float equality for ties; absent errors stay absent."""
    valid = {}
    for method in order:
        value = errors.get(method)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("only finite nonnegative measured errors are admissible")
            valid[method] = value
    if not valid:
        return None, []
    minimum = min(valid.values())
    tied = [method for method in order if method in valid and valid[method] == minimum]
    return tied[0], tied


def validate_record(record, plan_sha, method, metadata):
    verify_seal(record)
    if (record.get("schema") != RECORD_SCHEMA or record.get("plan_sha256") != plan_sha
            or record.get("method") != method or record.get("config") != CONFIG):
        raise ValueError("saved record contract/plan mismatch")
    for key in ("equation_id", "expression_hash", "family_id", "structure_signature"):
        if record.get(key) != metadata[key]:
            raise ValueError("saved record identity mismatch: " + key)
    expected_inputs = {key: metadata[key] for key in ("x0", "x1", "y0", "v0")}
    if record.get("inputs") != expected_inputs:
        raise ValueError("saved initial conditions differ from input")
    if record.get("status") not in {"success", "solver_failure", "solver_exception", "invalid_target", "evaluation_failure"}:
        raise ValueError("unknown terminal status")
    if record.get("usable"):
        result = record.get("result") or {}
        if (record["status"] != "success" or result.get("success") is not True
                or result.get("failure_reason") != "" or result.get("n_steps") != 128
                or result.get("h") != 1 / 64):
            raise ValueError("unusable solver return marked usable")
        if not all(len(result.get(key, [])) == 129 for key in ("x", "y", "dy")):
            raise ValueError("incomplete trajectory marked usable")
        if result["x"] != [-1 + i / 64 for i in range(129)] or result["x"][64] != 0.0:
            raise ValueError("wrong numerical grid")
        if result["y"][0] != metadata["y0"] or result["dy"][0] != metadata["v0"]:
            raise ValueError("wrong initial state in trajectory")
        if not all(math.isfinite(v) for key in ("x", "y", "dy") for v in result[key]):
            raise ValueError("nonfinite trajectory marked usable")
        error = abs(result["y"][64] - record["exact0"])
        if (record["y0_target"] != result["y"][64] or record["E0"] != error
                or record["accuracy_met"] != (error <= 0.05)):
            raise ValueError("saved zero-point metric mismatch")
    elif any(record.get(key) is not None for key in ("E0", "y0_target", "accuracy_met")):
        raise ValueError("failed solve has a fabricated measurement")
    return record
