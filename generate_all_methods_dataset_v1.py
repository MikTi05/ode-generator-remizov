#!/usr/bin/env python3
"""Generate the isolated all_methods_equations_v1 dataset artifacts.

The production defaults create 200 structural families with five numerical
parameterizations per family (1000 equations).  This command only constructs
and validates equations; it never runs an ODE solver or creates method labels.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Sequence

from src.all_methods_dataset_v1 import (
    AllMethodsConfig,
    GenerationError,
    generate_dataset,
    validate_dataset_artifacts,
    write_dataset,
)


def _nonnegative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def _positive_int(text: str) -> int:
    value = _nonnegative_int(text)
    if value == 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _positive_finite_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=_nonnegative_int, default=2488)
    parser.add_argument("--n-families", type=_positive_int, default=200)
    parser.add_argument("--samples-per-family", type=_positive_int, default=5)
    parser.add_argument("--ml-grid-size", type=_positive_int, default=256)
    parser.add_argument("--validation-grid-size", type=_positive_int, default=2049)
    parser.add_argument("--max-attempts", type=_positive_int, default=100)
    parser.add_argument("--max-abs-safety", type=_positive_finite_float, default=1.0e6)
    parser.add_argument("--a-min", type=_positive_finite_float, default=1.0e-3)
    parser.add_argument("--residual-rtol", type=_positive_finite_float, default=1.0e-10)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> AllMethodsConfig:
    return AllMethodsConfig(
        seed=args.seed,
        n_families=args.n_families,
        samples_per_family=args.samples_per_family,
        ml_grid_size=args.ml_grid_size,
        validation_grid_size=args.validation_grid_size,
        max_attempts=args.max_attempts,
        max_abs_safety=args.max_abs_safety,
        a_min=args.a_min,
        residual_rtol=args.residual_rtol,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    try:
        config.validate()
    except ValueError as exc:
        parser.error(str(exc))

    try:
        dataset = generate_dataset(config)
        written = write_dataset(dataset, args.output_dir)
        validation = validate_dataset_artifacts(
            args.output_dir,
            expected_config=config,
        )
    except (GenerationError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {"valid": False, "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1

    output_paths = {
        name: str(Path(path).resolve()) for name, path in written["paths"].items()
    }
    summary = {
        "valid": bool(validation["valid"]),
        "generation_version": dataset.report["generation_version"],
        "master_seed": config.seed,
        "total_equations": dataset.report["total_equations"],
        "total_families": dataset.report["total_families"],
        "samples_per_family": config.samples_per_family,
        "counts_by_complexity": dataset.report["counts_by_complexity"],
        "counts_by_regime": dataset.report["counts_by_regime"],
        "counts_by_split": dataset.report["counts_by_split"],
        "total_rejections": dataset.report["total_rejections"],
        "rejection_counts": dataset.report["rejection_counts"],
        "max_generation_attempts": dataset.report["max_generation_attempts"],
        "output_paths": output_paths,
        "sha256": written["sha256"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
