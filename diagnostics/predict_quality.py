"""Unified frozen-model CLI: coefficients only, no saved outcomes or ODE solve."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from ml.quality_prototype import QualityPrototype, DEFAULT_REGISTRY
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, default=DEFAULT_REGISTRY)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--ids', nargs='+')
    source.add_argument('--input-json', type=Path,
                        help='x_grid, coefficients [a,b,c,g], coefficient metadata; no exact needed')
    parser.add_argument('--dataset', type=Path, default=ROOT/'data/all_methods_v1_1000_2026-10-18')
    parser.add_argument('--profiles', nargs='+')
    parser.add_argument('--planned-workers', type=int, choices=(1, 8))
    args = parser.parse_args()
    prototype = QualityPrototype(args.registry)
    answers = []
    if args.input_json:
        data = json.loads(args.input_json.read_text())
        answers.append(prototype.predict(data['coefficients'], data['x_grid'], data['metadata'],
            profiles=args.profiles, planned_workers=args.planned_workers,
            domain=data.get('domain', 'all_methods_v1')))
    else:
        meta = {m['equation_id']: m for m in map(json.loads,
            (args.dataset/'all_methods_equations_v1_metadata.jsonl').read_text().splitlines())}
        with np.load(args.dataset/'all_methods_equations_v1_ml.npz', allow_pickle=False) as z:
            if list(z['channel_order']) != ['a', 'b', 'c', 'g']:
                raise ValueError('unknown coefficient channel order')
            ids = z['equation_id'].tolist()
            for pid in args.ids:
                if meta[pid]['generation_version'] != 'all_methods_v1':
                    raise ValueError('unsupported generation version')
                answer = prototype.predict(z['X'][ids.index(pid)], z['x_grid'], meta[pid],
                    profiles=args.profiles, planned_workers=args.planned_workers)
                answers.append(dict(equation_id=pid, **answer))
    print(json.dumps(answers, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
