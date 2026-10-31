"""Certify/count 1000 metadata records; never run any numerical solver."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main(out):
    from mc_optimized import prepare_all_methods_mc
    from mc_solver import auto_T
    from diagnostics.compare_chernov import write_json
    path=ROOT/'data/all_methods_v1_1000_2026-10-18/all_methods_equations_v1_metadata.jsonl'
    began=time.perf_counter();rows=[]
    for line in path.read_text().splitlines():
        meta=json.loads(line);p=prepare_all_methods_mc(meta)
        T=math.ceil(auto_T(p.problem.lambda_value,p.problem.c_abs_upper_bound,p.forcing_bound.abs_upper_bound,1e-4)*8)/8
        N=int(T*128);updates=N*1024*33
        rows.append(dict(problem_id=p.problem.problem_id,family_id=meta['family_id'],split=meta.get('split'),
            expression_hash=meta.get('expression_hash'),G=p.forcing_bound.abs_upper_bound,T=T,N=N,
            coefficient_nodes=len(p.coeff_tape),solution_nodes=len(p.y_tape),
            weighted_nodes=len(p.coeff_tape)+3*len(p.y_tape),
            path_steps_one_seed_33_points=updates,
            fits_default_step_budget=N<=100000,fits_default_path_budget=updates<=100000000,
            ast_preparation_seconds=p.preparation_seconds))
    write_json(out/'preflight_1000.json',dict(dataset_metadata_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        equations=len(rows),numerical_solves=0,seconds=time.perf_counter()-began,M=1024,h=1/128,points=33,seeds=1,tail_eps=1e-4,rows=rows))
    print(json.dumps(dict(equations=len(rows),numerical_solves=0,seconds=time.perf_counter()-began,
        path_steps=sum(r['path_steps_one_seed_33_points'] for r in rows),
        max_T=max(r['T'] for r in rows),max_N=max(r['N'] for r in rows))),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    main(ap.parse_args().out.resolve())
