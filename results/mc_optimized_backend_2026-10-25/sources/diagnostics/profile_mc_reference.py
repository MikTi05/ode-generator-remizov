"""Short fixed baseline profile; never an accuracy benchmark or a dataset run."""
from __future__ import annotations
import argparse
import cProfile
from dataclasses import asdict
import json
import os
from pathlib import Path
import pstats
import sys
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    from all_methods_dataset_loader import load_all_methods_equations
    from all_methods_bounds import certify_all_methods_forcing
    from mc_solver import MCConfig,solve_monte_carlo
    from diagnostics.compare_chernov import write_json
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    out=ap.parse_args().out.resolve();protocol=json.loads((out/'protocol.json').read_text())['profiling']
    data=ROOT/'data/generated'
    equations={e.equation_id:e for e in load_all_methods_equations(
        data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl')};summaries=[]
    for pid in protocol['ids']:
        eq=equations[pid];bound=certify_all_methods_forcing(eq.remizov_problem,eq.metadata).forcing_bound
        cfg=MCConfig(M=protocol['M'],dt=protocol['dt'],T=protocol['T'],seed=protocol['seed'],chunk=256)
        profile=cProfile.Profile();began=time.perf_counter()
        result=profile.runcall(solve_monte_carlo,eq.remizov_problem,[0.],cfg,forcing_bound=bound)
        elapsed=time.perf_counter()-began
        if not result.success:raise RuntimeError(result.failure_reason)
        profile.dump_stats(str(out/'profiling'/f'{pid}.pstats'))
        stats=pstats.Stats(profile);rows=[]
        for (filename,line,function),(primitive,total,self_sec,cum_sec,callers) in stats.stats.items():
            rows.append(dict(file=filename,line=line,function=function,primitive_calls=primitive,
                total_calls=total,self_seconds=self_sec,cumulative_seconds=cum_sec))
        rows.sort(key=lambda r:r['cumulative_seconds'],reverse=True)
        record=dict(problem_id=pid,config=asdict(cfg),profiled_wall_seconds=elapsed,result=asdict(result),functions=rows)
        write_json(out/'profiling'/f'{pid}.json',record)
        summary=dict(problem_id=pid,profiled_seconds=elapsed,
            expression_jet_seconds=sum(r['self_seconds'] for r in rows if r['function'] in ('_evaluate_array','_finite_jet','evaluate_jet2','validate_expression')),
            ast_cumulative_seconds=next((r['cumulative_seconds'] for r in rows if r['function']=='evaluate_jet2'),0),
            top_functions=rows[:12])
        summaries.append(summary)
        print(json.dumps({k:v for k,v in summary.items() if k!='top_functions'}),flush=True)
    write_json(out/'profiling/summary.json',summaries)


if __name__=='__main__':main()
