"""Analyze saved backend measurements only; no numerical solver imports."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import statistics


def read(p):return json.loads(p.read_text())


def table(rows):
    names=list(rows[0]);s='| '+' | '.join(names)+' |\n| '+' | '.join('---' for _ in names)+' |\n'
    for r in rows:s+='| '+' | '.join(str(r[n]) for n in names)+' |\n'
    return s


def save_csv(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main(out):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plan=read(out/'execution_plan.json');run=read(out/'run_summary.json')
    if run['reason'] is not None:raise ValueError('incomplete run; inspect and report failures before aggregation')
    records=[read(p) for p in sorted((out/'measurements').glob('*.json'))]
    if len(records)!=48:raise ValueError('missing measurements')
    rows=[];rates=[];weighted_rates=[];equivalence=[]
    for task in plan['tasks']:
        pid=task['problem_id'];groups={engine:[r for r in records if r['task']['problem_id']==pid and r.get('series')!='grid' and r['engine']==engine] for engine in ('reference','numpy','numba')}
        med={k:statistics.median(r['external_solve_seconds'] for r in v) for k,v in groups.items()}
        ast=groups['numba'][0]['ast_preparation_seconds']
        ref=groups['reference'][0]['result'];node=task['coefficient_nodes']+3*task['solution_nodes']
        rate=med['numba']/ref['n_path_steps'];rates.append(rate);weighted_rates.append(rate/node)
        rows.append(dict(problem_id=pid,T=task['T'],coefficient_nodes=task['coefficient_nodes'],solution_nodes=task['solution_nodes'],
            preparation_seconds=ast,reference_seconds=med['reference'],numpy_seconds=med['numpy'],numba_seconds=med['numba'],
            speedup_numpy=med['reference']/med['numpy'],speedup_numba=med['reference']/med['numba'],
            reference_min=min(r['external_solve_seconds'] for r in groups['reference']),reference_max=max(r['external_solve_seconds'] for r in groups['reference']),
            numba_min=min(r['external_solve_seconds'] for r in groups['numba']),numba_max=max(r['external_solve_seconds'] for r in groups['numba']),
            numba_rng_median=statistics.median(r['result']['rng_seconds'] for r in groups['numba']),
            numba_kernel_median=statistics.median(r['result']['numerical_seconds'] for r in groups['numba']),
            path_steps=ref['n_path_steps'],peak_rss_mib=max(r['process_peak_rss_mib'] for g in groups.values() for r in g),
            error_min=min(r['error_infinity'] for r in groups['numba']),error_max=max(r['error_infinity'] for r in groups['numba'])))
        diffs={}
        for field in ('f','stderr','ci95'):
            diffs[field]=max(float(np.max(abs(np.array(next(r for r in groups[engine] if r['repeat']==i)['result'][field])-np.array(next(r for r in groups['reference'] if r['repeat']==i)['result'][field])))) for engine in ('numpy','numba') for i in range(3))
        replay=read(out/'replay'/f'{pid}.json')
        equivalence.append(dict(problem_id=pid,coefficient_difference=replay['coefficient_max_abs_difference'],
            path_difference=max(replay['path_max_abs_difference'].values()),mean_difference=diffs['f'],stderr_difference=diffs['stderr'],ci_difference=diffs['ci95']))
    grids=[]
    for r in records:
        if r.get('series')!='grid':continue
        point=next(v for v in rows if v['problem_id']==r['task']['problem_id']);z=r['result']
        grids.append(dict(problem_id=r['task']['problem_id'],seconds=r['external_solve_seconds'],
            point_to_grid_ratio=r['external_solve_seconds']/point['numba_seconds'],
            error_infinity=r['error_infinity'],max_pointwise_ci95=max(z['ci95']),
            path_steps=z['n_path_steps'],peak_rss_mib=r['process_peak_rss_mib'],success=z['success'],
            observed_tolerance_met=r['error_infinity']<=.01,total_error_certified=z['total_error_certified']))
    preflight=read(out/'preflight_1000.json');q=preflight['rows'];work=sum(r['path_steps_one_seed_33_points'] for r in q)
    baseline=sum(r['path_steps_one_seed_33_points']*statistics.median(weighted_rates)*r['weighted_nodes'] for r in q)
    forecast=dict(equations=len(q),points=33,M=1024,h=1/128,seeds=1,total_path_steps=work,
        observed_rate_range_seconds=[work*min(rates),work*max(rates)],
        ast_weighted_scenario_seconds=baseline,
        weighted_rate_sensitivity_seconds=[sum(r['path_steps_one_seed_33_points']*v*r['weighted_nodes'] for r in q) for v in (min(weighted_rates),max(weighted_rates))],
        ast_preparation_seconds=sum(r['ast_preparation_seconds'] for r in q),
        max_T=max(r['T'] for r in q),median_T=statistics.median(r['T'] for r in q),
        max_N=max(r['N'] for r in q),over_default_steps=sum(not r['fits_default_step_budget'] for r in q),
        over_default_path_budget=sum(not r['fits_default_path_budget'] for r in q),
        weighted_nodes_range=[min(r['weighted_nodes'] for r in q),max(r['weighted_nodes'] for r in q)],
        measured_weighted_nodes_range=[min(t['coefficient_nodes']+3*t['solution_nodes'] for t in plan['tasks']),max(t['coefficient_nodes']+3*t['solution_nodes'] for t in plan['tasks'])],
        beyond_measured_node_range=sum(r['weighted_nodes']>max(t['coefficient_nodes']+3*t['solution_nodes'] for t in plan['tasks']) for r in q),
        interpretation='Scenarios, not confidence bounds or guaranteed .01 cost; AST opcode mix, paths, hardware, seed and longer horizons vary. One persistent worker; JIT once. Larger M and seed count scale work approximately linearly.')
    for name,obj in [('timing_summary',rows),('equivalence_summary',equivalence),('grid_summary',grids),('forecast_1000',forecast)]:
        (out/f'{name}.json').write_text(json.dumps(obj,indent=2)+'\n')
        if isinstance(obj,list):save_csv(out/f'{name}.csv',obj)
    (out/'plots').mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');xs=np.arange(len(rows));width=.25
    for i,engine in enumerate(('reference','numpy','numba')):
        axes[0].bar(xs+(i-1)*width,[r[f'{engine}_seconds'] for r in rows],width,label=engine)
    axes[0].set(yscale='log',ylabel='Solve wall time, seconds (median of 3)',xticks=xs,xticklabels=[r['problem_id'][5:10] for r in rows]);axes[0].legend()
    axes[1].bar(xs,[r['speedup_numba'] for r in rows]);axes[1].set(ylabel='Reference / Numba warm solve',xticks=xs,xticklabels=[r['problem_id'][5:10] for r in rows]);axes[1].axhline(1,color='black',lw=.7)
    fig.savefig(out/'plots/timing.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for g in grids:
        record=next(r for r in records if r.get('series')=='grid' and r['task']['problem_id']==g['problem_id'])
        ax.plot(record['result']['x'],np.abs(np.array(record['result']['f'])-record['y_exact']),label=g['problem_id'][5:10])
    ax.axhline(.01,color='black',ls='--',label='target .01');ax.set(yscale='log',xlabel='x',ylabel='Observed absolute error, one seed');ax.legend()
    fig.savefig(out/'plots/grid_errors.png',dpi=160);plt.close(fig)
    compact=[dict(ID=r['problem_id'][5:10],T=r['T'],AST=f"{r['coefficient_nodes']}+{r['solution_nodes']}",
        prep_ms=f"{1000*r['preparation_seconds']:.3f}",reference_s=f"{r['reference_seconds']:.4f}",
        numpy_s=f"{r['numpy_seconds']:.4f}",numba_s=f"{r['numba_seconds']:.4f}",speedup=f"{r['speedup_numba']:.2f}×") for r in rows]
    (out/'TABLES.md').write_text(table(compact)+'\n'+table([dict(ID=g['problem_id'][5:10],grid_seconds=f"{g['seconds']:.3f}",grid_point_ratio=f"{g['point_to_grid_ratio']:.2f}",E_inf=f"{g['error_infinity']:.6f}",max_pointwise_CI95=f"{g['max_pointwise_ci95']:.6f}") for g in grids]))
    print(json.dumps(dict(timings=compact,grids=grids,forecast=forecast,equivalence=equivalence),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);main(ap.parse_args().out.resolve())
