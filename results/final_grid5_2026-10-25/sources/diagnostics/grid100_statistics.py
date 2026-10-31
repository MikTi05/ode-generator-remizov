"""Saved-only statistics for a 33-node fixed-profile pilot; no solvers."""
import math
import numpy as np
from scipy.stats import norm,t


def aggregate_mc(records,alpha=.05,looks=2):
    """Bonferroni across 33 coordinates and two predeclared possible looks.

    Normal bands are asymptotic. Hoeffding uses the analytic path envelope
    |Z|<=G*(T+h) for the ideal discrete scheme, and is deliberately separate.
    Neither band controls EM/trapezoid bias relative to the continuous problem.
    """
    count=len(records);M=records[0]['result']['M'];n=M*count
    means=np.asarray([r['result']['f'] for r in records]);se=np.asarray([r['result']['stderr'] for r in records])
    mean=np.mean(means,axis=0)
    variance=((M-1)*np.sum(se*se*M,axis=0)+M*np.sum((means-mean)**2,axis=0))/(n-1)
    stderr=np.sqrt(variance/n);K=mean.size;z=float(norm.isf(alpha/(2*K*looks)))
    half=z*stderr
    p=records[0]['result'];B=p['forcing_abs_upper_bound']*(p['T']+p['dt_actual'])
    hoeffding=B*math.sqrt(2*math.log(2*K*looks/alpha)/n)
    exact=np.asarray(records[0]['y_exact']);error=np.abs(mean-exact);tail=p['tail_error_bound']
    lower=max(0.,float(np.max(error-half))-tail);upper=float(np.max(error+half))+tail
    hlow=max(0.,float(np.max(error))-hoeffding-tail);hhigh=float(np.max(error))+hoeffding+tail
    answer=dict(f=mean.tolist(),stderr=stderr.tolist(),normal_simultaneous_half=half.tolist(),normal_z=z,
        alpha=alpha,coordinates=K,maximum_looks=looks,paths_per_point=n,seeds=count,
        E_inf=float(np.max(error)),error_mean_interval=[lower,upper],
        statistical_accuracy='supported_pass_discrete' if upper<=.01 else ('supported_fail_discrete' if lower>.01 else 'uncertain'),
        hoeffding_half=hoeffding,analytic_path_envelope=B,hoeffding_error_interval=[hlow,hhigh],
        finite_sample_accuracy='supported_pass_discrete' if hhigh<=.01 else ('supported_fail_discrete' if hlow>.01 else 'uncertain'),
        interval_scope='discrete truncated expectation; normal asymptotic, Hoeffding ideal real arithmetic; no time-bias certificate',
        total_accuracy_certified=False,tail_bound=tail)
    if count>1:
        s=np.std(means,axis=0,ddof=1)/math.sqrt(count)
        answer['seed_mean_stderr']=s.tolist()
        answer['seed_t_simultaneous_half']=(float(t.isf(alpha/(2*K*looks),count-1))*s).tolist()
    return answer


def adaptive_ids(rows,maximum=12):
    """Six R-failing uncertain cases, then smallest-band unresolved cases.

    Input rows are primary fixed-seed summaries only. No h/M/T changes.
    Selection is intentionally enriched, not representative of all train.
    """
    eligible=[r for r in rows if r.get('both_completed') and r['mc']['statistical_accuracy']=='uncertain']
    first=sorted([r for r in eligible if not r['R_pass']],key=lambda r:(r['mc']['error_mean_interval'][1],r['problem_id']))[:maximum//2]
    selected=[r['problem_id'] for r in first]
    rest=sorted([r for r in eligible if r['problem_id'] not in selected],key=lambda r:(r['mc']['error_mean_interval'][1],r['problem_id']))
    selected.extend(r['problem_id'] for r in rest[:maximum-len(selected)])
    return selected
