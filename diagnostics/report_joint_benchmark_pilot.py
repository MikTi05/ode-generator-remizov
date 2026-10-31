"""Analyze only saved joint-pilot predictions, scores and resource logs."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,float):return v if math.isfinite(v) else None
    if hasattr(v,'tolist'):return clean(v.tolist())
    if hasattr(v,'item'):return clean(v.item())
    return v


def write(path,value):
    Path(path).write_text(json.dumps(clean(value),indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')


def csv_write(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def cp_bounds(k,n,alpha=.05):
    from scipy.stats import beta
    if not 0<=k<=n or n<1:raise ValueError('invalid complete binomial sample')
    return (0. if k==0 else float(beta.ppf(alpha,k,n-k+1)),
            1. if k==n else float(beta.ppf(1-alpha,k+1,n-k)))


def holm(pvalues):
    ordered=sorted(range(len(pvalues)),key=lambda i:pvalues[i]);answer=[0.]*len(pvalues);last=0.
    for rank,index in enumerate(ordered):
        last=max(last,min(1.,(len(pvalues)-rank)*pvalues[index]));answer[index]=last
    return answer


def log_time_ratio(mc,r):
    import numpy as np
    from scipy.stats import t
    if len(mc)<2 or len(r)<2 or min(*mc,*r)<=0:return None
    a,b=np.log(mc),np.log(r)
    va,vb=float(np.var(a,ddof=1))/len(a),float(np.var(b,ddof=1))/len(b)
    se=math.sqrt(va+vb);d=float(np.mean(a)-np.mean(b))
    if se==0:return dict(ratio=math.exp(d),lower=math.exp(d),upper=math.exp(d),df=None)
    df=(va+vb)**2/(va*va/(len(a)-1)+vb*vb/(len(b)-1))
    half=float(t.ppf(.975,df))*se
    return dict(ratio=math.exp(d),lower=math.exp(d-half),upper=math.exp(d+half),df=df)


def collect(out):
    import numpy as np
    plan=read(out/'plan.json');summary=read(out/'summary.json')
    if summary['status']!='completed':raise ValueError('cannot present an unfinished pilot as completed')
    if sha(out/'plan.json')!=summary['plan_sha256']:raise ValueError('plan changed')
    rows=[];raw={}
    for spec,terminal in zip(plan['attempts'],summary['runs']):
        aid=spec['attempt_id'];trial=out/'trials'/aid
        if terminal['attempt_id']!=aid:raise ValueError('attempt ordering mismatch')
        row=dict(attempt_id=aid,mode=spec['mode'],problem_id=spec['problem_id'],method=spec['method'],
            repeat=spec['repeat'],**spec['identity'],status=terminal['status'],reason=terminal.get('reason'),
            T=spec['config']['T'],M=spec['config'].get('M'),h=spec['config'].get('dt'),
            seed=spec['config'].get('seed'),target_points=len(spec['target_x']),
            worker_response_seconds=terminal.get('worker_response_seconds'),solver_seconds=terminal.get('solver_seconds'),
            worker_process_seconds=terminal.get('worker_process_seconds'),peak_rss_mib=terminal.get('peak_rss_mib'))
        if (trial/'prediction.json').exists():
            prediction,score,ready=read(trial/'prediction.json'),read(trial/'score.json'),read(trial/'ready.json')
            checksum=sha(trial/'prediction.json')
            if checksum!=score['prediction_sha256'] or checksum!=ready['prediction_sha256']:
                raise ValueError('prediction changed after scoring')
            if ready['prediction_frozen_utc']>score['scored_utc']:raise ValueError('exact score precedes frozen answer')
            result=prediction['result']
            if score['finite_complete']:
                err=np.asarray(result['f'])-np.asarray(score['y_exact'])
                E=float(np.max(np.abs(err)))
                if E!=score['max_abs_error'] or bool(E<=.01)!=score['observed_accuracy_pass']:
                    raise ValueError('exact error/accuracy label mismatch')
                if spec['method']=='monte_carlo':
                    expected=spec['work']
                    if result['n_path_steps']!=expected['path_steps'] or result['n_normal_draws']!=expected['normal_draws']:
                        raise ValueError('completed MC work differs from fixed config')
                    if sum(result['coefficient_calls'].values())!=expected['coefficient_calls']:
                        raise ValueError('MC callback count mismatch')
                    if result['completed_paths']!=[spec['config']['M']]*len(spec['target_x']):
                        raise ValueError('MC incomplete paths')
                    if result['dt_actual']!=1/128 or result['tail_tolerance_met'] is not True:
                        raise ValueError('changed dt or uncertified tail')
                elif result['n_operator_applications']!=65536 or result['n_spline_builds']!=65537:
                    raise ValueError('Chernov work mismatch')
            row.update(max_abs_error=score['max_abs_error'],relative_error=score['relative_error'],
                observed_accuracy_pass=score['observed_accuracy_pass'],numerical_success=score['finite_complete'],
                tail_bound=prediction['common_tail_bound'],G=prediction['common_forcing_certificate']['forcing_bound']['abs_upper_bound'],
                n_path_steps=result.get('n_path_steps'),n_normal_draws=result.get('n_normal_draws'),
                n_operator_applications=result.get('n_operator_applications'),
                coefficient_calls=sum(result.get('coefficient_calls',{}).values()),
                coefficient_values=sum(result.get('coefficient_values',{}).values()),
                max_stderr=max(result.get('stderr',[]),default=0.),
                max_pointwise_ci95_half=max(result.get('ci95',[]),default=0.),
                max_simultaneous_sampling_half=max(score.get('simultaneous_sampling_halfwidth',[]),default=0.),
                total_error_certified=False)
            raw[aid]=dict(spec=spec,prediction=prediction,score=score,ready=ready)
        rows.append(row)
    return plan,summary,rows,raw


def summarize_profiles(plan,rows,raw):
    import numpy as np
    from scipy.stats import t,binom
    groups=defaultdict(list)
    for row in rows:groups[(row['mode'],row['problem_id'],row['method'])].append(row)
    profiles=[];vectors={}
    for (mode,pid,method),rr in sorted(groups.items()):
        spec=next(a for a in plan['attempts'] if a['mode']==mode and a['problem_id']==pid and a['method']==method)
        good=[r for r in rr if r['status']=='completed' and r.get('numerical_success')]
        all_complete=len(good)==len(rr)
        evaluable=all(r['status'] in ('completed','solver_failure') for r in rr)
        timed=[r for r in rr if r.get('worker_response_seconds') is not None]
        base=dict(mode=mode,problem_id=pid,method=method,complexity=spec['identity']['complexity'],
            regime=spec['identity']['regime'],original_split=spec['identity']['original_split'],
            requested_repeats=len(rr),complete_repeats=len(good),
            accuracy_successes=sum(bool(r.get('observed_accuracy_pass')) for r in rr),
            all_repeats_complete=all_complete,all_trials_evaluable=evaluable,T=spec['config']['T'],M=spec['config'].get('M'),
            median_response_seconds=statistics.median(r['worker_response_seconds'] for r in timed) if timed else None,
            min_response_seconds=min((r['worker_response_seconds'] for r in timed),default=None),
            max_response_seconds=max((r['worker_response_seconds'] for r in timed),default=None),
            median_solver_seconds=statistics.median(r['solver_seconds'] for r in timed) if timed else None,
            max_rss_mib=max(r['peak_rss_mib'] for r in rr),
            median_error=statistics.median(r['max_abs_error'] for r in good) if good else None,
            max_error=max((r['max_abs_error'] for r in good),default=None),
            sum_response_seconds=math.fsum(r['worker_response_seconds'] or 0 for r in rr),
            sum_solver_seconds=math.fsum(r['solver_seconds'] or 0 for r in rr),
            rmse_E=math.sqrt(statistics.mean(r['max_abs_error']**2 for r in good)) if good else None,
            policy_development_exposed=True,training_eligible=False)
        if good:
            records=[raw[r['attempt_id']] for r in good]
            F=np.asarray([r['prediction']['result']['f'] for r in records])
            exact=np.asarray(records[0]['score']['y_exact']);errors=F-exact
            mean=np.mean(errors,axis=0)
            vec=dict(target_x=spec['target_x'],y_exact=exact,seed_solutions=F,mean_error=mean,
                     mean_solution=np.mean(F,axis=0),aggregate_paths_per_point=None,
                     aggregate_is_not_single_call_profile=True)
            base['max_abs_mean_bias']=float(np.max(np.abs(mean)))
            base['signed_mean_bias_x0']=float(mean[len(mean)//2])
            if method=='solve':
                consistent=all(f.tobytes()==F[0].tobytes() for f in F)
                base['bitwise_reproducible']=consistent
                base['accuracy_state']='qualified_observed' if all_complete and consistent and all(r['observed_accuracy_pass'] for r in good) else 'accuracy_not_met'
                vec.update(bias_lower=mean,bias_upper=mean)
            else:
                n=len(good);K=F.shape[1]
                se=np.std(errors,axis=0,ddof=1)/math.sqrt(n) if n>1 else np.full(K,np.nan)
                half=float(t.ppf(.975,n-1))*se if n>1 else np.full(K,np.nan)
                joint=float(t.ppf(1-.05/(2*K),n-1))*se if n>1 else np.full(K,np.nan)
                lower,upper=mean-joint,mean+joint
                base.update(max_simultaneous_bias_half=float(np.max(joint)),
                    mean_bias_norm_ci_lower=float(np.max(np.maximum(0.,np.maximum(lower,-upper)))),
                    mean_bias_norm_ci_upper=float(np.max(np.maximum(np.abs(lower),np.abs(upper)))),
                    bias_ci_state=('inside_tolerance' if np.all(lower>=-.01) and np.all(upper<=.01) else
                                   'outside_tolerance' if np.any(lower>.01) or np.any(upper<-.01) else 'unresolved'),
                    median_max_sampling_half=statistics.median(r['max_pointwise_ci95_half'] for r in good),
                    median_max_simultaneous_sampling_half=statistics.median(r['max_simultaneous_sampling_half'] for r in good))
                vec.update(seed_mean_stderr=se,bias_pointwise_lower=mean-half,bias_pointwise_upper=mean+half,
                    bias_lower=lower,bias_upper=upper,aggregate_paths_per_point=n*spec['config']['M'],
                    internal_stderr=[r['prediction']['result']['stderr'] for r in records])
            vectors[f'{mode}:{pid}:{method}']=vec
        if method=='monte_carlo':
            if evaluable:
                k=base['accuracy_successes'];n=len(rr);L,U=cp_bounds(k,n)
                base.update(p_lower=L,p_upper=U,
                    reliability_state='qualified' if L>=.95 else 'below_target' if U<.95 else 'unresolved',
                    below_target_p=float(binom.cdf(k,n,.95)))
            else:base.update(p_lower=None,p_upper=None,reliability_state='missing_or_censored_design',below_target_p=1.)
        elif not evaluable:base['accuracy_state']='infrastructure_unresolved'
        elif not all_complete:base['accuracy_state']='numerical_failure'
        profiles.append(base)
    mc=[p for p in profiles if p['method']=='monte_carlo']
    adjusted=holm([p.get('below_target_p',1.) for p in mc])
    for p,a in zip(mc,adjusted):p['below_target_holm_p']=a
    return profiles,vectors


def compare(profiles,rows):
    pairs=defaultdict(dict)
    for p in profiles:pairs[(p['mode'],p['problem_id'])][p['method']]=p
    result=[]
    for (mode,pid),pair in sorted(pairs.items()):
        r,m=pair['solve'],pair['monte_carlo']
        rq=r.get('accuracy_state')=='qualified_observed'
        rb=r.get('accuracy_state') in ('accuracy_not_met','numerical_failure')
        mq=m.get('reliability_state')=='qualified';mb=m.get('reliability_state')=='below_target'
        winner,reason='unknown','accuracy_unresolved'
        if rq and mb:winner,reason='remizov','only_accuracy_qualified'
        elif rb and mb:winner,reason='neither','both_fixed_profiles_below_target'
        elif rb and mq:winner,reason='monte_carlo','only_accuracy_qualified'
        times={method:[x['worker_response_seconds'] for x in rows if x['mode']==mode and x['problem_id']==pid
                       and x['method']==method and x['status']=='completed'] for method in ('solve','monte_carlo')}
        ratio=log_time_ratio(times['monte_carlo'],times['solve'])
        if rq and mq:
            if ratio and ratio['lower']>1.10:winner,reason='remizov','qualified_and_faster'
            elif ratio and ratio['upper']<1/1.10:winner,reason='monte_carlo','qualified_and_faster'
            elif ratio and ratio['lower']>=1/1.10 and ratio['upper']<=1.10:winner,reason='tie_practical','qualified_and_time_equivalent'
            else:reason='time_unresolved'
        result.append(dict(mode=mode,problem_id=pid,complexity=r['complexity'],regime=r['regime'],
            T=r['T'],r_E=r['max_error'],r_reproducible=r.get('bitwise_reproducible'),
            r_accuracy_state=r.get('accuracy_state'),r_response_seconds=r['median_response_seconds'],
            r_solver_seconds=r['median_solver_seconds'],r_rss_mib=r['max_rss_mib'],
            mc_successes=m['accuracy_successes'],mc_repeats=m['requested_repeats'],
            mc_median_E=m['median_error'],mc_max_E=m['max_error'],mc_rmse_E=m['rmse_E'],
            mc_mean_bias_max=m.get('max_abs_mean_bias'),mc_bias_ci_state=m.get('bias_ci_state'),
            mc_median_sampling_half=m.get('median_max_sampling_half'),
            mc_median_simultaneous_sampling_half=m.get('median_max_simultaneous_sampling_half'),
            mc_response_seconds=m['median_response_seconds'],mc_solver_seconds=m['median_solver_seconds'],
            mc_rss_mib=m['max_rss_mib'],mc_p_lower=m.get('p_lower'),mc_p_upper=m.get('p_upper'),
            mc_reliability_state=m.get('reliability_state'),mc_below_target_holm_p=m.get('below_target_holm_p'),
            descriptive_mc_over_r_time_ratio=None if ratio is None else ratio['ratio'],
            descriptive_time_ratio_lower=None if ratio is None else ratio['lower'],
            descriptive_time_ratio_upper=None if ratio is None else ratio['upper'],
            winner=winner,winner_reason=reason,
            winner_survives_holm26=bool(winner in ('remizov','neither') and m.get('below_target_holm_p',1.)<=.05),
            confidence_scope='per profile; additional Holm26 flag only for below-target MC claims',
            fixed_profile_not_optimal_time=True,training_eligible=False))
    return result


def plots(out,comparison,vectors,rows):
    import numpy as np
    os.environ['MPLCONFIGDIR']=str(out/'mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.25,'savefig.dpi':160})
    folder=out/'figures';folder.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(folder/(name+'.png'),bbox_inches='tight')
        fig.savefig(folder/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    for mode in ('point','grid33'):
        rr=[r for r in comparison if r['mode']==mode];x=np.arange(len(rr))
        fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True)
        for i,r in enumerate(rr):
            for method,color,offset in [('solve','#c84b31',-.12),('monte_carlo','#2464ae',.12)]:
                values=[v for v in rows if v['mode']==mode and v['problem_id']==r['problem_id']
                        and v['method']==method and v['status']=='completed']
                axes[0].scatter(np.full(len(values),i+offset),[v['max_abs_error'] for v in values],
                    color=color,alpha=.65,s=24,label=('Remizov' if method=='solve' else 'MC individual seeds') if i==0 else None)
                times=[v['worker_response_seconds'] for v in values]
                if times:
                    med=np.median(times)
                    axes[1].errorbar(i+offset,med,yerr=[[med-min(times)],[max(times)-med]],
                        fmt='o',color=color,capsize=3,label=None)
        axes[0].axhline(.01,color='black',ls='--',label='Target 0.01')
        axes[0].set_yscale('log');axes[0].set_ylabel('Observed absolute error E0' if mode=='point' else 'Observed grid E_inf')
        axes[0].legend(ncol=3);axes[0].set_title(f'Fixed profiles, {mode}; every completed answer retained')
        axes[1].set_yscale('log');axes[1].set_ylabel('Cold response, seconds\nmedian and min/max, not confidence intervals')
        axes[1].set_xticks(x,[r['problem_id'].split('_')[1] for r in rr],rotation=60)
        axes[1].set_xlabel('Small dataset equation (AMV1_..._S00)')
        fig.tight_layout();save(fig,'accuracy_time_'+mode)
    rr=[r for r in comparison if r['mode']=='point']
    fig,ax=plt.subplots(figsize=(13,5))
    for i,r in enumerate(rr):
        v=vectors[f"point:{r['problem_id']}:monte_carlo"]
        mu=float(v['mean_error'][0]);lo=float(v['bias_pointwise_lower'][0]);hi=float(v['bias_pointwise_upper'][0])
        ax.errorbar(i+.10,mu,yerr=[[max(0,mu-lo)],[max(0,hi-mu)]],fmt='o',color='#2464ae',capsize=3,
                    label='MC mean error, t95 across 8 independent seeds' if i==0 else None)
        vr=vectors[f"point:{r['problem_id']}:solve"]
        ax.scatter(i-.10,float(vr['mean_error'][0]),marker='x',color='#c84b31',
                   label='Remizov deterministic error' if i==0 else None)
    ax.axhspan(-.01,.01,color='green',alpha=.12,label='Tolerance band')
    ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(len(rr)),[r['problem_id'].split('_')[1] for r in rr],rotation=60)
    ax.set_ylabel('Signed error at x=0');ax.set_title('Bias of discrete truncated expectation plus tail; pointwise intervals, no full-error certificate')
    ax.legend(ncol=2);fig.tight_layout();save(fig,'point_bias_uncertainty')
    rr=[r for r in comparison if r['mode']=='grid33']
    fig,axes=plt.subplots(len(rr),2,figsize=(13,3*len(rr)),squeeze=False)
    for i,r in enumerate(rr):
        mc=vectors[f"grid33:{r['problem_id']}:monte_carlo"]
        rem=vectors[f"grid33:{r['problem_id']}:solve"]
        xx=np.asarray(mc['target_x']);exact=np.asarray(mc['y_exact'])
        axes[i,0].plot(xx,exact,'k-',lw=1.8,label='Exact')
        axes[i,0].plot(xx,rem['seed_solutions'][0],color='#c84b31',lw=1.4,label='Remizov')
        for j,f in enumerate(mc['seed_solutions']):
            axes[i,0].plot(xx,f,color='#2464ae',alpha=.45,lw=.9,label='MC individual seed' if j==0 else None)
            axes[i,1].plot(xx,np.abs(np.asarray(f)-exact),color='#2464ae',alpha=.45,lw=.9)
        axes[i,1].plot(xx,np.abs(np.asarray(rem['seed_solutions'][0])-exact),color='#c84b31',lw=1.5)
        axes[i,1].axhline(.01,color='black',ls='--');axes[i,1].set_yscale('symlog',linthresh=1e-4)
        axes[i,0].set_title(r['problem_id']+f"; MC {r['mc_successes']}/{r['mc_repeats']} grid successes")
        axes[i,1].set_title('Absolute error at all 33 target points')
        axes[i,0].set_ylabel('y(x)');axes[i,1].set_ylabel('|error|')
        axes[i,0].set_xlabel('x');axes[i,1].set_xlabel('x')
        if i==0:axes[i,0].legend(ncol=3)
    fig.tight_layout();save(fig,'grid_solutions_and_errors')
    fig,ax=plt.subplots(figsize=(8,5))
    for mode,marker in [('point','o'),('grid33','^')]:
        for level,color in [('L1','#2464ae'),('L2','#228833'),('L3','#aa3377'),('L4','#ee7733')]:
            rr=[r for r in rows if r['method']=='monte_carlo' and r['mode']==mode and r['complexity']==level and r['status']=='completed']
            if rr:ax.scatter([r['n_path_steps']/1e6 for r in rr],[r['solver_seconds'] for r in rr],
                            marker=marker,color=color,alpha=.6,label=f'{level}, {mode}',s=22)
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('MC path updates, millions');ax.set_ylabel('Measured solver seconds')
    ax.set_title('Actual work/time; AST and target points affect throughput');ax.legend(ncol=2)
    fig.tight_layout();save(fig,'mc_work_and_time')


def telemetry_summary(out,plan):
    peak=0.;thermal_counts=Counter();max_cpu=0.;samples=0;guards=0
    for spec in plan['attempts']:
        trial=out/'trials'/spec['attempt_id']
        previous=None
        for line in (trial/'worker.resources.jsonl').read_text().splitlines():
            r=json.loads(line);peak=max(peak,r['peak_rss_mib']);samples+=1
            if previous and r['monotonic']>previous['monotonic']:
                max_cpu=max(max_cpu,100*(r['cpu_seconds']-previous['cpu_seconds'])/(r['monotonic']-previous['monotonic']))
            previous=r
        for line in (trial/'controller.resources.jsonl').read_text().splitlines():
            r=json.loads(line);thermal_counts[str(r.get('thermal_state'))]+=1
    for p in (out/'control').rglob('*.guardian.json'):
        g=read(p)
        if g.get('finished') and g.get('cleanup_success') and g.get('worker_returncode')==0:guards+=1
    return dict(sampled_peak_rss_mib=peak,thermal_counts=dict(thermal_counts),worker_resource_samples=samples,
                maximum_sampled_worker_cpu_percent=max_cpu,guardians_clean=guards,temperature_celsius=None)


def create_report(out,plan,summary,rows,profiles,vectors,comparisons,telemetry):
    import numpy as np
    modes={}
    for mode in ('point','grid33'):
        rr=[r for r in rows if r['mode']==mode]
        mode_stats={}
        for method in ('solve','monte_carlo'):
            ss=[r for r in rr if r['method']==method]
            mode_stats[method]=dict(attempts=len(ss),completed=sum(r['status']=='completed' for r in ss),
                observed_accuracy_successes=sum(bool(r.get('observed_accuracy_pass')) for r in ss),
                solver_seconds_sum=math.fsum(r.get('solver_seconds') or 0 for r in ss),
                response_seconds_sum=math.fsum(r.get('worker_response_seconds') or 0 for r in ss),
                median_response_seconds=statistics.median(r['worker_response_seconds'] for r in ss),
                min_response_seconds=min(r['worker_response_seconds'] for r in ss),
                max_response_seconds=max(r['worker_response_seconds'] for r in ss),
                max_rss_mib=max(r['peak_rss_mib'] for r in ss),
                path_steps=sum(r.get('n_path_steps') or 0 for r in ss),
                operator_applications=sum(r.get('n_operator_applications') or 0 for r in ss))
        modes[mode]=mode_stats
    cross=[]
    for r in comparisons:
        if r['mode']!='grid33':continue
        pid=r['problem_id'];a=np.asarray(vectors[f'point:{pid}:solve']['seed_solutions'][0])
        b=np.asarray(vectors[f'grid33:{pid}:solve']['seed_solutions'][0])
        cross.append(dict(problem_id=pid,point_equals_grid_center_bitwise=a.tobytes()==b[16:17].tobytes()))
    total_wall=math.fsum(s['wall_seconds'] for s in summary['sessions'])
    totals=dict(attempts=len(rows),outcomes=dict(Counter(r['status'] for r in rows)),
        wall_seconds=total_wall,solver_seconds_sum=math.fsum(r.get('solver_seconds') or 0 for r in rows),
        response_seconds_sum=math.fsum(r.get('worker_response_seconds') or 0 for r in rows),
        worker_cpu_seconds_sum=math.fsum(r.get('worker_cpu_seconds') or 0 for r in summary['runs']),
        modes=modes,telemetry=telemetry,
        observed_winner_labels=dict(Counter(r['winner'] for r in comparisons)),
        winner_labels_passing_holm26=sum(r['winner_survives_holm26'] for r in comparisons),
        mc_reliability_states=dict(Counter(r['mc_reliability_state'] for r in comparisons)),
        cross_mode_remizov=cross,fixed_profiles_not_optimal_cost=True,new_production1000_solves=0,new_ml_runs=0)
    write(out/'statistics.json',totals)
    fmt=lambda x: '—' if x is None else f'{x:.6g}'
    L=['# Сравнительный pilot Ремизов / Монте-Карло — 2026-10-25','',
       f"**Выполнены {len(rows)} заранее заданные попытки**, включая все 20 задач в x=0 и все 6 выбранных задач на33 узлах. Новых отдельных исследований F0002, расчёта1000 уравнений и ML не было.",'',
       f"Статусы: `{dict(Counter(r['status'] for r in rows))}`. Фактическое время управляемой сессии: **{total_wall:.3f}с / {total_wall/60:.2f}мин**; сумма времени решателей {totals['solver_seconds_sum']:.3f}с, сумма cold response {totals['response_seconds_sum']:.3f}с. Разность с session wall включает guardian, polling, запись и управление; это не время численной схемы.",'',
       '## Исполнение и сопоставимость','',
       'Параметры и seed перенесены из [предложения](../joint_benchmark_plan_2026-10-25/PLAN.md). Исполняемый [plan.json](plan.json) заморожен до первого worker. По новой инструкции пользователя удалены attempt/phase/session wall-cutoffs: **нет лимитов60с и90мин**, нет PAR2 с прежним B=60. Ни один seed не отбирался по exact. RSS-guard512МиБ обоснован прежним пиком179.5МиБ; один worker, BLAS/OpenMP=1.','',
       'Ремизов — `ChernovSolver.solve`,n128,Q513,dx=.0125,padding12,cubic/zero,3 свежих процесса на equation/mode. MC — production EM с двумя трапециями, h=1/128,chunk1024; point:8 независимых seed×2048; grid:4 других seed×1024 на каждую точку. Это фиксированные диагностические профили, **не оптимальное время достижения0.01**.','',
       'Общий T=ceil(8*T_certificate)/8, вычислен по аналитическому G и eps_tail=1e-4, явно передан обоим методам. Сертификат заново проверялся в каждом worker. Полная ошибка не сертифицирована: граница относится к непрерывному хвосту, интервалы MC — к дискретному усечённому ожиданию. Point оценивается по E0; grid по E∞ на x_j=-1+j/16,j=0,…,32.','',
       'Cold response — от запуска worker через guardian до замораживания prediction, включая import/load/certificate/решатель/запись. После prediction+SHA+ready впервые вычисляется exact для score. Проверены SHA и временной порядок всех ответов. Загрузчик отдельно выполняет штатную валидацию входных данных; эти проверки не выбирают параметры.','',
       'Внешний оператор продлевал120-секундный lease после свежей проверки контроллера. Независимый guardian следил за pipe и heartbeat контроллера, владел группой процессов и освобождал её при потере управления. Watchdog обнаруживал потерю телеметрии/CPU-прогресса/завершённых единиц работы, а не медленное продвижение; фиксированные длительности не ограничивались. Прозрачные диагностические observers только считали завершённые исходные MC chunks и узлы Чернова, не меняя арифметику и RNG.','',
       f"Завершённая очистка guardian: {telemetry['guardians_clean']}/{len(rows)}. Максимум sampled RSS {telemetry['sampled_peak_rss_mib']:.2f}МиБ; максимум конечных process peaks {max(r['peak_rss_mib'] for r in rows):.2f}МиБ. ThermalState-сэмплы: {telemetry['thermal_counts']}; это не температура в°C. Максимальный sampled CPU worker {telemetry['maximum_sampled_worker_cpu_percent']:.2f}% одного ядра.",'',
       '## Ошибки и измеренное время','',
       'В таблицах время — медиана cold response одного вызова; MC error — медиана и максимум **индивидуальных** seed-ответов. Успех MC означает E≤.01 для всей целевой сетки данного запуска. Повторы Ремизова — замеры времени и воспроизводимости, не независимые испытания случайной точности. Времена не отбираются только по успешным E.','']
    for mode,title in [('point','20 задач, x=0'),('grid33','6 задач, общая сетка33')]:
        L += ['### '+title,'','| ID | R E | MC median / max E | MC успехи | R,c | MC,c | MC p,L/U* | Исход профилей |',
              '|---|---:|---:|---:|---:|---:|---|---|']
        for r in comparisons:
            if r['mode']!=mode:continue
            L.append(f"| {r['problem_id'].split('_')[1]} | {fmt(r['r_E'])} | {fmt(r['mc_median_E'])} / {fmt(r['mc_max_E'])} | {r['mc_successes']}/{r['mc_repeats']} | {r['r_response_seconds']:.3f} | {r['mc_response_seconds']:.3f} | {fmt(r['mc_p_lower'])} / {fmt(r['mc_p_upper'])} | {r['winner']} |")
        L.append('')
    L += ['*L/U — две односторонние95% границы Clopper–Pearson для вероятности успеха фиксированного профиля; не совместный двухсторонний95% CI. `remizov` в колонке исхода обычно означает единственный accuracy-qualified профиль, **не доказанное ускорение при одинаковой успешности**. Полные причины, Holm26 и интервалы — [comparisons.csv](comparisons.csv).','',
          '## Статистика и смещение','',
          f"Состояния MC-надёжности при p*=.95: {totals['mc_reliability_states']}. Метки по одному профилю: {totals['observed_winner_labels']}; {totals['winner_labels_passing_holm26']} сравнений имеют отрицательный вывод о MC-надёжности, сохраняющийся после Holm по26 заранее заданным equation/mode профилям. Положительная надёжность MC при столь малом R в принципе не подтверждается:8/8 дают L=.68766,4/4 —.47287; минимум для L≥.95 —59/59 без множественной поправки.",'',
          'Среднее по seed и t95 ошибки относительно exact сохранены отдельно: [profile_vectors.json](profile_vectors.json). Для33 точек t-критическое значение скорректировано Bonferroni по пространственным точкам. Это оценка смещения дискретного усечённого ожидания вместе с хвостом, не изолированная EM-ошибка. Приближённые интервалы не учитывают доказанно все источники ошибки. Отсутствие различимого смещения не доказывает его малость.','',
          'В [profiles.csv](profiles.csv) сохранены median/max errors, RMS(E) по seed, mean-bias, его разрешённость, обычная и совместная статистическая полуширина, k/R и интервалы. Наблюдаемое попадание конкретного MC-ответа в .01 не заменяется утверждением о повторяемости. Усреднение повторов имеет суммарные M и стоимость и не используется как бесплатный ответ одного вызова. Никаких выводов о порядке сходимости или экстраполяции здесь нет.','',
          'Полная таблица статистики смещения находится в [BIAS.md](BIAS.md). Интервал нормы среднего смещения получен из совместных пространственных t-интервалов; он не является интервалом ошибки будущего отдельного ответа.','',
          '### Измеренная цена','',
          '| Режим/метод | Попытки | Exact-успехи | Сумма solver,c | Сумма response,c | Median response,c | RSS max,МиБ |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for mode,mm in modes.items():
        for method,s in mm.items():
            L.append(f"| {mode}/{method} | {s['attempts']} | {s['observed_accuracy_successes']} | {s['solver_seconds_sum']:.3f} | {s['response_seconds_sum']:.3f} | {s['median_response_seconds']:.3f} | {s['max_rss_mib']:.2f} |")
    L += ['',f"Работа MC: {sum(r.get('n_path_steps') or 0 for r in rows):,} обновлений и столько же нормальных значений. Ремизов: {sum(r.get('n_operator_applications') or 0 for r in rows):,} применений S. Единицы этих счётчиков несопоставимы; стоимость сравнивается по времени и ресурсам.",'',
          'Для шести общих уравнений проверено побайтовое совпадение R(x=0) и центрального значения сеточного ответа: '+str(all(x['point_equals_grid_center_bitwise'] for x in cross))+'. Это не перенос сеточной точности или времени MC из одной точки. В R сохраняется один рабочий интервал [-13,13] и2081 узел для обоих mode, поэтому делить время на33 нельзя. У MC профили имеют разные M и независимые seed; время и качество измерены отдельно.','',
          '## Графики','',
          '- [Точность и время, все20 в x=0](figures/accuracy_time_point.png).','- [Точность и время,6 сеток](figures/accuracy_time_grid33.png).',
          '- [Смещение и неопределённость в x=0](figures/point_bias_uncertainty.png).','- [Все сеточные решения и ошибки каждого seed](figures/grid_solutions_and_errors.png).',
          '- [Фактическая работа и время MC](figures/mc_work_and_time.png).','',
          'PDF-версии лежат рядом. Whiskers времени — min/max, не доверительный интервал. Bias-график показывает точечные t95, не совместное95% утверждение по20 уравнениям.','',
          '## Практический вывод о1000 задачах','',
          'Pilot проверяет техническую способность данной инфраструктуры последовательно получать и сохранять ответы на длинном сертифицированном T, включая полные33 точки. Численное завершение, observed .01 и надёжность профиля — разные критерии; точный выбор наиболее эффективного метода этим фиксированным диагностическим бюджетом не решён. Отношения времени в comparisons описательные, пока оба профиля не квалифицированы.','',
          'Для массового accuracy-benchmark при .01 текущие MC-параметры нельзя считать обоснованными только из-за завершения pilot. При недостаточном M рост времени не означает контролируемое временное смещение; повышение M автоматически не разрешено и здесь не выполнялось. Сеточная выборка6 преднамеренно включает длинные T/сложные AST, не является случайной репрезентативной выборкой1000; её среднее время нельзя просто умножать на1000. Также нельзя переносить point-результат наgrid.','',
          'Технические ограничения перед разрешением1000: нынешний исполнитель намеренно ограждён scope20+6; понадобится явная привязка другого dataset fingerprint, preflight его сертификатов/объёма, контроль работающих процессов, проверяемое восстановление после прерывания без повтора уже завершённых ответов и отдельный бюджет по фактически измеренным AST. Общие ID small/production не означают одинаковые уравнения. Это требования эксплуатации, не новый массовый запуск.','',
          'В частности, оставленный work-guard300млн обновлений для grid33/M1024/h128 допускает общий T лишь до69.25 с учётом округления до1/8. Нельзя молча переносить его на1000 неизвестных сертификатов и объявлять будущие work_limit_exceeded отказами математического метода. `load_all_methods_equation` штатно валидирует всю пару файлов при каждом cold-вызове; стоимость подготовки1000 также отдельно не измерена этим pilot.','',
          'Small20 содержит0 train,10 validation,10 test и уже использован для разработки политики. Семейные split и provenance сохранены, все seed/точки/профили одного уравнения остаются в одной группе. Метки pilot `training_eligible=false`; неизвестный победитель не заполнен самым быстрым удачным seed. Данные точного решения не включаются в признаки селектора.','',
          '## Артефакты и проверки','',
          '[262 попытки](attempts.csv),[52 профиля](profiles.csv),[26 сравнений](comparisons.csv),[сводка](statistics.json),[подробные векторы](profile_vectors.json),[протокол анализа](analysis_spec.json). Каждый trial содержит request,prediction,ready соSHA,score,worker log,resource log иterminal. Исторические результаты и датасеты не перезаписывались. Итоговая сохранность/регрессия записываются в verification.json и test logs.']
    (out/'REPORT.md').write_text('\n'.join(L)+'\n')
    mc_profiles=[p for p in profiles if p['method']=='monte_carlo']
    B=['# Смещение и статистическая неопределённость фиксированного MC','',
       'Среднее по независимым seed оценивает дискретное усечённое ожидание. Интервалы t95: для одной точки — обычные; для33 точек — Bonferroni по пространству. Таблица относится к одному equation/mode; семейная поправка по26 для этих интервалов не применялась. Норма среднего смещения и ошибка отдельного ответа — разные величины. Полная временная ошибка не сертифицирована.','',
       '| Режим/ID | max abs mean bias | Совместный интервал нормы bias | Состояние | Median max sampling CI half* |',
       '|---|---:|---|---|---:|']
    for p in mc_profiles:
        B.append(f"| {p['mode']}/{p['problem_id'].split('_')[1]} | {fmt(p.get('max_abs_mean_bias'))} | [{fmt(p.get('mean_bias_norm_ci_lower'))}, {fmt(p.get('mean_bias_norm_ci_upper'))}] | {p.get('bias_ci_state')} | {fmt(p.get('median_max_simultaneous_sampling_half'))} |")
    B += ['', '*Это статистическая полуширина одного production-ответа, приближённая нормальная и совместная по точкам. Она не включает временное смещение и хвост.','',
          'Состояния bias-CI: '+str(dict(Counter(p.get('bias_ci_state') for p in mc_profiles)))+'.',
          'При `outside_tolerance` временное/усечённое среднее имеет различимое отклонение за пределами .01 в данном профиле; увеличение только M не исправляет такое ожидание. При `unresolved` данные не отделяют допустимое смещение от недопустимого. Даже `inside_tolerance` не подтверждает надёжность новой реализации с тем же M.']
    (out/'BIAS.md').write_text('\n'.join(B)+'\n')
    return totals


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args();out=args.out.resolve()
    plan,summary,rows,raw=collect(out)
    profiles,vectors=summarize_profiles(plan,rows,raw)
    comparisons=compare(profiles,rows)
    csv_write(out/'attempts.csv',rows);csv_write(out/'profiles.csv',profiles);csv_write(out/'comparisons.csv',comparisons)
    write(out/'profile_vectors.json',vectors)
    plots(out,comparisons,vectors,rows)
    telemetry=telemetry_summary(out,plan)
    totals=create_report(out,plan,summary,rows,profiles,vectors,comparisons,telemetry)
    print(json.dumps(clean(totals),ensure_ascii=False,indent=2))


if __name__=='__main__':main()


