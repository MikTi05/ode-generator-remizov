"""Read immutable MC attempts and historical Chernov33; compare x=0 only."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def seed_summary(values,stderr,M,exact,tail,rerror):
    """Statistics for equal-M independent seeds, never a total-error bound."""
    import numpy as np
    from scipy.stats import t
    z=np.asarray(values);n=len(z);mean=float(np.mean(z));var=float(np.var(z,ddof=1))
    within=np.asarray(stderr)**2*M
    pooled_variance=float(((M-1)*sum(within)+M*sum((z-mean)**2))/(n*M-1))
    pooled_se=math.sqrt(pooled_variance/(n*M));seed_se=math.sqrt(var/n)
    half=float(t.ppf(.975,n-1))*seed_se
    lower=mean-half-tail;upper=mean+half+tail
    emin=0. if lower<=exact<=upper else min(abs(lower-exact),abs(upper-exact))
    emax=max(abs(lower-exact),abs(upper-exact));label='uncertain'
    if rerror is not None:
        if emax+.0025<rerror:label='mc_mean_closer_exploratory'
        elif emin>rerror+.0025:label='remizov_closer_exploratory'
    return dict(mc_mean=mean,mc_aggregate_error=abs(mean-exact),mc_seed_mean_std=math.sqrt(var),
        mc_pooled_stderr=pooled_se,mc_pooled_ci95=1.96*pooled_se,mc_seed_t95=half,
        mc_seed_bias=mean-exact,mc_bias_t95_low=mean-exact-half,mc_bias_t95_high=mean-exact+half,
        mc_error_interval_low=emin,mc_error_interval_high=emax,quality_label=label)


def analyze(out):
    import numpy as np
    import pandas as pd
    from diagnostics.train_remizov_ml import write
    plan=read(out/'plan.json');plan_sha=sha(out/'plan.json');metadata={m['equation_id']:m for m in (json.loads(s) for s in (ROOT/plan['data_dir']/'all_methods_equations_v1_metadata.jsonl').read_text().splitlines())}
    # Bind historical input bytes to this dataset, not merely matching IDs.
    oldplan=read(ROOT/'results/chernov_solve1000_2026-10-18/plan.json')
    for n,h in oldplan['protected_sha256'].items():
        if n.startswith('data/') and sha(ROOT/n)!=h:raise ValueError('historical dataset SHA differs')
    attempts=[];rows=[];checked=0
    for pid in plan['equation_ids']:
        meta=metadata[pid];histpath=ROOT/'results/chernov_solve1000_2026-10-18/solutions'/f'{pid}.json';old=read(histpath)
        row=dict(equation_id=pid,family_id=meta['family_id'],split=meta['split'],expression_hash=meta['expression_hash'],
            complexity=meta['complexity_level'],regime=meta['regime'],remizov_status=old['status'],
            remizov_known=False,remizov_f0=None,remizov_error0=None,remizov_accuracy0=None,
            historical_remizov33_seconds=old.get('solver_seconds'),historical_source=str(histpath.relative_to(ROOT)),historical_sha256=sha(histpath))
        if old.get('result') and old['result']['success']:
            if old['implementation']!='chernov_solver.ChernovSolver.solve' or old['metadata']['expression_hash']!=meta['expression_hash']:raise ValueError('historical identity differs')
            x=np.asarray(old['result']['x']);ix=np.flatnonzero(x==0.)
            if len(ix)!=1 or old['target_x']!=x.tolist():raise ValueError('saved x=0 missing or mismatched')
            j=int(ix[0]);f=float(old['result']['f'][j]);exact=float(old['y_exact'][j]);error=abs(f-exact)
            row.update(remizov_known=True,remizov_f0=f,remizov_error0=error,remizov_accuracy0=int(error<=.01),
                exact0=exact,remizov_zero_index=j,remizov_T=old['result']['T']);checked+=1
        values=[];ses=[];errors=[];times=[];tails=[];work=[];outcomes=[]
        for i,seed in enumerate(plan['seeds']):
            path=out/'attempts'/f'{pid}_seed{i}.json'
            flat=dict(equation_id=pid,family_id=meta['family_id'],split=meta['split'],seed_index=i,seed=seed,status='pending',
                value=None,exact=None,abs_error=None,accuracy_met=None,stderr=None,ci95=None,T=None,N=None,
                solver_seconds=None,task_wall_seconds=None,peak_rss_mib=None,path_steps=None,normal_draws=None,tail_bound=None,
                failure_reason=None,source=None,sha256=None)
            if path.exists():
                r=read(path)
                if r['plan_sha256']!=plan_sha or r['expression_hash']!=meta['expression_hash'] or r['config']!=dict(plan['config'],seed=seed) or r['target_x']!=[0.]:raise ValueError('MC identity/profile differs')
                flat.update(status=r['status'],task_wall_seconds=r['task_wall_seconds'],peak_rss_mib=r['process_peak_rss_mib'],
                    failure_reason=r.get('failure_reason'),source=str(path.relative_to(ROOT)),sha256=sha(path))
                if r.get('result'):
                    z=r['result'];flat.update(T=z['T'],N=z['n_steps'],solver_seconds=z['runtime_seconds'],
                        path_steps=z['n_path_steps'],normal_draws=z['n_normal_draws'],tail_bound=z['tail_error_bound'])
                if r['status']=='completed':
                    if not z['success'] or z['backend']!='numba' or not z['tail_tolerance_met'] or z['x']!=[0.] or z['M']!=1024:raise ValueError('invalid completed MC')
                    f=float(z['f'][0]);y=float(r['y_exact']);error=abs(f-y)
                    if abs(error-r['abs_error'])>1e-13 or not math.isfinite(error):raise ValueError('MC error differs')
                    if row['remizov_known'] and abs(y-row['exact0'])>1e-12:raise ValueError('MC and historical exact differ at x=0')
                    row['exact0']=y
                    flat.update(value=f,exact=y,abs_error=error,accuracy_met=int(error<=.01),stderr=z['stderr'][0],ci95=z['ci95'][0])
                    values.append(f);ses.append(z['stderr'][0]);errors.append(error);times.append(z['runtime_seconds']);tails.append(z['tail_error_bound']);work.append(z['n_path_steps'])
            outcomes.append(flat['status']);attempts.append(flat)
        row.update(mc_completed=len(values),mc_terminal=sum(s!='pending' for s in outcomes),mc_all_four=len(values)==4,
            mc_success_count=sum(e<=.01 for e in errors),mc_seed_success_fraction=sum(e<=.01 for e in errors)/len(errors) if errors else None,
            mc_any_observed_success=any(e<=.01 for e in errors),mc_all_observed_success=len(errors)==4 and all(e<=.01 for e in errors),
            mc_solver_seconds=sum(times),mc_path_steps=sum(work),mc_mean=None,mc_aggregate_error=None,quality_label='insufficient_seeds')
        if len(values)==4:
            row.update(seed_summary(values,ses,1024,row['exact0'],max(tails),row['remizov_error0']))
            row['mc_aggregate_accuracy']=int(row['mc_aggregate_error']<=.01)
        rows.append(row)
    analysis=out/'analysis';analysis.mkdir(exist_ok=True)
    eq=pd.DataFrame(rows);att=pd.DataFrame(attempts);eq.to_csv(analysis/'equations.csv',index=False);att.to_csv(analysis/'attempts.csv',index=False)
    eligible=eq.mc_all_four & eq.remizov_known
    census={s:dict(rows=int((eq.split==s).sum()),paired=int(((eq.split==s)&eligible).sum()),families=eq.loc[eq.split==s,'family_id'].nunique()) for s in ('train','validation','test')}
    good=att.status.eq('completed');complete=eq.mc_all_four
    metrics=dict(attempts=4000,terminal=int(att.status.ne('pending').sum()),pending=int(att.status.eq('pending').sum()),
        statuses=att.status.value_counts().to_dict(),equations_all_four=int(complete.sum()),paired_equations=int(eligible.sum()),historical_identities_checked=checked,
        split_census=census,mc_seed_accuracy_rate=float(att.loc[good,'accuracy_met'].mean()),
        mc_aggregate_accuracy_rate=float(eq.loc[complete,'mc_aggregate_accuracy'].mean()) if complete.any() else None,
        remizov_point_accuracy_rate=float(eq.loc[eq.remizov_known,'remizov_accuracy0'].mean()),
        mc_all_four_pass=int(eq.mc_all_observed_success.sum()),mc_at_least_one_pass=int(eq.mc_any_observed_success.sum()),
        paired_aggregate_contingency={f'R{r}_MC{m}':int((eligible & eq.remizov_accuracy0.eq(r)&eq.mc_aggregate_accuracy.eq(m)).sum()) for r in (0,1) for m in (0,1)},
        exploratory_labels=eq.loc[eligible,'quality_label'].value_counts().to_dict(),
        error_quantiles={name:{str(q):float(eq.loc[mask,name].quantile(q)) for q in (.1,.5,.9,.95,.99,1.)} for name,mask in [('remizov_error0',eq.remizov_known),('mc_aggregate_error',complete)]},
        solver_seconds_sum=float(att.solver_seconds.sum()),task_seconds_sum=float(att.task_wall_seconds.sum()),
        path_steps=int(att.path_steps.sum()),peak_rss_mib=float(att.peak_rss_mib.max()),
        independent_seed_success95_lower_bound_if4of4=.05**.25,
        scope='x=0 only; MC 1024 per seed vs 4096-path aggregate distinguished; timing not directly comparable with saved33-point Remizov',
        dataset_sha256={n:sha(ROOT/n) for n in plan['protected_sha256'] if n.startswith('data/')})
    write(analysis/'summary.json',metrics)
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);analyze(ap.parse_args().out.resolve())
