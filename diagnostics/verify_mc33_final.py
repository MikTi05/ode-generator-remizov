"""Independent saved-only scalar verification of the final MC33 release.

No builder, solver, numerical backend, or ML imports. JSON/CSV/hashlib/math/
statistics only. Recompute every seed error and mean function from raw f.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
PROD=Path('results/mc33_parallel_2026-11-05/production')


def require(value,message):
    if not value: raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def csv_rows(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def close(a,b,label):
    require(math.isclose(float(a),float(b),rel_tol=2e-13,abs_tol=2e-15),label)


def scalar_stats(values):
    a=sorted(values)
    def q(p):
        pos=(len(a)-1)*p;i=math.floor(pos);j=math.ceil(pos)
        return a[i]+(a[j]-a[i])*(pos-i)
    return dict(n=len(a),pass_005=sum(v<=.05 for v in a),pass_001=sum(v<=.01 for v in a),
        rate_005=sum(v<=.05 for v in a)/len(a),minimum=min(a),q25=q(.25),median=statistics.median(a),
        q75=q(.75),q90=q(.9),q95=q(.95),q99=q(.99),maximum=max(a),mean=statistics.mean(a),
        std=statistics.stdev(a))


def verify(out,root=ROOT):
    plan=read(root/PROD/'plan.json');ph=sha(root/PROD/'plan.json')
    require(ph==read(root/PROD/'plan.sha256.json')['sha256'],'plan SHA')
    for name,h in plan['protected_sha256'].items():require(sha(root/name)==h,'pin '+name)
    for name,h in read(out/'input_sha256.json').items():require(sha(root/name)==h,'source '+name)
    summary=read(out/'summary.json');validation=read(out/'validation.json')
    require(validation['passed'] and validation['problem_equations']==0,'validation failed')
    published=csv_rows(out/'mc33_all1000.csv');rows={r['equation_id']:r for r in published}
    require(len(rows)==len(published)==1000,'table duplicates/count')
    jobs={j['key']:j for j in plan['jobs']};pairs=Counter((j['equation_id'],j['episode']) for j in jobs.values())
    require(len(jobs)==4000 and all(n==1 for n in pairs.values()),'manifest duplicates')
    calls=csv_rows(out/'mc33_calls4000.csv');call_index={(r['equation_id'],int(r['episode'])):r for r in calls}
    require(len(calls)==len(call_index)==4000,'calls CSV duplicates')
    nodes=csv_rows(out/'mc33_grid33000.csv');node_index={(r['equation_id'],float(r['x'])):r for r in nodes}
    require(len(nodes)==len(node_index)==33000,'grid CSV duplicates')
    primary=[];means=[];all_errors=[];episodes=[[] for _ in range(4)];old_errors=[];new_zeros=[];mean_zeros=[]
    observed_repeats=Counter();checked_arrays=0;details={}
    for pid,row in rows.items():
        fs=[];exact=None;errors=[];zeros=[]
        for ep in range(4):
            j=jobs[f'{pid}_MC1_episode{ep}'];path=root/PROD/'records'/(j['key']+'.json')
            wrapped=read(path);r=wrapped['record'];z=r['result']
            require(hashlib.sha256(json.dumps(r,sort_keys=True,allow_nan=False).encode()).hexdigest()==wrapped['sha256'],'raw checksum')
            require(r['job']==j and r['plan_sha256']==ph and r['status']=='completed' and z['success'] and not z['failure_reason'],'raw identity')
            require(j['config']['M']==4096 and j['config']['dt']==1/128 and j['config']['seed']==248800030100+ep,'MC1 config')
            require(z['x']==[-1+i/16 for i in range(33)] and len(z['f'])==33 and
                    z['completed_paths']==[4096]*33,'raw complete grid')
            require(all(math.isfinite(v) for v in z['f']+r['y_exact']),'raw finite')
            require(exact is None or exact==r['y_exact'],'exact across repeats');exact=r['y_exact']
            err=max(abs(a-b) for a,b in zip(z['f'],exact));zero=abs(z['f'][16]-exact[16])
            require(err==r['E_grid'] and err>=zero and r['accuracy_met']==(err<=.01),'raw saved errors')
            require(float(row[f'E33_{ep}'])==err and float(row[f'E0_{ep}'])==zero and
                    int(row[f'seed_{ep}'])==248800030100+ep,'per-seed CSV')
            c=call_index[(pid,ep)];require(c['record_sha256']==sha(path) and float(c['E33'])==err and
                    float(c['E0'])==zero and c['checksum']==wrapped['sha256'],'calls CSV content')
            fs.append(z['f']);errors.append(err);zeros.append(zero);episodes[ep].append(err);all_errors.append(err);checked_arrays+=1
        mean=[sum(fs[ep][i] for ep in range(4))/4 for i in range(33)]
        me=max(abs(a-b) for a,b in zip(mean,exact));mz=abs(mean[16]-exact[16])
        close(row['mean4_E33'],me,'mean function max error');close(row['mean4_E0'],mz,'mean function zero error')
        require(float(row['MC_E33'])==errors[0] and float(row['MC_E0'])==zeros[0] and
                (row['MC_pass_005']=='True')==(errors[0]<=.05) and (row['mean4_pass_005']=='True')==(me<=.05),'primary/aggregate threshold')
        require(int(row['repeats_pass_005'])==sum(v<=.05 for v in errors),'repeat successes')
        close(row['repeat_error_mean'],statistics.mean(errors),'mean seed errors diagnostic')
        close(row['repeat_error_median'],statistics.median(errors),'median seed errors diagnostic')
        for i in range(33):
            n=node_index[(pid,-1+i/16)];require(float(n['y_exact'])==exact[i],'node exact')
            for ep in range(4):require(float(n[f'f_{ep}'])==fs[ep][i],'node seed f')
            close(n['f_mean4'],mean[i],'node mean f');close(n['mean4_abs_error'],abs(mean[i]-exact[i]),'node mean error')
            require(float(n['episode0_abs_error'])==abs(fs[0][i]-exact[i]),'node primary error')
        old=[]
        for ep in range(4):
            r=read(root/'results/mc1000_x0_2026-10-25/attempts'/f'{pid}_seed{ep}.json')
            require(r['y_exact']==exact[16] and r['config']['M']==1024 and
                    r['config']['seed']==202610250101+ep,'old x0 identity/config')
            old.append(r['result']['f'][0])
        olde=abs(sum(old)/4-exact[16]);close(row['old_mean4_1024_E0'],olde,'old zero aggregate')
        # Recompute the other three methods directly, without the old builder.
        for label,dirname in [('RK4','rk4'),('PI','product_integral')]:
            r=read(root/'results/four_method_accuracy_1000_2026-11-01/records'/dirname/(pid+'.json'))
            require(r['exact_grid33']==exact,'IVP/MC exact')
            err=max(abs(r['result']['y'][4*i]-exact[i]) for i in range(33))
            require(float(row[label+'_E33'])==err,'IVP E33')
        r=read(root/'results/chernov_solve1000_2026-10-18/solutions'/(pid+'.json'))
        if r['computation_status']=='success':
            require(r['y_exact']==exact,'R/MC exact');err=max(abs(a-b) for a,b in zip(r['result']['f'],exact))
            require(float(row['R_E33'])==err,'R E33')
        else:require(row['R_E33']=='' and r['computation_status']=='worker_exception','R missing')
        primary.append(errors[0]);means.append(me);old_errors.append(olde);new_zeros.append(zeros[0]);mean_zeros.append(mz)
        observed_repeats[sum(v<=.05 for v in errors)]+=1
        details[pid]=dict(primary=errors[0],mean4=me)
    for key,values in [('primary',primary),('mean4_supplementary',means),('all_calls',all_errors),
                       ('old_mean4_1024_E0',old_errors),('new_episode0_E0',new_zeros),('new_mean4_E0',mean_zeros)]:
        for name,val in scalar_stats(values).items():close(summary[key][name],val,'summary '+key+' '+name)
    for ep,values in enumerate(episodes):
        for name,val in scalar_stats(values).items():close(summary['episodes'][str(ep)][name],val,'episode stats')
    require(summary['repeats_pass_005_histogram']=={str(k):v for k,v in observed_repeats.items()},'repeat histogram')
    comp=csv_rows(out/'comparison.csv')
    for c in comp:
        sub=[r for r in rows.values() if c['scope']=='available' or r['R_E33']!='']
        values=[float(r[c['method']+'_E33']) for r in sub if r[c['method']+'_E33']!='']
        for name,val in scalar_stats(values).items():close(c[name],val,'comparison '+c['method']+' '+name)
    require(csv_rows(out/'four_methods_all1000.csv')==published,'four method table')
    require(csv_rows(out/'common997.csv')==[r for r in published if r['R_E33']!=''],'common997 table')
    require(sum(all(float(r['RK4_E33'])<float(r[m+'_E33']) for m in ('R','PI','MC'))
                for r in rows.values() if r['R_E33']!='')==summary['four_method_unique_RK4_winners'],'winner counts')
    distributions=csv_rows(out/'distribution.csv')
    for d in distributions:
        a=primary if d['estimator']=='episode0' else means;lo=float(d['lower']);hi=None if d['upper']=='' else float(d['upper'])
        count=sum((v>=lo if lo==0 else v>lo) and (hi is None or v<=hi) for v in a)
        require(count==int(d['count']),'distribution count')
    result=dict(passed=True,independent_implementation='stdlib scalar max/abs/sum/statistics; no builder/solver imports',
        raw_final_calls_checked=4000,old_x0_calls_checked=4000,other_raw_records_checked=3000,
        equations=1000,node_rows_checked=33000,canonical_exact_cross_source_match=True,
        primary_pass_005=sum(v<=.05 for v in primary),mean4_pass_005=sum(v<=.05 for v in means),
        source_hashes_checked=len(read(out/'input_sha256.json')),new_solver_calls=0,new_fit_calls=0)
    with (out/'independent_verification.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    verify(p.parse_args().out.resolve())
