#!/usr/bin/env python3
"""Independently verify the saved 60-task cost pilot; never invoke a solver."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
METHODS=('rk4','product_integral');NS=(32,64,128,256)


def require(condition,message):
    if not condition:raise ValueError(message)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(p):
    with p.open(newline='',encoding='utf-8') as stream:return list(csv.DictReader(stream))
def close(a,b):return math.isclose(a,b,rel_tol=2e-13,abs_tol=1e-15)
def checkrow(observed,expected,label):
    for key,v in expected.items():
        got=observed[key]
        if v is None:require(got=='',label+'/'+key)
        elif isinstance(v,bool):require(got==str(v),label+'/'+key)
        elif isinstance(v,(int,float)):require(close(float(got),v),label+'/'+key)
        elif isinstance(v,(list,dict)):require(json.loads(got)==v,label+'/'+key)
        else:require(got==v,label+'/'+key)


def verify(out):
    plan=read(out/'plan.json');session=read(out/'session.json');ids=plan['equation_ids'];plan_sha=sha(out/'plan.json')
    require(plan['epsilon']==.01 and plan['N']==list(NS) and plan['measured_repeats']==5 and plan['maximum_solver_calls']==2880,'fixed protocol')
    require(all(sha(ROOT/p)==h for p,h in plan['source_sha256'].items()),'pinned sources')
    require(sha(out/'selection.csv')==plan['selection_sha256'] and sha(out/'candidate_pool.csv')==plan['candidate_pool_sha256'],'selection hashes')
    require(read(out/'started.json')['plan_sha256']==plan_sha,'plan started seal')
    require(plan['created_utc']<=read(out/'started.json')['utc'],'plan must precede computation')
    metadata=[json.loads(line) for line in (ROOT/plan['dataset_dir']/'all_methods_equations_v1_metadata.jsonl').read_text().splitlines()]
    meta={m['equation_id']:m for m in metadata};chosen=[meta[pid] for pid in ids]
    require(len(ids)==len(set(ids))==len({m['family_id'] for m in chosen})==len({m['structure_signature'] for m in chosen})==60,'60 distinct families/topologies')
    require(all(m['split']=='train' for m in chosen),'nontrain selection')
    require(Counter((m['complexity_level'],m['regime']) for m in chosen)=={(l,r):3 for l in ('L1','L2','L3','L4') for r in ('constant','slow_varying','moderate_varying','oscillatory','high_contrast')},'stratum coverage')
    splits=read(ROOT/'results/ml_selector_hypotheses_2026-10-31/splits.json')
    held=set().union(*(set(s['test']['ids']) for s in splits['scenarios'].values()))
    heldfamilies={meta[pid]['family_id'] for pid in held}
    require(not {m['family_id'] for m in chosen}&heldfamilies,'historical Test family')
    other=[json.loads(line) for line in (ROOT/'results/ml_selector_final_2026-10-27/final_experiment/test_inputs/all_methods_equations_v1_metadata.jsonl').read_text().splitlines()]
    require(not {m['structure_signature'] for m in chosen}&{m['structure_signature'] for m in other},'other Test topology')
    inputs={pid:read(out/'inputs'/(pid+'.json')) for pid in ids}
    for pid,q in inputs.items():
        require(q['expression_hash']==meta[pid]['expression_hash'] and q['family_id']==meta[pid]['family_id'],'input identity')
        require(q['inputs']=={k:meta[pid][k] for k in ('x0','x1','y0','v0')},'input IC')
        require(q['target_x']==[-1+i/16 for i in range(33)] and len(q['exact33'])==33,'input target')
        old=read(ROOT/'results/four_method_accuracy_1000_2026-11-01/records/rk4'/(pid+'.json'))
        require(q['exact33']==old['exact_grid33'],'saved exact changed')
    records=[read(p) for p in sorted((out/'records').glob('*.json'))]
    require(len(records)==2880 and {r['sequence'] for r in records}==set(range(1,2881)),'call budget/count')
    groups=defaultdict(list);keys=set();rawrows={int(r['sequence']):r for r in table(out/'records.csv')}
    require(len(rawrows)==2880,'raw CSV cardinality')
    for r in records:
        pid=r['equation_id'];n=r['N'];method=r['method'];episode=r['episode']
        require(pid in ids and method in METHODS and n in NS and episode in range(-1,5),'unexpected computation')
        require(r['phase']==('warmup' if episode==-1 else 'measured') and r['plan_sha256']==plan_sha,'record phase/plan')
        key=(pid,method,n,episode);require(key not in keys,'duplicated computation');keys.add(key)
        require(r['solver_nanoseconds']>0 and r['solver_seconds']==r['solver_nanoseconds']/1e9,'timing units')
        q=r['result']
        if r['usable']:
            require(q['success'] and q['failure_reason']=='' and q['n_steps']==n and q['h']==2/n,'result status')
            require(q['x']==[-1+2*i/n for i in range(n+1)],'fixed solver grid')
            require(len(q['y'])==len(q['dy'])==n+1 and all(math.isfinite(v) for k in ('y','dy') for v in q[k]),'finite trajectory')
            require(q['y'][0]==inputs[pid]['inputs']['y0'] and q['dy'][0]==inputs[pid]['inputs']['v0'],'trajectory IC')
            require(q['x'][::n//32]==inputs[pid]['target_x'],'nested grid extraction')
            e=max(abs(q['y'][i*n//32]-inputs[pid]['exact33'][i]) for i in range(33))
            require(e==r['E33'] and r['accuracy_met']==(e<=.01),'independent E33')
            require(q['nfev']==4*n if method=='rk4' else q['expm_evals']==q['matvec_evals']==q['steps_completed']==n,'method counters')
        else:require(r['E33'] is None and r['accuracy_met'] is None,'fabricated failed error')
        checkrow(rawrows[r['sequence']],{k:r[k] for k in rawrows[r['sequence']]},'raw CSV')
        groups[(pid,method,n)].append(r)
    require(len(groups)==480,'profile count')
    profiles={};published={(r['equation_id'],r['method'],int(r['N'])):r for r in table(out/'profiles.csv')}
    require(len(published)==480,'profile CSV cardinality')
    for key,rr in groups.items():
        require({r['episode'] for r in rr}==set(range(-1,5)),'warmup/repeats')
        rr=sorted(rr,key=lambda r:r['episode'])
        require(all(r['result']==rr[0]['result'] and r['E33']==rr[0]['E33'] for r in rr),'deterministic six-call answers')
        a=[r['solver_seconds'] for r in rr if r['episode']>=0];median=statistics.median(a)
        mad=statistics.median(abs(v-median) for v in a);pid,method,n=key
        valid=all(r['usable'] for r in rr)
        p=dict(equation_id=pid,family_id=meta[pid]['family_id'],method=method,N=n,configuration=f'{method}_N{n}',
            usable=valid,repeat_identical=True,E33=rr[0]['E33'] if valid else None,
            accurate=valid and all(r['E33']<=.01 for r in rr),solver_median_seconds=median,
            solver_min_seconds=min(a),solver_max_seconds=max(a),solver_MAD_seconds=mad,
            relative_MAD=mad/median,timing_stable=mad/median<=.20,times=a,
            failure_reason='' if valid else '|'.join(sorted({r['status'] for r in rr[1:] if not r['usable']})))
        checkrow(published[key],p,'profile');profiles[key]=p
        if n==128 and valid:
            old=read(ROOT/'results/four_method_accuracy_1000_2026-11-01/records'/method/(pid+'.json'))
            require(all(close(a,b) for a,b in zip(rr[0]['result']['y'],old['result']['y'])),'N128 historical trajectory mismatch')
    def best(rows):
        a=[p for p in rows if p['accurate']]
        return min(a,key=lambda p:(p['solver_median_seconds'],METHODS.index(p['method']),p['N'])) if a else None
    def robust(a,b):
        return bool(a and b and a['timing_stable'] and b['timing_stable'] and
            a['solver_median_seconds']<=.9*b['solver_median_seconds'] and a['solver_max_seconds']<b['solver_min_seconds'])
    constant_rows=table(out/'configuration_summary.csv');const=[]
    for method in METHODS:
        for n in NS:
            a=[profiles[pid,method,n] for pid in ids]
            errors=[p['E33'] for p in a if p['E33'] is not None]
            c=dict(method=method,N=n,configuration=f'{method}_N{n}',successful=sum(p['usable'] for p in a),
                accurate=sum(p['accurate'] for p in a),total_solver_seconds=sum(p['solver_median_seconds'] for p in a),
                mean_solver_seconds=statistics.mean(p['solver_median_seconds'] for p in a),
                mean_E33=statistics.mean(errors) if errors else None,max_E33=max(errors) if errors else None,
                stable_profiles=sum(p['timing_stable'] for p in a))
            checkrow(next(r for r in constant_rows if r['method']==method and int(r['N'])==n),c,'constant');const.append(c)
    feasible=[c for c in const if c['accurate']==60]
    constant=min(feasible,key=lambda c:(c['total_solver_seconds'],METHODS.index(c['method']),c['N'])) if feasible else None
    winners=table(out/'winners.csv');require(len(winners)==60 and [r['equation_id'] for r in winners]==ids,'winner IDs')
    wins=Counter();robust_counts=Counter();none=[];unknown=[];derived=[]
    for row in winners:
        pid=row['equation_id'];group=[profiles[pid,m,n] for m in METHODS for n in NS]
        oracle=best(group);rk=best([p for p in group if p['method']=='rk4']);pi=best([p for p in group if p['method']=='product_integral'])
        rkmin=min((p for p in group if p['method']=='rk4' and p['accurate']),key=lambda p:p['N'],default=None)
        piwin=robust(pi,rk) if rk else pi is not None; rkwin=robust(rk,pi) if pi else rk is not None
        cons=pi if piwin else rk if rk else pi
        if oracle:wins[oracle['method']]+=1
        elif all(p['usable'] for p in group):none.append(pid)
        else:unknown.append(pid)
        if piwin:robust_counts['product_integral']+=1
        elif rkwin:robust_counts['rk4']+=1
        elif oracle:robust_counts['practical_tie_or_unstable']+=1
        c=next((p for p in group if constant and p['configuration']==constant['configuration']),None)
        expected=dict(equation_id=pid,family_id=meta[pid]['family_id'],status='solved' if oracle else 'none_meets_tolerance' if pid in none else 'unknown_due_to_failure',
            admissible_configurations=[p['configuration'] for p in sorted(group,key=lambda p:(p['method'],p['N'])) if p['accurate']],
            oracle_configuration=oracle['configuration'] if oracle else None,oracle_method=oracle['method'] if oracle else None,
            oracle_N=oracle['N'] if oracle else None,oracle_E33=oracle['E33'] if oracle else None,
            oracle_seconds=oracle['solver_median_seconds'] if oracle else None,rk4_only_configuration=rk['configuration'] if rk else None,
            rk4_only_seconds=rk['solver_median_seconds'] if rk else None,rk4_minN_configuration=rkmin['configuration'] if rkmin else None,
            rk4_minN_seconds=rkmin['solver_median_seconds'] if rkmin else None,PI_best_configuration=pi['configuration'] if pi else None,
            PI_best_seconds=pi['solver_median_seconds'] if pi else None,PI_robust_win=piwin,RK4_robust_win=rkwin,
            conservative_configuration=cons['configuration'] if cons else None,conservative_seconds=cons['solver_median_seconds'] if cons else None,
            constant_configuration=c['configuration'] if c else None,constant_seconds=c['solver_median_seconds'] if c else None,
            gain_vs_constant=1-oracle['solver_median_seconds']/c['solver_median_seconds'] if oracle and c else None,
            gain_vs_rk4_only=1-oracle['solver_median_seconds']/rk['solver_median_seconds'] if oracle and rk else None,
            PI_gain_vs_rk4_only=1-pi['solver_median_seconds']/rk['solver_median_seconds'] if pi and rk else None)
        checkrow(row,expected,'winner');derived.append(expected)
    summary=read(out/'summary.json')
    require(summary['raw_method_winners']=={m:wins[m] for m in METHODS} and summary['robust_method_winners']==dict(robust_counts),'method winners')
    if constant is None:require(summary['best_constant'] is None,'best constant')
    else:
        require(all(close(summary['best_constant'][k],v) if isinstance(v,(float,int)) else summary['best_constant'][k]==v
                    for k,v in constant.items()),'best constant')
    require(summary['no_configuration_meets_tolerance']==none and summary['unknown_due_to_failure']==unknown,'unresolved cases')
    require(table(out/'no_solution.csv')==[r for r in winners if r['status']!='solved'],'no-solution CSV')
    for label,g in summary['gains'].items():
        choice='conservative_seconds' if label.startswith('conservative') else 'oracle_seconds'
        baseline='constant_seconds' if label.endswith('best_constant') else 'rk4_only_seconds' if label.endswith('rk4_only_oracle') else 'rk4_minN_seconds'
        pairs=[(r[baseline],r[choice]) for r in derived if r[baseline] is not None and r[choice] is not None]
        if not pairs:require(g is None,'empty cost comparison');continue
        a,b=zip(*pairs);rel=[1-y/x for x,y in pairs]
        expected=dict(n=len(a),baseline_mean_seconds=statistics.mean(a),selected_mean_seconds=statistics.mean(b),
            mean_absolute_saving_seconds=statistics.mean(x-y for x,y in pairs),pooled_saving_fraction=1-sum(b)/sum(a),
            mean_relative_saving_fraction=statistics.mean(rel),median_relative_saving_fraction=statistics.median(rel),
            total_baseline_seconds=sum(a),total_selected_seconds=sum(b))
        require(all(close(g[k],v) for k,v in expected.items()),'paired cost gain '+label)
    stable=sum(p['timing_stable'] for p in profiles.values())/480
    gates=dict(all60_solved=sum(wins.values())==60,no_unknown_minimum=not unknown,constant_meets_tolerance_on60=constant is not None,
        rk4_only_covers60=all(r['rk4_only_seconds'] is not None for r in derived),
        heterogeneous_raw_winners=all(wins[m]>=12 for m in METHODS),robust_each_method=all(robust_counts[m]>=12 for m in METHODS),stable_profile_fraction=stable>=.90)
    for name in ('best_constant','rk4_only_oracle'):
        g=summary['gains']['conservative_oracle_vs_'+name]
        gates['practical_gain_vs_'+name]=bool(g and g['n']==60 and g['pooled_saving_fraction']>=.10 and g['mean_relative_saving_fraction']>=.10)
    require(summary['GO_gates']==gates and summary['recommendation']==('GO' if all(gates.values()) else 'NO-GO'),'independent GO decision')
    require(summary['GO_thresholds']==plan['GO_policy'],'preregistered thresholds')
    require(session['solver_calls']==2880 and session['warmup_calls']==480 and session['measured_calls']==2400 and session['complete'],'session counters')
    require(session['forbidden_calls']==dict(MC=0,remizov=0,fit=0,inference=0),'forbidden calls')
    before=read(out/'environment.before.json');require(all(p['num_threads']==1 for p in before['threadpools']),'single numeric thread')
    require(all(sha(ROOT/p)==h for p,h in plan['source_sha256'].items()),'final source integrity')
    result=dict(passed=True,selection_train_families=60,selection_strata=20,historical_Test_overlap=0,
        raw_attempts_checked=2880,measured_times_checked=2400,profiles_checked=480,winners_checked=60,
        nested_grid_errors_recomputed=2880,historical_N128_trajectories_crosschecked=120,
        repeat_answers_identical=True,all_CSV_cells_checked=True,GO_independently_recomputed=True,
        recommendation=summary['recommendation'],new_solver_calls=0,source_checksums_unchanged=True,
        artifacts_sha256={name:sha(out/name) for name in ('plan.json','selection.csv','records.csv','profiles.csv','winners.csv','configuration_summary.csv','summary.json','REPORT.md')})
    atomic=out/'verification.json';require(not atomic.exists(),'verification already exists; choose separate verification output instead of overwriting')
    atomic.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='artifacts_sha256'},ensure_ascii=False))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    verify(parser.parse_args().out.resolve())
