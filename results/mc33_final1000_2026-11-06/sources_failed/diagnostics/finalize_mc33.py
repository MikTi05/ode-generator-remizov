"""Verify saved production MC1 and publish final grid33 accuracy artifacts.

No numerical solve, training, or interpolation. Episode0 remains primary;
mean-of-four functions is explicitly a supplementary 16384-path estimator.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PRODUCTION = Path('results/mc33_parallel_2026-11-05/production')
RELEASE = Path('results/saved_grid33_1000_2026-11-04')
DATA = Path('data/all_methods_v1_1000_2026-10-18')
CONTRACT = Path('reports/ml_selector_contract/plan.json')
X0 = Path('results/mc1000_x0_2026-10-25')
GRID = [-1+i/16 for i in range(33)]
SEEDS = [248800030100+i for i in range(4)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_sha(record):
    # Exact serialization used by the frozen runner (including default spaces).
    return hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()


def unique_pairs(pairs):
    result = {}
    for k, v in pairs:
        require(k not in result, 'duplicate JSON key: '+k)
        result[k] = v
    return result


def finite_tree(value):
    if isinstance(value, float):
        require(math.isfinite(value), 'NaN/Inf in JSON')
    elif isinstance(value, dict):
        for v in value.values(): finite_tree(v)
    elif isinstance(value, list):
        for v in value: finite_tree(v)


def strict_json(data):
    def bad(value): raise ValueError('non-standard JSON constant: '+value)
    obj = json.loads(data, object_pairs_hook=unique_pairs, parse_constant=bad)
    finite_tree(obj)
    return obj


class Reader:
    def __init__(self, root=ROOT):
        self.root = root
        self.hashes = {}

    def bytes(self, path):
        path = Path(path)
        if path.is_absolute(): path = path.relative_to(self.root)
        data = (self.root/path).read_bytes()
        h = hashlib.sha256(data).hexdigest()
        require(str(path) not in self.hashes or self.hashes[str(path)] == h, 'input changed: '+str(path))
        self.hashes[str(path)] = h
        return data

    def json(self, path): return strict_json(self.bytes(path))


def vector(value, size, label):
    require(isinstance(value, list) and len(value) == size, label+': length')
    require(all(type(v) in (int, float) and math.isfinite(v) for v in value), label+': finite numbers')
    return value


def validate_mc(job, raw, exact, expected_G, expected_T):
    """Scalar independent checks; called for all 4000 final records."""
    from mc_solver import _spawn_key, _gap_lower, _tail_bound
    z = raw['result']; c = job['config']; pid = job['equation_id']
    require(raw['job'] == job and raw['config'] == c, pid+': full job/config')
    require(raw['status'] == 'completed' and raw['solver_called'] is True and
            z['success'] is True and z['failure_reason'] == '', pid+': status')
    require(raw['equation_id'] == z['problem_id'] == pid and
            raw['expression_hash'] == job['expression_hash'], pid+': identity')
    require(c == dict(M=4096, dt=1/128, T=None, tail_eps=1e-4,
        seed=SEEDS[job['episode']], stream_id=0, chunk=1024, max_steps=100000,
        max_path_steps=job['numerical_work'], max_batch_paths=1024, max_runtime_seconds=None), pid+': MC1 config')
    require(job['target_x'] == z['x'] == GRID, pid+': grid')
    require(job['expected_T'] == z['T'] == expected_T and job['forcing_G'] == expected_G ==
            z['forcing_abs_upper_bound'], pid+': analytical G/T')
    n = max(1, math.ceil(expected_T/(1/128)))
    if expected_T/n > 1/128: n += 1
    require(z['n_steps'] == job['expected_steps'] == n and z['dt_requested'] == 1/128 and
            z['dt_actual'] == expected_T/n <= 1/128, pid+': step/grid')
    work = 33*4096*n
    require(job['numerical_work'] == z['n_path_steps'] == z['n_normal_draws'] == work, pid+': work')
    require(z['M'] == 4096 and z['completed_paths'] == [4096]*33 and
            z['n_batches'] == 132 and z['peak_batch_paths'] == 1024, pid+': paths/chunks')
    require(z['seed_entropy'] == c['seed'] and z['stream_id'] == 0 and
            z['point_spawn_keys'] == [_spawn_key(pid, x, 0) for x in GRID] and
            z['rng_scheme'] == 'PCG64/SeedSequence/point_address_v1', pid+': RNG')
    require(z['backend'] == 'numba' and z['block_steps'] == 64 and
            z['numba_import_seconds'] == z['compilation_seconds'] == 0, pid+': warm backend')
    f = vector(z['f'], 33, pid+': f')
    se = vector(z['stderr'], 33, pid+': stderr')
    for name in ('ci95', 'ci_lower', 'ci_upper'): vector(z[name], 33, pid+': '+name)
    require(raw['y_exact'] == list(exact), pid+': canonical exact')
    require(all(v >= 0 for v in se), pid+': negative stderr')
    require(z['ci95'] == [1.96*v for v in se] and z['ci_lower'] == [a-b for a,b in zip(f,z['ci95'])]
            and z['ci_upper'] == [a+b for a,b in zip(f,z['ci95'])], pid+': confidence arithmetic')
    require(z['tail_eps'] == 1e-4 and z['tail_tolerance_met'] is True and
            0 <= z['tail_error_bound'] <= 1e-4 and z['total_error_certified'] is False,
            pid+': tail/certification contract')
    require(z['coefficient_values'] == dict(a=work,b=work,c_R=work+33*4096,g_R=work+33*4096), pid+': counters')
    error = max(abs(a-b) for a,b in zip(f,exact))
    e0 = abs(f[16]-exact[16])
    require(raw['E_grid'] == error and raw['accuracy_met'] == (error <= .01) and error >= e0,
            pid+': saved max error')
    require(raw['finished_epoch'] >= raw['started_epoch'] and raw['method_wall_seconds'] > 0 and
            raw['peak_rss_mib'] > 0, pid+': timing/resources')
    return error, e0


def stats(values):
    import numpy as np
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), pass_005=int(sum(a <= .05)), pass_001=int(sum(a <= .01)),
        rate_005=float(np.mean(a <= .05)), minimum=float(a.min()), q25=float(np.quantile(a,.25)),
        median=float(np.median(a)), q75=float(np.quantile(a,.75)), q90=float(np.quantile(a,.9)),
        q95=float(np.quantile(a,.95)), q99=float(np.quantile(a,.99)), maximum=float(a.max()),
        mean=float(a.mean()), std=float(a.std(ddof=1)))


def aggregate(fs, exact):
    import numpy as np
    a = np.asarray(fs, dtype=np.float64); y = np.asarray(exact, dtype=np.float64)
    require(a.shape == (4,33) and y.shape == (33,), 'aggregate shape')
    require(np.isfinite(a).all() and np.isfinite(y).all(), 'aggregate finite')
    mean = a.mean(axis=0, dtype=np.float64)
    return mean, float(np.max(np.abs(mean-y))), abs(float(mean[16])-float(y[16]))


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')


def table(path, rows):
    with Path(path).open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def validate(out):
    import numpy as np
    from all_methods_dataset_loader import load_all_methods_equations
    from mc_optimized import prepare_all_methods_mc
    from mc_solver import auto_T, _gap_lower, _tail_bound
    from diagnostics.build_saved_grid33 import classify_row
    reader=Reader(); plan=reader.json(PRODUCTION/'plan.json')
    ph=reader.hashes[str(PRODUCTION/'plan.json')]
    require(reader.json(PRODUCTION/'plan.sha256.json')['sha256'] == ph,'plan checksum')
    # Verify against the pre-run artifact manifest, as well as its internal SHA.
    old=reader.json(Path('results/mc33_parallel_2026-11-05/artifact_sha256.json'))
    require(old['production/plan.json'] == ph, 'plan differs from pre-run frozen version')
    for name,h in plan['protected_sha256'].items():
        require(hashlib.sha256(reader.bytes(name)).hexdigest()==h,'pinned source/input: '+name)
        snap=PRODUCTION/'sources'/name
        if (ROOT/snap).is_file(): require(sha(ROOT/snap)==h,'source snapshot: '+name)
    contract=reader.json(CONTRACT)
    require(plan['final_seed_schedule']==contract['final_seed_schedule']['MC1']==SEEDS,'final seeds')
    require(plan['aggregation']=='none; E_grid for each seed; episode0 primary','aggregation policy changed')
    eqs={e.equation_id:e for e in load_all_methods_equations(ROOT/DATA/'all_methods_equations_v1.csv',
         ROOT/DATA/'all_methods_equations_v1_metadata.jsonl')}
    require(len(eqs)==1000,'canonical equations')
    jobs=plan['jobs']; lookup={j['key']:j for j in jobs}
    require(len(jobs)==len(lookup)==4000,'manifest duplicates/count')
    expected={(pid,i) for pid in eqs for i in range(4)}
    require({(j['equation_id'],j['episode']) for j in jobs}==expected,'manifest pair coverage')
    require({p.stem for p in (ROOT/PRODUCTION/'records').glob('*.json')}==set(lookup),'result coverage')
    old_release=reader.json(RELEASE/'verification.json')
    require(old_release['passed'] is True,'old release not verified')
    for name,h in old_release['artifacts_sha256'].items():
        require(hashlib.sha256(reader.bytes(RELEASE/name)).hexdigest()==h,'old release artifact changed: '+name)
    old_sources=reader.json(RELEASE/'input_sha256.json')
    for name,h in old_sources.items():
        require(hashlib.sha256(reader.bytes(name)).hexdigest()==h,'old numerical source changed: '+name)
    previous={r['equation_id']:r for r in csv.DictReader(reader.bytes(RELEASE/'grid33_all1000.csv').decode().splitlines())}
    requests={p.name[:-len('.request.json')]:reader.json(p) for p in
              (ROOT/PRODUCTION/'attempts').glob('*.request.json')}
    result_attempts={p.name[:-len('.result.json')]:p for p in
                     (ROOT/PRODUCTION/'attempts').glob('*.result.json')}
    interruptions={p.name[:-len('.interrupted.json')]:reader.json(p) for p in
                    (ROOT/PRODUCTION/'attempts').glob('*.interrupted.json')}
    require(set(requests)==set(result_attempts)|set(interruptions) and
            not set(result_attempts)&set(interruptions),'unaccounted/ambiguous physical attempts')
    for attempt,req in requests.items():
        require(req['attempt']==attempt and req['job']==lookup[req['job']['key']] and
                req['plan_sha256']==ph,'request identity')
    for attempt,r in interruptions.items():
        require(r['request']==requests[attempt] and r['reason']=='interrupted' and
                r['partial_solver_counters']=='unknown','interruption contract')
    rows=[]; calls=[]; nodes=[]; records={}; seen_attempts=set(); total_work=0
    ivp=Path('results/four_method_accuracy_1000_2026-11-01/records')
    for k,pid in enumerate(sorted(eqs)):
        e=eqs[pid];meta=e.metadata;exact=np.asarray(e.y_exact(np.asarray(GRID)))
        vector(exact.tolist(),33,'canonical exact')
        prepared=prepare_all_methods_mc(meta); problem=prepared.problem
        G=prepared.forcing_bound.abs_upper_bound
        T=auto_T(problem.lambda_value,problem.c_abs_upper_bound,G,1e-4)
        group=[]; e33=[]; e0=[]
        for episode in range(4):
            j=next_job=lookup[f'{pid}_MC1_episode{episode}']
            require(j['family_id']==meta['family_id'] and j['split']==meta['split'] and
                    j['expression_hash']==meta['expression_hash'] and j['profile']=='MC1','job metadata')
            path=PRODUCTION/'records'/(j['key']+'.json');wrapped=reader.json(path);r=wrapped['record']
            require(record_sha(r)==wrapped['sha256'],'result checksum: '+j['key'])
            require(r['plan_sha256']==ph and r['attempt'] not in seen_attempts,'provenance/duplicate attempt')
            attempt=r['attempt'];seen_attempts.add(attempt)
            require(attempt in result_attempts and requests[attempt]['job']==j and
                    requests[attempt]['session']==r['session'],'result/request link')
            require(reader.bytes(result_attempts[attempt])==reader.bytes(path),'record/attempt-result differ')
            err,zero=validate_mc(j,r,exact,G,T)
            require(r['result']['lambda_c_gap']==_gap_lower(problem.lambda_value,problem.c_abs_upper_bound) and
                    r['result']['tail_error_bound']==_tail_bound(G,r['result']['lambda_c_gap'],T),'gap/tail arithmetic')
            require(r['result']['forcing_bound_source']==prepared.forcing_bound.source,'forcing certificate source')
            group.append(r);e33.append(err);e0.append(zero);total_work+=j['numerical_work']
            calls.append(dict(equation_id=pid,episode=episode,seed=SEEDS[episode],status=r['status'],
                E33=err,E0=zero,pass_005=err<=.05,pass_001=err<=.01,record_sha256=reader.hashes[str(path)],
                checksum=wrapped['sha256'],attempt=attempt,session=r['session'],record_path=str(path),
                method_wall_seconds=r['method_wall_seconds'],path_updates=j['numerical_work']))
        records[pid]=group
        mean,mean_error,mean_e0=aggregate([r['result']['f'] for r in group],exact)
        old_mc=[reader.json(X0/'attempts'/f'{pid}_seed{i}.json') for i in range(4)]
        old_r=reader.json(Path('results/chernov_solve1000_2026-10-18/solutions')/(pid+'.json'))
        other=classify_row(meta,old_r,reader.json(ivp/'rk4'/(pid+'.json')),
                           reader.json(ivp/'product_integral'/(pid+'.json')),old_mc)
        for field in ('R_E33','RK4_E33','PI_E33','MC_E0_lower_bound','MC_mean4_0'):
            require((None if previous[pid][field]=='' else float(previous[pid][field]))==other[field],
                    'old release scalar mismatch: '+pid+' '+field)
        for old_record in old_mc:
            require(old_record['result']['T']==T and old_record['forcing_bound']['abs_upper_bound']==G,
                    'old x0 G/T differs')
        row=dict(equation_id=pid,family_id=meta['family_id'],split=meta['split'],
                 complexity=meta['complexity_level'],regime=meta['regime'],expression_hash=meta['expression_hash'],
                 valid_repeats=4,MC_E33=e33[0],MC_E0=e0[0],MC_pass_005=e33[0]<=.05,MC_pass_001=e33[0]<=.01)
        for episode in range(4):
            row.update({f'seed_{episode}':SEEDS[episode],f'E33_{episode}':e33[episode],f'E0_{episode}':e0[episode]})
        row.update(repeats_pass_005=sum(v<=.05 for v in e33),repeats_pass_001=sum(v<=.01 for v in e33),
            repeat_error_mean=float(np.mean(e33)),repeat_error_median=float(np.median(e33)),
            repeat_error_min=min(e33),repeat_error_max=max(e33),
            mean4_E33=mean_error,mean4_E0=mean_e0,mean4_pass_005=mean_error<=.05,
            mean4_pass_001=mean_error<=.01,old_mean4_1024_E0=other['MC_E0_lower_bound'],
            old_mean4_1024_f0=other['MC_mean4_0'],new_episode0_f0=group[0]['result']['f'][16],
            new_mean4_f0=float(mean[16]),exact0=float(exact[16]),R_E33=other['R_E33'],
            R_status=other['R_status'],RK4_E33=other['RK4_E33'],PI_E33=other['PI_E33'])
        available={name:row[name+'_E33'] for name in ('RK4','R','PI','MC') if row[name+'_E33'] is not None}
        row['best_available_method']='|'.join(name for name,v in available.items() if v==min(available.values()))
        row['four_method_complete']=row['R_E33'] is not None
        rows.append(row)
        for i,x in enumerate(GRID):
            nodes.append(dict(equation_id=pid,x=x,y_exact=float(exact[i]),
                **{f'f_{ep}':group[ep]['result']['f'][i] for ep in range(4)},
                f_mean4=float(mean[i]),episode0_abs_error=abs(group[0]['result']['f'][i]-float(exact[i])),
                mean4_abs_error=abs(float(mean[i])-float(exact[i]))))
        if (k+1)%250==0: print(f'Verified {k+1}/1000 equations',flush=True)
    require(seen_attempts==set(result_attempts),'attempt/result coverage')
    for attempt,r in interruptions.items():
        key=r['request']['job']['key'];completed=next(c for c in calls if c['record_path'].endswith('/'+key+'.json'))
        require(completed['attempt']!=attempt and lookup[key]['config']==r['request']['job']['config'],'retry changed seed/config')
    sessions=[reader.json(p) for p in sorted((ROOT/PRODUCTION/'sessions').glob('*.json'))]
    require(sum(s['new_terminal'] for s in sessions)==4000,'session total terminal')
    for s in sessions:
        actual=[r for group in records.values() for r in group if r['session']==s['session']]
        require(s['new_terminal']==len(actual),'session new_terminal discrepancy')
        if actual:
            require(s['workers']==4 and s['stop_reason'] is None and s['completed']==4000 and s['pending']==0,
                    'unexpected final session')
    for p in (ROOT/PRODUCTION/'control').rglob('*.guardian.json'):
        g=reader.json(p);require(g['finished'] and g['cleanup_success'],'guardian not cleaned')
    for p in (ROOT/PRODUCTION/'workers').glob('*.warmup.json'):
        w=reader.json(p);require(w['numba_threads']==1 and w['environment']==plan['environment'] and
            all(pool['num_threads']==1 for pool in w['threadpools']),'worker thread/environment mismatch')
    progress=reader.json(PRODUCTION/'progress.json')
    require(progress['terminal']==4000 and progress['pending']==0 and progress['statuses']=={'completed':4000}
            and progress['in_flight']==[],'final progress')
    validation=dict(passed=True,canonical_equations=1000,valid_equations=1000,problem_equations=0,
        completed_calls=4000,pending=0,unique_semantic_keys=4000,final_seeds=SEEDS,nodes_per_call=33,
        values_checked=132000,request_count=len(requests),complete_attempts=len(result_attempts),
        interrupted_attempts=len(interruptions),unaccounted_attempts=0,duplicate_completed_keys=0,
        issues=[],checks=dict(strict_json_no_duplicate_keys=True,no_nan_inf=True,plan_matches_pre_run_sha=True,
            all_input_and_source_pins_valid=True,record_and_attempt_checksums_valid=True,
            canonical_exact_recomputed=True,analytical_G_T_recomputed=True,RNG_addresses_verified=True,
            raw_max_error_recomputed=True,old_sources_and_release_unchanged=True,all_retry_seeds_preserved=True,
            old_x0_identity_exact_G_T_match=True,new_E33_ge_new_E0_all=True,guardians_clean=True),
        total_path_updates=total_work,sessions=sessions,
        aggregation='episode0 primary; mean4 function diagnostic, never mean/min of seed errors',
        old_x0_comparison='different seeds and M; exact equality not required; old E0 not a lower bound for new f',
        new_solver_calls=0,new_fit_calls=0)
    dump(out/'validation.json',validation)
    return reader,rows,calls,nodes,records,validation


def summarize(rows,calls,validation):
    comparisons=[]
    for scope,subset in [('available',rows),('common997',[r for r in rows if r['R_E33'] is not None])]:
        for method in ('RK4','R','PI','MC'):
            values=[r[method+'_E33'] for r in subset if r[method+'_E33'] is not None]
            comparisons.append(dict(scope=scope,method=method,total_scope=len(subset),missing=len(subset)-len(values),**stats(values)))
    primary=stats([r['MC_E33'] for r in rows]);mean4=stats([r['mean4_E33'] for r in rows])
    episodes={str(i):stats([r[f'E33_{i}'] for r in rows]) for i in range(4)}
    old0=stats([r['old_mean4_1024_E0'] for r in rows]);new0=stats([r['MC_E0'] for r in rows])
    common=[r for r in rows if r['four_method_complete']]
    def dominance(subset,key):
        return {method:sum(r['RK4_E33']<r[col] for r in subset if r[col] is not None)
                for method,col in [('R','R_E33'),('PI','PI_E33'),('MC','MC_E33'),('MC_mean4','mean4_E33')]}
    summary=dict(schema='MC33_final1000_v1',threshold=.05,secondary_frozen_threshold=.01,
        primary_estimator='MC1 episode0, M4096 per node, seed248800030100',
        primary=primary,mean4_supplementary=mean4,episodes=episodes,
        all_calls=stats([c['E33'] for c in calls]),old_mean4_1024_E0=old0,new_episode0_E0=new0,
        new_mean4_E0=stats([r['mean4_E0'] for r in rows]),
        repeats_pass_005_histogram=dict(sorted(Counter(r['repeats_pass_005'] for r in rows).items())),
        old0_new33_transitions={f'old_{a}_new_{b}':sum((r['old_mean4_1024_E0']<=.05)==a and (r['MC_E33']<=.05)==b for r in rows)
                                 for a in (False,True) for b in (False,True)},
        new_E0_pass_but_E33_fail=sum(r['MC_E0']<=.05<r['MC_E33'] for r in rows),
        new_mean4_E0_pass_but_E33_fail=sum(r['mean4_E0']<=.05<r['mean4_E33'] for r in rows),
        four_method_common=997,missing_R=[r['equation_id'] for r in rows if r['R_E33'] is None],
        best_available_counts=dict(Counter(r['best_available_method'] for r in rows)),
        strict_RK4_dominance_available=dominance(rows,'all'),strict_RK4_dominance_common997=dominance(common,'common'),
        four_method_unique_RK4_winners=sum(all(r['RK4_E33']<r[m+'_E33'] for m in ('R','PI','MC')) for r in common),
        comparison=comparisons,validation=validation,
        conclusion='priority4 complete: full MC33 verified; RK4 accuracy lead confirmed on these fixed profiles; ML NO-GO unchanged',
        timing_note='parallel throughput timings are not isolated per-method speed comparisons')
    return summary,comparisons


def plots(out,rows,summary):
    os.environ['MPLCONFIGDIR']=str(out/'mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    dest=out/'figures';dest.mkdir()
    def save(fig,name):
        fig.savefig(dest/(name+'.png'),dpi=180,facecolor='white')
        fig.savefig(dest/(name+'.pdf'),facecolor='white')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.8,5.2),layout='constrained')
    for label,values,color,style in [('MC1: основной episode0',[r['MC_E33'] for r in rows],'#C76B20','-'),
        ('Среднее 4 MC1: дополнительно',[r['mean4_E33'] for r in rows],'#1D7780','--')]:
        a=np.sort(values);ax.step(a,np.arange(1,len(a)+1)/len(a),where='post',label=label,color=color,ls=style,lw=2)
    ax.axvline(.05,color='#AB2940',ls=':',label='Допуск 0,05');ax.set(xscale='log',xlabel='Максимальная ошибка E33',ylabel='Доля уравнений',ylim=(0,1.02),title='Распределение MC33: все 1000 уравнений')
    ax.grid(alpha=.2);ax.legend(loc='lower right');save(fig,'mc33_ecdf')
    fig,ax=plt.subplots(figsize=(9,5.3),layout='constrained')
    colors={'RK4':'#287A50','R':'#306CA7','PI':'#8756A2','MC':'#C76B20'}
    labels={'RK4':'RK4','R':'Ремизов (997)','PI':'Продукт-интеграл','MC':'MC1 episode0'}
    for method in colors:
        a=np.sort([r[method+'_E33'] for r in rows if r[method+'_E33'] is not None]);ax.step(a,np.arange(1,len(a)+1)/len(a),where='post',label=labels[method],color=colors[method],lw=1.8)
    ax.axvline(.05,color='#AB2940',ls=':',label='Допуск 0,05');ax.set(xscale='log',xlabel='Максимальная ошибка E33',ylabel='Доля доступных ответов',ylim=(0,1.02),title='Четыре метода на одной сетке из 33 узлов')
    ax.grid(alpha=.2);ax.legend(loc='lower right');save(fig,'four_method_ecdf')
    fig,ax=plt.subplots(figsize=(8.8,5.2),layout='constrained')
    comp=[r for r in summary['comparison'] if r['scope']=='common997'];order=['RK4','R','MC','PI']
    bars=[]
    for i,m in enumerate(order):
        s=next(r for r in comp if r['method']==m);bars.append(s['pass_005']);ax.bar(i,s['pass_005'],color=colors[m]);ax.text(i,s['pass_005']+10,f"{s['pass_005']}/997",ha='center')
    ax.set(xticks=range(4),xticklabels=['RK4','Ремизов','MC1 episode0','Продукт-интеграл'],ylim=(0,1090),ylabel='Число задач с E33 ≤ 0,05',title='Общая выборка четырёх методов: 997 уравнений');save(fig,'four_method_accuracy_common997')
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    axes[0].scatter([r['MC_E0'] for r in rows],[r['MC_E33'] for r in rows],s=11,alpha=.45,color='#C76B20')
    axes[1].scatter([r['old_mean4_1024_E0'] for r in rows],[r['MC_E33'] for r in rows],s=11,alpha=.45,color='#306CA7')
    for ax,title,xlabel in zip(axes,['Одна новая реализация: E33 ≥ E0','Исторический MCx0 и новый MC33'],['Новый E0, MC1 episode0','Старый E0, среднее 4×MC1024']):
        ax.set(xscale='log',yscale='log',xlabel=xlabel,ylabel='Новый E33, MC1 episode0',title=title)
        lim=(1e-7,2);ax.plot(lim,lim,color='gray',ls='--',lw=.8);ax.axhline(.05,color='#AB2940',ls=':');ax.axvline(.05,color='#AB2940',ls=':');ax.grid(alpha=.2)
    save(fig,'x0_vs_mc33')
    fig,ax=plt.subplots(figsize=(8.8,5),layout='constrained')
    counts=summary['repeats_pass_005_histogram'];ax.bar(range(5),[counts.get(i,0) for i in range(5)],color='#1D7780')
    for i in range(5):ax.text(i,counts.get(i,0)+5,str(counts.get(i,0)),ha='center')
    ax.set(xticks=range(5),xlabel='Число повторов с E33 ≤ 0,05 из четырёх',ylabel='Число уравнений',title='Разброс MC1 по четырём финальным seeds');save(fig,'repeat_success_counts')


def report(out,s,comparison):
    v=s['validation'];final=next(r for r in v['sessions'] if r['new_terminal']==4000)
    lines=['# Итог MC33: приоритет №4 завершён','',
        'Проект №2488. Анализ сохранённых результатов, 2026-11-06 (Europe/Moscow).',
        'Новых solver-вызовов и ML-fit нет; основной LaTeX не изменён.','',
        '## Проверка полноты и корректности','',
        'Проверены все **4000 завершённых вызовов / 1000 уникальных уравнений / 4 финальных seed / 33 узла**. '
        'Проблемных уравнений **0**. Строгий JSON отвергает повторяющиеся ключи, NaN и Inf. '
        'Проверены SHA замороженного плана относительно снимка до запуска, исходники и датасет, '
        'checksum каждого результата, идентичность его attempt-result, request и semantic key. '
        'Пропущенных и дублированных завершённых заданий нет.','',
        f"Физических requests **{v['request_count']}**: 4000 complete и **{v['interrupted_attempts']} interrupted**. "
        'Ранняя сессия8 workers прервана без готовых ответов; те же задания выполнены в итоговой сессии4 workers '
        'с неизменными seeds/config. Восемь прерываний — сохранённая история, не восемь проблемных результатов. '
        'Все guardians завершены и подтверждают cleanup.','',
        f"Итоговая сессия: **4 workers, {final['wall_seconds']:.4f} с / {final['wall_seconds']/3600:.3f} ч**, "
        'stop_reason=null, completed4000, pending0. Её параллельные времена не являются '
        'изолированным сравнением скорости четырёх методов.','',
        '## Зафиксированная методика','',
        '`M=4096` на узел и один вызов, requested h=1/128, T=auto_T(λ,C,G,1e-4), '
        'chunk1024, Numba/block_steps64, seeds248800030100…103. '
        'G/T заново проверены аналитически, exact — безопасным каноническим AST-загрузчиком на33 узлах. '
        'Операторы/интеграторы и RNG не исполнялись заново для получения численного f.','',
        'Для каждой реплики `E33_s=max_j|f_s(x_j)-exact(x_j)|`. '
        '**Основной показатель — episode0**, как в frozen plan/CONTRACT. '
        'Все четыре E33 и число попаданий сохранены, без выбора лучшего seed. '
        'При пороге0,05 ошибка пересчитана без округления; прежний флаг accuracy_met относится к0,01 и проверен отдельно.','',
        'Дополнительный агрегат: `f_mean4(x)=sum_s f_s(x)/4`, '
        '`E33_mean4=max_j|f_mean4(x_j)-exact(x_j)|`. Это **не среднее/медиана четырёх ошибок** '
        'и не замена основного target. Его бюджет —4×4096=16384 траектории на узел. '
        'Такое усреднение соответствует прежней формуле mean-of-functions в x0, но прежний бюджет был4×1024.','',
        '## Точность MC33','',
        '| Показатель | E33≤0,05 | E33≤0,01 | Среднее | Медиана | p95 | Максимум |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,key in [('MC1 episode0 (основной)','primary'),('Среднее4 MC1 (дополнительный)','mean4_supplementary')]:
        a=s[key];lines.append(f"| {name} | {a['pass_005']}/1000 | {a['pass_001']}/1000 | {a['mean']:.9g} | {a['median']:.9g} | {a['q95']:.9g} | {a['maximum']:.9g} |")
    lines+=['','Параметры распределений, квантили и bins — summary.json/distribution.csv. '
        'Разброс по seeds и клеткам complexity×regime — repeat/group tables. '
        'Четыре повтора не подтверждают вероятность успеха ≥0,95; CI MC описывает дискретное среднее, '
        'а не полную ошибку ОДУ. E33 — сеточный максимум, не непрерывная L∞-гарантия.','',
        '## Четыре метода: единая сетка и допуск0,05','',
        '| Метод | Доступно | E33≤0,05 | Среднее E33 | Медиана | p95 | Максимум |',
        '|---|---:|---:|---:|---:|---:|---:|']
    names={'RK4':'RK4','R':'Ремизов n128, исторический taper','PI':'Продукт-интеграл','MC':'MC1 episode0'}
    for a in comparison:
        if a['scope']=='available':lines.append(f"| {names[a['method']]} | {a['n']} | {a['pass_005']} | {a['mean']:.9g} | {a['median']:.9g} | {a['q95']:.9g} | {a['maximum']:.9g} |")
    lines+=['','У Ремизова три прежних worker_exception: '+', '.join(s['missing_R'])+'. '
        'Они оставлены missing, без E=0/∞. Честная полная парная выборка —997.','',
        '| Метод | E33≤0,05 на общих997 | Доля |','|---|---:|---:|']
    for a in comparison:
        if a['scope']=='common997':lines.append(f"| {names[a['method']]} | {a['pass_005']}/997 | {a['rate_005']:.4%} |")
    lines+=['','Для RK4/PI выбраны каждый четвёртый из129 исходных узлов; интерполяции нет. '
        'Исторические raw R/RK4/PI повторно проверены и SHA совпали с выпуском saved_grid33. '
        'Сравнение относится к этим фиксированным профилям; оно не ранжирует математические методы при всех возможных бюджетах.','',
        '## Согласованность с прежним x0','',
        'На всех1000 подтверждены те же ID/expression_hash/семейства, exact(0), аналитические G/T/h. '
        'Для каждого нового seed **E33≥E0**, то же верно для нового среднего4. '
        'Но MC1024/seeds202610250101…104 и MC4096/seeds248800030100…103 — разные реализации: '
        'новые значения в нуле не обязаны совпадать со старыми, и старый E0 не служит нижней границей нового E33.','',
        f"Прежнее среднее4×MC1024: E0≤0,05 на **{s['old_mean4_1024_E0']['pass_005']}/1000**. "
        f"Новый episode0: E0≤0,05 на **{s['new_episode0_E0']['pass_005']}/1000**, "
        f"E33≤0,05 на **{s['primary']['pass_005']}/1000**. "
        f"У **{s['new_E0_pass_but_E33_fail']}** задач точка0 проходит допуск, а вся сетка — нет. "
        'Это показывает ограниченность оценки по одной точке. Числа не доказывают изменение точности из-за одного параметра: '
        'одновременно различаются seed schedule, M и пространственный охват.','',
        '## Изменились ли выводы','',
        f"RK4 имеет строго меньшую E33, чем MC1 episode0 на **{s['strict_RK4_dominance_available']['MC']}/1000**, "
        f"чем среднее4 MC1 на **{s['strict_RK4_dominance_available']['MC_mean4']}/1000**; "
        f"на **{s['four_method_unique_RK4_winners']}/997** полных четвёрках RK4 — единственный победитель по ошибке. "
        'Прежний вывод о лидерстве RK4 на этих данных подтверждён прямым MC33. '
        'Прежние x0-метрики сохранены как исторические; условное доказательство по старому E0 '
        'не переносится на новые seeds, но новые прямые результаты дают самостоятельную проверку. '
        'Уточнение касается полной MC33-точности и её разброса; оснований отменять ML NO-GO нет.','',
        '**Приоритет №4 можно считать завершённым:** все1000×4 вызова получены, проверены, '
        'выпущены итоговые таблицы и графики. Новые массовые вычисления не требуются. '
        'Три missing R относятся к прежней серии другого метода и не мешают завершению MC33; '
        'они ограничивают полное четырёхметодное сравнение997 задачами.','',
        '## Файлы и воспроизведение','',
        '- validation.json — полнота, SHA/identity/числа, история попыток.',
        '- mc33_all1000.csv — основной ответ, все4 ошибки и дополнительный mean4.',
        '- mc33_calls4000.csv — каждый вызов и ссылка на raw/checksum.',
        '- mc33_grid33000.csv — четыре f, mean4, exact и ошибки в узлах.',
        '- four_methods_all1000.csv / common997.csv — сопоставление.',
        '- comparison.csv, distribution.csv, groups.csv, summary.json — агрегаты.',
        '- figures/ —5 графиков PNG и векторные PDF; FIGURES.md — подписи.',
        '- independent_verification.json, checks.json, integrity.json — проверки.',
        '- input_sha256.json, artifact_sha256.json, sources/ — происхождение и воспроизводимость.','',
        '```bash',
        'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/finalize_mc33.py --out results/mc33_final1000_reproduction',
        'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/verify_mc33_final.py --out results/mc33_final1000_reproduction',
        '```','']
    (out/'REPORT.md').write_text('\n'.join(lines))


def build(out):
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'PROJECT_STATUS.before.md').write_bytes((ROOT/'PROJECT_STATUS.md').read_bytes())
    # Protect the full existing research state, including production, rather
    # than treating results generated in Terminal as a change by this task.
    protected={}
    for folder in ('data','results','reports','ml'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and out not in p.parents and not any(x in p.parts for x in ('__pycache__','.DS_Store')) and not p.name.endswith('.lock'):
                protected[str(p.relative_to(ROOT))]=sha(p)
    for name in ('MIGRATION_MANIFEST.json','mc_solver.py','mc_optimized.py','mc_ast_kernels.py',
                 'remizov_solver.py','all_methods_dataset_loader.py','src/all_methods_expressions.py','src/all_methods_dataset_v1.py'):
        protected[name]=sha(ROOT/name)
    dump(out/'protected_before.json',protected)
    try:
        reader,rows,calls,nodes,records,validation=validate(out)
    except Exception as exc:
        if not (out/'validation.json').exists(): dump(out/'validation.json',dict(passed=False,error=str(exc),new_solver_calls=0))
        raise
    s,comparison=summarize(rows,calls,validation)
    table(out/'mc33_all1000.csv',rows);table(out/'mc33_calls4000.csv',calls);table(out/'mc33_grid33000.csv',nodes)
    table(out/'four_methods_all1000.csv',rows);table(out/'common997.csv',[r for r in rows if r['four_method_complete']])
    table(out/'comparison.csv',comparison);dump(out/'summary.json',s)
    bounds=[0,.01,.025,.05,.1,.2,.5,1,None];dist=[]
    for estimator,key in [('episode0','MC_E33'),('mean4_supplementary','mean4_E33')]:
        for i,(lo,hi) in enumerate(zip(bounds[:-1],bounds[1:])):
            values=[r[key] for r in rows];count=sum((v>=lo if i==0 else v>lo) and (hi is None or v<=hi) for v in values)
            dist.append(dict(estimator=estimator,lower=lo,upper=hi,count=count,fraction=count/1000))
    table(out/'distribution.csv',dist)
    groups=[]
    for kind in ('split','complexity','regime','complexity_regime'):
        labels=sorted({(r['complexity']+'/'+r['regime']) if kind=='complexity_regime' else r[kind] for r in rows})
        for label in labels:
            sub=[r for r in rows if ((r['complexity']+'/'+r['regime']) if kind=='complexity_regime' else r[kind])==label]
            groups.append(dict(group_type=kind,group=label,**stats([r['MC_E33'] for r in sub])))
    table(out/'groups.csv',groups)
    plots(out,rows,s);report(out,s,comparison)
    (out/'FIGURES.md').write_text('''# Подписи к графикам

1. mc33_ecdf: эмпирическое распределение E33 на1000; основной episode0 и дополнительное среднее4 (бюджеты4096 и16384 на узел).
2. four_method_ecdf: одинаковые33 узла/допуск0,05; R997, остальные1000; доли от доступных ответов.
3. four_method_accuracy_common997: четыре метода на одних997 задачах; MC только заранее выбранный episode0.
4. x0_vs_mc33: слева одна новая реализация (E33≥E0); справа разные seeds и M старого x0/нового MC33, без покоординатного равенства.
5. repeat_success_counts: число успешных повторов из4 на1000 уравнений; не сертификат вероятности успеха.

PNG предназначены для просмотра, PDF сохраняют векторные элементы для научного отчёта. Основной LaTeX не изменён.
''')
    dump(out/'input_sha256.json',reader.hashes)
    for name in ('diagnostics/finalize_mc33.py','diagnostics/verify_mc33_final.py','tests/test_mc33_final.py'):
        if (ROOT/name).exists():
            dest=out/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    dump(out/'environment.json',dict(python=sys.version,platform=sys.platform,
        versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','matplotlib')},
        new_solver_calls=0,new_fit_calls=0))
    changed=[name for name,h in protected.items() if not (ROOT/name).is_file() or sha(ROOT/name)!=h]
    dump(out/'integrity.json',dict(passed=not changed,checked=len(protected),changed=changed))
    require(not changed,'historical inputs changed')
    print(json.dumps(dict(validated=4000,equations=1000,primary=s['primary'],mean4=s['mean4_supplementary']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    build(p.parse_args().out)
