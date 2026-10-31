"""Postprocess saved arrays/logs only. Never import or call numerical solvers."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BATCH=ROOT/'results/chernov_solve1000_2026-10-18'
DATA=ROOT/'data/all_methods_v1_1000_2026-10-18'


def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def distribution(values):
    if not values:return {'count':0}
    a=np.array(values,dtype=float)
    assert np.all(np.isfinite(a))
    return dict(count=len(a),min=float(a.min()),median=float(np.median(a)),mean=float(a.mean()),
                p90=float(np.percentile(a,90)),p95=float(np.percentile(a,95)),
                p99=float(np.percentile(a,99)),max=float(a.max()),sum=float(a.sum()))


def grouped(rows,key):
    groups=defaultdict(list)
    for r in rows:groups[r[key]].append(r)
    return {k:dict(total=len(g),outcomes=dict(Counter(r['status'] for r in g)),
                   error=distribution([r['error'] for r in g if r['error'] is not None]),
                   relative=distribution([r['relative'] for r in g if r['relative'] is not None]))
            for k,g in sorted(groups.items())}


def csv_write(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def analyze(out):
    out.mkdir(parents=True,exist_ok=False)
    plan=read(BATCH/'plan.json');identity=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    meta={r['equation_id']:r for r in map(json.loads,(DATA/'all_methods_equations_v1_metadata.jsonl').read_text().splitlines())}
    preflight={r['problem_id']:r for r in read(BATCH/'preflight.json')}
    raw={pid:read(BATCH/'solutions'/f'{pid}.json') for pid in plan['equation_ids']}
    assert len(raw)==len(meta)==len(list((BATCH/'solutions').glob('*.json')))==1000
    rows=[]
    for pid,r in raw.items():
        assert r['plan_sha256']==identity and r['config']==plan['config']
        assert r['implementation']=='chernov_solver.ChernovSolver.solve'
        assert digest(BATCH/'solutions'/f'{pid}.json')==digest(BATCH/'attempts'/r['attempt_id']/'result.json')
        m=meta[pid];z=r['result'];p=preflight[pid]
        T=z['T'] if z else p['T'];d=max(2*math.sqrt(m['a_upper_cert']*T/128),2*m['b_abs_cert']*T/128)
        row=dict(problem_id=pid,family=m['family_id'],complexity=m['complexity_level'],regime=m['regime'],
            cell=m['complexity_level']+'/'+m['regime'],split=m['split'],status=r['status'],
            error=r['max_abs_error'],relative=r['relative_error'],solver_seconds=r['solver_seconds'],
            T=T,lam=m['remizov_lambda'],C=m['remizov_c_abs_cert'],gap=m['remizov_gap_cert'],
            a_upper=m['a_upper_cert'],B=m['b_abs_cert'],tau_max=T/128,
            dmax=d,conservative_half_padding=128*d,
            old_initial_points_at_same_dx=math.ceil((2+256*d)/.0125)+1,
            outside=z['outside_query_values'] if z else None,boundary=z['boundary_state_max_abs'] if z else None,
            f_max=None,exact_max=None,error_x=None,log10_boundary=None,
            support_left=m['support_left'],support_right=m['support_right'],attempt_id=r['attempt_id'])
        if z:
            f=np.asarray(z['f']);y=np.asarray(r['y_exact']);x=np.asarray(z['x'])
            assert z['success'] and z['scheme']=='solve' and len(f)==len(y)==33
            assert z['x']==r['target_x']==plan['target_x'] and np.all(np.isfinite(f))
            e=float(np.max(np.abs(f-y)));relative=e/float(np.max(np.abs(y)))
            assert e==r['max_abs_error'] and relative==r['relative_error']
            assert r['status']==('accuracy_pass' if e<=.01 else 'accuracy_not_met')
            assert z['n_operator_applications']==65536 and z['n_spline_builds']==65537
            assert z['work_grid_points']==2081 and z['actual_dx']==.0125 and z['time_nodes']==513
            assert not z['total_error_certified'] and not z['refinement_controls_performed']
            row.update(f_max=float(np.max(np.abs(f))),exact_max=float(np.max(np.abs(y))),
                       error_x=float(x[np.argmax(np.abs(f-y))]),
                       log10_boundary=math.log10(z['boundary_state_max_abs']) if z['boundary_state_max_abs']>0 else None)
        else:assert r['status']=='worker_exception' and row['error'] is None
        rows.append(row)
    assert Counter(r['status'] for r in rows)==Counter(accuracy_pass=871,accuracy_not_met=126,worker_exception=3)
    finite=[r for r in rows if r['error'] is not None];failed=[r for r in finite if r['error']>.01]
    ranked=sorted(finite,key=lambda r:r['error'],reverse=True)
    features={}
    for status in ('accuracy_pass','accuracy_not_met'):
        g=[r for r in finite if r['status']==status]
        features[status]={k:distribution([r[k] for r in g if r[k] is not None])
                          for k in ('T','gap','boundary','log10_boundary','outside','conservative_half_padding')}
        features[status]['boundary_over_1']=sum(r['boundary']>1 for r in g)
        features[status]['boundary_over_1e20']=sum(r['boundary']>1e20 for r in g)
        features[status]['outside_nonzero']=sum(r['outside']>0 for r in g)
    exceptions=[]
    session=read(BATCH/'sessions/2026-10-18T080113558750+0000_596389.json')
    for r in rows:
        if r['status']!='worker_exception':continue
        record=raw[r['problem_id']];worker=next(w for w in session['worker_startups'] if w['pid']==record['worker_pid'])
        state_path=BATCH/worker['state_file'];rss_path=state_path.with_suffix('.rss.jsonl');log=state_path.with_suffix('.log')
        lines=rss_path.read_text().splitlines();samples=list(map(json.loads,lines));state=read(state_path)
        req=read(BATCH/'attempts'/record['attempt_id']/'request.json')
        history=sorted([q for q in raw.values() if q.get('worker_pid')==record['worker_pid'] and q['status']!='worker_exception'],key=lambda q:q['finished_utc'])
        previous=history[-1] if history else None
        later=[w for w in session['worker_startups'] if w['slot']==worker['slot'] and w['pid']!=worker['pid']]
        exceptions.append(dict(**r,worker_pid=record['worker_pid'],slot=worker['slot'],
            request_utc=req['requested_utc'],finished_utc=record['finished_utc'],
            finished_msk=datetime.fromisoformat(record['finished_utc']).astimezone(timezone(timedelta(hours=3))).isoformat(),
            task_wall_seconds=record['task_wall_seconds'],peak_rss_mib=record['process_peak_rss_mib'],
            traceback_present='traceback' in record,returncode_present='returncode' in record,
            state_file=str(state_path.relative_to(ROOT)),rss_file=str(rss_path.relative_to(ROOT)),
            log_file=str(log.relative_to(ROOT)),log_bytes=log.stat().st_size,
            last_state=state,rss_samples=len(lines),last_rss=samples[-1],last_rss_line=len(lines),
            previous_completed_id=previous['problem_id'] if previous else None,
            previous_completed_utc=previous['finished_utc'] if previous else None,
            replacement_workers=later))
    comparison=[]
    integration=ROOT/'results/chernov_integration_2026-10-18'
    summary=read(integration/'final/summary.json');by_spec=defaultdict(list)
    for run in summary['runs']:
        result=read(integration/'final'/run['result_file']);z=result['result'];spec=result['spec']
        assert z['success'] and result['max_abs_error']<=.01
        assert float(np.max(np.abs(np.array(z['f'])-np.array(result['y_exact']))))==result['max_abs_error']
        by_spec[(spec['problem_id'],spec['method'])].append((run,result))
    for (pid,method),values in sorted(by_spec.items()):
        assert len(values)==3
        z=values[0][1]['result'];cfg=values[0][1]['config']
        assert all(v[1]['result']['f']==z['f'] for v in values)
        comparison.append(dict(problem_id=pid,method=method,repetitions=3,
            median_seconds=statistics.median(v[1]['result']['runtime_seconds'] for v in values),
            min_seconds=min(v[1]['result']['runtime_seconds'] for v in values),
            max_seconds=max(v[1]['result']['runtime_seconds'] for v in values),
            error=values[0][1]['max_abs_error'],config=cfg,
            operators=z['n_operator_applications'],splines=z['n_spline_builds'],
            grid_peak=z.get('spatial_grid_points_peak',z.get('work_grid_points')),
            nt_final=z.get('final_nt',z.get('time_nodes')),
            quadrature_nodes_total=z.get('n_quadrature_node_evaluations',z.get('time_nodes')),
            quadrature_history=z.get('quadrature_history'),spatial_levels=z.get('spatial_dx_levels'),
            interpolated_values=z.get('n_interpolated_values'),
            source_result=str((integration/'final'/values[0][0]['result_file']).relative_to(ROOT))))
    bins=[('<=0.001',0,.001),('(0.001,0.01]',.001,.01),('(0.01,0.02]',.01,.02),
          ('(0.02,0.1]',.02,.1),('(0.1,1]',.1,1),('(1,10]',1,10),('>10',10,math.inf)]
    result=dict(created_date='2026-10-20',scope='saved artifacts only, no solver calls',
        total=1000,outcomes=dict(Counter(r['status'] for r in rows)),
        absolute=distribution([r['error'] for r in finite]),relative=distribution([r['relative'] for r in finite]),
        errors_without_top3=distribution([r['error'] for r in ranked[3:]]),
        top3_share_of_error_sum=sum(r['error'] for r in ranked[:3])/sum(r['error'] for r in finite),
        bins={label:sum(lo<r['error']<=hi for r in finite) for label,lo,hi in bins},
        by_complexity=grouped(rows,'complexity'),by_regime=grouped(rows,'regime'),by_cell=grouped(rows,'cell'),
        by_split=grouped(rows,'split'),by_family=grouped(rows,'family'),feature_summaries=features,
        top20=ranked[:20],exceptions=exceptions,comparison=comparison,
        old_geometric_grid_estimate=distribution([r['old_initial_points_at_same_dx'] for r in rows]),
        old_geometric_grid_over_100000=sum(r['old_initial_points_at_same_dx']>100000 for r in rows),
        conservative_half_padding=distribution([r['conservative_half_padding'] for r in rows]),
        no_global_domain_certificate_for_fixed_window=sum(r['conservative_half_padding']>12 for r in rows),
        profile=read(ROOT/'results/remizov_n4_followup_2026-10-12/F0000_profile_analysis.json'),
        cross_dataset_identity={pid:dict(small=next(q for q in read(integration/'final/summary.json')['runs'] if q['spec']['problem_id']==pid)['spec']['problem_id'],
            same_expression_hash=next(read(p)['metadata']['expression_hash'] for p in (integration/'final').glob('*.result.json') if read(p)['spec']['problem_id']==pid)==meta[pid]['expression_hash'])
            for pid in ('AMV1_F0005_S00','AMV1_F0006_S00','AMV1_F0007_S00')})
    with (out/'analysis.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    csv_write(out/'all_1000.csv',rows);csv_write(out/'accuracy_not_met_126.csv',sorted(failed,key=lambda r:-r['error']))
    csv_write(out/'outliers_top20.csv',ranked[:20])
    with (out/'exceptions_evidence.json').open('x') as f:json.dump(exceptions,f,ensure_ascii=False,indent=2);f.write('\n')
    with (out/'verification.json').open('x') as f:json.dump(dict(unique_records=1000,all_errors_recomputed=True,
        arrays_finite=997,canonical_raw_hash_matches=1000,comparison_repeats_checked=36,numerical_solver_calls=0,
        analyzer_sha256=digest(Path(__file__)),source_sha256={str(p.relative_to(ROOT)):digest(p) for p in
            [ROOT/'remizov_solver.py',ROOT/'chernov_solver.py',ROOT/'all_methods_dataset_loader.py',DATA/'all_methods_equations_v1.csv',DATA/'all_methods_equations_v1_metadata.jsonl']}),f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ('total','outcomes','bins','top3_share_of_error_sum')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True,type=Path)
    analyze(p.parse_args().output_dir.resolve())
