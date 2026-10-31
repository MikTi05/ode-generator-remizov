"""Independent saved-only audit. Never call solver, JIT, fit or inference."""
import csv,hashlib,io,json,math,struct,sys
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent.parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def audit():
 p=read(OUT/'plan.json');ph=sha(OUT/'plan.json');analysis=read(OUT/'analysis.json')
 for n,h in p['protected_sha256'].items():assert sha(ROOT/n)==h,n
 for n,h in read(OUT/'artifact_sha256.json').items():assert sha(OUT/n)==h,n
 release=ROOT/'results/ml_selector_final_2026-10-27'
 for root in (release,ROOT/'results/ml_selector_budget_review_2026-10-27'):
  for n,h in read(root/'artifact_sha256.json').items():assert sha(root/n)==h,n
 from all_methods_dataset_loader import load_all_methods_equations
 d=ROOT/p['data_dir'];eqs={e.equation_id:e for e in load_all_methods_equations(d/'all_methods_equations_v1.csv',d/'all_methods_equations_v1_metadata.jsonl')}
 jobs={j['key']:j for j in p['jobs']}
 requests=[read(f) for f in (OUT/'events').glob('*.request.json')]
 records={f.stem:read(f) for f in (OUT/'records').glob('*.json')}
 assert len(jobs)==140 and len(requests)<=140
 assert len({r['job']['key'] for r in requests})==len(requests)
 for r in requests:assert r['job']==jobs[r['job']['key']] and r['plan_sha256']==ph
 assert set(records)<={r['job']['key'] for r in requests}
 addresses=set();calls=[];outcomes={};pids=set()
 for key,r in records.items():
  j=jobs[key];assert r['job']==j and r['plan_sha256']==ph
  assert all(r[n]==j[n] for n in ('equation_id','profile','family_id','expression_hash','config','split'))
  assert r['seed']==j['config'].get('seed')
  row=dict(key=key,equation_id=j['equation_id'],profile=j['profile'],status=r['status'],seed=r['seed'],E_grid=None,accurate=None,method_wall_seconds=r.get('method_wall_seconds'),peak_rss_mib=r.get('peak_rss_mib'))
  if r.get('worker_pid'):pids.add(r['worker_pid'])
  if r['status']=='completed':
   z=r['result'];assert z['success'] and not z['failure_reason'] and r['solver_called']
   assert z['problem_id']==j['equation_id'] and z['T']==j['expected_T']
   x=np.array(z['x']);f=np.array(z['f']);assert np.array_equal(x,np.array([-1+i/16 for i in range(33)]))
   exact=eqs[j['equation_id']].y_exact(x)
   assert np.array_equal(exact,r['y_exact']) and f.shape==(33,) and np.isfinite(f).all()
   error=max(abs(float(a)-float(b)) for a,b in zip(f,exact));good=error<=.01
   assert error==r['E_grid'] and good==r['accuracy_met']
   elapsed=(r['timer_stop_ns']-r['timer_start_ns'])/1e9
   assert elapsed>0 and math.isclose(elapsed,r['method_wall_seconds'],rel_tol=1e-12,abs_tol=1e-12)
   assert r['peak_rss_mib']<=512
   if j['profile']=='MC1':
    N=math.ceil(z['T']*128);assert N==j['expected_steps']==z['n_steps']
    assert z['M']==4096 and z['dt_requested']==1/128 and 0<z['dt_actual']<=1/128
    assert z['seed_entropy']==248800100100 and z['backend']=='numba' and z['block_steps']==64
    assert z['n_path_steps']==z['n_normal_draws']==33*4096*N==j['numerical_work']
    assert z['completed_paths']==[4096]*33 and z['forcing_abs_upper_bound']==j['forcing_G'] and z['tail_tolerance_met']
    assert z['compilation_seconds']==z['numba_import_seconds']==0
    for x,keyvec in zip(z['x'],z['point_spawn_keys']):
     h=hashlib.sha256(b'mc_point_v1\0'+j['equation_id'].encode()+struct.pack('<d',0. if x==0 else x)).digest()
     assert keyvec==[*struct.unpack('<8I',h),0]
     address=(z['seed_entropy'],tuple(keyvec));assert address not in addresses;addresses.add(address)
   else:
    assert z['n']==j['config']['n'] and z['n_operator_applications']==512*z['n']==j['numerical_work']
    assert z['n_spline_builds']==j['numerical_work']+1 and z['work_grid_points']==2081 and z['time_nodes']==513
    assert z['work_domain']==[-13.,13.] and z['actual_dx']==.0125 and z['scheme']=='solve'
   row.update(E_grid=error,accurate=int(good))
  calls.append(row);outcomes[j['equation_id'],j['profile']]=row
 pred={r['equation_id']:r['prediction'] for r in read(OUT/'frozen_predictions.json')['rows']}
 assert Counter(r['selected_profile'] for r in pred.values())=={'R128':21,'R256':39}
 primary={};family=[]
 for name,profile in [('selector',None),('constant_R128','R128'),('constant_R256','R256')]:
  values=[];costs=[];violations=0
  for c in p['cases']:
   pid=c['equation_id'];selected=pred[pid]['selected_profile'] if name=='selector' else profile
   r=outcomes.get((pid,selected));overhead=pred[pid]['inference_seconds'] if name=='selector' else 0.
   values.append(r['accurate'] if r else None)
   costs.append(r['method_wall_seconds']+overhead if r and r['method_wall_seconds'] is not None and r['accurate'] is not None else None)
   if r and r['accurate']==0 and r['status']=='completed':violations+=1
  known=[v for v in values if v is not None];total=math.fsum(costs) if all(t is not None for t in costs) else None
  primary[name]=dict(confirmed_correct=sum(known),known_outcomes=len(known),finite_violations=violations,total_warm_cost=total)
  actual=analysis['primary'][name]
  assert actual['confirmed_correct']==sum(known) and actual['known_outcomes']==len(known) and actual['finite_violations']==violations
  if total is not None:assert math.isclose(total,actual['total_warm_cost'],rel_tol=1e-12,abs_tol=1e-9)
  else:assert actual['total_warm_cost'] is None
 for c in p['cases']:
  pid=c['equation_id'];r128=outcomes.get((pid,'R128'));r256=outcomes.get((pid,'R256'));mc=outcomes.get((pid,'MC1'))
  row=dict(equation_id=pid,identity=c['identity'],cell=c['complexity_level']+'/'+c['regime'],selected_profile=pred[pid]['selected_profile'],MC_measured=pid in p['MC_equation_ids'])
  for profile,r in [('R128',r128),('R256',r256),('MC1',mc)]:
   row.update({profile+'_accurate':r['accurate'] if r else None,profile+'_E_grid':r['E_grid'] if r else None,profile+'_seconds':r['method_wall_seconds'] if r else None})
  family.append(row)
 assert len(family)==60 and sum(r['MC_measured'] for r in family)==20
 assert analysis['MC_secondary']['not_measured_families']==40 and not analysis['MC_secondary']['conditional_MC_choice_quality_confirmed']
 for row in family:
  if not row['MC_measured']:assert row['MC1_accurate'] is None and row['MC1_seconds'] is None
 thermal=[json.loads(l) for l in (OUT/'thermal.jsonl').open()]
 assert thermal and all(r['thermal_state'] is not None and r['thermal_state']<2 for r in thermal)
 rss=[]
 for f in (OUT/'workers').glob('*.rss.jsonl'):
  rss.extend(json.loads(l)['peak_rss_mib'] for l in f.open())
 assert rss and max(rss)<=512
 active=read(OUT/'active.json');guards=[read(f) for f in Path(active['control_dir']).glob('*.guardian.json')]
 assert active['finished'] and guards and all(g['finished'] and g['cleanup_success'] for g in guards)
 budget=read(OUT/'budget.json');assert budget['plan_sha256']==ph and budget['approved_cap_seconds']==7464 and budget['maximum_physical_calls']==140
 assert math.isclose(budget['deadline_epoch']-budget['started_epoch'],7464,abs_tol=1e-6)
 assert math.isclose(budget['deadline_monotonic']-budget['started_monotonic'],7464,abs_tol=1e-6)
 end=max(g['finished_monotonic'] for g in guards)
 assert end<=budget['deadline_monotonic']
 renewals=[json.loads(l) for l in (OUT/'execution_checks/external_owner.jsonl').open()]
 good=[r for r in renewals if r.get('renewal',{} ) and r['renewal']['returncode']==0]
 stamps=[budget['started_monotonic']]+[r['observed_monotonic']-r['lease_age_seconds'] for r in good]+[end]
 stamps=sorted(t for t in stamps if budget['started_monotonic']<=t<=end)
 gaps=[b-a for a,b in zip(stamps,stamps[1:])]
 assert good and max(gaps)<300
 warmups=[read(f) for f in (OUT/'workers').glob('*.warmup.json')]
 assert warmups and all(w['extra_solver_calls']==0 and w['no_RNG_in_warmup'] and w['numba_threads']==1 for w in warmups)
 assert len(pids)==1,'one persistent numerical worker expected in uninterrupted series'
 result=dict(passed=True,plan_sha256=ph,requests=len(requests),records=len(records),complete_140=len(requests)==len(records)==140,
  statuses=dict(Counter(r['status'] for r in records.values())),profiles=dict(Counter(r['profile'] for r in records.values())),
  independent_primary=primary,unique_MC_point_addresses=len(addresses),MC_choices=0,MC_choice_quality_confirmed=False,
  numerical_worker_pids=sorted(pids),maximum_worker_rss_mib=max(rss),thermal_samples=len(thermal),maximum_thermal_state=max(r['thermal_state'] for r in thermal),
  guardian_cleanup_verified=True,external_CLI_renewals=len(good),maximum_renewal_gap_seconds=max(gaps),
  elapsed_to_guardian_cleanup_seconds=end-budget['started_monotonic'],budget_cap_seconds=7464,protected_sources_model_inputs_and_original_artifacts_unchanged=True,
  new_solver_calls_by_audit=0,new_model_training=0,production_blocked=True)
 for name,rows in [('call_results.csv',sorted(calls,key=lambda r:next(i for i,j in enumerate(p['jobs']) if j['key']==r['key']))),('family_results.csv',family)]:
  stream=io.StringIO(newline='');w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  content=stream.getvalue().encode()
  if (OUT/name).exists():assert (OUT/name).read_bytes()==content,'changed derived CSV: '+name
  else:(OUT/name).write_bytes(content)
 target=OUT/'final_independent_verification.json'
 if target.exists():assert read(target)==result,'changed independent audit result'
 else:target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 return result

if __name__=='__main__':print(json.dumps(audit(),ensure_ascii=False))
