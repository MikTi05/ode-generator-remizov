import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from decimal import Decimal
from fractions import Fraction as F
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent
EVIDENCE = REPORT_DIR / 'evidence'
TEX = (REPORT_DIR / 'report.tex').read_text()
OUT = REPORT_DIR / 'numeric_audit.json'
checks = []

def read_csv(rel):
    with (EVIDENCE / rel).open() as stream:
        return list(csv.DictReader(stream))

def read_json(rel):
    return json.loads((EVIDENCE / rel).read_text())

def check(name, passed, observed=None, expected=None, detail=None):
    row = {'check': name, 'passed': bool(passed)}
    for key, value in [('observed', observed), ('expected', expected), ('detail', detail)]:
        if value is not None:
            row[key] = value
    checks.append(row)

def near(a, b):
    return math.isclose(a, b, rel_tol=2e-14, abs_tol=2e-15)

def table(label):
    tables = re.findall(r'\\begin\{table\}.*?\\end\{table\}', TEX, re.S)
    matches = [s for s in tables if '\\label{' + label + '}' in s]
    check('unique table ' + label, len(matches) == 1, len(matches), 1)
    return matches[0] if matches else ''

def cells(block):
    body = block.split(r'\midrule')[-1].split(r'\bottomrule')[0]
    return [[c.strip() for c in row.split('&')] for row in body.split(r'\\') if '&' in row]

def number(cell):
    cell = cell.strip().strip('$')
    sci = re.fullmatch(r'([0-9.+-]+)\\cdot10\^\{(-?\d+)\}', cell)
    if sci:
        mant, exponent = sci.groups()
        exponent = int(exponent)
    else:
        mant, exponent = cell, 0
    value = Decimal(mant) * Decimal(10) ** exponent
    digits = len(mant.split('.')[1]) if '.' in mant else 0
    unit = Decimal(10) ** (exponent-digits)
    return value, unit

def rounded(name, printed, actual):
    value, unit = number(printed)
    error = abs(value - Decimal(str(actual)))
    check(name, error <= abs(unit)/2,
          {'printed': printed, 'actual': actual, 'absolute_rounding_error': str(error)},
          {'maximum_absolute_rounding_error': str(abs(unit)/2)})

def boolval(row, name):
    return row[name].lower() == 'true'

cost_plan = read_json('cost60/plan.json')
cost_sum = read_json('cost60/summary.json')
cost_sel = read_csv('cost60/selection.csv')
profiles = read_csv('cost60/profiles.csv')
winners = read_csv('cost60/winners.csv')
records = read_csv('cost60/records.csv')
configs = read_csv('cost60/configuration_summary.csv')
check('cost selection dimensions and strata',
      len(cost_sel) == 60 and len({r['family_id'] for r in cost_sel}) == 60
      and len({r['structure_signature'] for r in cost_sel}) == 60
      and Counter(Counter((r['complexity_level'], r['regime']) for r in cost_sel).values()) == {3:20},
      {'tasks':len(cost_sel), 'families':len({r['family_id'] for r in cost_sel}),
       'topologies':len({r['structure_signature'] for r in cost_sel}),
       'cell_sizes':dict(Counter(Counter((r['complexity_level'],r['regime']) for r in cost_sel).values()))})
check('cost original train selection and no canonical Test families',
      {r['split'] for r in cost_sel} == {'train'}
      and not {r['family_id'] for r in cost_sel}.intersection(cost_plan['selection_audit']['canonical_test_families']))
check('cost plan counts', (cost_plan['warmup_per_configuration'],cost_plan['measured_repeats'],
                            cost_plan['expected_warmup_calls'],cost_plan['expected_measured_calls']) == (1,5,480,2400))
check('cost actual call counts', len(profiles)==480 and len(records)==2880
      and Counter(r['phase'] for r in records)=={'warmup':480,'measured':2400}, dict(Counter(r['phase'] for r in records)))
by_id = defaultdict(list)
by_cfg = defaultdict(list)
profile_errors = []
for r in profiles:
    by_id[r['equation_id']].append(r)
    by_cfg[r['configuration']].append(r)
    times = json.loads(r['times'])
    median = statistics.median(times)
    mad = statistics.median(abs(t-median) for t in times)
    passed = (len(times)==5 and near(median,float(r['solver_median_seconds']))
              and near(min(times),float(r['solver_min_seconds']))
              and near(max(times),float(r['solver_max_seconds']))
              and near(mad,float(r['solver_MAD_seconds']))
              and near(mad/median,float(r['relative_MAD']))
              and boolval(r,'accurate') == (float(r['E33']) <= .01)
              and boolval(r,'timing_stable') == (mad/median <= .20))
    if not passed:
        profile_errors.append((r['equation_id'],r['configuration']))
check('cost 480 profile medians MAD and accuracy flags from five saved times', not profile_errors, profile_errors)
check('cost 8 profiles per selected ID',len(by_id)==60 and all(len(v)==8 for v in by_id.values()))
check('cost all deterministic usable profiles stable',all(boolval(r,'usable') and boolval(r,'repeat_identical') and boolval(r,'timing_stable') for r in profiles))

cost_table = cells(table('tab:costpilot-configurations'))
check('cost table rows',len(cost_table)==8,len(cost_table),8)
config_calc = {}
for cfg in configs:
    rows = by_cfg[cfg['configuration']]
    calc = {'accurate':sum(float(r['E33'])<=.01 for r in rows),
            'mean':statistics.mean(float(r['solver_median_seconds']) for r in rows),
            'total':sum(float(r['solver_median_seconds']) for r in rows),
            'maximum':max(float(r['E33']) for r in rows)}
    config_calc[cfg['configuration']] = calc
    check('cost independent configuration aggregation '+cfg['configuration'],
          calc['accurate']==int(cfg['accurate']) and near(calc['mean'],float(cfg['mean_solver_seconds']))
          and near(calc['total'],float(cfg['total_solver_seconds'])) and calc['maximum']==float(cfg['max_E33']))
    row = [r for r in cost_table if r[0]==('RK4' if cfg['method']=='rk4' else 'PI') and int(r[1])==int(cfg['N'])][0]
    check('cost printed coverage '+cfg['configuration'],row[2]==f"{calc['accurate']}/60",row[2],f"{calc['accurate']}/60")
    rounded('cost printed milliseconds '+cfg['configuration'],row[3],calc['mean']*1000)
    rounded('cost printed max E33 '+cfg['configuration'],row[4],calc['maximum'])

oracle_rows = []
rk_rows = []
conservative_rows = []
robust = Counter()
winner_fail = []
pi_feasible = 0
for equation_id,rows in by_id.items():
    good=[r for r in rows if boolval(r,'usable') and float(r['E33'])<=.01]
    oracle=min(good,key=lambda r:float(r['solver_median_seconds'])) if good else None
    rks=[r for r in good if r['method']=='rk4']
    pis=[r for r in good if r['method']=='product_integral']
    rk=min(rks,key=lambda r:float(r['solver_median_seconds'])) if rks else None
    pi=min(pis,key=lambda r:float(r['solver_median_seconds'])) if pis else None
    pi_feasible+=bool(pi)
    def faster(a,b):
        return bool(a and b and float(a['solver_median_seconds'])<=.9*float(b['solver_median_seconds'])
                    and float(a['solver_max_seconds'])<float(b['solver_min_seconds'])
                    and float(a['relative_MAD'])<=.20 and float(b['relative_MAD'])<=.20)
    pi_robust=faster(pi,rk) if rk else bool(pi)
    rk_robust=faster(rk,pi) if pi else bool(rk)
    robust['product_integral' if pi_robust else 'rk4' if rk_robust else 'practical_tie_or_unstable']+=1
    selected=pi if pi_robust else rk if rk else pi
    w=next(w for w in winners if w['equation_id']==equation_id)
    ok=(oracle and rk and w['oracle_configuration']==oracle['configuration']
        and w['rk4_only_configuration']==rk['configuration']
        and w['conservative_configuration']==selected['configuration']
        and w['rk4_minN_configuration']==min(rks,key=lambda r:int(r['N']))['configuration']
        and boolval(w,'PI_robust_win')==pi_robust and boolval(w,'RK4_robust_win')==rk_robust)
    if not ok: winner_fail.append(equation_id)
    oracle_rows.append(oracle);rk_rows.append(rk);conservative_rows.append(selected)
check('cost 60 winners and comparator choices independently recomputed',not winner_fail,winner_fail)
raw=Counter(r['method'] for r in oracle_rows)
configuration_wins=Counter(r['configuration'] for r in oracle_rows)
check('cost raw method winner counts',raw=={'rk4':60},dict(raw),{'rk4':60,'product_integral':0})
check('cost oracle N counts',configuration_wins=={'rk4_N32':59,'rk4_N64':1},dict(configuration_wins))
check('cost robust counts',robust=={'rk4':56,'practical_tie_or_unstable':4},dict(robust))
check('cost feasible PI tasks',pi_feasible==7,pi_feasible,7)
check('cost no unresolved or unusable outcomes',len(oracle_rows)==60 and all(oracle_rows) and all(rk_rows)
      and not cost_sum['no_configuration_meets_tolerance'] and not cost_sum['unknown_due_to_failure'])
best=min((k for k,v in config_calc.items() if v['accurate']==60),key=lambda k:config_calc[k]['total'])
check('cost best all-accurate constant',best=='rk4_N64',best,'rk4_N64')
constant_times=[float(next(r for r in by_id[e] if r['configuration']==best)['solver_median_seconds']) for e in by_id]
oracle_times=[float(r['solver_median_seconds']) for r in oracle_rows]
rk_times=[float(r['solver_median_seconds']) for r in rk_rows]
mean_relative=statistics.mean(1-a/b for a,b in zip(oracle_times,constant_times))
pooled=1-sum(oracle_times)/sum(constant_times)
absolute=statistics.mean(b-a for a,b in zip(oracle_times,constant_times))
check('cost oracle equals RK4-only and minimum feasible N',all(a['configuration']==b['configuration'] for a,b in zip(oracle_rows,rk_rows))
      and all(r['configuration']==min((p for p in by_id[r['equation_id']] if p['method']=='rk4' and float(p['E33'])<=.01),key=lambda p:int(p['N']))['configuration'] for r in oracle_rows))
rounded('cost mean relative percent in prose','49.1967',100*mean_relative)
rounded('cost pooled percent in prose','48.2849',100*pooled)
rounded('cost mean absolute milliseconds in prose','57.923901',absolute*1000)
check('cost gain numbers actually present in report',all(t in TEX for t in ['49.1967\\%','48.2849\\%','57.923901~мс']))
selection_table=cells(table('tab:costpilot-selection'))
check('cost selector comparison table rows',len(selection_table)==3,len(selection_table),3)
for i, times in enumerate([constant_times,oracle_times,rk_times]):
    rounded('cost selector mean row '+str(i),selection_table[i][1],statistics.mean(times)*1000)
    rounded('cost selector total row '+str(i),selection_table[i][2],sum(times))
gates={
 'all60_solved':len(oracle_rows)==60 and all(oracle_rows),
 'no_unknown_minimum':all(boolval(r,'usable') for r in profiles),
 'constant_meets_tolerance_on60':config_calc[best]['accurate']==60,
 'rk4_only_covers60':len(rk_rows)==60 and all(rk_rows),
 'heterogeneous_raw_winners':all(raw[m]>=12 for m in ['rk4','product_integral']),
 'robust_each_method':all(robust[m]>=12 for m in ['rk4','product_integral']),
 'stable_profile_fraction':sum(float(r['relative_MAD'])<=.20 for r in profiles)/len(profiles)>=.90,
 'practical_gain_vs_best_constant':mean_relative>=.10 and pooled>=.10,
 'practical_gain_vs_rk4_only_oracle':statistics.mean(1-a/b for a,b in zip(oracle_times,rk_times))>=.10 and 1-sum(oracle_times)/sum(rk_times)>=.10,
}
check('cost all NO-GO gates independently match summary',gates==cost_sum['GO_gates'],gates,cost_sum['GO_gates'])
check('cost independent recommendation NO-GO',not all(gates.values()) and cost_sum['recommendation']=='NO-GO')
special={int(r['N']):float(r['E33']) for r in by_id['AMV1_F0186_S00'] if r['method']=='rk4'}
rounded('cost exceptional ID N32 printed error','0.0136620881782',special[32])
rounded('cost exceptional ID N64 printed error','0.000210115440205',special[64])
prep=read_json('cost60/preparation.json')
for key,textval in [('import_seconds','0.208656'),('dataset_load_seconds','2.885793'),('selected_input_exact_preparation_and_IO_seconds','0.913157')]:
    rounded('cost preparation '+key,textval,prep[key])
env=read_json('cost60/environment.before.json')
check('cost instrumentation limitations faithfully stated',env['threadpools']==[] and 'Error:' in env['thermal']['stdout']
      and 'пустой список' in TEX and 'Тепловые и performance-сведения' in TEX)

boundary_plan=read_json('boundary_dev/plan.json')
boundary_sum=read_json('boundary_dev/summary.json')
boundary_rows=read_csv('boundary_dev/before_after.csv')
boundary_sel=read_csv('boundary_dev/selection.csv')
regressions=read_csv('boundary_dev/material_regressions.csv')
all_increases=read_csv('boundary_dev/regressions.csv')
groups={}
flag_fail=[]
for r in boundary_rows:
    before=float(r['E_before']);after=float(r['E_after']);delta=after-before;tol=1e-10*max(1,before)
    calc={'pass_before':before<=.05,'pass_after':after<=.05,'improved':delta < -tol,
          'equal_within_tolerance':abs(delta)<=tol,'increased_unrounded':delta>0,
          'material_regression':delta>tol,'pass_lost':before<=.05<after,'pass_gained':after<=.05<before,
          'catastrophe_before':before>1,'catastrophe_after':after>1,'new_catastrophe':before<=1<after}
    if not (delta==float(r['delta_error']) and tol==float(r['comparison_tolerance'])
            and all(boolval(r,key)==value for key,value in calc.items())):
        flag_fail.append(r['equation_id'])
check('boundary all 52 delta thresholds accuracy and regression flags',not flag_fail,flag_fail)
for category in ['case','control','all']:
    rs=[r for r in boundary_rows if category=='all' or r['category']==category]
    calc={'n':len(rs),'before_pass':sum(boolval(r,'pass_before') for r in rs),
          'after_pass':sum(boolval(r,'pass_after') for r in rs),
          'improved':sum(boolval(r,'improved') for r in rs),
          'equal':sum(boolval(r,'equal_within_tolerance') for r in rs),
          'increased_unrounded':sum(float(r['E_after'])>float(r['E_before']) for r in rs),
          'material_regressions':sum(boolval(r,'material_regression') for r in rs),
          'pass_losses':sum(boolval(r,'pass_lost') for r in rs),
          'pass_gains':sum(boolval(r,'pass_gained') for r in rs),
          'catastrophes_before':sum(float(r['E_before'])>1 for r in rs),
          'catastrophes_after':sum(float(r['E_after'])>1 for r in rs),
          'maximum_before':max(float(r['E_before']) for r in rs),
          'maximum_after':max(float(r['E_after']) for r in rs)}
    groups[category]=calc
    check('boundary independent aggregation '+category,calc==boundary_sum['groups'][category],calc,boundary_sum['groups'][category])
group_table=cells(table('tab:boundarydev-groups'))
check('boundary group table row count',len(group_table)==3,len(group_table),3)
for printed,category in zip(group_table,['case','control','all']):
    group=groups[category]
    actual=[str(group['n']),f"{group['before_pass']}/{group['after_pass']}",str(group['improved']),str(group['equal']),
            str(group['material_regressions']),f"{group['catastrophes_before']}/{group['catastrophes_after']}"]
    check('boundary group printed counts '+category,printed[1:]==actual,printed[1:],actual)
case_families={r['family_id'] for r in boundary_sel if r['category']=='case'}
control_families={r['family_id'] for r in boundary_sel if r['category']=='control'}
check('boundary selection dimensions and disjoint families',len(boundary_sel)==52 and len(case_families)==14
      and len(control_families)==30 and not case_families.intersection(control_families)
      and all(r['split'] in ['train','validation'] for r in boundary_sel))
check('boundary control coverage 20 strata',len({(r['complexity_level'],r['regime']) for r in boundary_sel if r['category']=='control'})==20)
check('boundary new and reused run counts',sum(boolval(r,'zero_reused') for r in boundary_rows)==2
      and len(boundary_rows)-sum(boolval(r,'zero_reused') for r in boundary_rows)==50
      and boundary_sum['new_taper_calls']==0)
check('boundary only parameter change',
      {k:(v,boundary_plan['zero_config'][k]) for k,v in boundary_plan['baseline_config'].items() if v!=boundary_plan['zero_config'][k]}=={'boundary':('taper','zero')})
check('boundary fixed grid and time counts',boundary_plan['baseline_config']['n']==128
      and boundary_plan['baseline_config']['num_t']==513 and boundary_plan['baseline_config']['dx']==.0125
      and boundary_plan['baseline_config']['padding']==12 and 26/.0125+1==2081)
catastrophes={r['equation_id']:r for r in boundary_rows if float(r['E_before'])>1}
extreme_table=cells(table('tab:boundarydev-extremes'))
check('boundary former catastrophic IDs',set(catastrophes)=={'AMV1_F0141_S00','AMV1_F0184_S01','AMV1_F0186_S02'})
for printed in extreme_table:
    code=re.search(r'F\d{4}/S\d{2}',printed[0]).group().replace('/','_')
    r=catastrophes['AMV1_'+code]
    rounded('boundary catastrophic before '+code,printed[1],float(r['E_before']))
    rounded('boundary catastrophic after '+code,printed[2],float(r['E_after']))
    check('boundary catastrophic pass after '+code,printed[3]==('да' if float(r['E_after'])<=.05 else 'нет'))
reg_table=cells(table('tab:boundarydev-regressions'))
check('boundary exactly three above-threshold regressions',len(reg_table)==len(regressions)==3)
for printed in reg_table:
    code=re.search(r'F\d{4}/S\d{2}',printed[0]).group().replace('/','_')
    r=next(r for r in regressions if r['equation_id']=='AMV1_'+code)
    rounded('boundary exact delta '+code,printed[1],float(r['delta_error']))
    rounded('boundary relative delta divided by before '+code,printed[2],float(r['delta_error'])/float(r['E_before']))
check('boundary all 19 positive deltas published',len(all_increases)==19 and groups['all']['increased_unrounded']==19)
check('boundary 21 of 22 cases remain outside tolerance',groups['case']['n']-groups['case']['after_pass']==21)
rounded('boundary prose global maximum before','19704.5019109',groups['all']['maximum_before'])
rounded('boundary prose global maximum after','0.596851548154',groups['all']['maximum_after'])
bgates={'all_zero_results_usable':all(boolval(r,'usable_zero') for r in boundary_rows),
        'no_material_regressions':groups['all']['material_regressions']==0,
        'no_pass_losses':groups['all']['pass_losses']==0,
        'no_new_catastrophes':not any(float(r['E_before'])<=1<float(r['E_after']) for r in boundary_rows),
        'former_catastrophes_removed':all(float(r['E_after'])<=1 for r in catastrophes.values()),
        'pass_count_increases':groups['all']['after_pass']>groups['all']['before_pass'],
        'at_least_one_material_improvement':groups['all']['improved']>0}
check('boundary all NO-GO gates independently match summary',bgates==boundary_sum['GO_gates'],bgates,boundary_sum['GO_gates'])
check('boundary independent recommendation and sole failure',not all(bgates.values())
      and [k for k,v in bgates.items() if not v]==['no_material_regressions'] and boundary_sum['recommendation']=='NO-GO')
old=next(r for r in read_csv('old_boundary/comparison.csv') if r['equation_id']=='AMV1_F0180_S04')
rounded('boundary historical Test F0180 before','19134.405',float(old['cubic_taper_error']))
rounded('boundary historical Test F0180 after','0.0704519',float(old['cubic_zero_error']))
check('boundary historical Test excluded from 52',not any(r['equation_id']=='AMV1_F0180_S04' for r in boundary_rows))
p=lambda x:F(3,2)*x-F(1,2)*x*x
q=lambda x:-(x-1)*(x-2)*(x-3)/6
eval_shift=lambda fun:F(1,4)*fun(F(5,4))+F(1,4)*fun(F(3,4))+F(1,2)*fun(F(3,2))
check('boundary exact arithmetic nodal counterexamples without solver',
      eval_shift(p)==F(67,64) and eval_shift(q)==F(-1,64),[str(eval_shift(p)),str(eval_shift(q))],['67/64','-1/64'],
      'Fraction arithmetic applied to the already specified cubic polynomials; no solver or new numerical experiment.')
check('boundary counterexample interpretation and strict NO-GO caveat retained',
      'дискретной кубической реконструкции в узловой максимальной норме' in TEX
      and 'а не для точного оператора' in TEX and 'не означает доказанной практической' in TEX)

failed=[c for c in checks if not c['passed']]
result={'schema':'integrated_numeric_audit_v1','report':str(REPORT_DIR/'report.tex'),
        'report_sha256':hashlib.sha256((REPORT_DIR/'report.tex').read_bytes()).hexdigest(),
        'scope':'Read-only independent numerical audit of new cost60 and boundary tables/prose against copied CSV/JSON evidence. No solver/MC/ML calls.',
        'passed':not failed,'checks_total':len(checks),'checks_passed':len(checks)-len(failed),'checks_failed':len(failed),
        'new_solver_calls':0,'MC_calls':0,'ML_calls':0,
        'recomputed_cost':{'raw_method_winners':dict(raw),'configuration_winners':dict(configuration_wins),
                           'robust_method_winners':dict(robust),'pi_feasible_tasks':pi_feasible,'best_constant':best,
                           'mean_relative_saving':mean_relative,'pooled_saving':pooled,'mean_absolute_saving_seconds':absolute,
                           'GO_gates':gates},'recomputed_boundary':{'groups':groups,'GO_gates':bgates},
        'failures':failed,'checks':checks}
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'passed':result['passed'],'checks':len(checks),'failures':failed,'output':str(OUT)},ensure_ascii=False))
