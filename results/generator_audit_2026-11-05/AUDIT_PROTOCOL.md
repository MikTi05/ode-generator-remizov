# Протокол воспроизведения аудита

Рабочий каталог — корень репозитория. Все прежние файлы читаются; результаты записываются в новый каталог аудита. Численные решатели не вызываются. До проверок создан before_sha256.json и побайтовая копия PROJECT_STATUS.md.

## Тесты

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_all_methods_expressions.py tests/test_all_methods_dataset_v1.py tests/test_all_methods_loader.py
```

## Одноразовая проверочная программа

Программа исполнялась из /private/tmp; исходники проекта не менялись. Для повторения извлеките блок в новый временный файл, выберите новый OUT и создайте каталог заранее. Время в facts.json включает проверки и повторение семей/отказов; оно не является чистым временем генерации.

```python
from pathlib import Path
import sys, json, csv, hashlib, collections, time, platform
from dataclasses import asdict
import numpy as np
import sympy as sp
import mpmath as mp
ROOT=Path.cwd();sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
import src.all_methods_dataset_v1 as gen
from src.all_methods_expressions import SafeExpressionNode,evaluate_solution
from test_all_methods_expressions import _sympy_expression
from all_methods_dataset_loader import load_all_methods_equations
from all_methods_bounds import certify_all_methods_forcing
from remizov_solver import _problem_conditions_valid
OUT=ROOT/'results/generator_audit_2026-11-05'
def save(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
dirs=['data/generated','data/all_methods_v1_1000_2026-10-18','results/ml_development8_2026-10-26/input_pool','results/ml_selector_final_2026-10-27/final_experiment/input_pool']
facts={'environment':{'python':sys.version,'numpy':np.__version__,'sympy':sp.__version__,'platform':platform.platform()},'datasets':{},'solver_calls':0}
for d in dirs:
 start=time.monotonic();p=ROOT/d;report=json.loads((p/gen.OUTPUT_FILENAMES['report']).read_text());meta=[json.loads(l) for l in (p/gen.OUTPUT_FILENAMES['metadata']).read_text().splitlines()]
 result=gen.validate_dataset_artifacts(p)
 save('validation_'+str(len(meta))+'.json',result)
 cfg=gen.AllMethodsConfig(**report['configuration']);families=gen._generate_families(cfg)
 rejected=[]
 for m in meta:
  for attempt in range(m['accepted_attempt']-1):
   try: gen._candidate_record(cfg,families[int(m['family_id'][1:])],m['sample_index'],attempt)
   except gen.RecordValidationError as e: rejected.append({'id':m['equation_id'],'attempt':attempt+1,'reason':e.reason,'detail':e.detail,'level':m['complexity_level'],'regime':m['regime'],'attempt_seed':gen._stable_seed(cfg.seed,21,int(m['family_id'][1:]),m['sample_index'],attempt)})
 ops=collections.Counter(); actual={}
 for m in meta:
  for role,t in m['tree_metrics']['per_tree'].items():
   ops.update(t['operator_counts'])
   if m['regime']=='constant' and role!='y_shape': continue
   a=actual.setdefault(m['complexity_level'],{'nodes':[],'depth':[]})
   a['nodes'].append(t['nodes']);a['depth'].append(t['depth'])
 facts['datasets'][d]={'report':report,'validation_seconds':time.monotonic()-start,'all_valid':result['valid'],'replayed_rejections':rejected,'operation_counts':dict(ops),'actual_nonconstant_tree_limits':{k:{s:[min(v),max(v)] for s,v in a.items()} for k,a in actual.items()},'attempts':sum(m['accepted_attempt'] for m in meta),'rejections_by_cell':dict(collections.Counter(x['level']+'|'+x['regime'] for x in rejected))}
 print('validated',d,'n=',len(meta),'seconds',facts['datasets'][d]['validation_seconds'],flush=True)
 save('facts.json',facts)
prod=ROOT/dirs[1];meta=[json.loads(l) for l in (prod/gen.OUTPUT_FILENAMES['metadata']).read_text().splitlines()];byid={m['equation_id']:m for m in meta}
with (prod/gen.OUTPUT_FILENAMES['csv']).open(newline='') as f:rows={r['equation_id']:r for r in csv.DictReader(f)}
ids=['AMV1_F0000_S00','AMV1_F0010_S00','AMV1_F0060_S00','AMV1_F0141_S00','AMV1_F0180_S04']
eqs={e.equation_id:e for e in load_all_methods_equations(prod/gen.OUTPUT_FILENAMES['csv'],prod/gen.OUTPUT_FILENAMES['metadata'])}
examples=[];mp.mp.dps=60;x=sp.Symbol('x',real=True);cfg=gen.AllMethodsConfig();families=gen._generate_families(cfg)
for eid in ids:
 m=byid[eid];e=eqs[eid];p=m['serialized_ast']['construction'];nodes={k:SafeExpressionNode.from_dict(m['serialized_ast'][k]) for k in ['a_raw','b_raw','c_raw','y_shape']}
 def sf(f):return sp.Rational(float(f))
 # Independent symbolic chain/product differentiation with exact binary64 constants.
 def sy(n):
  op=n.op;pa={k:sf(v) for k,v in n.params.items()};ch=[sy(t) for t in n.children]
  if op=='constant':return pa['value']
  if op in ['sin_affine','cos_affine','tanh_affine']:return {'sin_affine':sp.sin,'cos_affine':sp.cos,'tanh_affine':sp.tanh}[op](pa['omega']*x+pa['phi'])
  if op=='gaussian':return sp.exp(-pa['alpha']*(x-pa['mu'])**2)
  if op=='lorentz':return 1/(1+pa['alpha']*(x-pa['mu'])**2)
  if op=='saturating':return (x-pa['mu'])/sp.sqrt(1+(x-pa['mu'])**2)
  if op=='lincomb':return pa['alpha']*ch[0]+pa['beta']*ch[1]
  if op=='mul':return ch[0]*ch[1]
  if op=='exp_neg_square':return sp.exp(-ch[0]**2)
  if op=='rational_saturate':return ch[0]/(1+ch[0]**2)
  return {'sin':sp.sin,'cos':sp.cos,'tanh':sp.tanh}[op](ch[0])
 z=(x-sf(p['bump_mu']))/sf(p['bump_radius']);Y=sf(p['solution_scale'])*sp.exp(1-1/(1-z*z))*sy(nodes['y_shape'])
 oracle=[sp.lambdify(x,sp.diff(Y,x,k),'mpmath',cse=True) for k in range(3)]
 co=[sf(p['a_mid'])+sf(p['a_half'])*sp.tanh(sy(nodes['a_raw'])),sf(p['b_scale'])*sp.tanh(sy(nodes['b_raw'])),sf(p['c_scale'])*sp.tanh(sy(nodes['c_raw']))-sf(p['lambda'])]
 cof=[sp.lambdify(x,v,'mpmath',cse=True) for v in co]
 points=np.array([-1.,-.37,0.,.43,1.]);actual=gen._all_values(nodes,p,points);errs={k:0. for k in ['y','dy','d2y','g']}
 for i,point in enumerate(points):
  xx=mp.mpf(float(point));oy=[f(xx) for f in oracle];oc=[f(xx) for f in cof];og=oc[0]*oy[2]+oc[1]*oy[1]+oc[2]*oy[0]
  for key,val in zip(['y','dy','d2y','g'],oy+[og]):errs[key]=max(errs[key],float(abs(mp.mpf(float(actual[key][i]))-val)))
 rawmax=m['target_amplitude']/p['solution_scale'];candidate=gen._candidate_record(cfg,families[int(m['family_id'][1:])],m['sample_index'],m['accepted_attempt']-1)
 checks={'csv_exactly_replayed':candidate.csv_row==rows[eid],'metadata_except_hashes_exactly_replayed':candidate.metadata=={k:v for k,v in m.items() if k not in ['feature_hash','expression_hash']},'solver_conditions':_problem_conditions_valid(e.remizov_problem),'independent_points':points.tolist(),'independent_precision_decimal_digits':60,'independent_absolute_errors':errs}
 assert all(checks[k] for k in ['csv_exactly_replayed','metadata_except_hashes_exactly_replayed','solver_conditions']);assert max(errs.values())<1e-9
 outside=np.array([m['support_left']-1,m['support_left'],m['support_right'],m['support_right']+1]);v=gen._all_values(nodes,p,outside);checks['outside_max_abs']=max(float(np.max(np.abs(v[k]))) for k in ['y','dy','d2y','g'])
 examples.append({'id':eid,'metadata_jsonl_line':int(m['family_id'][1:])*5+m['sample_index']+1,'csv_line':int(m['family_id'][1:])*5+m['sample_index']+2,'raw_normalization_max_recovered':rawmax,'csv_row':rows[eid],'metadata':m,'checks':checks,'forcing_certificate':asdict(certify_all_methods_forcing(e.remizov_problem,m))})
 print('independent example',eid,errs,flush=True)
 save('examples.json',examples)
# Inventory actual CSV/metadata subsets; same ID is scoped to a dataset.
inventory=[]
for p in sorted(ROOT.rglob('all_methods_equations_v1_metadata.jsonl')):
 if '.venv' in p.parts or OUT in p.parents:continue
 ms=[json.loads(l) for l in p.read_text().splitlines()]
 inventory.append({'path':str(p.relative_to(ROOT)),'count':len(ms),'seeds':sorted(set(m['master_seed'] for m in ms)),'splits':dict(collections.Counter(m['split'] for m in ms)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
facts['inventory']=inventory;save('facts.json',facts)
print('DONE',flush=True)
```

## Дополнительная точная проверка параметров

Для каждой строки четырёх metadata-файлов построены Fraction(a_mid), Fraction(a_half), Fraction(lambda), Fraction(c_scale), то есть точные рациональные значения сохранённых binary64. Без допуска проверены a_mid-a_half>0 и lambda-c_scale>0. Максимальные расхождения с a_lower_cert/a_upper_cert/gap и минимальные строгие запасы добавлены в facts.json. Проверка не вычисляла функции на сетке и не вызывала solver.
