"""Verify the research materials against saved evidence; never solve an ODE."""
import ast,csv,json,re
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ml_report_illustrations_2026-10-27'
read=lambda p:json.loads(p.read_text())

def test_no_solver_or_fit_calls_in_builder():
 tree=ast.parse((ROOT/'diagnostics/build_ml_report_illustrations.py').read_text())
 forbidden={'solve','solve_monte_carlo_optimized','solve_monte_carlo','fit','fit_model','compile_backend','predict'}
 for n in ast.walk(tree):
  if isinstance(n,ast.Call):assert getattr(n.func,'attr',getattr(n.func,'id','')) not in forbidden

def test_symbolic_values_and_explicit_derivatives_independently_checked():
 v=read(OUT/'internal/verification.json');assert v['passed'] and v['solver_calls']==v['new_training']==v['new_inference']==0
 assert len(v['examples'])==7
 assert all(r['passed'] and r['factored_jet_checked'] and r['analytic_comparison_points']==515 for r in v['examples'])
 assert max(max(r['maximum_absolute_differences'].values()) for r in v['examples'])<3e-10

def test_original_sources_untouched():
 before=read(OUT/'internal/source_hashes.before.json');after=read(OUT/'internal/source_hashes.after.json')
 assert before==after and len(before)>1500
 for n,h in before.items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n

def test_selected_scenarios_are_real_and_distinct_from_final_statistics():
 rows=list(csv.DictReader((OUT/'example_results.csv').open()))
 assert len(rows)==26 and len({r['example'] for r in rows})==7
 by={i:{p:[r for r in rows if int(r['example'])==i and r['method']==p] for p in ('Ремизов','Монте-Карло')} for i in range(1,8)}
 for i in (1,2):assert all(float(r['E_grid'])<=.01 for r in by[i]['Ремизов'])
 for i in (3,4):
  assert next(float(r['E_grid']) for r in by[i]['Ремизов'] if r['n']=='128')>.01
  assert next(float(r['E_grid']) for r in by[i]['Ремизов'] if r['n']=='256')<=.01
 for i in (5,6):assert all(float(r['E_grid'])>.01 for r in by[i]['Ремизов'])
 assert len(by[7]['Монте-Карло'])==4 and sum(float(r['E_grid'])<=.01 for r in by[7]['Монте-Карло'])==1
 a=read(ROOT/'results/ml_selector_final140_protocol_2026-10-27/analysis.json')
 assert len(a['cases'])==60 and a['MC_secondary']['confirmed_successes']==0
 assert a['MC_secondary']['not_measured_families']==40

def test_public_science_contains_no_internal_identifiers_or_file_names():
 paths=list((OUT/'equations').glob('*'))+[OUT/n for n in ('MATHEMATICAL_EXAMPLES.md','EXPLANATIONS.md','DISCUSSION_NOTES.md','NUMERICAL_CONTEXT.md','REPORT_STRUCTURE.md','FIGURE_RECOMMENDATIONS.md','TABLES.md','CAPTIONS.md')]
 for p in paths:
  s=p.read_text();assert not re.search(r'AMV1_|F\d{4}_S\d+|\.json|\.py\b|\.csv|MC1\b|R128\b|R256\b|API\b|serialized_ast',s),p

def test_figures_complete_and_numeric_curves_are_saved_nodes():
 f=read(OUT/'internal/figures_manifest.json');assert len(f)==34
 assert sum(r['atlas']=='examples_atlas.pdf' for r in f)==29
 assert sum(r['atlas']=='final_sample_atlas.pdf' for r in f)==5
 assert len(list((OUT/'figures').glob('*.png')))==34
 assert all((OUT/'figures'/(r['name']+'.png')).stat().st_size>15000 for r in f)
 trace=read(OUT/'internal/provenance.json')
 for ex in trace:
  for path,h in ex['records'].items():
   r=read(ROOT/path);assert len(r['result']['x'])==len(r['result']['f'])==33
   assert r['expression_hash']==ex['expression_hash']

def test_prior_reports_and_global_bounds_not_claimed_tight():
 prior=read(OUT/'internal/previous_reports.json');assert [r['pages'] for r in prior]==[10,6]
 for r in prior:
  assert len(r['sha256'])==64
  if Path(r['path']).exists():assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']
 for p in (OUT/'equations').glob('*.md'):
  s=p.read_text();assert r'\lambda-C\approx' in s and r'|c_R|_{\max}' not in s
  assert 'могут быть неточными' in s and 'задача Коши' in s
