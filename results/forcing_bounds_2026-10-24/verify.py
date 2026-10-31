"""Read-only independent rational/high-precision and preservation checks."""
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


before = read(OUT/'before_sha256.json')
allowed = {'README.md','PROJECT_STATUS.md','docs/MONTE_CARLO.md','mc_solver.py',
           'examples/solve_mc_equation.py','diagnostics/compare_chernov.py',
           'tests/test_all_methods_mc_pipeline.py'}
changed = [p for p,h in before.items() if not (ROOT/p).is_file() or digest(ROOT/p)!=h]
assert not set(changed)-allowed, changed
backups = {p:digest(OUT/'before'/p)==before[p] for p in changed}
assert all(backups.values())
manifest = read(ROOT/'MIGRATION_MANIFEST.json')
core = {p:digest(ROOT/p)==manifest['source_sha256'][p] for p in manifest['unchanged_mathematical_core']}
assert all(core.values())
survey = OUT/'survey'
environment = read(survey/'environment.json')
sources = {p:digest(survey/'sources'/p)==h for p,h in environment['source_sha256'].items()}
assert all(sources.values())
data = ROOT/'data/generated'
inputs = {p:digest(data/p)==h for p,h in environment['inputs_sha256'].items()}
assert all(inputs.values())
metadata = {m['equation_id']:m for m in map(json.loads,(data/'all_methods_equations_v1_metadata.jsonl').read_text().splitlines())}
rows = read(survey/'certificates.json')
plan = read(survey/'plan.json')
assert [r['problem_id'] for r in rows]==plan['equation_ids']
checked=[]
for row in rows:
    proof=row['certificate'];m=metadata[row['problem_id']];p=m['serialized_ast']['construction']
    payload=json.dumps(m['serialized_ast'],sort_keys=True,separators=(',',':'),allow_nan=False)
    assert hashlib.sha256(payload.encode()).hexdigest()==proof['serialized_ast_sha256']
    H=proof['shape_bounds'];Y=proof['solution_bounds'];P=proof['bump_bounds']
    s,R=Fraction(abs(p['solution_scale'])),Fraction(p['bump_radius'])
    P1,P2=Fraction(3)/R,Fraction(1809,64)/(R*R)
    assert Fraction(P['d1'])>=P1 and Fraction(P['d2'])>=P2
    assert Fraction(Y['value'])>=s*Fraction(H['value'])
    assert Fraction(Y['d1'])>=s*(P1*Fraction(H['value'])+Fraction(H['d1']))
    assert Fraction(Y['d2'])>=s*(P2*Fraction(H['value'])+2*P1*Fraction(H['d1'])+Fraction(H['d2']))
    C=proof['coefficient_bounds'];terms=proof['term_bounds']
    analytical_terms=[Fraction(C['a_upper'])*Fraction(Y['d2']),
                      Fraction(C['b_abs'])*Fraction(Y['d1']),
                      (Fraction(abs(p['lambda']))+Fraction(C['c_R_abs']))*Fraction(Y['value'])]
    for name,value in zip(('diffusion','drift','potential'),analytical_terms):
        assert Fraction(terms[name])>=value
    assert Fraction(row['G'])>=sum(map(Fraction,terms.values()))
    assert row['G']==proof['forcing_bound']['abs_upper_bound']
    assert Fraction(row['gap'])<=Fraction(m['remizov_lambda'])-Fraction(m['remizov_c_abs_cert'])
    with localcontext() as ctx:
        ctx.prec=200
        lam,cb,G,T,eps=map(Decimal.from_float,(m['remizov_lambda'],m['remizov_c_abs_cert'],row['G'],row['T'],plan['tail_eps']))
        gap=lam-cb
        required=max(Decimal(0),(G/(gap*eps)).ln()/gap)
        reference_tail=G/gap*(-gap*T).exp()
        assert Decimal.from_float(row['T'])>=required
        assert Decimal.from_float(row['tail_error_bound'])>=reference_tail
        assert Decimal.from_float(row['tail_error_bound'])<=eps
    for name,cost in row['costs'].items():
        cfg=plan['profiles'][name]
        N=cost['time_steps']
        assert cost['path_steps']==cfg['points']*cfg['config']['M']*N
        assert cost['normal_draws']==cost['path_steps']
    checked.append(row['problem_id'])
verification=dict(old_files=len(before),unchanged_old_files=len(before)-len(changed),
    permitted_changes=sorted(changed),unexpected_changes=[],backups=backups,
    mathematical_core=core,survey_source_snapshots=sources,dataset_inputs=inputs,
    independently_checked_equations=checked,
    independent_checks='exact Fraction product-rule and forcing inequalities; 200-digit horizon and tail oracle',
    survey_monte_carlo_trajectories=environment['monte_carlo_trajectories'],
    tests=(OUT/'pytest.txt').read_text().splitlines()[-1])
(OUT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(unchanged=verification['unchanged_old_files'],changed=sorted(changed),
    checked=len(checked),tests=verification['tests']),ensure_ascii=False))
