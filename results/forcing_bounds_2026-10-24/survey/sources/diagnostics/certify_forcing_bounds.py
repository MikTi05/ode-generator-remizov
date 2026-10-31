"""Deterministic 20-equation certificate/cost survey; never simulate MC paths."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
THREADS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')
for name in THREADS:
    os.environ[name] = '1'

import numpy as np
from all_methods_bounds import (DerivativeBounds, certify_all_methods_forcing,
    expression_bounds, forcing_bound_from_solution_bounds, solution_bounds)
from all_methods_dataset_loader import load_all_methods_equations
from mc_solver import MCConfig, _gap_lower, _tail_bound, auto_T
from src.all_methods_expressions import SafeExpressionNode


def write(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def costs(T, cfg, K):
    N = max(1,math.ceil(T/cfg.dt)) if T else 0
    if N and T/N > cfg.dt:
        N += 1
    work = K*cfg.M*N
    batch = min(cfg.M,cfg.chunk)
    return dict(points=K,time_steps=N,actual_dt=T/N if N else 0.,
        path_steps=work,normal_draws=work,peak_batch_paths=batch,
        coefficient_values=K*cfg.M*(4*N+2) if N else 0,
        coefficient_calls=K*math.ceil(cfg.M/cfg.chunk)*(4*N+2) if N else 0,
        within_work_budget=(N<=cfg.max_steps and work<=cfg.max_path_steps and batch<=cfg.max_batch_paths))


def analytic_examples(eps):
    args = dict(a_upper=1.,b_abs=0.,c_abs=0.,lambda_value=2.)
    gaussian = SafeExpressionNode('gaussian',params={'alpha':1.,'mu':0.})
    constant = DerivativeBounds(.5,0.,0.)
    rows = []
    for name,y,exact_G in [('constant_y_half',constant,1.),('gaussian_y',expression_bounds(gaussian),4.)]:
        bound = forcing_bound_from_solution_bounds(y,**args)
        T = auto_T(2.,0.,bound.abs_upper_bound,eps)
        rows.append(dict(name=name,solution_bounds=asdict(y),G=bound.abs_upper_bound,
                         exact_G=exact_G,T=T,tail_bound=_tail_bound(bound.abs_upper_bound,2.,T)))
    sine = SafeExpressionNode('sin_affine',params={'omega':1.,'phi':0.})
    eta = 2.**-40
    cancellation = SafeExpressionNode('lincomb',(sine,sine),{'alpha':1.,'beta':-(1-eta)})
    small = forcing_bound_from_solution_bounds(solution_bounds(expression_bounds(sine),1.,2.),**args).abs_upper_bound
    large = forcing_bound_from_solution_bounds(solution_bounds(expression_bounds(cancellation),1/eta,2.),**args).abs_upper_bound
    return dict(examples=rows,cancellation=dict(identity='2^40*(sin(x)-(1-2^-40)*sin(x))=sin(x)',
        G_reduced=small,G_unreduced=large,proven_conservatism_lower_bound=large/small,
        T_reduced=auto_T(2.,0.,small,eps),T_unreduced=auto_T(2.,0.,large,eps)))


def survey(out, data):
    out.mkdir(parents=True,exist_ok=False)
    profiles = [('point_default',MCConfig(),1), ('curve_default',MCConfig(),33),
                ('curve_moderate',MCConfig(M=2000,dt=.02,chunk=2000),33)]
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),tail_eps=1e-4,
        equation_ids=[f'AMV1_F{i:04d}_S00' for i in range(20)],
        profiles={name:dict(config=asdict(cfg),points=K) for name,cfg,K in profiles},
        sample_points=[4097,16385],data_dir=str(data),
        policy='certificates and budgets computed before diagnostic samples; no parameter tuning; no Monte Carlo paths',
        cost_model='K*M*N path steps and K*M*(4*N+2) coefficient values, excluding nested AST work; no wall-time extrapolation',
        scope='all existing small-dataset equations, two analytical forcing norms and one cancellation identity')
    write(out/'plan.json',plan)
    inputs={name:digest(data/name) for name in ['all_methods_equations_v1.csv','all_methods_equations_v1_metadata.jsonl']}
    equations=load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl')
    assert [e.equation_id for e in equations] == plan['equation_ids']
    rows=[]
    for e in equations:
        began=time.perf_counter()
        proof=certify_all_methods_forcing(e.remizov_problem,e.metadata)
        elapsed=time.perf_counter()-began
        p=e.remizov_problem;G=proof.forcing_bound.abs_upper_bound
        gap=_gap_lower(p.lambda_value,p.c_abs_upper_bound)
        T=auto_T(p.lambda_value,p.c_abs_upper_bound,G,plan['tail_eps'])
        tail=_tail_bound(G,gap,T)
        assert tail <= plan['tail_eps']
        rows.append(dict(problem_id=e.equation_id,complexity=e.metadata['complexity_level'],
            regime=e.metadata['regime'],certificate=asdict(proof),G=G,gap=gap,T=T,
            tail_error_bound=tail,certification_seconds=elapsed,
            costs={name:costs(T,cfg,K) for name,cfg,K in profiles}))
    write(out/'certificates.json',rows)  # freeze analytical output before any samples
    observations=[]
    for e,row in zip(equations,rows):
        left,right=row['certificate']['support']
        sample_max=[]
        for count in plan['sample_points']:
            x=np.linspace(left,right,count)
            values=e.remizov_problem.g_remizov(x)
            assert np.all(np.isfinite(values))
            sample_max.append(float(np.max(abs(values))))
        outside=np.array([left-1.,right+1.])
        assert np.all(e.remizov_problem.g_remizov(outside)==0.)
        maximum=sample_max[-1]
        assert maximum<=row['G']
        optimistic=auto_T(e.remizov_problem.lambda_value,e.remizov_problem.c_abs_upper_bound,maximum,plan['tail_eps'])
        observations.append(dict(problem_id=e.equation_id,sampled_maxima=sample_max,
            G_over_sampled_max=row['G']/maximum,
            optimistic_T_from_sample_NOT_CERTIFIED=optimistic,
            optimistic_curve_moderate_path_steps_NOT_CERTIFIED=costs(optimistic,profiles[2][1],33)['path_steps']))
    write(out/'observations.json',observations)
    write(out/'analytical_examples.json',analytic_examples(plan['tail_eps']))
    write(out/'environment.json',dict(python=sys.version,platform=platform.platform(),
        versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','sympy','pytest')},
        thread_environment={name:os.environ[name] for name in THREADS},inputs_sha256=inputs,
        source_sha256={name:digest(ROOT/name) for name in ['all_methods_bounds.py','mc_solver.py',
            'all_methods_dataset_loader.py','src/all_methods_expressions.py',str(Path(__file__).relative_to(ROOT))]},
        monte_carlo_trajectories=0,monte_carlo_path_steps=0))
    print(json.dumps(dict(certified=len(rows),G_range=[min(r['G'] for r in rows),max(r['G'] for r in rows)],
        T_range=[min(r['T'] for r in rows),max(r['T'] for r in rows)],
        budget_passes={name:sum(r['costs'][name]['within_work_budget'] for r in rows) for name,_,_ in profiles},
        monte_carlo_trajectories=0)))


def report(out):
    rows=json.loads((out/'certificates.json').read_text())
    observations=json.loads((out/'observations.json').read_text())
    examples=json.loads((out/'analytical_examples.json').read_text())
    plan=json.loads((out/'plan.json').read_text())
    lines=['# Аналитические границы forcing — 2026-10-24','',
        'Все 20 существующих задач сертифицированы по AST. Сертификаты записаны до сеточных наблюдений. Траектории MC не моделировались.',
        'Доказательство: [ALL_METHODS_FORCING_BOUNDS.md](../../../docs/ALL_METHODS_FORCING_BOUNDS.md).',
        'План, полные компоненты и provenance: [plan.json](plan.json), [certificates.json](certificates.json), [environment.json](environment.json).','',
        'tail_eps=1e-4, delta — нижнее округление lambda-C из существующего RemizovProblem. C и lambda не перенастраивались.',
        'В таблице профиль curve_moderate: K=33, M=2000, dt=.02, max_path_steps=100000000. Стоимость — число обновлений траекторий, не время.','',
        '| ID | G | delta | T | N | K*M*N | Бюджет | max на сетке | G/max сетки |',
        '|---|---:|---:|---:|---:|---:|---|---:|---:|']
    for row,obs in zip(rows,observations):
        c=row['costs']['curve_moderate']
        lines.append(f"| {row['problem_id']} | {row['G']:.9g} | {row['gap']:.6g} | {row['T']:.6f} | {c['time_steps']} | {c['path_steps']} | {'да' if c['within_work_budget'] else 'превышен'} | {obs['sampled_maxima'][-1]:.7g} | {obs['G_over_sampled_max']:.5g} |")
    lines+=['','Все границы хвоста <=1e-4. Это не сертификат временной ошибки или полной точности MC.',
        'Сеточные максимумы получены на 4097 и 16385 точках полного носителя плюс проверены внешние нули. Они не заменяют глобальные границы.',
        'G/max_сетки — диагностическое отношение, не доказанный коэффициент консервативности. Оптимистические T от sampled maximum не используются в MC.','',
        '## Аналитические примеры','', '| Задача | G выведенный | Точная sup-норма forcing | T |','|---|---:|---:|---:|']
    for a in examples['examples']:
        lines.append(f"| {a['name']} | {a['G']:.10g} | {a['exact_G']:.10g} | {a['T']:.8f} |")
    stress=examples['cancellation']
    lines+=['','## Практичность и консервативность','']
    for name in plan['profiles']:
        cfg=plan['profiles'][name]
        passes=sum(row['costs'][name]['within_work_budget'] for row in rows)
        lines.append(f"- {name}: K={cfg['points']}, M={cfg['config']['M']}, dt={cfg['config']['dt']}: в бюджете {passes}/20. Полные N и стоимость каждой задачи находятся в certificates.json.")
    exceeded=[r['problem_id'] for r in rows if not r['costs']['curve_moderate']['within_work_budget']]
    even_optimistic=[o['problem_id'] for o in observations if o['optimistic_curve_moderate_path_steps_NOT_CERTIFIED']>100000000]
    lines+=['', 'Превышают умеренный профиль: '+', '.join(exceeded)+'.',
        'Даже оптимистический сеточный ориентир превышает этот бюджет: '+', '.join(even_optimistic)+'. Для этих задач уменьшение только запаса G не выглядит достаточным при фиксированных delta, M, dt, K.',
        'Это оценка работы, не измерение времени и не заявление о достаточной статистике при M=2000. Стоимость вложенной AST не приравнивается к nfev.',
        f"Доказанный стресс-пример сокращения: {stress['identity']}. Для одинакового решения G_reduced={stress['G_reduced']:.9g}, G_unreduced={stress['G_unreduced']:.9g}; отношение >={stress['proven_conservatism_lower_bound']:.9g}, так как истинная норма <=G_reduced.",
        f"Горизонты в нём: {stress['T_reduced']:.6f} и {stress['T_unreduced']:.6f}. Правила треугольника не отслеживают сокращения; искусственного ограничения G или уменьшения T нет.",
        'Большой T на малом датасете зависит также от малого delta; большой G не означает автоматически самый дорогой расчёт.', '',
        '## Проверки и ограничения','',
        'Регрессия и сохранность: [pytest.txt](../pytest.txt), [verification.json](../verification.json).',
        'Тесты используют независимые SymPy-производные всех 14 операций, символическую формулу bump, точные Fraction и 200-значный эталон горизонта/хвоста.',
        'Численная схема EM/трапеции не менялась. Исправлено только направленное вычисление T и хвоста: прежний один nextafter оказался недостаточен на F0002.',
        'Небольшой интеграционный тест с реальными MC-путями выполняется только на одной задаче; обзор 20 задач содержит ровно ноль траекторий.',
        'Изолированные h/T/M, вероятность достижения допуска, сравнительный benchmark и ML остаются отдельными задачами.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(out/'REPORT.md'),tasks=len(rows))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--input-dir',type=Path,default=ROOT/'data/generated')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    if args.report_only:
        report(args.output_dir.resolve())
    else:
        survey(args.output_dir.resolve(),args.input_dir.resolve())
