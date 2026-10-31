"""Independently verify the inert TRAIN-switch40 plan and write its review report."""
from __future__ import annotations
import argparse,csv,hashlib,math,statistics,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from diagnostics.audit_probability_selector import read,write,sha
from diagnostics.plan_switch5_r_timing import IDS,GRID,NESTED,CONTRACT


def verify(out):
    from all_methods_dataset_loader import load_all_methods_equations
    plan=read(out/'plan.json');b=read(out/'budget.json');approval=read(out/'approval_request.json')
    assert not plan['execution_authorized'] and not b['approved'] and not approval['approved']
    assert approval['plan_sha256']==sha(out/'plan.json') and approval['maximum_calls']==40
    assert plan['blockers']==['no_approved_budget']
    assert [c['equation_id'] for c in plan['cases']]==list(IDS)
    jobs=plan['jobs'];assert len(jobs)==len({j['key'] for j in jobs})==40
    assert Counter(j['profile'] for j in jobs)=={'R128':20,'R256':20}
    assert all(j['split']=='train' and j['seed'] is None and not j['execution_authorized'] for j in jobs)
    assert set(j['equation_id'] for j in jobs)==set(IDS)
    for name,h in plan['protected_sha256'].items():assert sha(ROOT/name)==h,name
    registry=read(NESTED/'registry.json');assert not registry['deployment_ready'] and registry['no_production_dispatch']
    contract=read(CONTRACT/'plan.json')
    assert all(contract['profiles'][p]==read(ROOT/'results/ml_probability_selector_2026-10-26/registry.json')['profile_context'][p] for p in ('R128','R256'))
    data=ROOT/plan['data_dir'];sub=out/'training_inputs'
    for name in ('all_methods_equations_v1.csv',):
        def selected(path):
            with path.open(newline='') as f:return {r['equation_id']:r for r in csv.DictReader(f) if r['equation_id'] in IDS}
        assert selected(data/name)==selected(sub/name)
    def metadata(path):
        import json
        return {r['equation_id']:r for line in path.open() if (r:=json.loads(line))['equation_id'] in IDS}
    assert metadata(data/'all_methods_equations_v1_metadata.jsonl')==metadata(sub/'all_methods_equations_v1_metadata.jsonl')
    eqs={e.equation_id:e for e in load_all_methods_equations(sub/'all_methods_equations_v1.csv',sub/'all_methods_equations_v1_metadata.jsonl')}
    errors=[]
    for c in plan['cases']:
        pid=c['equation_id'];e=eqs[pid]
        assert c['split']==e.metadata['split']=='train' and c['expression_hash']==e.metadata['expression_hash']
        exact=np.asarray(e.y_exact(np.asarray(GRID)))
        for p,ref in c['references'].items():
            raw=read(ROOT/ref['source']);f=np.asarray(raw['result']['f']);error=float(np.max(abs(f-exact)))
            assert error==ref['E_grid'] and ref['accurate']==(error<=.01)
            assert sha(ROOT/ref['source'])==ref['source_sha256']
            assert hashlib.sha256(f.astype('<f8').tobytes()).hexdigest()==ref['f_float64_sha256']
            assert hashlib.sha256(exact.astype('<f8').tobytes()).hexdigest()==ref['exact_float64_sha256']
            assert raw['job']['config']==dict(contract['profiles'][p]['config'],T=c['T'])
            errors.append((pid,p,error))
    assert len(errors)==10 and sum(E<=.01 for _,_,E in errors)==8
    assert {pid for pid,p,E in errors if E>.01}=={'AMV1_F0172_S03'}
    for c in plan['cases']:
        pid=c['equation_id'];own=[j for j in jobs if j['equation_id']==pid]
        assert Counter((j['profile'],j['episode']) for j in own)==Counter((p,i) for p in ('R128','R256') for i in range(4))
        first=[jobs[i] for i in range(0,40,2) if jobs[i]['equation_id']==pid]
        assert Counter(j['profile'] for j in first)=={'R128':2,'R256':2}
    assert sum(j['numerical_work'] for j in jobs)==3932160
    deltas=[]
    for row in read(out/'overhead_sources.json'):
        values=[]
        for name in row['source_paths']:
            raw=read(ROOT/name);values.append(raw['method_wall_seconds']-raw['result']['runtime_seconds'])
        deltas.append(statistics.median(values))
    overhead=statistics.median(deltas)
    scenario=4*sum(c['historical_R128_native_seconds']+overhead+c['historical_R256_full_seconds'] for c in plan['cases'])
    assert math.isclose(scenario,b['central_scenario_seconds'],abs_tol=1e-9)
    assert b['proposed_cap_seconds']==math.ceil(2*scenario+300)==1766
    assert math.isclose(sum(j['scenario_seconds'] for j in jobs),scenario,abs_tol=1e-9)
    assert plan['resources_proposed']['workers']==plan['resources_proposed']['threads']==1
    assert plan['resources_proposed']['rss_mib']==512
    assert not (out/'records').exists() and not (out/'events').exists() and not (out/'active.json').exists()
    write(out/'independent_verification.json',dict(passed=True,identities=5,canonical_errors_recomputed=10,
        original_selected_input_rows_equal=True,profiles_verified=2,proposed_calls=40,paired_order_balanced=True,
        budget_independently_recomputed=True,approved=False,new_numerical_calls=0,MC_calls=0,ML_fits=0,
        validation_test_outcomes_used=False,expected_accurate_cases=4,known_inaccurate='AMV1_F0172_S03',
        model_policy_production_guard_unchanged=True))
    sum128=sum(c['historical_R128_native_seconds']+overhead for c in plan['cases'])
    sum256=sum(c['historical_R256_full_seconds'] for c in plan['cases'])
    lines=['# Префлайт пяти train-переключений:40 измерений R128/R256', '',
'Дата:2026-10-26. **Расчёты не запущены. Бюджет не утверждён.** '
'Подготовлены5 фиксированных train-ID,40 jobs (20 R128+20 R256), '
'проверены10 сохранённых ответов и утверждённые профили. MC/ML/benchmark1000 calls=0. '
'Validation/test outcomes не использованы.', '',
'## Какой вопрос проверяем', '',
'Получить сопоставимое полное wall-clock для пяти ранее зафиксированных **OOF** переключений '
'R256→R128 при q=.95: F0016/S01,F0043/S00,F0119/S00,F0153/S01,F0172/S03. '
'Это решения outer-fold1 из nested-калибровки, а не новые решения финального refit или '
'production API. Модели, калибраторы и политика не переоцениваются/не изменяются. '
'Все5 выбраны ранее и не заменяются после просмотра точности/времени.', '',
'**Важное ограничение:** F0172 уже неточен у обоих профилей. Ожидается4/5 точных '
'доставок для обоих, а не5/5. Случай сохраняется как известный неточный контроль; '
'его скорость не считается победой точного метода. Ожидаемое сохранение точности '
'означает отсутствие потерь относительно R256 на четырёх baseline-correct задачах. '
'Четыре детерминированных повтора — timing-повторы, не новые независимые accuracy trials.', '',
'| Задача | E_grid R128 | E_grid R256 | Оба≤.01 | Старое native R128,с | Старое full R256,с |',
'|---|---:|---:|---|---:|---:|']
    for c in plan['cases']:
        a=c['references']['R128'];z=c['references']['R256']
        lines.append(f'| {c["equation_id"]} | {a["E_grid"]:.9f} | {z["E_grid"]:.9f} | {a["accurate"] and z["accurate"]} | {c["historical_R128_native_seconds"]:.6f} | {c["historical_R256_full_seconds"]:.6f} |')
    lines += ['', 'Все E_grid независимо пересчитаны по каноническим exact только этих5 train-задач; '
'проверены expression/family/split hashes, AST, глобальный G, единый auto-T, configs, '
'reference-файлы и их прежние замороженные SHA-256. Только5 канонических train-строк '
'скопированы в training_inputs/ и загружены безопасным loader; исходный набор не изменён '
'и не перегенерирован. Это отбор входов для preflight, не новая версия датасета.', '',
'## Профили, измерение, порядок', '',
'Remizov ChernovSolver.solve, R128/R256: num_t513, t_grading=quadratic, '
'interp=cubic, boundary=zero, dx=.0125, padding12, tail_eps=1e−4, n_jobs1, '
'рабочая сетка2081 узел [-13,13];33 target-узла [-1,1]. Полные configs и '
'аналитические G/T сохранены для каждой задачи. n_steps32768 — сохранённое '
'поле config; streaming не запускается.', '',
'20 пар:4 episodes×5 задач. В каждом episode case-порядок циклически сдвигается; '
'для каждогоID порядок соседних R128/R256 чередуется с заранее фиксированной '
'hash-фазой: ровно2 R128-first и2 R256-first. Seeds отсутствуют: Remizov '
'детерминирован; RNG/MC/JIT/численных warmup-вызовов нет.', '',
'perf_counter_ns включает certify_G+auto_T+config+новый constructor+solve '
'(warm_end_to_end_method_wall_v1). Общие импорты/загрузка отдельно; '
'exact/scoring/reference comparison/serialization после таймера. Все4 результата '
'сохраняются, первый запуск не отбрасывается. ML-inference в этот эксперимент '
'не входит: это измерение стоимости заранее зафиксированных OOF действий, не online dispatch.', '',
'## Бюджет для согласования', '',
f'**Предлагаемый cap:1766 с =29 мин26 с, максимум40 вызовов.** '
f'Сценарная сумма method wall **{scenario:.3f} с =12 мин12.7 с** '
f'(R128 {b["R128_scenario_seconds"]:.3f} с; R256 {b["R256_scenario_seconds"]:.3f} с). '
'Cap=ceil(2×сценарий+300); это запас на setup/IO/guardian/нагрев и исторические различия, '
'не доверительный интервал и не доказанная верхняя оценка.', '',
'R256: по одному ранее измеренному full method wall именно каждого из5 ID, '
'умноженному на4. R128: native runtime именно каждогоID плюс типичный внешний '
f'factory/G/T-overhead **{overhead:.9f} с**, измеренный в4 других train-семействах '
'с настоящим full wall, затем×4. Нативные времена используются **только для сметы**; '
'полное наблюдаемое время R128 на этих5 остаётся отсутствующим до будущего эксперимента.', '',
f'Альтернатива, использующая только full-wall R128-median других семейств, '
f'даёт {b["full_wall_only_R128_generic_alternative_seconds"]:.3f} с. '
f'Прежняя грубая смета {b["previous_generic_scenario_seconds"]:.3f} с/cap1738 с '
'из NEXT_STEPS сохранена исторической; теперь она уточнена конкретными ID. '
'Увеличение cap на28 с не является переносом/расширением старого разрешённого бюджета: '
'новый бюджет пока вообще не утверждён.', '',
f'Ожидаемый локальный method-only выигрыш одной доставки на каждом из5 ID: '
f'сумма R128≈{sum128:.3f} с против R256≈{sum256:.3f} с, '
f'около{100*(1-sum128/sum256):.1f}% ({sum256-sum128:.3f} с). '
'Это прогноз по разным историческим cohorts, а не установленная экономия. '
'Затрата самой исследовательской серии40 calls отличается от стоимости5 доставок.', '',
'Ресурсы:1 compute-worker/поток; RSS≤512 MiB; guardian с внешней lease300 с; '
'RSS=.25 с, thermal каждые10 с, stop при NSProcessInfo state≥2 или недоступной '
'телеметрии; перед стартом актуальные power/thermal/locks/disk проверки. '
'Это thermal pressure, не измерение температуры в°C. Startup cap90 с; '
'watchdog R128120 с/R256164 с, каждый ограничен оставшимся глобальным временем. '
'Работа40 calls:3,932,160 применений S и3,932,200 spline builds. '
'Предварительный снимок окружения/питания сохранён; он не заменяет проверку перед запуском.', '',
'## Сохранение и безопасное возобновление', '',
'При будущем разрешении — новый results/ml_switch5_r_timing_2026-10-26/, '
'новый execution plan с отдельным authorization и ссылкой на этот preflight. '
'Текущие plan/approval_request с approved=false остаются историческими. '
'Нужен проверенный R-only40 runner с exact scope; существующие8-call F0047 '
'и96-call validation drivers напрямую запускать нельзя.', '',
'Исходные epoch+monotonic deadlines сохраняются и не сбрасываются при resume. '
'Каждый отправленный request занимает слот из40, даже при прерывании. '
'Сохранённые completed/failure ответы не перезаписываются; interrupted/unknown '
'не перезапускаются автоматически. Полностью завершённый resume —0 новых calls. '
'Повторы/смена параметров/добавление задач ради улучшения статистики запрещены. '
'G/T/config/hash/identity mismatch, неожиданное отличие детерминированного '
'результата от reference, unexpected solver failure, превышение ресурса/температуры/lease '
'останавливают расчёт с сохранением raw/partial. Заранее известный F0172 accuracy=false '
'таким несоответствием не является.', '',
'## Заранее фиксированная оценка после40 calls', '',
'1. Для каждого возврата E_grid≤.01, конечность и точное совпадение канонических '
'reference-векторов/configs; доля точных задач и потерянных baseline-доставок '
'считается по5 задачам, не по40 повторным вызовам.',
'2. Время каждого профиля: все4 raw, median/min/max/IQR/MAD, paired episode '
'разности. Основная локальная экономия=1−Σmedian_R128/Σmedian_R256, '
'а абсолютная=Σ(median_R256−median_R128). Показать отдельно все5 (включая '
'неточный контроль), четыре точные задачи, стоимость на точную доставку по method wall '
'с включением времени неточного ответа. ML-inference и прочие outside-method затраты '
'показывать отдельно; без их измерения online-стоимость не заявлять.',
'3. Поддержанное timing-преимущество на точной задаче: '
'median_R128≤.9×median_R256, max_R128<min_R256, MAD/median≤.2 у обоих; '
'иначе ничья/нестабильность. Это критерий отчёта, не изменение замороженной ML-политики. '
'Четыре замера дают описательный разброс, не сильное статистическое доказательство.',
'4. F0172 получает neither_profile_accurate независимо от скорости. '
'Не объявлять4/5 точность95%-надёжностью. Не переносить локальную экономию '
'на весь benchmark/новые семейства и не выдавать её за подтверждение MC.',
'5. Эти5 замеров закроют5 пропущенных R128-time targets, однако4 других '
'R256 costs в общей paired40-таблице останутся неизвестны. Поэтому можно '
'оценить локальную добавочную экономию пяти переключений; полная измеренная '
'процентная экономия/стоимость всей40-task политики ещё не определена. '
'Все35 непереключённых действий остаются фиксированными R256; новые timing-данные '
'не используются для refit, выбора q или пересоставления пяти ID.', '',
'## Проверки и артефакты', '',
'[plan.json](plan.json), [manifest.csv](manifest.csv), [budget.json](budget.json), '
'[approval_request.json](approval_request.json), [preflight.json](preflight.json), '
'[independent_verification.json](independent_verification.json). '
'Первый технический запуск планировщика остановился из-за ошибочного ожидания '
'artifact_sha256.json у старого joint_grid100: там используются замороженные '
'per-file SHA из первоначального ML-пакета. Предположение исправлено, все10 '
'референсов проверены; незавершённая попытка сохранена отдельно, численных вызовов0. '
'Это не несовпадение численных данных.', '',
'Production API остаётся research_only/abstain; ML-модели/политика/методы и '
'прежние результаты сохранены. Unit-проверки: [tests.json](tests.json); '
'окончательная сохранность: [final_verification.json](final_verification.json), '
'[artifact_sha256.json](artifact_sha256.json).', '']
    with (out/'REPORT.md').open('x') as f:f.write('\n'.join(lines))
    print(dict(passed=True,reference_errors=10,calls=0,proposed_cap_seconds=1766,approved=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);verify(p.parse_args().out.resolve())
