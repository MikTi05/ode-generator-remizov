"""Independent saved-only metrics/bound audit and the prospective review report."""
from __future__ import annotations
import argparse,decimal,json,math,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnostics.audit_probability_selector import read,write,sha,MODEL
from ml.probabilistic_selector import load_head


def report(out):
    import numpy as np
    nested=ROOT/'results/ml_nested_calibration_2026-10-26';val=ROOT/'results/ml_probability_validation_2026-10-26'
    r=read(out/'retrospective_results.json');pred=read(out/'retrospective_predictions.json')['rows']
    primary=r['policies']['0.95']['baseline_default'];base=r['policies']['constant_R256']
    assert all(p['primary_action']==p['choices']['0.95']['baseline_default']=='R256' for p in pred)
    assert primary['actions']=={'R256':8} and primary['abstentions']==0
    records=[read(p) for p in (val/'records').glob('*.json')];assert len(records)==96
    correct=0;cost=0.;bad_ids=[]
    for p in pred:
        own=[z for z in records if z['equation_id']==p['equation_id'] and z['profile']=='R256']
        assert len(own)==4
        zero=next(z for z in own if z['job']['episode']==0)
        error=float(np.max(abs(np.asarray(zero['result']['f'])-zero['y_exact'])))
        correct+=error<=.01;cost+=statistics.median(z['method_wall_seconds'] for z in own)
        if error>.01:bad_ids.append(p['equation_id'])
    assert correct==primary['correct']==base['correct']==7 and bad_ids==['AMV1_F0041_S02']
    assert math.isclose(cost,primary['method_cost_seconds'],abs_tol=1e-12) and cost==base['method_cost_seconds']
    infer=sum(p['feature_and_prediction_seconds'] for p in pred)
    assert math.isclose(primary['warm_cost_seconds'],cost+infer,abs_tol=1e-12)
    assert math.isclose(primary['seconds_per_correct'],(cost+infer)/7,abs_tol=1e-12)
    # Independent Decimal proof for the actual frozen MC model; no fit/prediction tuning.
    old=read(MODEL/'registry.json');head=load_head(old['heads']['MC1:probability'],MODEL/'models');model=head.models[0]
    values=model.get_leaf_values();counts=model.get_tree_leaf_counts();scale,bias=model.get_scale_and_bias();assert scale==1. and bias==0.
    with decimal.localcontext() as ctx:
        ctx.prec=80;position=0;upper=decimal.Decimal(0)
        for count in counts:
            upper+=max(decimal.Decimal.from_float(float(v)) for v in values[position:position+count]);position+=count
        threshold=decimal.Decimal(19).ln()
        assert upper+decimal.Decimal('0.000001')<threshold
        decimal_upper=str(upper);threshold_text=str(threshold)
    bound=read(out/'model_reachability.json')['heads']['MC1']
    assert float(upper)<=bound['guarded_logit_upper'] and bound['probability_upper']<.95
    scan=read(out/'never_fit_input_scan.json');assert scan['actions']=={'R256':250}
    assert all(row['split']=='train' and not row['labels_read'] for row in scan['rows'])
    proposal=read(out/'prospective_experiment.json');pilot=proposal['minimum_next_experiment'];formal=proposal['final_evidence_lower_bound']
    assert pilot['maximum_calls']==96 and pilot['families']==8 and pilot['proposed_cap_seconds']==5460
    assert not proposal['execution_authorized'] and not proposal['budget_approved']
    assert formal['maximum_calls']==708 and formal['not_universal_minimum']
    for name,h in read(out/'source_sha256.json').items():assert sha(ROOT/name)==h,name
    write(out/'independent_verification.json',dict(passed=True,validation_records=96,primary_decisions=8,
        primary_accuracy_and_cost_independently_recomputed=True,MC_Decimal_tree_bound_below_log19=True,
        MC_exact_leaf_box_logit_upper=decimal_upper,logit_q95=threshold_text,never_fit_input_scan=250,
        no_solver_calls=True,no_fit=True,no_policy_tuning=True,test_used=False,old_validation_not_independent=True))
    mc=[z for z in records if z['profile']=='MC1'];mc_good=sum(z['accuracy_met'] for z in mc)
    assert mc_good==4
    cases=read(val/'cases.json');positive=next(c for c in cases if c['profiles']['MC1']['accurate']==4)
    ceiling=read(out/'model_reachability.json')['heads']
    warm_delta=primary['warm_cost_seconds']-base['warm_cost_seconds']
    lines=['# Обобщение замороженного ML-селектора: повторная диагностика и план', '',
'Дата:2026-10-26. Новых solver/MC/benchmark1000 calls и ML-fit — **0**. '
'Test не использован. Проверены неизменность моделей/калибраторов/политики и '
'сохранённые validation96 и train-switch40. **Текущая версия пока не решает '
'исходную задачу полноценного выбора R128/R256/MC1.**', '',
'## Что именно заморожено', '',
'Основная версия — final raw R128/R256 головы и final sigmoid-калибраторы из '
'ml_nested_calibration_2026-10-26, прежние MC/cost головы, baseline_default q=.95. '
'R256 — unconditional fallback; q ограничивает переключения, не гарантирует '
'95% точности fallback. Production API по-прежнему research_only/abstain. '
'Новые train timing пяти переключений **не подставлялись в ML targets/cost heads**. '
'Никакой q, калибратор или версия не выбирались по восьми validation.', '',
'Прогнозы/решения записаны до присоединения старых ошибок и времён, но это '
'не делает8 семей независимыми: их исходы уже изучались и повлияли на историю '
'исследования. Ниже — **ретроспективная диагностика текущей версии**, не финальный test. '
'Финальный refit не подменяется outer-fold1 моделями, давшими5 train-переключений.', '',
'## Восемь уже рассчитанных validation-семейств', '',
'Текущий calibrated R256-default выбрал **R256 на8/8**. '
'**7/8 точных доставок (87.5%),1/8 нарушений E_grid≤.01 (12.5%), '
'0 отказов, coverage100%** — ровно как constant_R256.', '',
'| Задача | Решение | E_grid выбранного | Median4 method wall,с | Быстрейший наблюдённый точный профиль |',
'|---|---|---:|---:|---|']
    for row in primary['details']:
        c=next(z for z in cases if z['equation_id']==row['equation_id'])
        lines.append(f'| {row["equation_id"]} | {row["action"]} | {c["profiles"]["R256"]["primary_error"]:.9f} | {row["method_seconds"]:.3f} | {row["fastest_accurate_profile"] or "none"} |')
    lines += ['', '**F0041:** R256 E=.0172385414, неверный ответ сохраняется. На этой задаче '
'все3 испытанных профиля не достигают порога в primary episode0; корректный '
'выбор здесь — отсутствие пригодного профиля, а не уверенный R256. '
'**F0189:** прежний ложный отказ устранён самим fallback: R256 E=.0051778312 точен. '
'Калибровка не установила допустимость — она оценивает p_R256 слишком низко, '
'но default всё равно запускает R256.', '',
'Из7 задач, где наблюдался точный профиль, доставлены все7. Однако самый быстрый '
'точный профиль выбран лишь на **1/7** (F0189): на остальных6 точный R128 '
'быстрее R256. Таким образом, сохранение baseline-доставок достигнуто ценой '
'отсутствия экономии. Наблюдаемый oracle по episode0 не является истинным '
'oracle вероятности нового MC-вызова.', '',
'| Политика q=.95 | Точные/8 | Нарушения | Отказы | Method cost,с | Warm cost,с | Секунд/точный |',
'|---|---:|---:|---:|---:|---:|---:|']
    for name,z in [('constant_R256',base),('Текущая calibrated default',primary),('Calibrated risk с отказом',r['policies']['0.95']['risk_constraint'])]:
        lines.append(f'| {name} | {z["correct"]} | {z["violations"]} | {z["abstentions"]} | {z["method_cost_seconds"]:.3f} | {z["warm_cost_seconds"]:.3f} | {z["seconds_per_correct"]:.3f} |')
    lines += ['', f'Методическая экономия текущей политики — **0%**: '
f'{cost:.6f} с против{base["method_cost_seconds"]:.6f} с. '
f'Измеренные feature/head/calibration/primary-choice затраты добавили '
f'**{infer:.6f} с** на8 запросов; warm стоимость{primary["warm_cost_seconds"]:.6f} с '
f'против{base["warm_cost_seconds"]:.6f} с, то есть примерно '
f'**+{100*warm_delta/base["warm_cost_seconds"]:.3f}%**. '
'Общие model loading и загрузка массива входов вне warm-таймера; model loading '
f'{r["models_loaded_once_seconds"]:.3f} с показан отдельно. '
'Это offline сумма сохранённых полных median4 method times и нового измерения '
'ML-инференса; новых solver-dispatch и online-сравнения не было. Неточный '
'F0041 оплачивается в cost-per-correct, отказ не считается успехом.', '',
'Production guard даёт8 отказов/0 доставок; стоимость на точный ответ '
'не определена. API сейчас не исполняет эту исследовательскую политику. '
'В JSON для вспомогательных вариантов использован один общий диагностический '
'inference overhead текущего стека; это не отдельный benchmark legacy API/raw-версии. '
'Для них надёжно сравнивать прежде всего method-cost и решения.', '',
'## Исходная сетка q и альтернативные диагностические варианты', '',
'| q | Версия | R128/R256/MC | Точные | Отказы | Method cost,с |',
'|---:|---|---|---:|---:|---:|']
    for q in (.5,.8,.95):
        for name in ('baseline_default','risk_constraint','raw_baseline_default'):
            z=r['policies'][str(q)][name];ac=z['actions']
            lines.append(f'| {q} | {name} | {ac.get("R128",0)}/{ac.get("R256",0)}/{ac.get("MC1",0)} | {z["correct"]} | {z["abstentions"]} | {z["method_cost_seconds"]:.3f} |')
    lines += ['', 'Меньший q или отключение R128-калибратора могли бы дать другие действия '
'на этих8; **ни одна такая альтернатива не назначена новой политикой**. '
'Оптимизация по этой таблице была бы подгонкой по раскрытым validation. '
'Ранее train-only nested OOF уже показал ухудшение R128 от sigmoid: возможная '
'будущая revision должна определяться только train/nested данными и затем '
'проверяться на новом holdout.', '',
'## Почему нынешняя версия не выбирает MC', '',
f'Для binary MC CatBoost point-головы сумма максимальных leaf values всех '
f'120 деревьев даёт консервативную верхнюю границу **p_MC≤{bound["probability_upper"]:.10f}<.95**. '
'Каждый вход проходит ровно в один leaf каждого дерева, поэтому его logit '
'не может превысить эту сумму. Scale=1,bias=0; учтён численный запас. '
'Независимый Decimal расчёт сравнил верхнюю сумму с log(19)=logit(.95). '
'Суммы routed leaves воспроизвели реальные logits на250 входах. '
'Следовательно, **MC-действие глобально недостижимо для текущей point-политики '
'при q=.95**, даже на новых семействах.', '',
'Это ограничение **модели прогноза**, не верхняя граница реальной MC-точности '
'и не доказательство бесполезности численного метода. У Remizov/MC могли бы '
'существовать иные режимы, где MC выгоден; текущая модель не сможет их выбрать '
'при этом q независимо от затрат на новый solver-test.', '',
f'R128 после final-калибровки тоже сильно ограничен: глобальный leaf-box upper '
f'≈{ceiling["R128"]["calibrated_probability_upper"]:.6f}; вблизи q=.95 '
'остаётся очень узкий допустимый диапазон. На входах8 validation R128 не '
'прошёл q. На всех250 уравнениях из50 original-train семейств, не вошедших '
'ни в одну голову/scaler/calibrator, **выбран только R256**; maximum '
'calibrated p_R128≈.93593, p_MC≈.36653. Метки этих250 не читались и '
'параметры не менялись. Такой input-only scan проверяет наличие действий, '
'но не точность и не является test. Never-fit также не означает исторически '
'нераскрытое семейство: на исходном наборе были старые x0-эксперименты.', '',
f'В validation96 MC точен в{mc_good}/32 seed-вызовов, все4 на F0143/S04. '
f'Там median_MC={positive["profiles"]["MC1"]["timing"]["median"]:.3f} с, '
f'а точный R128={positive["profiles"]["R128"]["timing"]["median"]:.3f} с. '
'Наблюдаемого MC-winner среди8 нет.4/4 имеет индивидуальную двухстороннюю '
'CP95 нижнюю границу≈.3976, не95% надёжность следующего seed. '
'3 positive train-family/40 и7/160 seed-success недостаточны для high-p_MC '
'калибровки. Дополнительные seeds оценивают вероятность фиксированного MC1, '
'а не увеличивают M или точность следующего одиночного вызова.', '',
'## Минимальный перспективный опыт: что можно проверить честно', '',
'**Для нынешнего freeze нет финального solver-эксперимента, способного '
'подтвердить трёхметодный выбор:** MC action недостижим, а на имеющемся '
'never-fit input-frame нет и R128-переключений. Не следует расходовать '
'полный benchmark, чтобы ещё раз измерить фактически constant_R256.', '',
'Минимальный предлагаемый следующий **численный** этап — ограниченный '
'**MC1 utility/futility pilot на8 действительно новых независимых семействах**. '
'Это сбор недостающего научного свидетельства о самом профиле и диагностика '
'frozen прогнозов, а не обещание завершить ML-задачу.', '',
'1. Сначала отдельно подготовить новый input-only all_methods_v1 pool '
'(предложен master_seed2488000901,40 семей,1 образец, прежняя грамматика/диапазоны), '
'не запускать его сейчас. Использовать только train-роль нового pool, '
'исключить topology signatures fit/scaler/calibration и раскрытых семейств; '
'зафиксировать8 семей по input-only hash с балансом complexity/regime '
'до просмотра numerical outcomes. Если8 допустимых семейств не найдены, '
'остановить preflight, не менять выбор по solver-результатам. '
'Идентичность — dataset SHA+topology signature+канонический ID: Fxxxx/ID '
'могут повторяться между cohorts и сами по себе не доказывают независимость.',
'2. Freeze прогнозов, q=.95, primary default и baseline до вызовов. '
'Сопоставимые R128/R256/MC1 по4 возврата насемейство, **96 вызовов максимум**. '
'Remizov параметры неизменны, MC1 M4096,h1/128; grid33/eps.01, '
'единый аналитический G/T, один worker/поток,RSS512,guardian/thermal/deadline. '
'Seeds-предложение248800080100…103 с обязательной проверкой nonreuse; '
'MC point_address включает ID/точку, новый entropy namespace обязателен.',
'3. Primary episode0 заранее фиксирован. Сохранить все4 E_grid/seed outcomes, '
'median/ranges/IQR/MAD, временные отказы/неполноту. Не выбирать лучший seed, '
'не объединять4 вызова в метод сM16384, не менять профили/q/состав задач '
'после результата. Наблюдаемый MC accuracy+≥10% speedup относительно '
'самого быстрого точного R — лишь кандидат полезного режима, не p_new≥.95. '
'Считать deliveries/violations/abstain, feasible/none, cost-per-correct и '
'fastest-accurate выбор отдельно. При нуле новых MC-success/opportunities '
'остановиться на зарегистрированном cap, а не добирать wins.', '',
f'Бюджетная оценка: **{pilot["central_scenario_seconds"]:.1f} с '
f'({pilot["central_scenario_seconds"]/60:.2f} мин)**; '
f'предлагаемый отдельный cap **{pilot["proposed_cap_seconds"]} с (91 мин)**. '
'Формула4×8×(matched R128-median+R256-median+coverage20 MC1 mean), '
'cap=ceil(2×scenario+300). Использованы только сохранённые train timing; '
f'с median MC сценарий{pilot["median_MC_sensitivity_seconds"]:.1f} с, '
f'с maximum coverage20 MC{pilot["maximum_historical_MC_sensitivity_seconds"]:.1f} с. '
'Stress-cohort содержит MC до314.7 с на вызов: это ещё раз показывает, '
'что cap не является доказанной верхней границей. Полный ID/G/T/N/seed '
'preflight и уточнение case-specific бюджета обязательны до разрешения. '
'Сейчас budget_approved=false/execution_authorized=false. Остатки старых '
'бюджетов не переносятся.', '',
'При историческом ever-success3/40 в8 семьях ожидается всего0.6 positive-family; '
'по простому условному сценарию вероятность вообще не увидеть success≈53.6%. '
'Train40 был stress-enriched, так что это не достоверный прогноз новой '
'популяции. Равномерный pilot8 — недорогая проверка полезности/бесполезности '
'профиля, а не достаточный train/calibration dataset. Если цель — только '
'обобщение R-переключений, сначала нужна отдельная train-only revision '
'модели/калибровки с ненулевым input-only action coverage; текущая задача '
'такую revision не выполняла. Нельзя снижать q по нынешним8/250 входам, '
'чтобы принудительно получить MC или R128 действия.', '',
'## Что будет окончательной проверкой исходной задачи', '',
'После появления реальных MC-режимов возможна новая отдельная train-only '
'версия с проверенной калибровкой/coverage и новой заморозкой. Затем — '
'**новый независимый семейный holdout**, не восемь старых validation и '
'не тот pilot, который стал train/calibration. Заранее определить домен, '
'цену отказа/coverage, допустимый риск, сравнительную стоимость и критерий '
'полезности MC. MC не нужно выбирать искусственно: если достоверно показано '
'отсутствие его выгоды в заявленном домене, отрицательный вывод тоже научный '
'результат. Но нынешний structural cap этого не доказывает.', '',
'Универсального числа случаев для финальной проверки нет. Для примера: '
'если заявлена≥95% точность **среди выбранных ответов** с односторонней95% '
'нижней границей, при нуле ошибок нужно **59/59 независимых принятых '
'семейных исходов**; при одной ошибке —92/93. Это не59 seeds одного ОДУ, '
'не число всего запрошенных задач и не гарантия coverage/полезности MC. '
'Глобальные59 не обосновывают отдельный редкий MC-action: MC-specific '
'надёжность требует своей достаточной preregistered MC-selected поддержки '
'(при том же нулевом-error критерии также59 независимых MC-selected исходов). '
'Для нескольких одновременных утверждений нужны дополнительные поправки, '
'для>0 ошибок/отказов/других рисков — иной заранее рассчитанный объём.', '',
f'59 трёхпрофильных семейных блоков — **708 calls**, ориентировочный '
f'scenario{formal["central_scenario_seconds"]:.1f} с '
f'({formal["central_scenario_seconds"]/3600:.2f} ч), '
f'cap{formal["proposed_cap_seconds"]} с ({formal["proposed_cap_seconds"]/3600:.2f} ч). '
'Это **условная нижняя иллюстрация бюджета, не предложенный готовый финальный '
'запуск**: модель пока не проходит MC-action gate, новые IDs не подготовлены, '
'число принятых/MC-selected ответов и статистическая мощность неизвестны. '
'Дать обещание «8 или20 семей окончательно закроют трёхметодную задачу» '
'по текущим данным было бы необоснованно.', '',
'## Насколько близко к цели', '',
'Численные профили, воспроизводимые признаки/OOF, сохранение результатов и '
'локальные R128/R256 timing-переключения работают; train5 дали49.92% '
'локальной экономии без потерь baseline-доставок. Однако замороженный '
'final-селектор на разобранных новых для fit входах сводится кR256 и '
'не может выбирать MC при q=.95. **Научная задача ещё не завершена.** '
'Ближайшая проверяемая цель — оценить наличие настоящего MC1 accuracy/cost '
'режима на свежих семьях и отдельно исправить train-only action/calibration '
'деградацию, затем проверять новую версию независимо. Ни один из этих '
'следующих расчётов/fit здесь не запускался.', '',
'[Protocol](protocol.json), [прогнозы8](retrospective_predictions.json), '
'[метрики](retrospective_results.json), [глобальные границы](model_reachability.json), '
'[input-only250](never_fit_input_scan.json), [перспективный план](prospective_experiment.json), '
'[независимая проверка](independent_verification.json), [тесты](tests.json), '
'[итоговая сохранность](final_verification.json), [SHA](artifact_sha256.json).', '']
    with (out/'REPORT.md').open('x') as f:f.write('\n'.join(lines))
    write(out/'conclusion.json',dict(original_scientific_task_complete=False,current_q95_policy_beats_R256=False,
        retrospective_correct=7,requests=8,violations=1,abstentions=0,method_saving_percent=0.,
        MC_q95_choice_structurally_impossible=True,MC_probability_model_upper=bound['probability_upper'],
        true_MC_reliability_bound=False,new_numerical_calls=0,ML_fits=0,test_used=False,
        pilot96_is_final_test=False,pilot_proposed_unapproved_cap_seconds=5460,
        current_freeze_final_three_method_experiment_possible=False))
    print(dict(passed=True,primary_correct='7/8',primary_method_saving=0,MC_model_upper=bound['probability_upper'],pilot_cap=5460))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);report(p.parse_args().out.resolve())
