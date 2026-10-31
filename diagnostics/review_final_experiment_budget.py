"""Read-only design arithmetic. No solver, inference, training or simulation."""
from __future__ import annotations
import hashlib,json,math,shutil,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RELEASE=ROOT/'results/ml_selector_final_2026-10-27'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')
def clock(seconds):
    seconds=round(seconds);return f'{seconds//3600}:{seconds%3600//60:02d}:{seconds%60:02d}'


def review(out):
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(ROOT/'PROJECT_STATUS.md',out/'PROJECT_STATUS.before.md')
    original=read(RELEASE/'artifact_sha256.json')
    for n,h in original.items():assert sha(RELEASE/n)==h,n
    exp=RELEASE/'final_experiment';plan=read(exp/'plan.json');pred=read(exp/'frozen_predictions.json');budget=read(exp/'budget_proposal.json')
    assert len(plan['cases'])==60 and len(plan['jobs'])==720 and not list((exp/'records').iterdir()) and not list((exp/'events').iterdir())
    choices={r['equation_id']:r['prediction']['selected_profile'] for r in pred['rows']};cases=plan['cases']
    assert set(choices.values())=={'R128','R256'} and sum(p=='R128' for p in choices.values())==21
    stages=[[c for i,c in enumerate(cases) if i%3==rank] for rank in range(3)]
    assert all(len(g)==20 and len({(c['complexity_level'],c['regime']) for c in g})==20 for g in stages)
    def proposal(name,selected,MC_ids):
        ids={c['equation_id'] for c in selected};refs=[j['key'] for j in plan['jobs'] if j['equation_id'] in ids and j['episode']==0 and (j['profile']!='MC1' or j['equation_id'] in MC_ids)]
        scenario=sum(c['R128_scenario_seconds']+c['R256_scenario_seconds']+(c['MC_scenario_seconds'] if c['equation_id'] in MC_ids else 0) for c in selected)
        cap=math.ceil(2*scenario+300)
        return dict(name=name,families=len(selected),R128_calls=len(selected),R256_calls=len(selected),MC1_calls=len(MC_ids),maximum_calls=len(refs),
            scenario_seconds=scenario,proposed_cap_seconds=cap,scenario_hms=clock(scenario),cap_hms=clock(cap),
            referenced_primary_job_keys=refs,equation_ids=[c['equation_id'] for c in selected],
            prediction_choices={p:sum(choices[c['equation_id']]==p for c in selected) for p in ('R128','R256','MC1')},
            execution_authorized=False,proposal_only=True,requires_separate_versioned_protocol_and_executor=True,
            single_call_realized_warm_cost_replaces_median4=True,model_policy_numerical_parameters_and_source_data_unchanged=True)
    a=proposal('A_fixed20_all3_once',stages[0],{c['equation_id'] for c in stages[0]})
    b=proposal('B_staged20_40_60_all3_once',cases,{c['equation_id'] for c in cases})
    c=proposal('C_fixed60_R_once_MC20_once',cases,{c['equation_id'] for c in stages[0]})
    b['stages']=[]
    cumulative=[]
    for stage in stages:
        cumulative+=stage
        z=proposal('cumulative',cumulative,{r['equation_id'] for r in cumulative})
        b['stages'].append({k:z[k] for k in ('families','maximum_calls','scenario_seconds','proposed_cap_seconds','scenario_hms','cap_hms','prediction_choices')})
    b['stopping']=dict(one_final_study_not_retraining_pilots=True,positive_early_claim=False,
        practical_futility_example='if loss of R256-correct deliveries exceeds3 at a20/40-family look, cannot finish within an observed5percent individual-loss limit over maximum60; stop negative/inconclusive',
        practical_rule_is_not_significance_test=True,margin_requires_preregistration=True,
        inference='simultaneous look-adjusted bounds/confidence sequence for outcome-dependent stopping; ordinary per-look95percent intervals are not optional-stopping valid',
        no_model_or_threshold_tuning=True,no_new_families_seeds_or_replacements=True,original_elapsed_budget_never_reset=True)
    assert a['maximum_calls']==60 and b['maximum_calls']==180 and c['maximum_calls']==140
    assert math.isclose(b['scenario_seconds']*4,budget['scenario_seconds'],rel_tol=0,abs_tol=1e-8)
    assert [z['maximum_calls'] for z in b['stages']]==[60,120,180]
    stats=[]
    for n in (20,40,60):
        stats.append(dict(families=n,zero_loss_one_sided95_upper=1-.05**(1/n),
            zero_loss_two_comparison_Bonferroni_upper=1-.025**(1/n),zero_loss_3looks_2comparisons_upper=1-(.05/6)**(1/n),
            approximate_accuracy_halfwidth_at_p825=1.95996398454*math.sqrt(.825*.175/n),
            assumptions='binomial/iid reference and normal planning approximation only; not a certificate for stratified families or sequential stopping'))
    sample_size=dict(zero_losses_needed_for5percent_one_endpoint=math.ceil(math.log(.05)/math.log(.95)),
        two_baselines=math.ceil(math.log(.025)/math.log(.95)),three_looks_two_baselines=math.ceil(math.log(.05/6)/math.log(.95)),
        caveat='best-case zero-loss illustration, not power calculation, and relative loss is different from absolute accuracy')
    detection=[dict(hypothetical_family_event_rate=q,no_events20=(1-q)**20,no_events60=(1-q)**60) for q in (.01,.05,.1)]
    # Best possible exact two-sided paired-binomial p when every discordance favours one method.
    paired=[dict(discordances=d,best_two_sided_p=2**(1-d)) for d in range(5,9)]
    write(out/'options.json',dict(original=dict(families=60,calls=720,scenario_seconds=budget['scenario_seconds'],cap_seconds=49780),options=[a,b,c],recommended='B'))
    write(out/'statistical_limits.json',dict(reference_intervals=stats,zero_loss_sample_size=sample_size,rare_MC_event_scenarios=detection,paired_binomial_best_case=paired,
        baseline_R256_only21_possible_disagreement_families=True,first20_only5_R128_switches=True,
        no_power_or_population_reliability_claim=True,no_MC_selected_on_this_frame=True))
    sources=[dict(title='NIST exact binomial confidence limits',url='https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm'),
        dict(title='NIST McNemar paired test',url='https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/mcnemar.htm'),
        dict(title='Howard et al. Time-uniform nonparametric confidence sequences (Annals of Statistics,2021)',url='https://arxiv.org/abs/1810.08240')]
    write(out/'references.json',sources)
    lines=['# Критический пересмотр финального бюджета', '', '2026-10-27. Только чтение сохранённых планов и аналитическая арифметика. Новых solver/inference/ML-fit/симуляций0. Модели, frozen choices, исходные данные и исполнительный plan720 не изменены.',
        '', '## Вывод', '',
        '**60 семейств — допустимый объём оценки, но не доказанный необходимый минимум.720 calls и cap49780с не обязательны. Рекомендуется сохранить максимум60 независимых семейств, убрав лишние повторы: один R128,R256,MC1 на семью, максимум180calls/cap12670с.** Это сохраняет полноту сравнения трёх профилей и число семей, сокращая solver work примерно в4раза. Заранее заданные20→40→60 этапы могут закончить этот же финальный эксперимент раньше отрицательным/неопределённым выводом; это не новые обучающие пилоты.',
        '', '## Что действительно требует расчётов', '',
        'Frozen policy выбирает21R128/39R256/0MC на неизменном frame60. Для accuracy сравнения с R25639 решений совпадают тождественно; лишь21семейство может дать discordance. В первой20-cell пачке переключенийR128 только5. Ни60,ни720 не обеспечивают сами по себе мощность для малой разности accuracy. R128/R256 по4повторам прежних экспериментов детерминированы/bitwise; новые повторы на том же уравнении не увеличивают n независимых accuracy outcomes.',
        '', 'Один R128 и один R256 на семью дают оба baseline и результат селектора: отдельно повторно запускать выбранный профиль для ML не нужно. Старые train/development результаты можно использовать для бюджета/оценки jitter, но не добавлять к новому независимому test как свежие наблюдения. Входы новогоframe ещё не решались; их результаты остаются независимыми от обучения.',
        '', 'MC нельзя исключить из исходной научной задачи. Однако даже240MCcalls не проверяют conditional качество MC-selected действий, поскольку выбранныхMC здесь0. Они проверяют accuracy/cost прогнозы и missed counterfactual opportunities. Один заранее зафиксированный MC-seed на новой семье соответствует primary new-call event;4seeds нужны для within-family variability/диагностики, но это не4 независимых семьи.4/4 successes не удостоверяют p_new≥.95.',
        '', '## Три варианта', '', '|Вариант|Семейств|R128/R256/MC1 calls|Всего|max сценарий|Предлагаемый cap|', '|---|---:|---|---:|---:|---:|']
    for v in (a,b,c):lines.append(f"|{v['name']}|{v['families']}|{v['R128_calls']}/{v['R256_calls']}/{v['MC1_calls']}|{v['maximum_calls']}|{v['scenario_hms']}|{v['proposed_cap_seconds']}с ({v['cap_hms']})|")
    lines += ['', 'A:60calls/20семей — минимальная сбалансированная независимая оценка с одним случаем на каждую из20cells. Подходит для крупного провала/описательного завершения; не для малой noninferiority margin, subgroup/generalization claims или per-familyMC reliability. Не надо автоматически объявлять её пилотом и требовать продолжения: допустим окончательный ограниченный/отрицательный вывод.',
        '', 'B (рекомендовано):60семей,все3профиля по1,180calls. Пачки20/40/60 — первый/второй/третий уже выбранный hash-ranked case в каждойcell; ни затраты,ни forecasts,ни outcomes не выбирают семьи. Сохраняется максимум60семей исходного плана,полный observed3-profile primary oracle,при75% меньшем числе вызовов. Фактическое качество текущей frozen policy и оба baseline оцениваются на одинаковых семьях.',
        '', 'C:120Rcalls на60семьях плюс20MCcalls на первой заранее фиксированной семье каждойcell. Это полноценная проверка именно фактической политики против двухR-baseline на60 семьях (все её действияR),при ещё меньшей цене. MC остаётся в научной проверке,но только на20случаях: нет полного3-profile oracle для остальных40,MC outcomes там unknown,а не none. Нельзя заявлять,что найдены/исключены все упущенные MC-победы. ВыбиратьC только при приоритете проверки текущих действий и жёсткой экономии.',
        '', 'Бюджеты пересчитаны по неизменным case-wise R/MC сценариям исходногоmanifest; одна попытка стоит четверть4повторов. Формулаcap=ceil(2*scenario+300),oneworker/thread,RSS512MiB,guardian/thermal/deadline как раньше. Это плановые резервы,не гарантированная длительность. Повторного budget reset междуэтапами нет. Исходный720-controller не может исполнять эти предложения без отдельной версионированной поправки; новых разрешений/исполнительных manifest здесь не создано.',
        '', '## Статистическая достаточность', '',
        'Исходноеn60 не выведено из ожидаемых paired discordances,дисперсии времени или заранее выбранной мощности. Число59 относится к идеальному условному примеру59/59успехов для one-sided95% lower≥.95. Это не обоснование автоматической95%-точности данной модели,которая в trainOOF имела33/40. Проверка noninferiority кbaseline и доказательство абсолютной accuracy≥.95 — разные estimands.',
        '', '|n семей|Upper95% потерь при0/ n,один endpoint|Две baseline,Bonferroni|3looks×2baseline|Ориентир95% halfwidth accuracy,p=.825|', '|---:|---:|---:|---:|---:|']
    for s in stats:lines.append(f"|{s['families']}|{s['zero_loss_one_sided95_upper']:.1%}|{s['zero_loss_two_comparison_Bonferroni_upper']:.1%}|{s['zero_loss_3looks_2comparisons_upper']:.1%}|±{s['approximate_accuracy_halfwidth_at_p825']:.1%}|")
    lines += ['', 'Upper=1−alpha^(1/n) — reference bound при нуле событий потери. Идеальные минимумы для upper≤5%:59(один endpoint),72(двеbaseline),94(3looks×2baseline). При потерях нужно больше. Это не power calculation; conditional-on-baseline-correct риск требует собственного меньшего знаменателя,а не всехn. Stratified fixed-frame design и неодинаковые family-risks требуют явно выбранных допущений; binomial reference здесь не автоматический сертификат.',
        '', 'Сравнивать accuracy надо попарно (discordant outcomes),а не как две независимые доли. Если всеdiscordances однонаправлены,exact two-sided paired-binomial p=2^(1−d). При≤5discordances p≥.0625: перваяпачка20,где5switchesR128,даже в наилучшем случае не покажет5%-двустороннее superiority кR256. Это не мешает измерить стоимость/крупные ошибки и не означает,что надо требовать superiority вообще. При60семьях малыедоли потерь и редкиеMC-события всёравно оцениваются неточно.',
        '', 'Приn20 однасемья наcell: исходный20-cell within-stratum bootstrap вырождается. Приn40 по2 наcell оценка также нестабильна. Нельзя запускать старыйevaluator и интерпретировать нулевую bootstrap-ширину как уверенность. Нужны заранее выбранные консервативные bounded/paired bounds с допущениями,либо описательная оценка; объединениеcells дляvariance требует дополнительных exchangeability assumptions.',
        '', '## Последовательность без подглядывания', '',
        'Предлагается один preregistered test: этапы20→40→60,модель/порог/MC-seed/входы неизменны. Не останавливать по первому красивому p-value или экономии. Early success не нужен. Resource/data stop остаётся немедленным. Для научного futility можно заранее принять практический criterion:если накоплено4потери R256-correct доставок,то даже при0новых потерь итог за максимум60 не уложится в наблюдаемый5% individual-loss limit; остановка с отрицательным/неопределённым выводом. Это proposed5pp loss criterion,не настройка модели и не статистическое доказательство вреда; gains не компенсируют индивидуальные потери в этом estimand.',
        '', 'Если критерий5pp не нужен для научного решения,не вводить его формально: зафиксировать полный60-case финальный анализ и промежуточные остановки только по ресурсам/критическим сбоям. Если нужны выводы на outcome-dependent остановке,использовать simultaneous alpha-spending bounds(простой консервативный вариантalpha=.05/6 для3looks×2loss-comparisons)или корректную confidence sequence именно для выбранного stratified estimand. Дополнительные confirmatory time-endpoints требуют своей alpha-allocation/совместного анализа; здесь timing inference остаётся описательным. Обычные95%CI накаждомпросмотре не дают95%покрытия у выбранного stop time. При futility-only stopping можно сохранить один заранее фиксированный final-success test,но early-stop estimates/CI всёравно требуют корректной трактовки; не объявлять tentative point estimate независимым подтверждением.',
        '', 'Плановые cumulative caps B:20семей60calls4563с;40семей120calls8565с;60семей180calls12670с. Это одна последовательная проверка,не три пилота. После отрицательного/неопределённого окончания модель не менять и новые семьи ради MC не добирать.',
        '', '## Время,повторы иMC', '',
        'При1call нельзя сохранить критерийmedian4/disjoint timing ranges или уверенный индивидуальный10%winner. Новыйendpoint — paired realised full warm time и агрегированная стоимость по семьям,со стоимостью неудачных решений/ML-overhead. Старыйforecast median-типичнойстоимости сравнивается с шумным одним вызовом лишь описательно; solver-profileM/h/n и error event не меняются. Обучение/модели не меняются.',
        '', 'ПорядокR128/R256 обязательно counterbalance. Нельзя просто взятьepisode0 старогопорядка дляrank0-пачки: всееёi кратны3,иR128 былбывезде передR256. Предлагается новое заранее фиксированное расписание6перестановокпрофилей,попарно меняющее порядокR,назначенное семьям input-hash независимоотcost/outcomes;в20case ровно10R128-before/10R256-before. Это поправка измерительного протокола,не модели/входов. Без численных warm-up вызовов:общиеimports/JIT отдельно,одинworker,полныйметодныйtimer,всёraw сохранить.',
        '', 'Если обязательно нужна воспроизводимость median-time конкретногоуравнения/MC reliability,повторы оправданы,ноэто дополнительный estimand,а не требованиеполитики,котораяMCневыбирает. Можно заранее назначить небольшие timing-sentinels,но это отдельный cap и не повод адаптивно повторять только “удачные”/быстрые результаты. Для минимальногоB дополнительныереплики не обязательны.',
        '', 'MC1 вA/B тестируется накаждой новойсемье,вC на20предвыбранных. При гипотетическом1% family-level detected-useful event вероятность не увидеть ниодного≈81.8%на20 и54.7%на60;при5%≈35.8%и4.6%. Это sensitivity-сценарии,не оценки из3trainpositivefamilies. НулевойMC-result не доказывает глобальнуюбесполезность;0MC-selected во всехвариантах не позволяет проверитьconditional качествоactualMC-choice. Не делатьfalseclaimsдаже после720calls.',
        '', '## Оптимальное завершение', '',
        'ВыбратьB как основной компромисс:семейный максимум60сохраняется,все3профиля присутствуют,повторы убраны,capснижен49780→12670с(≈−74.55%). Зафиксировать до запуска новыйmeasurement/analysis/stopping amendment иmatchingapproval;старые данные/модель/планнепереписывать. Запустить только после отдельногоразрешения. Вконце честно сообщить accuracy/cost/refusal tradeoff и пределыMCвывода;равенствоbaseline или отрицательныйрезультат — допустимое завершение исследования. Прижёсткомлимите≈1ч16мин выбратьA и завершитьсболееузкимвыводом;полныйn60 не обязателен.',
        '', 'Источники: [NIST exact intervals](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm), [NIST paired McNemar](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/mcnemar.htm), [Howard et al.,time-uniform confidence sequences](https://arxiv.org/abs/1810.08240). Конкретные budgets получены только из сохранённого projectmanifest.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(out/'checks.json',dict(passed=True,original_release_artifacts_verified=len(original),model_data_and_original_plan_unchanged=True,
        total_single_cost_equals_original_div4=True,proposal_reference_keys_are_subset_of_original=True,new_solver_calls=0,new_ML_fits=0,
        proposals_not_authorized=True,no_MC_simulation_or_new_input_generation=True))
    for v in (a,b,c):assert set(v['referenced_primary_job_keys'])<={j['key'] for j in plan['jobs']}
    for n,h in original.items():assert sha(RELEASE/n)==h,n
    return (a,b,c)


if __name__=='__main__':
    out=Path(sys.argv[1]).resolve();options=review(out)
    print(json.dumps([{k:v[k] for k in ('name','maximum_calls','scenario_seconds','proposed_cap_seconds')} for v in options]))
