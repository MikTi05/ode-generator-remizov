"""Saved-only proper scores, nested policy comparison and source audit."""
from __future__ import annotations
import argparse,csv,math,os,sys,time
from collections import Counter,defaultdict
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from ml.revision_evidence import read,sha,ORIGINAL,SOURCES,timing_contexts,legacy_inventory
from ml.revision_selector import PROFILES,MODES,action,candidate_grid,family_weights,ResearchRevision
from diagnostics.prepare_development8 import write_json


def probability_scores(rows,p):
    y=np.array([r['k']/r['n'] for r in rows]);p=np.asarray(p,float);w=family_weights(rows);clip=np.clip(p,1e-10,1-1e-10)
    brier=y*(1-p)**2+(1-y)*p*p;loss=-y*np.log(clip)-(1-y)*np.log1p(-clip)
    bins=[]
    for lo,hi in [(0,.2),(.2,.5),(.5,.8),(.8,.95),(.95,1.0000001)]:
        ix=(p>=lo)&(p<hi)
        if ix.any():bins.append(dict(lower=lo,upper=min(hi,1.),equations=int(ix.sum()),mean_p=float(np.average(p[ix],weights=w[ix])),observed_fraction=float(np.average(y[ix],weights=w[ix]))))
    return dict(equations=len(rows),families=len(set(r['family_id'] for r in rows)),trials=sum(r['n'] for r in rows),
        brier=float(np.average(brier,weights=w)),logloss=float(np.average(loss,weights=w)),mean_prediction=float(np.average(p,weights=w)),
        observed_fraction=float(np.average(y,weights=w)),high95_count=int((p>=.95).sum()),high95_failures=float(np.sum((1-y)[p>=.95])),reliability_bins=bins,
        family_Brier_SE=float(np.std([np.mean(brier[[i for i,r in enumerate(rows) if r['family_id']==g]]) for g in sorted({r['family_id'] for r in rows})],ddof=1)/math.sqrt(len(set(r['family_id'] for r in rows)))))


def bootstrap_delta(details,baseline):
    matrix=np.array([[r['correct']-baseline[i]['correct'],r['seconds']-baseline[i]['seconds']] for i,r in enumerate(details)])
    groups=defaultdict(list)
    for i,r in enumerate(details):groups[r['family_id']].append(matrix[i])
    matrix=np.array([np.mean(v,axis=0) for _,v in sorted(groups.items())]);rng=np.random.default_rng(248827)
    draw=matrix[rng.integers(0,len(matrix),size=(4000,len(matrix)))].mean(axis=1)
    return dict(mean_correct_delta=float(matrix[:,0].mean()),mean_seconds_delta=float(matrix[:,1].mean()),fixed_prediction_family_bootstrap95=np.quantile(draw,[.025,.975],axis=0).T.tolist(),
        columns=['correct_delivery_fraction_delta','cost_per_request_seconds_delta'],caveat='descriptive conditional on these shared-training OOF predictions and selected cost subset; not prospective noninferiority/coverage proof')


def delivery_delta(details,baseline):
    grouped=defaultdict(list)
    for r,b in zip(details,baseline):grouped[r['family_id']].append(r['correct']-b['correct'])
    values=np.array([np.mean(v) for _,v in sorted(grouped.items())]);rng=np.random.default_rng(248827)
    draws=values[rng.integers(0,len(values),size=(4000,len(values)))].mean(axis=1)
    return dict(mean_delta=float(values.mean()),fixed_OOF_family_bootstrap95=np.quantile(draws,[.025,.975]).tolist(),
        families=len(values),caveat='conditional fixed OOF uncertainty; zero empirical differences do not prove noninferiority')


def feature_overheads(out):
    from ml.remizov_features import coefficient_features
    accuracy=read(out/'accuracy.json');keys=set(read(out/'paired_keys.json'));meta={}
    for line in (out/'training_inputs/all_methods_equations_v1_metadata.jsonl').open():
        import json
        m=json.loads(line);meta[m['equation_id']]=m
    index={r['equation_key']:r['equation_id'] for r in accuracy};rows=[]
    with np.load(ORIGINAL/'all_methods_equations_v1_ml.npz',allow_pickle=False) as z:
        ids={str(v):i for i,v in enumerate(z['equation_id'])};x=z['x_grid'];coeff=z['X'][[ids[index[k]] for k in sorted(keys)]]
    infer={r['equation_key']:r['seconds'] for r in read(out/'inference_timing.json')['rows']}
    for key,values in zip(sorted(keys),coeff):
        start=time.perf_counter_ns();coefficient_features(values,x,meta[index[key]]);feature=(time.perf_counter_ns()-start)*1e-9
        rows.append(dict(equation_key=key,feature_seconds=feature,inference_seconds=infer[key],seconds=feature+infer[key]))
    write_json(out/'overhead.json',dict(rows=rows,scope='measured warm feature + final inference proxy; solver/JIT/data loading excluded; no executed online dispatch'))
    return {r['equation_key']:r['seconds'] for r in rows}


def report(out):
    accuracy=read(out/'accuracy.json');cost=read(out/'cost.json');oof=read(out/'outer_oof.json');paired=set(read(out/'paired_keys.json'));audit=read(out/'audit.json')
    ai={(r['equation_key'],r['profile']):r for r in accuracy};ci={(r['equation_key'],r['profile']):r for r in cost};oi={r['equation_key']:r for r in oof};overhead=feature_overheads(out)
    context,pins=timing_contexts(read(out/'raw.json'));write_json(out/'timing_compatibility.json',context);write_json(out/'additional_pins.json',pins)
    inventory,legacy_pins=legacy_inventory(out);write_json(out/'legacy_scope_audit.json',inventory);write_json(out/'legacy_plan_pins.json',legacy_pins)
    probability={};time_metrics={}
    for p in PROFILES:
        own=[r for r in accuracy if r['profile']==p];probability[p]={m:probability_scores(own,[oi[r['equation_key']]['variants'][m][p]['p'] for r in own]) for m in MODES}
        constant=[];blocks=read(out/'protocol.json')['outer_groups']
        for r in own:
            held=set(blocks[oi[r['equation_key']]['outer_fold']]);train=[z for z in own if z['family_id'] not in held]
            w=family_weights(train)*np.array([z['n'] for z in train]);constant.append(float(np.average([z['k']/z['n'] for z in train],weights=w)))
        probability[p]['outer_train_constant']=probability_scores(own,constant)
        known=[r for r in cost if r['profile']==p];est=np.array([oi[r['equation_key']]['variants']['binary_sigmoid'][p]['seconds'] for r in known]);truth=np.array([r['seconds'] for r in known])
        time_metrics[p]=dict(known_equations=len(known),MAPE=float(np.mean(abs(est/truth-1))),mean_absolute_log_error=float(np.mean(abs(np.log(est/truth)))),
            median_ratio=float(np.median(est/truth)),residual_band_coverage=sum(oi[r['equation_key']]['variants']['binary_sigmoid'][p]['time_low']<=r['seconds']<=oi[r['equation_key']]['variants']['binary_sigmoid'][p]['time_high'] for r in known)/len(known),
            missing_not_scored=sum(r['profile']==p for r in accuracy)-len(known),scope='family OOF on genuinely observed full method wall; pooled R n-scaling assumption explicit')
    complete=sorted(k for k in paired if all((k,p) in ci for p in PROFILES));assert complete
    def evaluate(policy=None,nested=False):
        details=[]
        for key in sorted(paired):
            row=oi[key];rule=row['nested_selected_policy'] if nested else policy;a=action(row['variants'][rule['mode']],rule)
            obs=ai[key,a] if a!='abstain' else None;correct=obs['k']/obs['n'] if obs else 0.
            c=ci.get((key,a));seconds=c['seconds'] if c else (0. if a=='abstain' else None)
            ov=0. if rule['rule'].startswith('constant_') else overhead[key]
            details.append(dict(equation_key=key,equation_id=row['equation_id'],family_id=row['family_id'],action=a,correct=correct,
                inaccurate=(1-correct if a!='abstain' else 0.),seconds=seconds,overhead=ov,profile_policy=rule['name'],paired_complete_cost=key in complete))
        selected=[r for r in details if r['action']!='abstain'];correct=sum(r['correct'] for r in details)
        missing=[r for r in details if r['seconds'] is None];known=sum(r['seconds'] for r in details if r['seconds'] is not None)
        common=[dict(r,seconds=r['seconds']+r['overhead']) for r in details if r['paired_complete_cost']]
        correct_common=sum(r['correct'] for r in common);common_cost=sum(r['seconds'] for r in common)
        baseline_correct=sum(ai[r['equation_key'],'R256']['k'] for r in details)
        return dict(tasks=len(details),actions=Counter(r['action'] for r in details),accurate_deliveries=correct,finite_accuracy_violations=sum(r['inaccurate'] for r in details),
            abstentions=len(details)-len(selected),delivery_accuracy=correct/len(details),conditional_accuracy=correct/len(selected) if selected else None,
            lost_vs_R256=sum(max(0.,ai[r['equation_key'],'R256']['k']-r['correct']) for r in details),
            gained_vs_R256=sum(max(0.,r['correct']-ai[r['equation_key'],'R256']['k']) for r in details),
            observed_total_method_seconds=known if not missing else None,missing_selected_times=len(missing),known_partial_method_seconds=known,
            full40_cost_per_accurate=(known+sum(r['overhead'] for r in details))/correct if not missing and correct else None,
            common_complete_cost=dict(tasks=len(common),keys=complete,accurate_deliveries=correct_common,cost_with_warm_ML_proxy=common_cost,
                cost_per_accurate=common_cost/correct_common if correct_common else None,details=common),details=details,
            caveat='MC accuracy is empirical seed fraction for a new-call estimand, not a best seed; no population reliability guarantee; policy overhead is final-model warm proxy')
    candidates=candidate_grid();policies={c['name']:evaluate(c) for c in candidates};policies['nested_selected']=evaluate(nested=True)
    for name,metric in policies.items():
        metric['comparisons_common_cost']={base:bootstrap_delta(metric['common_complete_cost']['details'],policies[base]['common_complete_cost']['details']) for base in ('constant_R128','constant_R256')}
        metric['delivery_comparisons_all40']={base:delivery_delta(metric['details'],policies[base]['details']) for base in ('constant_R128','constant_R256')}
    ranking={};ranking_details=[]
    for mode in MODES:
        eligible=0;missed=0;wrong_R_order=0;false_high=Counter();switches=Counter()
        for key in sorted(paired):
            r=oi[key];v=r['variants'][mode];a=action(v,dict(mode=mode,rule='q95_default'))
            good128=ai[key,'R128']['k']==1;good256=ai[key,'R256']['k']==1
            if good256 and not good128 and v['R128']['p']>=v['R256']['p']:wrong_R_order+=1
            for p in PROFILES:
                if v[p]['p']>=.95 and ai[key,p]['k']<ai[key,p]['n']:false_high[p]+=1
            known=(key,'R128') in ci and (key,'R256') in ci
            opportunity=good128 and known and ci[key,'R128']['seconds']<.9*ci[key,'R256']['seconds']
            if opportunity:eligible+=1;missed+=a!='R128'
            switches[a]+=1
            ranking_details.append(dict(equation_key=key,equation_id=r['equation_id'],mode=mode,OOF_p={p:v[p]['p'] for p in PROFILES},
                observed_fraction={p:ai[key,p]['k']/ai[key,p]['n'] for p in PROFILES},q95_action=a,known_R128_opportunity=opportunity,
                missed_reason=('p_R128_below_q95' if v['R128']['p']<.95 else 'predicted_time_order') if opportunity and a!='R128' else None))
        ranking[mode]=dict(known_R128_opportunities=eligible,q95_missed_R128=missed,q95_actions=switches,R_order_errors_when_only_R256_accurate=wrong_R_order,high95_inaccurate_equations=false_high)
    write_json(out/'ranking_details.json',ranking_details)
    metrics=dict(probability=probability,time=time_metrics,policies=policies,ranking=ranking,
        common_cost_tasks=len(complete),common_cost_subset_selection='all paired original TRAIN cases with real full times for all3; nine-case historical selected subset, not representative holdout',
        primary='nested_selected; outer policies selected only on separate inner nested-calibrated OOF',
        final_policy=read(out/'registry.json')['final_policy'],final_policy_score_is_not_primary=True,
        new_successes_not_manufactured=True,development_validation_test_used_for_model_choice=False)
    write_json(out/'metrics.json',metrics)
    with (out/'policy_quality.csv').open('w',newline='') as f:
        fields=['policy','tasks','accurate','violations','abstentions','common_cost_tasks','common_cost_seconds','common_correct','common_cost_per_correct','missing_full_times']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for name,z in policies.items():
            c=z['common_complete_cost'];w.writerow(dict(policy=name,tasks=z['tasks'],accurate=z['accurate_deliveries'],violations=z['finite_accuracy_violations'],abstentions=z['abstentions'],common_cost_tasks=c['tasks'],common_cost_seconds=c['cost_with_warm_ML_proxy'],common_correct=c['accurate_deliveries'],common_cost_per_correct=c['cost_per_accurate'],missing_full_times=z['missing_selected_times']))
    return metrics


def markdown(out,m):
    audit=read(out/'audit.json');verify=read(out/'independent_verification.json');summary=read(out/'run_summary.json')
    final_name=m['final_policy']['name'];final=m['policies'][final_name];primary=m['policies']['nested_selected'];b256=m['policies']['constant_R256'];b128=m['policies']['constant_R128']
    saving=1-final['common_complete_cost']['cost_with_warm_ML_proxy']/b256['common_complete_cost']['cost_with_warm_ML_proxy']
    lines=['# Train-only revision: R128/R256/MC1', '', '2026-10-27, Europe/Moscow. Новых ODE/MC/benchmark-вызовов0. ML-fit разрешён и выполнен; production API не разблокирован.',
        '', '## Итог', '',
        '**Подтверждённого превосходства над обоими baseline нет.** Строго вложенный выбор на40 парных train-задачах выбрал R256 на40/40:33 точных доставки,7 нарушений,0 отказов. Это точность constant_R256, при небольшом ML-overhead. Constant_R128 точен25/40, но дешевле на измеренном подмножестве. Общая стоимость40 неизвестна из-за пропусков полных времён.',
        '', f"Заморожен перспективный research-кандидат **{final_name}**. На той же общей train-OOF таблице он имеет11 R128/29 R256,33/40 точных,0 потерь против R256. На9 задачах с реальными полными временами для всех3 его cost-per-correct {final['common_complete_cost']['cost_per_accurate']:.6f} с против{b256['common_complete_cost']['cost_per_accurate']:.6f} с у R256 (экономия{saving*100:.2f}%). **Этот вариант выбран по общей train-OOF таблице; его оценка после выбора оптимистична и не заменяет основной nested результат.** Он не выигрывает у constant_R128 по цене на точное решение, но точнее его. Независимого подтверждения нет.",
        '', '## Допустимые данные и идентичность', '',
        'Задание прямо запрещает подбор по ранее изученным development. Поэтому, несмотря на прежнее предложение включить development8 в train, в этой ревизии все96 его вызовов только проверены/карантинированы. Прежние4 post-fit независимых MC-seed F0047 также исключены из fit/калибровки/выбора. Старые validation8 и весь test не загружались как исходы для этой работы. Инвентаризация иных сохранённых экспериментов отличает legacy taper/x0/другие M/h/T/backend от трёх разрешённых профилей; они не смешаны.',
        '', 'Ключ уравнения: SHA256 исходного канонического CSV + equation_id + expression_hash. Все8 строковых ID development пересекаются со старым набором, но выражения/наборы другие. Семейный group — глобальная topology signature: совпавшая топология никогда не делится даже между наборами. Join, признаки и разрешённость train проверяются по полному ключу.',
        '', '|Профиль|Уравнений / семейств|Успехов / наблюдений|Полное время, задач|', '|---|---:|---:|---:|']
    for p,c in audit['counts'].items():lines.append(f"|{p}|{c['equations']} / {c['families']}|{c['successes']} / {c['trials']}|{c['full_cost_equations']}|")
    lines += ['', 'Четыре повторные MC-записи с тем же semantic seed удалены из likelihood.20 старых final-grid MC-вызовов имеют тот же M4096/h1/128, target33 и аналитический T/G, поэтому их независимые seeds добавлены к уравнениям как k/n, а не новые X-строки. Получено180 испытаний на40 уравнениях вместо прежних160;12 успехов, но всего3 positive-family: F0005/S02=5/8,F0047/S00=6/8,F0136/S01=1/4. Повторные таймеры replay сохраняются как измерения времени, не как новые MC-исходы. Детерминированные повторы R свёрнуты до одного accuracy label.',
        '', 'Проверены profile cfg, сетка33, конечность f, exact/E_grid, work counters, аналитические G/T/N, RNG/point addresses, source/plan/dataset SHA. Resource caps не меняют физический профиль успешно завершённого вызова. Сохранены отказы/другие профили/карантин отдельными таблицами. Числа k/n не интерпретируются как частоты общей популяции: исторический train выбран неоднородно и частично адаптивно.',
        '', 'Только explicit nanosecond method_wall используется как full cost. Старые native runtimes и task_wall, включавшие scoring/asdict, исключены из cost targets. При нескольких timing blocks берётся median внутри cohort, затем median cohort-медиан: ни быстрый блок, ни успешный seed не отбираются. У новых full-wall источников совпадают записанные platform/Python/NumPy/SciPy/threads; старые native-only источники не сохраняли platform и дают только accuracy. Это историческое сравнение, не uniform prospective timing.',
        '', '## Модели и исключение утечек', '',
        'Используются прежние253 input-only признака a/b/c/g + whitelist коэффициентных AST/сертификатов. ID, split, seed, exact/solver/error/time отсутствуют в X. Ошибка и время являются только targets. Профиль n известен до выбора. Во всех fits CatBoost120/depth2/lr.05/l2_leaf10,1thread; scaler fit только на train текущего узла. Hyperparameter search нет.',
        '', 'Вероятности: (1) три binomial CrossEntropy-head; (2) sigmoid к их inner-family OOF, с L2=2, n-weighted likelihood; (3) R-challenger log(n·E_grid) с profile-specific residual ECDF по family OOF, MC остаётся binomial+sigmoid. Масштабирование ошибки1/n — проверяемая ML-гипотеза, не теорема/изменение метода. Регуляризация не добавляет искусственных success rows. Для стоимости pooled R-head обучен на45 настоящих equation/profile измерениях log(t/n), MC-head на40 настоящих полных временах. R-cost n-linearity также модельное предположение; отсутствующие R128 времена не превращаются в метки.',
        '', 'Три outer-fold покрывают103 R128 cases/90 topology families.40 парных семейств сохраняют прежнее разбиение;50 auxiliary-family добавлены фиксированным hash. В каждом outer-train — отдельные3 policy-inner folds; внутри каждого — ещё3 family splits для калибровки. Каждый ancestor-held family исключён из каждого scaler/base model/calibrator. Outer policy выбирается до чтения соответствующих outer outcomes. Final policy выбирается на всей train-OOF таблице и затем головы refit на разрешённом train; его in-sample прогнозы не выдаются за проверку.',
        '', 'Заранее заданы20 правил: constant baselines, q=.95 default/risk, относительное accuracy-parity δ=0/.02/.05 и стоимость/ожидаемый успех. MC1 всегда кандидат и получает10% time-tie правило в пользу R. Внутренний selection gate требует отсутствия наблюдаемых индивидуальных потерь к R256, затем максимизирует точные доставки и минимизирует **предсказанную**, а не выдуманную измеренную стоимость. Это эмпирический train gate, не гарантия будущей точности.',
        '', '## Вероятности и калибровка', '', '|Профиль|Raw Brier / logloss|Sigmoid Brier / logloss|Continuous-error Brier / logloss|', '|---|---|---|---|']
    for p in PROFILES:
        cells=[f"{m['probability'][p][mode]['brier']:.5f} / {m['probability'][p][mode]['logloss']:.5f}" for mode in MODES]
        lines.append('|'+p+'|'+'|'.join(cells)+'|')
    lines += ['', 'Контрольная outer-train constant вероятность (семейные held-out исходы не использованы в её оценке): '+
        '; '.join(f"{p} Brier={m['probability'][p]['outer_train_constant']['brier']:.5f},logloss={m['probability'][p]['outer_train_constant']['logloss']:.5f}" for p in PROFILES)+'.']
    lines += ['', 'Binary R128 и MC по proper scores не превосходят этот простой контроль. Нельзя называть их улучшенными только потому, что они используют CatBoost; ограниченность сигналов/семейной поддержки сохраняется. Continuous-error R128 лучше constant по Brier/logloss. MC observed family-equation fraction=.040625, тогда как12/180=.06667 — другой, trial-weighted знаменатель; ни один из них не является оценкой популяционной надёжности нового уравнения.']
    lines += ['', 'Family-weighted scores, bins и семейные SE сохранены. Sigmoid не улучшил Brier и не является автоматически более правильной калибровкой. У R128 logloss немного улучшился при худшем Brier; у R256 и MC ухудшились оба score. Continuous-error даёт лучшие R proper scores, но ошибается в относительном ранжировании на2 случаях, где точен только R256. Улучшение среднего Brier не гарантирует улучшения решений.',
        '', 'При calibrated binary q=.95 R128 не eligible ни на одном внешнем OOF-случае; на9 cost-complete задачах упущены все7 известных возможностей точного R128. Поэтому главным ограничением остаётся вероятность/абсолютный порог, а не распознавание двукратной разницы времени. Raw q95 переключает15 случаев, но теряет1 точную доставку против R256 (32 вместо33). Нельзя снимать калибровку только ради переключений. Calibrated risk-q95 отказывается40/40: это не решение исходной задачи. Относительная parity-policy пытается убрать эту структурную блокировку без заявления p≥.95; она остаётся research-only.',
        '', '## Стоимость и качество политики', '', '|Политика|Точных /40|Нарушений|Отказов|Стоимость на9 с полным timing, с|Точных /9|с / точный на9|', '|---|---:|---:|---:|---:|---:|---:|']
    for name in ['constant_R128','constant_R256','nested_selected',final_name]:
        z=m['policies'][name];c=z['common_complete_cost']
        lines.append(f"|{name}|{z['accurate_deliveries']:g}|{z['finite_accuracy_violations']:g}|{z['abstentions']}|{c['cost_with_warm_ML_proxy']:.6f}|{c['accurate_deliveries']:g}|{c['cost_per_accurate']:.6f}|")
    lines += ['', 'Для полной40-case стоимости недостаёт31 R128 и4 R256 method-wall targets. Всего9 cost-complete задач — прежний выбранный timing-поднабор, а не случайный holdout; экономию нельзя переносить на40 или будущую популяцию. Все расходы неточных выбранных решений учтены; отказ оплачивает ML-overhead. MC accuracy оценивается средним по реальным seeds, не лучшим seed. Приведённый ML-overhead измерен на warm final heads и используется как proxy для outer policies; реальные online dispatch/cold/JIT/fallback не выполнялись.',
        '', '|Head времени|Реальных OOF cases|MAPE|Покрытие descriptive10/90 residual-band|', '|---|---:|---:|---:|']
    for p,z in m['time'].items():lines.append(f"|{p}|{z['known_equations']}|{100*z['MAPE']:.2f}%|{100*z['residual_band_coverage']:.1f}%|")
    lines += ['', 'Разброс residual не является калиброванным интервалом будущего времени. MC-cost остаётся значительно труднее прогнозировать, чем фиксированный n·512 R-work. Для всех правил сохранены quality/cost tables и paired fixed-OOF family-bootstrap. Shared training, policy selection и исторический отбор не покрываются этим bootstrap; нулевая наблюдаемая разность не доказывает noninferiority.',
        '', '## MC и окончательная проверка', '',
        'MC не выбран ни одним из20 правил в outer OOF. Искусственные MC-winner/upsampling не создавались. Новый кандидат теоретически может выбрать MC при достаточной прогнозируемой точности и выгоде, что отдельно проверено unit-тестом; его участие не блокировано глобальным q=.95. Но три MC-positive семейства не обеспечивают калибровку вероятности на новых структурах, а допуски .625/.75/.25 на известных задачах не свидетельствуют о надёжности нового seed≥.95. Старый gate20+5 winner не нужен для binomial/cost heads; реальная MC-support/utility всё равно необходима.',
        '', 'Минимальный следующий шаг — сохранить эту заморозку и подготовить **новые** independent families/правила оценки. Development8, прежние validation и этот train-OOF повторно не считать независимыми. На новых задачах нужны сопоставимые R128/R256/MC1 и независимые MC-seeds, полное время и E_grid, заранее заданные критерии потерь точных доставок/экономии и отдельный достаточный MC-selected support. Без реальных полезных MC-случаев такой holdout проверит в основном Remizov; трёхметодное обобщение останется незавершённым.',
        '', 'До дорогого пилота следует train-only saved-data feasibility-проверкой обосновать область/профиль, где MC может одновременно достичь точности и выиграть у самого быстрого точного R. Слепой добор20 random-family не оправдан только требованием MC-класса. При смене M/h нужен отдельный профиль/контракт и бюджет. Для закрытия именно исторической cost-таблицы минимум35 missing equation/profile cells (31 R128+4 R256); одно измерение закрывает отсутствие,4 повтора дают140 calls, но не новый независимый тест. В этой задаче они не запускались и бюджет не запрашивался.',
        '', 'Универсального минимального n финального теста нет. Например,59/59 независимых принятых семейных исходов без ошибок дают одностороннюю95% binomial lower≈.95 при iid-дизайне; это не59 произвольных задач, не59 seeds одного семейства и не доказательство для редкого MC-action. Нужен отдельный preregistered риск/precision для MC-selected подгруппы.',
        '', '## Воспроизводимость и проверки', '',
        f"Сохранены312 узлов моделей/скейлеров,13 calibration bundles,39 sigmoid calibrators,103 полных family-OOF прогнозов и policy selection provenance. Независимо воспроизведены{verify['raw_OOF_values_replayed']} raw OOF values и вся графовая цепочка family exclusions; max sigmoid KKT residual {verify['maximum_calibrator_KKT']:.3g}. Реальные targets/weights/scaler means/vars, replay collapse, hashes и blocked dispatch проверены. Повторный аудит дал побитово те же X и те же training/quarantine tables. Training wall≈{summary['training_seconds']:.2f} с, один ML-worker/thread; численных solver calls0.",
        '', 'Команды из корня с локальной .venv (для нового out сначала --action prepare):', '',
        '```bash', 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/revise_train_selector.py --action prepare --out results/<new_revision>',
        'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/revise_train_selector.py --out results/<new_revision>',
        'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/verify_train_revision.py --out results/<new_revision>',
        'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/report_train_revision.py --out results/<new_revision>', '```',
        '', 'Ни один из этих драйверов не запускает решатель; audit восстанавливает только analytic exact/G/T. Предыдущие модели/данные/результаты сохранены. Production code unchanged; новая ResearchRevision всегда action=abstain со статусом research_only, выбор доступен только как diagnostic_choice.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);out=ap.parse_args().out.resolve();m=report(out);markdown(out,m)
    for name in ('constant_R128','constant_R256','nested_selected'):
        print(name,{k:v for k,v in m['policies'][name].items() if k not in ('details','common_complete_cost','comparisons_common_cost')},m['policies'][name]['common_complete_cost']['cost_per_accurate'])
