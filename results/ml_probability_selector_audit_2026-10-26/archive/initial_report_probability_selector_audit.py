"""Render and independently cross-check the saved audit; no fitting or solving."""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
import shutil
import sys
import tempfile
import os

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from diagnostics.audit_probability_selector import read,write,sha,MODEL,VALIDATION,PROFILES


def ranking_auc(scores,positive):
    pos=[s for s,y in zip(scores,positive) if y];neg=[s for s,y in zip(scores,positive) if not y]
    if not pos or not neg:return None
    return sum((p>n)+.5*(p==n) for p in pos for n in neg)/(len(pos)*len(neg))


def independent_check(out):
    import numpy as np
    a=read(out/'analysis.json');cv=read(MODEL/'family_cv.json');targets=read(MODEL/'accuracy_targets.json')
    lookup={r['index']:r for r in targets}
    with np.load(MODEL/'features_train.npz',allow_pickle=False) as z:
        X={int(i):row for i,row in zip(z['original_indices'],z['X'])}
    scaler_checks=0;maximum_mean_error=0.
    for fold in cv['folds']:
        held=set(fold['held_families'])
        for head in fold['heads'].values():
            assert all(lookup[i]['split']=='train' and lookup[i]['family_id'] not in held for i in head['indices'])
            mean=np.mean(np.stack([X[i] for i in head['indices']]),axis=0)
            delta=float(np.max(abs(mean-np.asarray(head['train_scaler_mean']))))
            assert np.allclose(mean,head['train_scaler_mean'],rtol=1e-14,atol=1e-12)
            maximum_mean_error=max(maximum_mean_error,delta);scaler_checks+=1
    records=[read(p) for p in (VALIDATION/'records').glob('*.json')]
    assert len(records)==96
    for r in records:
        E=float(np.max(abs(np.asarray(r['result']['f'])-r['y_exact'])))
        assert E==r['E_grid'] and (E<=.01)==r['accuracy_met']
    frozen=read(VALIDATION/'frozen_predictions.json')['rows'];cases=read(VALIDATION/'cases.json')
    byid={r['equation_id']:r for r in cases};ranking={}
    for p in PROFILES:
        ranking[p]=dict(OOF_AUROC=ranking_auc([r['estimates'][p]['probability']['point'] for r in cv['oof']],
            [r['observed'][p]['k']>0 for r in cv['oof']]),validation_AUROC=ranking_auc(
            [r['estimates'][p]['probability']['point'] for r in frozen],[byid[r['equation_id']]['profiles'][p]['accurate']>0 for r in frozen]),
            target='ever-success family over4 seeds, not per-seed probability' if p=='MC1' else 'deterministic accuracy')
    # Recompute costs independently from the 96 records and frozen actions.
    import statistics
    time={(c['equation_id'],p):statistics.median(r['method_wall_seconds'] for r in records if r['equation_id']==c['equation_id'] and r['profile']==p)
          for c in cases for p in PROFILES}
    primary={(r['equation_id'],r['profile']):r['accuracy_met'] for r in records if r['job']['episode']==0}
    overhead=sum(r['feature_and_inference_seconds'] for r in read(VALIDATION/'policy_overhead.json')['rows'])
    for q in (.5,.8,.95):
        actions={r['equation_id']:r['diagnostic_choices'][str(q)]['point']['action'] for r in frozen}
        correct=sum(primary[pid,p] for pid,p in actions.items() if p!='abstain')
        cost=sum(time[pid,p] for pid,p in actions.items() if p!='abstain')+overhead
        z=a['policies']['validation'][str(q)]['point']
        assert correct==z['accurate_deliveries'] and math.isclose(cost,z['warm_cost'],abs_tol=1e-9)
        assert math.isclose(cost/correct,z['seconds_per_accurate_delivery'],abs_tol=1e-9)
    plan=read(out/'acquisition_plan.json');fits=set().union(*(set(h['metadata']['families']) for h in read(MODEL/'registry.json')['heads'].values()))
    assert len(plan['cases'])==len({c['family_id'] for c in plan['cases']})==20
    assert all(c['split']=='train' and c['family_id'] not in fits for c in plan['cases'])
    for path,h in read(out/'source_sha256.json').items():assert sha(ROOT/path)==h,path
    result=dict(passed=True,OOF_scalers_checked=scaler_checks,scaler_mean_max_absolute_error=maximum_mean_error,
        saved_validation_errors_recomputed=96,policy_cost_per_success_independently_checked=True,
        proposed_train_families_unseen_by_all_fit_scalers=True,ranking=ranking,
        ML_fits=0,numerical_calls=0,test_outcomes_used=False,thresholds_changed=False)
    write(out/'independent_verification.json',result)
    return result


def plots(out,a):
    with tempfile.TemporaryDirectory(prefix='ode_audit_mpl_') as cache:
        os.environ['MPLCONFIGDIR']=cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(11,3.8),layout='constrained')
        for ax,p in zip(axes,PROFILES):
            ax.plot([0,1],[0,1],color='.6',lw=1,ls='--')
            for cohort,marker,color in [('OOF','o','#225EA8'),('validation','^','#D95F0E')]:
                bins=a['cohorts'][cohort][p]['probability']['bins']
                x=[b['mean_prediction'] for b in bins];y=[b['observed_rate'] for b in bins]
                ax.scatter(x,y,s=[20+3*b['equations'] for b in bins],label=cohort,marker=marker,color=color,alpha=.8)
                for xx,yy,b in zip(x,y,bins):ax.annotate(str(b['equations']),(xx,yy),xytext=(3,5),textcoords='offset points',fontsize=8)
            ax.set(xlim=(-.03,1.04),ylim=(-.03,1.04),title=p,xlabel='Mean predicted probability',ylabel='Observed accuracy rate')
            ax.grid(alpha=.15)
        axes[0].legend(fontsize=8)
        fig.suptitle('Frozen predictions: descriptive fixed-bin reliability (labels = equations)')
        fig.savefig(out/'calibration.png',dpi=170);fig.savefig(out/'calibration.svg');plt.close(fig)
        policies=a['policies']['validation']
        chosen=[('R256',policies['baselines']['R256']),('R128',policies['baselines']['R128']),
                ('point .5/.8',policies['0.5']['point']),('point .95',policies['0.95']['point'])]
        fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
        x=list(range(4));labels=[r[0] for r in chosen]
        good=[r[1]['delivery_accuracy'] for r in chosen];bad=[r[1]['violations']/8 for r in chosen];abstain=[r[1]['abstention_rate'] for r in chosen]
        axes[0].bar(x,good,label='Accurate delivery',color='#31A354')
        axes[0].bar(x,bad,bottom=good,label='Inaccurate',color='#DE2D26')
        axes[0].bar(x,abstain,bottom=[u+v for u,v in zip(good,bad)],label='Abstain',color='.7')
        axes[0].set(xticks=x,xticklabels=labels,ylim=(0,1.04),ylabel='Fraction of 8 requests')
        axes[0].legend(fontsize=8,loc='lower right')
        values=[r[1]['seconds_per_accurate_delivery'] for r in chosen]
        axes[1].bar(x,values,color='#3182BD')
        for xx,y in zip(x,values):axes[1].text(xx,y+.3,f'{y:.2f}',ha='center')
        axes[1].set(xticks=x,xticklabels=labels,ylabel='Warm seconds / accurate delivery',ylim=(0,32))
        fig.suptitle('Offline validation: costs include failures + all inference, no free abstention')
        fig.savefig(out/'policy_quality.png',dpi=170);fig.savefig(out/'policy_quality.svg');plt.close(fig)


def report(out):
    a=read(out/'analysis.json');plan=read(out/'acquisition_plan.json');sizes=read(out/'sample_size.json')
    independent=independent_check(out);plots(out,a)
    with (out/'calibration_bins.csv').open('x',newline='') as f:
        fields=['cohort','profile','bin','families','equations','mean_prediction','observed_rate','successes','trials']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for cohort,z in a['cohorts'].items():
            for p,m in z.items():
                for b in m['probability']['bins']:writer.writerow(dict(cohort=cohort,profile=p,**b))
    rows=[]
    for name,z in [('constant_R256',a['policies']['validation']['baselines']['R256']),
                   ('constant_R128',a['policies']['validation']['baselines']['R128']),
                   ('point q=.5/.8',a['policies']['validation']['0.5']['point']),
                   ('point q=.95',a['policies']['validation']['0.95']['point']),
                   ('model_spread q=.95',a['policies']['validation']['0.95']['model_spread']),
                   ('production_guard',a['policies']['validation']['production_guard'])]:
        rows.append(dict(policy=name,**{k:v for k,v in z.items() if k!='actions'}))
    with (out/'policy_quality.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Почему вероятностный селектор не превзошёл Remizov: аудит без новых расчётов', '',
'Дата: 2026-10-26. Новых solver/benchmark/ML-fit/calibrator-fit — **0**, test не использован. '
'Прочитаны сохранённые 96 validation-возвратов, замороженные прогнозы и 40 семейных train-OOF строк. '
'Исходные модели, профили, q=.5/.8/.95 и прежние результаты не изменены.', '',
'## Вывод и границы утверждения', '',
'На восьми validation-семействах точечная политика дала **6/8 точных доставок**, 1 неточный ответ и 1 отказ; '
'constant_R256 — **7/8**, 1 неточный и 0 отказов. Потеря доставки вызвана отказом F0189, '
'а неточный F0041 общий для селектора и baseline. Политика дешевле по секундам на точный ответ, '
'поэтому речь о потере качества/coverage при экономии, а не о доминировании baseline по всем критериям.', '',
'Точность E_grid≤0.01 и порог q=0.95 различны: первое — фактическая ошибка результата, '
'второе — желаемая вероятность её достижения. Ни один не следует из solver success=true. '
'Baseline 7/8 также не подтверждает 95% надёжность. Восемь семейств — диагностика, '
'не доказательство статистического превосходства или его отсутствия в генеральной совокупности.', '',
'## Причины ошибки, отказа и блокировки API', '',
'**F0041/S02: неверная высокая оценка допустимости.** R256 p=0.992167, но E=0.0172385414; '
'R128 p=0.920211, E=0.0343758165. При q=.95 политика исключает R128 и выбирает R256; '
'при q=.5/.8 выбирает более быстрый, но также неточный R128. Оба профиля детерминированы, '
'повторы совпали, конфигурации/seeds/входы проверены: случайность таймера не объясняет нарушение. '
'На F0032 у R256 ровно тот же p=0.992167, но E=.0004907603; '
'это показывает отсутствие различения трудного случая, не просто общий сдвиг среднего p.', '',
'**F0189/S01: ложноотрицательная допустимость.** R128 E=.0102662533 строго выше порога, '
'но R256 E=.0051778312 точен. Прогнозы p_R128=.223228 и p_R256=.189161 исключают оба '
'при всех исходных q; причина point-отказа q=.95 — no_predicted_feasible_method. '
'Даже верхний bootstrap-spread R256 .76118 ниже .95. Изменять q по этому случаю нельзя. '
'Монотонная калибровка не способна исправить обратный порядок F0041/F0189; нужны данные '
'и проверка ранжирования допустимости на новых семействах, помимо самой калибровки.', '',
'Сохранённое отношение E_R128/E_R256 около 1.98–2.03 на всех восьми задачах; '
'для F0041 1.9941, F0189 1.9827. Это согласуется с вкладом конечного n, '
'но двух n недостаточно для установления порядка или разложения ошибки. '
'refinement_controls_performed=false и total_error_certified=false. '
'На F0041 boundary_state_max_abs ≈1.17e6/4.48e4, но на точном F0032 ещё больше '
'≈8.82e6/1.74e7; это максимум не взвешенного по exp(-λt) граничного состояния, '
'не граница полной ошибки. Причинно отделить окно/dx/квадратуру/n по сохранённым двум профилям нельзя. '
'Индикаторы не подставляются в input-only ML-признаки и не выдаются за сертификат.', '',
'**Production API: явный запрет исполнения.** select_method создаёт action=abstain/status=research_only '
'и никогда не присваивает диагностический выбор полю action. Registry deployment_ready=false, '
'validation_calibrated=false. Reason: no_original_validation_calibration;cost_context_unvalidated;R128_time_only4families. '
'Появление 96 результатов не переобучает/калибрует модели, не обновляет registry и не снимает запрет; '
'одного изменения registry также недостаточно для изменения жёстко заданного action. Это не восемь ошибок классификации.', '',
'**Model-spread policy — другой слой.** При q=.95 шесть отказов имеют reason '
'potential_candidate_time_head_insufficient: любой R128 с high_p≥q и cost.supported=false '
'блокирует робастное решение, даже если point p_R128<q или другой профиль выглядит допустимым. '
'F0133: probability_requirement_unresolved (R256 low=.8770<q, high=.9917≥q); '
'F0189: no_predicted_feasible_method. R128-time обучен константой на 4 семействах; '
'новые validation-времена не считаются train-данными автоматически. Полные трассы всех исходных q '
'и 48 совпавших решений сохранены в [analysis.json](analysis.json).', '',
'## Качество политики и стоимость точной доставки', '',
'| Политика | Точные / запросы | Нарушения / выбранные | Отказы | Warm, с | Секунд / точный ответ |',
'|---|---:|---:|---:|---:|---:|']
    for r in rows:
        cps=r['seconds_per_accurate_delivery']
        lines.append(f'| {r["policy"]} | {r["accurate_deliveries"]:.0f}/8 | {r["violations"]:.0f}/{r["selected"]} | {r["abstention_rate"]:.1%} | {r["warm_cost"]:.3f} | {cps:.3f} |' if cps is not None else
                     f'| {r["policy"]} | 0/8 | 0/0 | 100% | {r["warm_cost"]:.3f} | не определено |')
    lines += ['', 'Секунд на точный ответ = **вся стоимость выбранных профилей, включая неточные ответы, '
'плюс inference всех запросов / число точных доставок**. Отказ не является успехом; при нуле успехов '
'отношение не определено, а не равно нулю. Warm-стоимость — offline сумма median4, '
'не фактический online dispatch; загрузка моделей .382659 с отдельно, inference всего .089570 с. '
'96 исследовательских вызовов стоили 2732.966 с wall и не входят в эти policy-costs.', '',
'q=.95: условная точность 6/7=85.71%, нарушение 14.29% выбранных (12.5% всех запросов), '
'coverage 87.5%, точная доставка 75%. Экономия 38.54% total warm, но на F0032/F0133 '
'более дорогой R256 не улучшил качество относительно точного R128. q=.5/.8 дали те же '
'6 успехов/1 нарушение/1 отказ и меньше затрат; это описательная чувствительность исходной сетки, '
'не основание выбрать меньший q задним числом.', '',
'На train-OOF q=.5/.8/.95: точных доставок 27/29/28 из40, отказов 2/6/10; '
'у constant_R256 33/40. При q=.95 точны 28/30 выбранных=93.33%. '
'Полная OOF-стоимость point-политики неизвестна: full wall отсутствует у 29/24/16 выбранных '
'соответственно. Ни solver-only время, ни прогноз времени не подставлены вместо наблюдения.', '',
'![Качество и стоимость политики](policy_quality.png)', '',
'## Калибровка: что говорят семейные OOF и validation', '',
'Проверены исключение held families для всех шести голов каждой из3 складок и 18 scaler means '
'по сохранённым train-признакам; соответствие120 OOF targets подтверждено. '
'OOF40 — paired подвыборка: это не OOF для всех103 R128 строк/90 семейств. '
'Исходный train40 включает stress enrichment, поэтому его частоты не считаются популяционными. '
'Восемь validation-семейств были выбраны заранее RNG seed2488 из30 исходных validation-семейств '
'с одним ID по хешу, без подбора по outcomes; небольшой размер ограничивает выводы.', '',
'| Набор / профиль | Среднее p | Наблюдаемая частота | Bias p−rate | ECE | Brier / constant |',
'|---|---:|---:|---:|---:|---:|']
    for cohort in ('OOF','validation'):
        for p in PROFILES:
            z=a['cohorts'][cohort][p]['probability'];s=z['scores']
            lines.append(f'| {cohort} {p} | {s["mean_prediction"]:.4f} | {s["observed_rate_equal_equation_weight"]:.4f} | {z["calibration_bias_mean_p_minus_rate"]:+.4f} | {z["descriptive_ECE"]:.4f} | {s["brier"]:.4f} / {z["constant_scores"]["brier"]:.4f} |')
    lines += ['', 'Brier/logloss рассчитаны по реальным trials: для MC '
's(1−p)²+(1−s)p², а не (s−p)², где s=k/4. Детерминированные R-повторы дедуплицированы. '
'ECE — описательная сумма по фиксированным bins с равным весом уравнений; '
'на столь малом наборе это нестабильная оценка, не критерий подбора калибратора. '
'OOF baselines — частоты только соответствующего train-fold; validation baseline — прежний train. '
'Bootstrap4000 ресэмплирует целые семейства фиксированных прогнозов; интервалы условные, '
'не учитывают переобучение/выбор дизайна и не доказывают prospective coverage.', '',
'| Профиль | OOF p≥.95: успехи / семьи | Среднее p в хвосте | Описательный CP95 хвоста | Validation p≥.95 |',
'|---|---:|---:|---|---:|']
    for p in PROFILES:
        z=a['cohorts']['OOF'][p]['probability']['tail']['0.95'];v=a['cohorts']['validation'][p]['probability']['tail']['0.95']
        ci=z['descriptive_binomial_CP95'];ci_text=f'[{ci[0]:.3f},{ci[1]:.3f}]' if ci else 'нет данных'
        pm=f'{z["mean_prediction"]:.4f}' if z['mean_prediction'] is not None else '—'
        lines.append(f'| {p} | {z["successes"]}/{z["families"]} | {pm} | {ci_text} | {v["successes"]}/{v["families"]} |')
    lines += ['', 'R128 систематически переоценивает точность на paired OOF: mean p=.7658 против .625, '
'в хвосте 13/15=86.67% при среднем p=.9699. R256 OOF лучше train-constant по Brier, '
'но все условные family-bootstrap95 интервалы разности Brier включают ноль; '
'26/27 в хвосте не подтверждают нижнюю границу ≥.95. '
'На validation среднее p_R256=.8905 близко к .875, однако ECE=.2182: '
'средняя калибровка скрывает уверенный провал F0041 и низкую оценку точного F0189. '
'AUROC R256: .855 OOF, .214 на validation с единственным отрицательным случаем. '
'Последнее не точная оценка общего AUC, но наглядное свидетельство ошибки ранжирования в этом пакете.', '',
'![Описательная проверка калибровки](calibration.png)', '',
'## MC и неопределённость', '',
'Train MC: 40 семейств, 160 реальных seed-outcomes, 7 успехов всего на3 семействах. '
'OOF mean p=.02571 против .04375; Brier .04488 хуже constant .04253, '
'logloss .20964 хуже .18889. OOF AUROC для ever-success за4 seeds ≈.523; '
'это отдельный семейный диагностический target, не оценка AUROC истинного per-seed p. '
'Все OOF p_MC≤.287715; данных о высоком p_MC≥.95 вообще нет.', '',
'На validation 4/32 успеха на F0143, p_MC=.06777, самый высокий из8; '
'ever-success AUC=1 на единственном положительном семействе не подтверждает обобщение. '
'При условии истинности point p вероятность 4/4 всего ≈2.11e−5: '
'это описательное несоответствие прогнозу, не утверждение о p-value после множественных просмотров. '
'4/4 даёт индивидуальный CP95=[.397635,1], не 95% надёжность нового запуска. '
'Сохранённый seen-family F0047 аудит также менялся с train4/4 на fresh1/4; '
'он не добавлен в calibration как новое семейство.', '',
'Разброс10/90 десяти bootstrap-моделей описывает чувствительность к переобучающим семейным весам, '
'а не MC seed-noise, 95% CI или одновременную нижнюю границу p. '
'У него нет гарантии покрытия систематической ошибки или редкого неувиденного режима. '
'MC не выбран ни в одном исходном q, а F0143 имеет более быстрый точный R128; '
'ни одного реально пропущенного MC-winner среди этих8 не найдено. '
'Снижать q ради появления MC-действий научно необоснованно.', '',
'## Ошибки времени и их роль', '',
'| Профиль | OOF known / missing | OOF median APE | Validation median APE | Вал. cost target в model-spread |',
'|---|---:|---:|---:|---:|']
    for p in PROFILES:
        o=a['cohorts']['OOF'][p]['time'];v=a['cohorts']['validation'][p]['time']
        lines.append(f'| {p} | {o["known_cases"]} / {o["missing_cases"]} | {o["median_APE"]:.2%} | {v["median_APE"]:.2%} | {v["target_in_uncalibrated_model_spread"]}/8 |')
    lines += ['', 'R128 — константа, основанная на4 full-wall семействах; R256 OOF '
'median APE1.48% чуть хуже constant1.41%. MC OOF26.29% лучше constant51.29%, '
'validation14.31% лучше constant90.28%; на F0133 прогноз93.337 с вместо141.690 '
'(−34.13%), F0106 +42.53%. Bootstrap-time spread не является интервалом нового запуска: '
'даже family-median target попадает лишь в1/8 для R128/R256 и4/8 для MC. '
'Точность p/отказ, а не неправильный порядок времени R128<R256, объясняют потерю доставки. '
'Для будущей cost-модели полезно проверить аналитически доступные до solve G/T/N и стоимость AST, '
'сначала сопоставив контексты/цензурирование; ничего не добавлено в253 признака текущих моделей. '
'exp(predicted log time) оценивает центральную стоимость, не автоматически E[wall].', '',
'## Нужен ли q=.95 и сколько данных не хватает', '',
'q=.95 разумен только как заранее заданная цель риска, если допустимы ≤5% неточных '
'среди выданных ответов. Этот риск, минимальная coverage и цена отказа должны быть заданы '
'до оценки; сам порог не решает задачу калибровки. При одном нарушении из7 нет достаточной '
'мощности для точной оценки истинного риска; CP95 выбранной доли точности примерно [.421,.996] '
'содержит .95. Следовательно, 95% уровень не подтверждён, но это и не точное доказательство '
'его статистического нарушения в популяции. Порог оставлен неизменным; поиск нового q не проводился.', '',
'**Единого минимального числа calibration-семейств нет.** Для простого R-калибратора данные обоих '
'исходов уже есть (R12875/103, R25633/40), но полный R128 family-OOF сейчас только40, '
'а высоковероятностный хвост мал. Новая полная семейная OOF по103 R128 случаям/90 семействам '
'может быть получена без новых ОДУ-данных, но требует отдельного будущего ML-fit. '
'Калибратор может исправить уровень вероятностей, но не ошибочное ранжирование F0041/F0189. '
'MC-калибратор на3 ever-success семействах не обоснует высокий p и легко зависит от единственного случая.', '',
'Количественные ориентиры, не новые gates:', '',
'- До существующего инженерного флага R128-time≥8 не хватает4 full-wall **семейств**. '
'Это нижний программный порог, не доказательство достаточности regression/calibration; '
'нет смысла выдавать переключение флага за готовность production.',
'- Для односторонней95% биномиальной нижней границы точности≥.95 нужно **59/59 независимых '
'принятых семейных исходов** после freeze; при1 неуспехе требуется **92/93**. '
'Это единицы принятых семейных задач, не число seeds и не число train-строк; '
'OOF с общими обучающими выборками нельзя просто присоединить как независимые испытания. '
'При coverage .75 даже идеальные59 доставок означают в среднем около79 запрошенных семей, без гарантии успехов.',
'- Для инженерной разведки с10 MC-success семействами надо ещё минимум7 **положительных** новых семейств. '
'Если условно сохраняется частота ever-success3/40=.075, среднее ожидание —93 новых семьи; '
'вероятность получить≥7 в100 новых —63%, для80% нужно120. Частота имеет описательный '
'CP95=[.0157,.2039] и взята из stress-enriched набора, поэтому эти числа — сценарий, не прогноз. '
'10 не является новым обязательным gate; важны разнообразие реальных успехов/неуспехов и неопределённость.',
'- Среди140 original-train семей90 уже вошли хотя бы в одну fit/scaler голову; осталось50 '
'never-fit. Поэтому даже59 свежих принятых семейных испытаний нельзя получить из этого остатка '
'одним расширением текущего train, не меняя роль уже использованных семей. Нужен новый '
'заранее разделённый набор либо будущий отдельно проверенный нетронутый evaluation split; '
'test в этой задаче не читался и не использовался.', '',
'## Ограниченный следующий план', '',
'1. **Без новых solver-вызовов:** в отдельной задаче получить полный R128 family-OOF на существующих '
'103 случаях; заранее зафиксировать простой регуляризованный sigmoid-калибратор R, '
'проверять его через вложенные семейные разбиения. Не применять гибкий isotonic на столь малом хвосте '
'и не обещать исправление ранжирования одной калибровкой. Для MC сохранять оценку/неопределённость '
'и отсутствие подтверждения высокого p, не создавать псевдоуспехи.',
'2. **Минимальный ограниченный pilot для новых train-данных — предложено20 never-fit семейств**, '
'две заранее перечисленные группы по10. Это практический размер первого сбора, '
'не утверждение о статистически достаточной калибровке. По одному R128 и R256 для детерминированной '
'точности/full warm time и4 независимых MC1 seed: **120 вызовов максимум**. '
'Семейства/ID отобраны только по regime/complexity/фиксированному хешу, до чтения старых timing '
'для сметы; outcomes этих20 не использовались для выбора. '
'Профили и сетка остаются прежними; 1 R-time — допустимый train-target по прежнему протоколу, '
'но не median4 и не подтверждение observed-winner. При необходимости такого подтверждения '
'полные повторения R увеличат объём до240 вызовов и требуют отдельного бюджета.',
f'   Сценарий по сохранённым x0-временам/P2-factor: **{plan["central_scenario_seconds"]:.1f} с '
f'({plan["central_scenario_seconds"]/60:.2f} мин)**; предлагаемый отдельный cap '
f'**{plan["proposed_cap_seconds"]} с ({plan["proposed_cap_seconds"]/60:.2f} мин)**. '
'Это не верхняя оценка времени и не утверждённый бюджет. Существующий validation-budget не переносится. '
'Перед запуском заново проверить AST/hash/G/T/N, seeds, реальные max_path_steps, '
'guardian/thermal/RSS≤512/1worker и лимит; сохранить все неточные/неоконченные результаты. '
'Не расширять пакет автоматически ради нужного числа успехов. '
'По сценарию3/40 шанс хотя бы одного нового MC-success семейства в20 ≈79%, '
'а ожидаемо новых всего1.5: такой пакет, скорее всего, не решит проблему редких MC-режимов.',
'3. Использовать этот pilot как **development**, а не независимый отчёт об улучшенной модели. '
'Для новой версии fit p/time и калибровка — только family-OOF/nested cross-fit, '
'все семейства целиком в одной роли на каждой складке. Зафиксировать q, coverage-цель, '
'способ учёта отказа/цензурирования, секунды на точный ответ и правила MC до следующего audit. '
'Старые8 validation не использовать для выбора калибратора/порога/признаков; '
'они остаются раскрытой диагностикой, не независимым holdout.',
'4. **Остальные30 never-fit train-семейств зарезервировать до нового freeze** для prospective '
'внутреннего семейного audit (не финальный test). Их не включать в новые train/scaler/calibration '
'до этого audit. Первые метрики — точные доставки, нарушения среди выбранных, coverage, '
'стоимость на точный ответ, отдельно MC-выборы. 30 семейств не сертифицируют q=.95: '
'для строгого заявления потребуется рассчитанный дополнительный независимый объём и достаточное '
'число реальных MC-выборов. Если pilot снова не даёт поддержки MC или потенциального выигрыша, '
'остановить дорогое расширение и честно зафиксировать ограничение профиля MC1; '
'не подменять научную цель выбором только Remizov/отказ.', '',
'Полный список20 ID, seeds-предложение, работа/смета и неутверждённый бюджет — '
'[acquisition_plan.json](acquisition_plan.json). Объём/границы достаточности — '
'[sample_size.json](sample_size.json). Никакие новые расчёты, калибровка или обучение этим планом не запущены.', '',
'## Проверки и воспроизводимость', '',
'Источник: прежние immutable ML101, validation254 и preflight19 артефактов проверены по SHA-256; '
'сохранённые входы и старые артефакты не менялись. Дополнительно независимо пересчитаны96 E_grid '
'по сохранённым canonical exact-векторам,18 train-scaler means и cost-per-success всех исходных point q. '
'ML-fit/solver/test APIs не вызываются. См. [verification.json](verification.json), '
'[independent_verification.json](independent_verification.json), [source_sha256.json](source_sha256.json), '
'[tests.json](tests.json), [artifact_sha256.json](artifact_sha256.json).', '',
'Новые исходники diagnostics/audit_probability_selector.py, report_probability_selector_audit.py '
'и tests/test_probability_selector_audit.py. Скрипты принимают --out с новым каталогом; '
'отчётный скрипт запускается после audit в том же каталоге. Все результаты — saved-only; '
'прогнозы моделей не пересчитываются и калибраторы не подгоняются.', '']
    with (out/'REPORT.md').open('x') as f:f.write('\n'.join(lines))
    for name in ('diagnostics/report_probability_selector_audit.py',):
        dest=out/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    print(dict(passed=True,scalers=independent['OOF_scalers_checked'],errors=96,report=str(out/'REPORT.md')))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    report(parser.parse_args().out.resolve())
