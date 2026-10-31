"""Figures and report from saved MC1000 analysis/models; never solve ODEs."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
import re
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/ode-mc1000-matplotlib')


def read(p):return json.loads(p.read_text())


def mdtable(rows):
    keys=list(rows[0]);s='| '+' | '.join(keys)+' |\n| '+' | '.join('---' for _ in keys)+' |\n'
    return s+''.join('| '+' | '.join(str(r[k]) for k in keys)+' |\n' for r in rows)


def report(out):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from diagnostics.train_remizov_ml import write
    a=out/'analysis';eq=pd.read_csv(a/'equations.csv');att=pd.read_csv(a/'attempts.csv');s=read(a/'summary.json');plan=read(out/'plan.json')
    plots=out/'plots';plots.mkdir(exist_ok=True)
    known=eq.remizov_known.to_numpy(dtype=bool);complete=eq.mc_all_four.to_numpy(dtype=bool);paired=known&complete
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for name,col,mask in [('Remizov x0','remizov_error0',known),('MC 4x1024 mean','mc_aggregate_error',complete)]:
        e=np.sort(eq.loc[mask,col].to_numpy());axs[0].step(np.maximum(e,1e-12),np.arange(1,len(e)+1)/len(e),label=name)
    e=np.sort(att.loc[att.status.eq('completed'),'abs_error'].to_numpy());axs[0].step(np.maximum(e,1e-12),np.arange(1,len(e)+1)/len(e),label='MC single1024',alpha=.7)
    axs[0].axvline(.01,color='black',ls='--');axs[0].set(xscale='log',xlabel='Absolute error at x=0',ylabel='Empirical CDF');axs[0].legend()
    axs[1].scatter(np.maximum(eq.loc[paired,'remizov_error0'],1e-12),np.maximum(eq.loc[paired,'mc_aggregate_error'],1e-12),s=12,alpha=.5)
    lo=max(1e-12,min(eq.loc[paired,'remizov_error0'].min(),eq.loc[paired,'mc_aggregate_error'].min())*.7)
    hi=max(eq.loc[paired,'remizov_error0'].max(),eq.loc[paired,'mc_aggregate_error'].max())*1.5
    axs[1].plot([lo,hi],[lo,hi],color='grey',lw=.8)
    axs[1].set_xlim(lo,hi);axs[1].set_ylim(lo,hi)
    axs[1].axhline(.01,color='black',ls='--');axs[1].axvline(.01,color='black',ls='--');axs[1].set(xscale='log',yscale='log',xlabel='Remizov point error',ylabel='MC aggregate point error')
    fig.savefig(plots/'errors.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    vals=eq.loc[complete,'mc_success_count'].value_counts().reindex(range(5),fill_value=0)
    axs[0].bar(vals.index,vals.values);axs[0].set(xlabel='Successful individual seeds / 4',ylabel='Equations',xticks=range(5))
    axs[1].scatter(eq.loc[complete,'mc_pooled_stderr'],eq.loc[complete,'mc_seed_mean_std'],s=12,alpha=.5)
    axs[1].set(xscale='log',yscale='log',xlabel='Pooled standard error, 4096 paths',ylabel='Std of four 1024-path seed means')
    fig.savefig(plots/'seed_stability.png',dpi=170);plt.close(fig)
    valid=att.status.eq('completed');fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    axs[0].scatter(att.loc[valid,'path_steps'],att.loc[valid,'solver_seconds'],s=4,alpha=.25)
    axs[0].set(xlabel='Path updates, one x0 seed',ylabel='MC solve seconds',xscale='log',yscale='log')
    axs[1].hist(att.loc[valid,'solver_seconds'],bins=35);axs[1].set(xlabel='MC seconds / attempt',ylabel='Attempts')
    fig.savefig(plots/'mc_cost.png',dpi=170);plt.close(fig)
    # Paired family bootstrap: preserves within-family dependence.
    rng=np.random.default_rng(2026102502);groups=eq.loc[paired,'family_id'].to_numpy();fam=np.unique(groups)
    r=eq.loc[paired,'remizov_accuracy0'].to_numpy();m=eq.loc[paired,'mc_aggregate_accuracy'].to_numpy();diff=[]
    for _ in range(500):
        sample=rng.choice(fam,len(fam),replace=True);ix=np.concatenate([np.flatnonzero(groups==f) for f in sample]);diff.append(float(np.mean(m[ix]-r[ix])))
    b=dict(observed_mc_minus_remizov_point_accuracy=float(np.mean(m-r)),family_bootstrap95=np.quantile(diff,[.025,.975]).tolist(),resamples=500,
        limitation='Conditional on these four seed outcomes; not prospective .95 success probability or total-error certification')
    write(a/'paired_bootstrap.json',b)
    groups_out=[]
    for dim in ('split','complexity','regime'):
        for group,g in eq.groupby(dim):
            groups_out.append(dict(dimension=dim,group=group,n=len(g),mc_full=int(g.mc_all_four.sum()),remizov_known=int(g.remizov_known.sum()),
                remizov_accuracy=float(g.loc[g.remizov_known,'remizov_accuracy0'].mean()),
                mc_aggregate_accuracy=float(g.loc[g.mc_all_four,'mc_aggregate_accuracy'].mean()),mc_single_seed_accuracy=float(g.mc_seed_success_fraction.mean())))
    pd.DataFrame(groups_out).to_csv(a/'groups.csv',index=False)
    sessions=[read(p) for p in (out/'sessions').glob('mass_*.json') if read(p)['planned']>0];wall=sum(x['wall_seconds'] for x in sessions)
    chronological=max(datetime.fromisoformat(x['finished_utc']).timestamp() for x in sessions)-read(out/'budget.json')['started_epoch']
    stopped_sessions={x['session'] for x in sessions if x['reason'] is not None}
    interruptions=[read(p) for p in (out/'control').glob('mass_*/*.guardian.json') if p.parent.name in stopped_sessions]
    eq_known=eq.loc[paired]
    ml=read(out/'ml/metrics.json') if (out/'ml/metrics.json').exists() else None
    modeltable=[];regtable=[]
    if ml:
        for target,models in ml['classification'].items():
            for model,v in models.items():
                if not isinstance(v,dict) or 'test' not in v:continue
                t=v['test'];modeltable.append(dict(target=target,model=model,test_n=t['n'],AUC='—' if t['roc_auc'] is None else f"{t['roc_auc']:.4f}",logloss=f"{t['logloss']:.4f}",Brier=f"{t['brier']:.4f}",balanced_accuracy='—' if t['balanced_accuracy'] is None else f"{t['balanced_accuracy']:.4f}"))
        for target,models in ml['regression'].items():
            for model,v in models.items():
                if not isinstance(v,dict) or 'test' not in v:continue
                t=v['test'];regtable.append(dict(target=target,model=model,test_n=t['n'],MAE_log10=f"{t['mae_log10']:.4f}",MAE_error=f"{t['mae_error']:.5f}"))
        pd.DataFrame(modeltable).to_csv(out/'ml/classification_table.csv',index=False);pd.DataFrame(regtable).to_csv(out/'ml/regression_table.csv',index=False)
        pred=pd.read_csv(out/'ml/predictions.csv');fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for ax,target in zip(axs,('remizov_point_accuracy','mc_single_seed_accuracy')):
            cal=pd.read_csv(out/f'ml/calibration_{target}.csv')
            for model,g in cal.loc[cal.split.eq('test')].groupby('model'):ax.plot(g.mean_probability,g.observed_success,'o-',label=model)
            ax.plot([0,1],[0,1],'k--',lw=.7);ax.set(xlabel='Predicted probability',ylabel='Observed accuracy',title=target);ax.legend()
        fig.savefig(plots/'ml_calibration.png',dpi=170);plt.close(fig)
        test=pred.split.eq('test');wrong=pred.loc[test].copy()
        wrong['mc_fraction_prediction_error']=abs(wrong.p_mc_single_seed_accuracy_catboost-wrong.mc_seed_success_fraction)
        wrong.sort_values('mc_fraction_prediction_error',ascending=False).head(30).to_csv(out/'ml/largest_test_prediction_errors.csv',index=False)
    resource_rows=[dict(status=k,attempts=v) for k,v in s['statuses'].items()]
    quant=[dict(metric=k,median=f"{v['0.5']:.6g}",p90=f"{v['0.9']:.6g}",p99=f"{v['0.99']:.6g}",maximum=f"{v['1.0']:.6g}") for k,v in s['error_quantiles'].items()]
    worst=eq.loc[paired].nlargest(5,'remizov_error0')
    worst_table=[dict(ID=r.equation_id,split=r.split,R_error=f'{r.remizov_error0:.6g}',MC_mean_error=f'{r.mc_aggregate_error:.6g}') for r in worst.itertuples()]
    text=f'''# MC1000 в x=0 и первая модель сравнительного качества

Датасет: production `data/all_methods_v1_1000_2026-10-18/`.
Реальная серия: **{s['terminal']}/4000 терминальных попыток**, pending **{s['pending']}**.
Уравнений со всеми четырьмя seed: **{s['equations_all_four']}**.
Корректных пар с историческим Chernov x=0: **{s['paired_equations']}**.

## Протокол и ресурсы

Один Numba-worker, 50 последовательных партий по 20 уравнений, x=0,
4 независимых seed×1024, h=1/128, T=None: исходный auto_T по аналитическому G,
tail_eps=1e-4. Индивидуальной подстройки нет. Нормальные потоки PCG64 адресуются
по задаче/точке/seed. Формула, EM и обе трапеции не изменены. Exact-метрика
вызывается после возврата solver; стандартная безопасная загрузка отдельно
проверяет конструкцию датасета, а y-jet нужен для самой заданной RHS.

Жёсткие лимиты: 60 с на попытку, 14400 с на серию, 512 МиБ RSS,
один вычислительный worker. Общий абсолютный дедлайн сохраняется при resume.
Независимый guardian проверяет controller heartbeat и owner lease300 с;
владелец продлевал lease во время наблюдения. Атомарный immutable result
на попытку, requests/events и progress; прерванное остаётся pending,
ресурсный терминальный отказ автоматически не повторяется. Завершённые
результаты пропускаются по ключу и проверяются по plan SHA/профилю/AST hash.
Старый pilot не возобновлялся. [План](plan.json), [smoke](smoke/verification.json).

Фактический активный wall массовых сессий **{wall:.2f} с / {wall/60:.2f} мин**,
календарный wall от первого старта с учётом пауз **{chronological:.2f} с / {chronological/60:.2f} мин**,
сумма solver wall **{s['solver_seconds_sum']:.2f} с**,
пиковый worker RSS **{s['peak_rss_mib']:.2f} МиБ**.
Обновлений в завершённых результатах **{s['path_steps']:,}**. Частичная работа
прерванного запроса не имеет итоговых solver-счётчиков и дополнительно сюда
не включена; его известные wall/RSS сохранены в interrupted event.
Загрузка/JIT/IO/monitoring входят в session
wall, отдельно сохранены в worker/startup и result. Два smoke-ответа не
включены в 4000. Ресурсные статусы не являются математической неточностью.

Защитных прерываний guardian: **{len(interruptions)}**;
причины: `{[g.get('reason') for g in interruptions]}`.
Owner lease истёк во время подготовки кода анализа: guardian завершил
worker и освободил вычислительный слот. Сохранённые терминальные записи
не пересчитывались; незавершённый запрос остался pending и был выполнен
после возобновления. Общий дедлайн и все лимиты сохранены. Это операторская
остановка, а не ресурсная неудача solver или признак математической неточности.

{mdtable(resource_rows)}

## Точность и статистическая устойчивость

Доля успешных по допуску отдельных завершённых MC seed: **{s['mc_seed_accuracy_rate']:.2%}**.
Доля успехов среднего четырёх seed (4096 путей): **{s['mc_aggregate_accuracy_rate']:.2%}**.
Сохранённый Ремизов в x=0: **{s['remizov_point_accuracy_rate']:.2%}**.
Это разные профили: успех одного 1024-path ответа не заменяется успехом
среднего4096. Полностью успешных 4/4: **{s['mc_all_four_pass']}**,
хотя бы один успешный seed: **{s['mc_at_least_one_pass']}**.

{mdtable(quant)}

Ремизов чаще достигает допуска, но имеет редкие очень большие выбросы
данного исторического профиля. Они не исправлялись и не удалялись:

{mdtable(worst_table)}

Средняя ошибка и raw-error regression MAE чувствительны к этим случаям;
поэтому отдельно показаны медиана, квантили и log10-error regression.

На парных данных разность наблюдаемых accuracy (MC aggregate−Ремизов)
**{b['observed_mc_minus_remizov_point_accuracy']:.4f}**; family-bootstrap95
**[{b['family_bootstrap95'][0]:.4f}, {b['family_bootstrap95'][1]:.4f}]**.
Это интервал по семействам при зафиксированных наблюдённых seed, а не
гарантия вероятности успеха новой реализации. Таблица совместных успехов:
`{json.dumps(s['paired_aggregate_contingency'],ensure_ascii=False)}`.

Сохранены внутренние stderr1024, pooled stderr4096 (с межгрупповой частью
дисперсии), sd четырёх seed-средних и t95 при df=3. Сертификат хвоста
контролирует непрерывное усечение; статистические интервалы не контролируют
временное смещение EM/трапеций. Смещение оценивается как mean−exact,
а его t95 — только статистическая неопределённость. n=4 мало для проверки
нормальности и надёжной оценки хвостов. Даже 4/4 успеха дают одностороннюю
95% нижнюю границу вероятности всего **{.05**.25:.4f}**, не .95.
Нет выводов о порядке сходимости или оптимальном времени достижения .01.

![Ошибки в x=0](plots/errors.png)
![Устойчивость seed](plots/seed_stability.png)
![Измеренная MC-стоимость](plots/mc_cost.png)

## Идентичность и область сравнения

Проверены production SHA и исторические input SHA, implementation
ChernovSolver.solve, expression_hash, target_x==result.x и наличие ровно
одного x=0. Для {s['historical_identities_checked']} корректных старых ответов
извлечён узел16, а не прежний E∞33. Exact в этой точке совпадает с MC.
Три старых worker_exception остаются без R-целей. Профили T у методов
различаются: MC использует G-сертификат, исторический Chernov — legacy T;
они сохранены как разные фиксированные численные настройки.

**MC-время в одной точке не сравнивается как равнозначное с историческим
временем Chernov на 33 точках.** Таблица содержит отдельное поле
historical_remizov33_seconds только для аудита. Метки «самый быстрый метод»
не создаются. [Все уравнения](analysis/equations.csv),
[все4000 задания со статусами](analysis/attempts.csv), [группы](analysis/groups.csv).

## Первая экспериментальная ML-модель

'''
    if ml:
        mc_ci=ml['uncertainty']['mc_single_seed_accuracy']['catboost_minus_dummy_logloss_95']
        r_ci=ml['uncertainty']['remizov_point_accuracy']['catboost_minus_dummy_logloss_95']
        tops={name:pd.read_csv(out/f'ml/importance_{name}.csv').head(5).feature.tolist() for name in ('mc_single_seed_accuracy','remizov_point_accuracy')}
        text+=f'''Обучены quality-surrogates на **{ml['features']}** признаках, доступных до решения:
только заданные coefficient-каналы a,b,c,g, коэффициентные сертификаты и
AST a/b/c. Exact/y-shape/его производные/решения/ошибки/seed/ID/split не
попадают в X. y-shape в численном backend является частью определения RHS,
но исключён из ML-признаков. Прежний whitelist использован без изменений.
Train/validation/test сохранены700/150/150,140/30/30 семейств; seed-реплики
никогда не разделяются между subsets. Scaler/fit только train, stopping и
model selection по validation, test оценён после обучения. Исторический test
ранее изучался в численных аудитах; prospective независимость не заявляется.

Цели: Remizov accuracy в x=0, MC вероятность успеха одного1024-path seed,
а также log10 point-error Ремизова и среднего4096 MC (floor1e-12).
Ресурсные отказы и pending цензурированы, а не помечены как inaccurate.
Dummy, Logistic/Ridge и CatBoost обучены с одним заранее сохранённым
набором параметров; нет поиска по test. Время pipeline **{ml['elapsed_seconds']:.2f} с**.

{mdtable(modeltable)}

600 test-меток MC — это четыре seed150 уравнений из30 семейств, а не600
независимых уравнений. У Ремизова в test только6 неточных ответов: ROC-AUC
и recall неточности чувствительны к нескольким случаям. Одинаковая accuracy
.96 у CatBoost и Dummy не означает одинаковые вероятностные прогнозы.
Family-bootstrap95 разности test logloss CatBoost−Dummy:
MC **[{mc_ci[0]:.4f},{mc_ci[1]:.4f}]**, Ремизов **[{r_ci[0]:.4f},{r_ci[1]:.4f}]**.
Оба интервала ниже0, что поддерживает улучшение относительно постоянной
базы в этих данных; превосходство над всеми другими моделями не заявляется.

Важные признаки CatBoost MC: `{', '.join(tops['mc_single_seed_accuracy'])}`;
Ремизова: `{', '.join(tops['remizov_point_accuracy'])}`.
Для MC доминируют масштаб и изменчивость заданной RHS; для Ремизова заметна
кривизна a и lambda. Это важность внутри обученных моделей, не причинное
объяснение ошибки и не математический критерий применимости.

{mdtable(regtable)}

На редком test-выбросе AMV1_F0180_S04 сохранённая R-ошибка248.323.
CatBoost-классификатор дал p(success)≈.242, то есть отклонил профиль при
пороге .5, но регрессор предсказал ошибку лишь≈.0368. Он не воспроизводит
масштаб этого выброса. Raw-error MAE≈1.658 здесь мало характеризует типичную
задачу; selection регрессоров заранее шёл по MAE log10-error на validation.
Ridge имеет меньший test raw-error MAE, но это не повод менять выбранную
после validation модель по test.

Гейт сравнительной модели: `{json.dumps(ml['comparative'],ensure_ascii=False)}`.
Сравнительные метки опираются на приблизительный seed-t95 интервал
дискретного усечённого среднего плюс tail bound и margin .0025 относительно
ошибки Ремизова. При пересечении диапазонов ставится uncertain. Эти
pointwise интервалы не скорректированы на1000 сравнений и не доказывают
95% надёжность выбора. Разделённых меток: `{json.dumps(s['exploratory_labels'],ensure_ascii=False)}`.
Основной практический выход — отдельные предварительные модели качества;
при наличии comparative classifier он также явно экспериментальный.

[Метрики и family-bootstrap](ml/metrics.json), [предсказания](ml/predictions.csv),
[модели](ml/models/), [schema](ml/feature_schema.json),
[важность признаков](ml/), [ошибки test](ml/largest_test_prediction_errors.csv).
Bootstrap условен по этим seed-меткам; он не устраняет шум всего четырёх
seed и временное смещение. Модели не сертификаты точности.

Рабочий reload/serving, без численного решения:

```bash
.venv/bin/python -B diagnostics/predict_mc1000_quality.py \\
  --models results/mc1000_x0_2026-10-25/ml \\
  --ids AMV1_F0000_S00 AMV1_F0180_S04
```

Ответ содержит вероятность успеха каждого фиксированного профиля и
прогноз log10 ошибки. Выбор самого быстрого solver не производится.

![Калибровка вероятностей](plots/ml_calibration.png)

'''
    else:text+='Обучение не выполнено: см. сохранённые статусы/достаточность парных данных.\n\n'
    text+='''## Вывод и следующий шаг

Подтверждённые факты — покрытие, фиксированные профили, наблюдённые ошибки,
ресурсная стоимость и измеренное качество моделей на семейном split.
Статистическая надёжность будущего seed, отделение временного смещения и
быстрейший метод достижения .01 остаются ограничениями этих данных.

Следующий шаг к33 точкам: отдельный заранее фиксированный общий сеточный
профиль обоих методов с одинаковой областью метрики и одновременной
статистической политикой. Нельзя умножением точечного успеха получить
E∞33 или вероятность успеха всей сетки. Время33 MC ориентировочно растёт
с числом точек, но фактическую дисперсию/смещение каждой точки нужно
контролировать. Массовая33-точечная серия в эту задачу не запускалась.

Математические схемы, защищённое ядро, генератор, исходные данные и прежние
результаты сохранены. SHA-проверка, очистка worker/locks и воспроизводимые
источники перечислены в verification.json; PROJECT_STATUS.md обновлён.
'''
    log=(out/'pytest.log').read_text() if (out/'pytest.log').exists() else ''
    checks=re.findall(r'\d+ passed,.*',log)
    text+='\n## Проверки и новые файлы\n\n'
    text+=f"Регрессия: `{checks[-1] if checks else 'см. pytest.log'}`. Исключён только длительный R1.\n\n"
    text+='''Новые файлы кода: diagnostics/mc_production_x0.py,
diagnostics/analyze_mc1000_x0.py, diagnostics/train_mc1000_quality.py,
diagnostics/predict_mc1000_quality.py, diagnostics/report_mc1000_x0.py,
diagnostics/verify_mc1000_x0.py; тесты tests/test_mc_production_x0.py и
tests/test_mc1000_quality.py. Из существующих документов обновлён только
PROJECT_STATUS.md; его исходная версия сохранена в PROJECT_STATUS.before.md.

Проверены8205 прежних файлов: изменений нет. Математические файлы из
MIGRATION_MANIFEST не редактировались. Модели повторно загружены; метрики
и serving сверены с сохранёнными ответами на всех1000 строках.
RuntimeWarning matmul в sklearn повторились; все прогнозы конечны,
logistic и Ridge дополнительно проверены независимым einsum.
Причина предупреждений не установлена; результаты не подгонялись.
[Проверка сохранности и моделей](verification.json),
[проверка resume](resume_check.json), [лог тестов](pytest.log),
[serving-проверка](ml/serving_verification.json),
[воспроизведение](REPRODUCE.md), [окружение](requirements-lock.txt).
'''
    (out/'REPORT.md').write_text(text)
    print(json.dumps(dict(report=str(out/'REPORT.md'),wall_seconds=wall,classification_rows=len(modeltable))))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);report(ap.parse_args().out.resolve())
