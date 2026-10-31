"""Build a Russian research report from frozen saved results; no fits or tuning."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def f(value):
    return f'{value:.8g}'


def pct(value):
    return f'{100*value:.2f}%'


def rank(p):
    return (p['mean_regret'],-p['success_rate'],-p['macro_f1'],p['mc_predictions'],
            p['candidate_id'],p['policy'])


def report(out):
    validation=read(out/'validation_metrics.json')
    test=read(out/'test_metrics.json'); lock=read(out/'selection_lock.json')
    candidates=read(out/'candidates.json');by_id={c['id']:c for c in candidates}
    splits=read(out/'splits.json')['scenarios'];cv=read(out/'cv_summary.json')
    features=read(out/'feature_analysis.json');verification=read(out/'independent_verification.json')
    test_map={(p['candidate_id'],p['policy']):p for p in test['policies']}
    def best(scenario='new',kind=None,feature_set='X',weight=None,default=False):
        return min((p for p in validation if p['scenario']==scenario
                    and (kind is None or by_id[p['candidate_id']]['kind']==kind)
                    and (feature_set is None or by_id[p['candidate_id']]['feature_set']==feature_set)
                    and (weight is None or by_id[p['candidate_id']]['weight']==weight)
                    and (p['policy']=='default' if default else True)),key=rank)
    def test_for(p):
        return test_map[p['candidate_id'],p['policy']]
    def compact(p,label):
        return [label,f(p['mean_regret']),f(p['mean_error']),pct(p['accuracy']),f(p['macro_f1']),
                f(p['mc_precision']),f(p['mc_recall']),f(p['mc_f1']),
                p['mc_predictions'],p['mc_correct'],p['mc_false'],p['mc_missed'],
                f"{p['success_count']}/{p['n']}"]
    selected=lock['selected']['new']; selected_test=test['selected']['new']
    oldsel=lock['selected']['old'];oldtest=test['selected']['old']
    always=next(p for p in test['policies'] if p['candidate_id']=='new__always_remizov')
    basetest=test_map['old__X__classifier__w1__baseline','default']
    newbase=test_map['new__X__classifier__w1__baseline','default']
    header=['Вариант','Mean regret','Mean E','Accuracy','Macro-F1','MC precision','MC recall','MC F1',
            'Выбрано MC','Верно MC','Ложно MC','Пропущено MC','E≤.05']
    text=['# Пять гипотез выбора R128 / MC mean4 в x=0',
          '', 'Исследование 2026-10-31; ретроспективное и разведочное. Все тестовые результаты относятся к прежним 150 задачам, уже раскрытым в предыдущей работе. Это не новая независимая оценка.',
          '',f"**Заранее выбранный вариант нового split:** `{selected['candidate_id']}`, policy `{selected['policy']}`, порог score>{selected['threshold']:.12g}. Выбор сохранён в selection_lock.json до единой итоговой оценки Test. На Test обнаружено **{selected_test['mc_correct']}/9** побед MC; ошибочно выбран MC **{selected_test['mc_false']}** раз. Mean regret **{f(selected_test['mean_regret'])}**, у постоянного R **{f(always['mean_regret'])}**.",
          '', '## Неизменный контракт',
          '', '`E_R=abs(saved_R128(0)-exact(0))`; `E_MC=abs(mean(four saved MC1024 answers at 0)-exact(0))`. MC winner тогда и только тогда, когда E_MC<E_R; равенства относятся к R. Порог ε=.05 определяет success, а не winner. Скорость и E∞ на интервале в эту цель не входят. MC остаётся средним четырёх прежних независимых запусков M=1024, dt=1/128, а не новым одиночным MC4096.',
          '', 'Основной критерий выбора модели/порога — минимум среднего regret на Validation. На фиксированной выборке это тот же порядок, что минимум средней абсолютной ошибки выбранного метода. При равенстве: success, macro-F1, меньше выборов MC, стабильные candidate_id/policy. Внутри точного порогового скана при равенстве берётся больший порог; глобальная развязка между сохранёнными политиками использует их стабильные ID. Это может давать разные пороги с одинаковыми Validation-решениями и не использует Test. Постоянные R и MC также допущены к сравнению. Labels, исходные ошибки и численные методы не изменены. Новых уравнений, решений ОДУ и траекторий — **0**.',
          '', '## Данные и split', '']
    text.append(table(['Сценарий','Split','Задач','R winner','MC winner','Семейств'],
       [[scenario,s,splits[scenario][s]['n'],splits[scenario][s]['classes'].get('remizov',0),
         splits[scenario][s]['classes'].get('monte_carlo',0),splits[scenario][s]['families']]
        for scenario in ('old','new') for s in ('train','validation','test')]))
    text.extend(['','997 уникальных пригодных ID и expression_hash из 1000 исходных задач; 881 R winner и 116 MC winner. Три прежних отсутствующих R-ответа остаются без метки. Все 200 семейств имеют разные structure_signature; между split нет пересечений ID, семей или топологий. Test ID совпадают побайтно по списку. Seed=2488.',
      '', 'Новый Validation состоит из 20 полных семей по пять задач. Три семьи с четырьмя пригодными задачами остаются в Train, поэтому размеры 747/100 достигаются без разрыва групп. Среди 167 полных development-семей страты по числу побед MC: 0→91, 1→52, 2→18, 3→6; квоты Validation по правилу наибольших остатков: 11/6/2/1. Внутри страт — seeded permutation. Split выбран без прогноза моделей и без test outcomes.',
      '', '## Конфигурации и A/B: веса и пороги',
      '', 'CatBoost: baseline=(120 итераций, depth2, l2=10), medium=(240,depth3,l2=10), deeper=(300,depth4,l2=20). Общие lr=.05, seed2488, bootstrap=No, random_strength=0, один поток, без early stopping/resampling. Веса R=1, MC∈{1,2,3,5,8,12}. Все конфигурации обучены только на Train. Исходный baseline воспроизведён точно: максимальная разница score на прежних Train/Validation — 0.',
      '', 'Для каждой модели исследованы пороги .05… .50 с шагом .05 и все различные разбиения по validation scores, включая постоянные политики. Нижний порог сам по себе не является улучшением. В таблице для каждого веса приведена лучшая конфигурация original253 по Validation; Test — диагностический результат уже зафиксированного варианта.', ''])
    weightrows=[]
    for w in (1,2,3,5,8,12):
        p=best(kind='classifier',weight=w);t=test_for(p)
        weightrows.append([w,by_id[p['candidate_id']]['config']['name'],f(p['threshold']),
                           f(p['mean_regret']),f(p['mc_precision']),f(p['mc_recall']),f(p['mc_f1']),
                           f"{p['mc_correct']}/{p['mc_false']}",f(t['mean_regret']),f"{t['mc_correct']}/{t['mc_false']}"])
    text.append(table(['Вес MC','Конфигурация','Порог','Val regret','Val P','Val R','Val F1','Val TP/FP','Test regret','Test TP/FP'],weightrows))
    baseval=next(p for p in validation if p['candidate_id']=='new__X__classifier__w1__baseline' and p['policy']=='default')
    text.extend(['',f"Новый baseline при пороге .5: Val mean regret={f(baseval['mean_regret'])}, MC TP/FP={baseval['mc_correct']}/{baseval['mc_false']}. Поиск порога и веса сравниваются с этим исходным решением, а не с желаемым числом выборов MC.",
                 '', '## C/D: разность качества и признаки',
                 '', 'Регрессия использует Δ=E_R−E_MC: положительный знак означает MC winner. CatBoostRegressor RMSE обучен на Δ и отдельно на asinh(Δ/s), где s=median_train|Δ|; s вычисляется только на Train. Прогноз asinh обратно преобразован s·sinh(score). Это меняет обучающий loss, но сохраняет исходные ошибки/метки, знак и целевой regret. Проверены знак и Validation-порог в исходных единицах Δ.', ''])
    ablations=[]
    for kind in ('classifier','raw','asinh'):
        for fs in ('X','augmented'):
            p=best(kind=kind,feature_set=fs);t=test_for(p)
            ablations.append([kind,253 if fs=='X' else 258,by_id[p['candidate_id']]['config']['name'],
                by_id[p['candidate_id']]['weight'],f(p['threshold']),f(p['mean_regret']),
                f"{p['mc_correct']}/{p['mc_false']}",f(t['mean_regret']),f"{t['mc_correct']}/{t['mc_false']}"])
    text.append(table(['Модель','Признаков','Config','Вес MC','Порог','Val regret','Val TP/FP','Test regret','Test TP/FP'],ablations))
    text.extend(['', 'Исходные 253 признака используют только заданные a,b,c,g, аналитические сертификаты и AST коэффициентов a,b,c. Нет y/exact, численных ответов, ошибок, success, фактического времени, ID или split в матрице модели. Правая часть g доступна до решения и допустима как вход.',
      '', 'Пять новых признаков описаны формулами в ml/selector_research_features.py: локальное отношение дрейфа к диффузии/реакции; относительная кривизна forcing; изменчивость диффузии; компенсация интеграла forcing; дифференциальный дефект реакционного приближения u=g_R/(λ−c_R). Это сеточные описания на [-1,1], без solver/exact, без глобальных сертификатов. Они переупаковывают доступный сигнал; новых независимых измерений не добавляют.', ''])
    top=features['strongest_descriptive_differences']['new_train'][:8]
    text.append(table(['Признак','Train median R','Train median MC','Одномерный AUC'],
                      [[p['feature'],f(p['R_median']),f(p['MC_median']),f(p['AUC'])] for p in top]))
    with (out/'feature_importance.csv').open(newline='') as stream:
        important=sorted([p for p in csv.DictReader(stream) if p['candidate_id']==selected['candidate_id']],key=lambda p:-float(p['importance']))[:10]
    text.extend(['','Важнейшие признаки выбранной модели (CatBoost PredictionValuesChange): '+', '.join(f"`{p['feature']}`={float(p['importance']):.3g}" for p in important)+'. Это описательная важность, без причинной интерпретации.',
      '', 'Полные распределения, квартили и AUC сохранены отдельно для каждого Train/Validation. Исследованы только реальные MC-wins внутри development. SHAP не понадобился для этого сравнительного аудита. Сравнения 258 признаков и корреляция семей ограничивают интерпретацию отдельных высоких AUC.',
      '', '## Достаточность данных, CV и калибровка', ''])
    text.append(table(['Сценарий','Split','MC задач','Семейств с MC'],
                      [[p['scenario'],p['split'],p['classes'].get('monte_carlo',0),p['mc_families']] for p in features['partitions']]))
    cvrows=[]
    for p in cv['policy_summary']:
        if p['policy'] in ('classifier_w1','classifier_w5','raw','asinh','always_remizov','nested_selection'):
            var=p['fold_variability'];pool=p['pooled']
            cvrows.append([p['scenario'],p['policy'],f(pool['mean_regret']),
                f"{f(var['mean_regret']['mean'])} ± {f(var['mean_regret']['sd'])}",
                f"{f(var['mc_recall']['mean'])} ± {f(var['mc_recall']['sd'])}",
                f"{f(var['mc_recall']['minimum'])}…{f(var['mc_recall']['maximum'])}",
                f"{pool['mc_correct']}/{pool['mc_false']}"])
    text.append('')
    text.append(table(['Сценарий','CV политика','OOF mean regret','Fold regret mean±SD','Fold recall mean±SD','Recall диапазон','OOF TP/FP'],cvrows))
    text.extend(['', 'CV: пять StratifiedGroupKFold внутри Train каждого сценария. Для nested_selection выбор модели и порога выполнен на отдельном внутреннем семейном holdout размером около четверти outer Train; для выбранной спецификации использован уже выполненный fit на всём outer Train. Внешние метки не участвуют в подборе. Восемь baseline-конфигураций: шесть весов классификатора, raw и asinh; плюс две постоянные политики. Средний fold regret и pooled OOF regret отличаются из-за неодинаковых размеров фолдов. Это проверка ограниченной baseline-сетки, а не внешняя оценка всех 96 кандидатов.',
      '', f"MC-support внешних фолдов: old={cv['outer_MC_supports']['old']}, new={cv['outer_MC_supports']['new']}. На Validation всего 13 либо 18 MC-побед, в 9 либо 11 семьях. Одно обнаружение меняет recall нового Validation на 7.69 п.п., Test — на 11.11 п.п. Это мало для устойчивого выбора среди сотен порогов и моделей. Пять коррелированных CV-фолдов не дают независимых повторов исследования.", ''])
    cal=read(out/'calibration.json')
    selected_cal=next((p for p in cal if p['candidate_id']==selected['candidate_id']),None)
    if selected_cal:
        text.append(f"Выбранная модель на Validation: Brier={f(selected_cal['brier'])}, logloss={f(selected_cal['logloss'])}, ECE десяти bins={f(selected_cal['ece'])}. Это описательная проверка при малой поддержке, без калибровочного fit. Взвешенные scores не трактуются как калиброванные вероятности. Полные bins всех классификаторов — calibration.json.")
    text.extend(['','## E и единая итоговая оценка Test',
      '', 'Модель нового split выбрана только по его Validation. Отдельная модель старого split выбрана по прежнему Validation для сравнения сценариев. Значения Validation разных split напрямую не ранжируются между собой. Ни один тестовый вариант не использован для последующего выбора модели, порога или признаков.', ''])
    testrows=[compact(basetest,'Старый baseline697, порог .5'),compact(newbase,'Новый baseline747, порог .5'),
              compact(always,'Всегда R'),compact(oldtest,'Выбран old697/150'),compact(selected_test,'Выбран new747/100'),
              compact(next(p for p in test['policies'] if p['candidate_id']=='new__always_monte_carlo'),'Всегда MC'),
              compact(test['oracle'],'Oracle')]
    text.append(table(header,testrows))
    text.extend(['', f"Выбран old: `{oldsel['candidate_id']}` / `{oldsel['policy']}` / threshold={oldsel['threshold']:.12g}.",
      '', f"Confusion matrix выбранной new-модели (строки факт R,MC; столбцы выбор R,MC): **{selected_test['confusion_matrix']}**. Выборов MC **{selected_test['mc_predictions']}**, правильных **{selected_test['mc_correct']}**, ложных **{selected_test['mc_false']}**, пропущено **{selected_test['mc_missed']}/9**. Δmean regret относительно постоянного R = **{f(selected_test['mean_regret']-always['mean_regret'])}**; отрицательное значение — улучшение.",
      '', '## Что помогло и чего результат не доказывает', ''])
    verdict=[]
    baseline_opt=min((p for p in validation if p['candidate_id']=='new__X__classifier__w1__baseline'),key=rank)
    for label,p in [('A: веса (включая вес1)',best(kind='classifier')),('B: порог baseline747',baseline_opt),
         ('C: raw Δ',best(kind='raw')),('C: asinh Δ',best(kind='asinh')),
         ('D: пять признаков',best(kind='classifier',feature_set='augmented'))]:
        t=test_for(p)
        verdict.append([label,f(p['mean_regret']),f(t['mean_regret']),f"{t['mc_correct']}/9",t['mc_false']])
    text.append(table(['Изменение; выбор только по Val','Val regret','Test regret (диагностика)','Test найдено MC','Test ложных MC'],verdict))
    text.extend(['', 'У пяти дополнительных признаков есть сильный **диагностический** результат: заранее зафиксированный augmented-классификатор с отдельным Validation-порогом обнаружил 6/9 MC-побед, допустил 14 ложных MC-выборов и получил Test mean regret≈0.001708. Он проиграл глобальному победителю на Validation (0.000876 против 0.000710), поэтому не объявляется выбранной моделью после раскрытия Test. Это перспективная гипотеза для нового независимого набора; переносимость улучшения и отдельный вклад каждого из пяти признаков не установлены. Веса также дают отдельные удачные тестовые варианты, но не устойчивое преимущество выбора по Validation.',
      '', 'Подтверждено: R выигрывает чаще по метке меньшей ошибки, а редкие крупные R-ошибки могут доминировать в средней E/regret. Поэтому высокая winner-accuracy и число предсказаний MC не равнозначны хорошему выбору по численной ошибке. В исходном input-only пространстве классы заметно перекрываются; признаки дрейфа дают описательный сигнал, но не сертификат отказа R. Стохастический шум конечного MC-агрегата — возможный источник нестабильности меток; его вклад этим опытом отдельно не установлен.',
      '', 'Рекомендация: не расширять слепой поиск гиперпараметров на этих же 997 задачах. Проверять найденную заранее фиксированную политику на новых независимых семействах с большим числом естественных MC-побед и сохранением исходного контракта. Текущие Validation/Test малочисленны по MC, а известный Test не может стать новой проспективной проверкой. Если приоритет — защита от редких больших ошибок, нужны дополнительные независимые данные и проверка такого поведения; менять winner или добавлять ручное правило по известным test cases нельзя.',
      '', '## Файлы и воспроизведение',
      '', 'plan.json — критерии и сетка; splits.json — все устойчивые ID и группы; features.npz — обе матрицы; development_targets.csv — Δ и исходные метки; candidates.json/models/ — 96 моделей; thresholds/ — все validation partitions; validation_metrics.csv и test_metrics.csv — все политики; validation_predictions.csv и test_predictions.csv — пообъектные predictions; calibration.json — проверка scores; feature_* — распределения/важности; cv_* — folds/OOF/внутренний выбор; selection_lock.json — выбор до Test; baseline_independent_reproduction.json — точное воспроизведение; independent_verification.json и preserved_sha256.json — независимый пересчёт и сохранность источников.',
      '', f"Основных fit **96**, CV fit **{cv['fit_count']}**, независимый baseline audit fit **1**: всего **{97+cv['fit_count']}**. Каждый fit CatBoost и BLAS однопоточный. Независимая проверка: `{verification.get('status','see JSON')}`. Исходные данные, прежние benchmark/модели и математические реализации сохранены. Оптимальность main Validation-порогов независимо пересчитана по scores; оптимальность внутренних CV-порогов отдельно не пересчитана, поскольку inner scores не сохранялись. Для CV подтверждены ranking сохранённых кандидатов, метрики и отсутствие утечек; внутренний подбор воспроизводится из кода и seed.",
      '', 'Запуск в новый каталог (существующий каталог по протоколу не перезаписывается):',
      '', '```bash',
      'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/run_selector_hypotheses.py prepare --out results/<new_run>',
      'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/analyze_selector_hypotheses.py features --out results/<new_run>',
      'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/run_selector_hypotheses.py train --out results/<new_run>',
      'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/analyze_selector_hypotheses.py cv --out results/<new_run>',
      'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/run_selector_hypotheses.py final --out results/<new_run>',
      'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/verify_selector_hypotheses.py --out results/<new_run>',
      'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/report_selector_hypotheses.py --out results/<new_run>',
      '```'])
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    allrows=[]
    for c in candidates:
        own=[p for p in validation if p['candidate_id']==c['id'] and p['policy'] in ('default','validation_optimal','constant')]
        for p in own:
            allrows.append([c['id'],p['policy'],f(p['threshold']),f(p['mean_regret']),f(p['mean_error']),
                           f(p['macro_f1']),pct(p['accuracy']),f(p['mc_precision']),f(p['mc_recall']),
                           f(p['mc_f1']),p['mc_predictions'],p['mc_correct'],p['mc_false'],p['confusion_matrix']])
    (out/'ALL_EXPERIMENTS.md').write_text('# Все конфигурации: Validation\n\n'+
        table(['ID','Policy','Порог','Regret','Mean E','MacroF1','Accuracy','MC P','MC R','MC F1','MC n','TP','FP','CM'],allrows)+'\n')
    print('REPORT',out/'REPORT.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    report(parser.parse_args().out.resolve())
