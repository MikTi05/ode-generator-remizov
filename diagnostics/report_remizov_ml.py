"""Report the saved first ML run; no fitting and no numerical solver calls."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'


def read(p):return json.loads(p.read_text())


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',required=True,type=Path)
    out=ap.parse_args().out.resolve()
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_curve,precision_recall_curve,mean_absolute_error,roc_auc_score
    os.environ['MPLCONFIGDIR']=str(out/'mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics=read(out/'metrics.json');meta=read(out/'models/metadata.json');audit=read(out/'data_audit.json')
    summary=read(out/'run_summary.json');boot=read(out/'test_family_bootstrap.json')
    table=pd.read_csv(out/'predictions.csv');test=table[(table.split=='test')&table.target_known].copy()
    selected_c=summary['selected_classifier'];selected_r=summary['selected_regressor']
    importance=pd.read_csv(out/'feature_importance.csv');perm=pd.read_csv(out/'validation_group_permutation.csv')
    # Paired held-out family uncertainty for small apparent differences.
    fam=test.family_id.to_numpy();groups=np.unique(fam);rng=np.random.default_rng(20261028);differences=[]
    for _ in range(500):
        ii=np.concatenate([np.flatnonzero(fam==g) for g in rng.choice(groups,len(groups),replace=True)])
        sub=test.iloc[ii]
        if sub.accuracy_met.nunique()<2:continue
        differences.append([roc_auc_score(sub.accuracy_met,sub.p_success_catboost)-roc_auc_score(sub.accuracy_met,sub.p_success_logistic),
            mean_absolute_error(sub.solver_seconds,sub.seconds_catboost_context)-mean_absolute_error(sub.solver_seconds,sub.seconds_dummy_context)])
    paired=dict(unit='paired family bootstrap, test only, no retuning',repetitions=500,
        catboost_minus_logistic_auc=np.quantile(np.asarray(differences)[:,0],[.025,.975]).tolist(),
        catboost_minus_context_median_mae=np.quantile(np.asarray(differences)[:,1],[.025,.975]).tolist())
    (out/'paired_model_comparison.json').write_text(json.dumps(paired,indent=2)+'\n')
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'axes.grid':True,'grid.alpha':.25,'font.size':9,'savefig.dpi':160})
    def save(fig,name):
        fig.savefig(figdir/(name+'.png'),bbox_inches='tight');fig.savefig(figdir/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(12,4.6));fail=1-test.accuracy_met.to_numpy()
    for name in ['dummy_prior','logistic','catboost']:
        p=1-test['p_success_'+name].to_numpy();fpr,tpr,_=roc_curve(fail,p);prec,rec,_=precision_recall_curve(fail,p)
        ax[0].plot(fpr,tpr,label=f"{name}: AUC={metrics['classification'][name]['test']['default_threshold']['roc_auc']:.3f}")
        label=f"{name}: AP={metrics['classification'][name]['test']['default_threshold']['failure_average_precision']:.3f}"
        if name=='dummy_prior':ax[1].axhline(float(np.mean(fail)),label=label)
        else:ax[1].step(rec,prec,where='post',label=label)
    ax[0].plot([0,1],[0,1],'k--',alpha=.3);ax[0].set(xlabel='False alarm rate',ylabel='Failure recall',title='Held-out ROC: failure E_inf > 0.01')
    ax[1].set(xlabel='Failure recall',ylabel='Failure precision',title='Held-out precision/recall (16 failures / 150)')
    for a in ax:a.legend()
    fig.tight_layout();save(fig,'classification')
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for a,operating,title in zip(ax,['default_threshold','validation_threshold'],['Default failure threshold 0.5','Validation-selected threshold']):
        cm=np.array(metrics['classification']['catboost']['test'][operating]['confusion_good_bad'])
        a.imshow(cm,cmap='Blues');a.set(xticks=[0,1],yticks=[0,1],xticklabels=['accept','reject'],yticklabels=['E<=.01','E>.01'],title=title,xlabel='Prediction',ylabel='Observed')
        for (i,j),v in np.ndenumerate(cm):a.text(j,i,str(v),ha='center',va='center',color='white' if v>cm.max()/2 else 'black',fontsize=14)
        a.grid(False)
    fig.tight_layout();save(fig,'operating_points')
    fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    for cohort,color in [('initial_parallel8','#2464ae'),('managed_serial1','#dd6633')]:
        s=test[test.execution_cohort==cohort]
        ax[0].scatter(s.solver_seconds,s.seconds_catboost_context,label=cohort,color=color,alpha=.65,s=18)
        ax[1].scatter(s.solver_seconds,s.seconds_catboost_context-s.solver_seconds,label=cohort,color=color,alpha=.65,s=18)
    limits=[float(test.solver_seconds.min()-.5),float(test.solver_seconds.max()+.5)]
    ax[0].plot(limits,limits,'k--');ax[0].set(xlabel='Measured solver seconds',ylabel='CatBoost predicted seconds',title='Historical cost, known planned concurrency')
    ax[1].axhline(0,color='black',ls='--');ax[1].set(xlabel='Measured solver seconds',ylabel='Prediction residual, seconds',title='Held-out cost residuals')
    for a in ax:a.legend()
    fig.tight_layout();save(fig,'runtime')
    fig,ax=plt.subplots(1,2,figsize=(13,6))
    for a,key in zip(ax,['classifier_catboost','regressor_catboost_context']):
        g=importance[importance.model==key].head(12).iloc[::-1]
        a.barh(g.feature,g.importance);a.set(title=key,xlabel='CatBoost PredictionValuesChange, %')
    fig.tight_layout();save(fig,'importance')
    L=['# Первая ML-модель Ремизова — 2026-10-25','',
       '**Обучены baseline и CatBoost по сохранённым результатам, без новых решений ОДУ.** Модель прогнозирует качество и стоимость конкретного исторического профиля `ChernovSolver.solve`, а не выбирает между методами и не находит оптимальные параметры.','',
       '## Данные и защита от утечки','',
       'Использованы существующие1000 уравнений production `all_methods_v1` и1000 первичных JSON. Все997 конечных33-точечных ошибок пересчитаны из сохранённых f/exact.871 ответ достиг .01,126 не достигли;3 worker_exception имеют неизвестные математические цели и исключены из fit/метрик. Они сохранены в targets.csv с null, а не превращены в отрицательные примеры. В регрессию входят все997 завершений, включая126 неточных ответов.','',
       'Исходные семейные split не менялись: train700/140семейств (697 известных целей),validation150/30,test150/30. Проверено отсутствие пересечения family_id и structure_signature. Прежние numerical audits уже рассматривали test; это сохранённый holdout текущего обучения, но не совершенно нетронутый prospective test выбора численной политики.','',
       f"В классификаторе {summary['feature_count']} признака: статистики и32 отсчёта входных a,b,c_R,g_R, их вариация/спектр/корреляции, аналитические коэффициентные сертификаты, предвычислимый T исторического профиля, размеры/операции только AST a_raw,b_raw,c_raw. Входной ML-grid256, целевой error-grid33 — разные сетки. c_R=c+lambda,g_R=-g. g — заданная правая часть ОДУ, не точное решение.",'',
       'Запрещены и не подаются в модель: y_exact/производные/y0/v0, y_shape, solution scale/support/target amplitude, полный tree_metrics/complexity, IDs/family/split/hash/seed, exact-ошибки, solver-time и диагностика решения. Regime/complexity используются только в анализе ошибок. scaler обучен только на train; ни feature importance, ни test-ошибки не использованы для отбора признаков или повторной настройки. [Схема](feature_schema.json),[аудит](data_audit.json).','',
       '## Обучение','',
       'Один заранее зафиксированный набор настроек, без поиска гиперпараметров: LogisticRegression(C=.1)+StandardScaler; Ridge(alpha=100) для log-времени; CatBoost depth4,learning_rate=.03,l2=10,max1200,early stopping80 по validation,seed20261025,thread_count1. Test не участвует в stopping/threshold/model selection. Модели не переобучались на train+validation после выбора.','',
       f"Число деревьев: {meta['fit_trees']}. Полный pipeline занял {summary['training_wall_seconds']:.2f}с без установки библиотек. Выбор по validation: classifier={selected_c},regressor={selected_r}. Native CBM и sklearn joblib сохранены в models/. Версии NumPy2.2.6 и SciPy1.15.3 не менялись.",'',
       '## Классификация на test','',
       'Положительный риск — E∞>.01. Threshold0.5 и порог, выбранный наvalidation по balanced accuracy, показаны отдельно. Выбор порога повышает обнаружение неточных решений ценой отказа от части точных.','',
       '| Модель | ROC-AUC | AP неточности | Accuracy при0.5 | Balanced accuracy при выбранном пороге | F1 неточности | Пропущено неточных | Ложно отклонено точных |',
       '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,d in metrics['classification'].items():
        a=d['test']['validation_threshold'];b=d['test']['default_threshold']
        L.append(f"| {name} | {a['roc_auc']:.4f} | {a['failure_average_precision']:.4f} | {b['accuracy']:.4f} | {a['balanced_accuracy']:.4f} | {a['failure_f1']:.4f} | {a['false_accept_bad']} | {a['false_reject_good']} |")
    c=metrics['classification']['catboost']['test']['validation_threshold']
    L += ['',f"CatBoost: failure threshold={c['failure_threshold']:.6f}; обнаружено14/16 неточных, ошибочно принято2, ошибочно отклонено34 точных. Среди102 принятых100 точных — наблюдаемая precision98.04%, **не гарантия**. При стандартном0.5 пороге accuracy91.33%, но пропущено11/16 неточных. Полные вероятности и оба operating point сохранены.",'',
       'Logistic наtest имеет немного больший ROC-AUC/AP, чем CatBoost; CatBoost выбран поvalidation, не переизбран поtest. Нет оснований утверждать строгое превосходство одного learned baseline над другим по этой малой выборке.','',
       f"95% family-bootstrap интервалы выбранного классификатора: ROC-AUC {boot['percentile95']['roc_auc']},balanced accuracy {boot['percentile95']['balanced_accuracy']}.500 resamples целых30 test-семейств; это условная описательная неопределённость, не независимые150 семейств.",'',
       '## Регрессия времени на test','',
       'Target — сохранённый wall `solver_seconds`, а не полное время worker и не минимальная стоимость .01. Первые700 задач планировались с8 внешними workers, последние300 — с1. Поэтому обучены coefficient-only и context-aware варианты. `planned_external_workers` доступен до вычисления, берётся из замороженных pre-run планов, не из времени/PID/готовых результатов; в классификатор он не входит. Нельзя выдавать этот target за изолированную CPU-стоимость или переносить без проверки на новый hardware/режим.','',
       '| Модель | Validation MAE,c | Test MAE,c | Test RMSE,c | Test MAPE,% | Test R² |',
       '|---|---:|---:|---:|---:|---:|']
    for name,d in metrics['regression'].items():
        q=d['test'];L.append(f"| {name} | {d['validation']['mae_seconds']:.4f} | {q['mae_seconds']:.4f} | {q['rmse_seconds']:.4f} | {q['mape_percent']:.3f} | {q['r2']:.4f} |")
    L += ['',
       'CatBoost с контекстом имеет наtest MAE0.652с против1.379с глобальной train-медианы. Однако контекстная медиана почти столь же хороша (0.677с) и была чуть лучше наvalidation; **serving сохраняет именно её**, а не выбирает CatBoost поtest. Преимущество CatBoost над глобальным constant в основном связано с известным режимом измерения. Без контекста CB MAE0.915с; важность размеров AST в этом варианте частично отражает совпадение сложности с порядком/режимом исторического запуска, а не причинную стоимость AST.','',
       f"Парные family-bootstrap95% разности: CB−Logistic ROC-AUC {paired['catboost_minus_logistic_auc']}; CB−context-median MAE,c {paired['catboost_minus_context_median_mae']}. При включении0 преимущество статистически не разрешено. Выбранная поvalidation контекстная медиана: MAE95% {boot['percentile95']['runtime_mae_seconds']}.",'',
       '## Важность и ошибки','']
    for key in ['classifier_catboost','regressor_catboost_context']:
        top=importance[importance.model==key].head(6)
        L += ['**'+key+'**: '+', '.join(f"{r.feature} ({r.importance:.2f}%)" for r in top.itertuples())+'.','']
    L += ['Это PredictionValuesChange, не причинный эффект. Дополнительно сохранена permutation importance групп коэффициентов наvalidation (10 повторов); коррелированные признаки и нарушение физических связей при перестановке ограничивают интерпретацию.','',
       'Два false accept выбранного CatBoost-порога: F0046/S01(E=.01749385,p_success=.96622),F0084/S03(E=.01251383,p_success=.92357). Ошибки не использовались для повторной настройки. [Все test-ошибки](test_error_analysis.csv),[подгруппы](test_subgroups.csv),[важность](feature_importance.csv),[перестановки](validation_group_permutation.csv).','',
       'В sklearn-baseline зарегистрированы RuntimeWarning от BLAS matmul. Отдельная проверка без BLAS (einsum) дала max разницу logistic probabilities1.23e-15,Ridge seconds1.07e-14; все признаки/прогнозы конечны, независимая max-norm градиента logistic8.84e-5 при tol1e-4,39 итераций. Ошибка результата этими проверками не обнаружена; точная причина предупреждений не установлена. Они не были скрыты и не исправлялись подстройкой метрик. [Аудит](numerical_warning_audit.json).','',
       '## Использование','',
       '```bash',
       '.venv/bin/python -B diagnostics/predict_remizov_ml.py \\',
       '  --models results/remizov_ml_v1_2026-10-25/models \\',
       '  --equation-id AMV1_F0000_S00 --planned-workers 8',
       '```','',
       'CLI читает только входные коэффициенты/сертификаты; targets/results ему не нужны. Возвращает вероятность достижения .01, risk-decision и прогноз секунд. Модель относится к n128/Q513/dx.0125/padding12/**cubic/taper**, legacy T и33 целевым узлам. Для cubic/zero, иного n/dx/T, MC или произвольного ОДУ она не аттестована. Probability не является математическим сертификатом точности.','',
       'Воспроизведение обучения в новом каталоге:','',
       '```bash','.venv/bin/python -m pip install -r requirements-ml.txt',
       '.venv/bin/python -B diagnostics/train_remizov_ml.py --out results/remizov_ml_v1_reproduction',
       '.venv/bin/python -B diagnostics/report_remizov_ml.py --out results/remizov_ml_v1_reproduction','```','',
       'Для независимого report при воспроизведении нужен аналогичный audit предупреждений; основной train сохраняет модели,метрики,прогнозы и importance без report. Полный environment-lock,исходники и проверки сохранены рядом.','',
       '## Артефакты','',
       '[features.npz](features.npz),[253 input-features](features.csv),[1000 targets/masks](targets.csv),[1000 predictions](predictions.csv),[metrics](metrics.json),[native/sklearn models](models/metadata.json),[protocol](training_protocol.json).','',
       'Графики: [ROC/PR](figures/classification.png),[два порога](figures/operating_points.png),[стоимость](figures/runtime.png),[важность](figures/importance.png); PDF рядом.','',
       '## MC отложен','',
       'После смены приоритета вычисления не возобновлялись. Pilot остался частичным:236 полных логических ответов,2 операторских остановки guardian,25 неначатых заданий; это не завершённое сравнение. Все артефакты сохранены. Отдельный [технический вариант ускорения MC](MC_OPTIMIZATION.md) использует только уже сохранённые измерения и исходный код; MC не менялся и новый profiling-run не выполнялся.']
    (out/'REPORT.md').write_text('\n'.join(L)+'\n')
    print(json.dumps({'report':str(out/'REPORT.md'),'figures':4,'paired_intervals':paired},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
