"""Publication-ready static plots from saved tables only; no model/ODE calls."""
from pathlib import Path
import json
import os
import csv
from collections import Counter

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
os.environ['MPLCONFIGDIR']=str(OUT/'.mplconfig')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
                     'axes.labelsize':10,'legend.fontsize':8,'pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'figure.facecolor':'white','savefig.facecolor':'white'})
HYP=ROOT/'results/ml_selector_hypotheses_2026-10-31'
FOUR=ROOT/'results/four_method_accuracy_1000_2026-11-01'
COLORS={'remizov':'#2874A6','monte_carlo':'#148F77','rk4':'#7D3C98','product_integral':'#CA6F1E',
        'TP':'#148F77','FP':'#B03A2E','FN':'#B9770E'}
LABELS={'remizov':'Remizov R128','monte_carlo':'MC mean4','rk4':'RK4','product_integral':'Product Integral'}
figures=[]
used=json.loads((OUT/'used_sources.json').read_text())


def source(path):
    import hashlib
    name=str(Path(path).relative_to(ROOT));used[name]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return path


def df(path):
    return pd.read_csv(source(path),float_precision='round_trip',keep_default_na=False)


def save(fig,stem,title,inputs,notes):
    fig.suptitle(title,fontsize=12,y=1.015)
    fig.tight_layout()
    for suffix in ('png','pdf'):
        fig.savefig(OUT/'figures'/f'{stem}.{suffix}',dpi=170,bbox_inches='tight')
    figures.append(dict(file=stem+'.png',pdf=stem+'.pdf',title=title,sources=inputs,notes=notes))
    plt.close(fig)


def ecdf(ax,values,label,color):
    values=np.sort(np.asarray(values,float))
    positive=values[values>0]
    assert len(positive)==len(values), 'zero values need an explicit plotting convention'
    ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',label=label,color=color,lw=1.8)


def main():
    common=df(FOUR/'release/ml_four_methods.csv')
    for method in LABELS:common['E_'+method]=common['E_'+method].astype(float)
    comparison=df(OUT/'tables/main_model_comparisons.csv')
    cases=df(OUT/'tables/diagnostic_258_cases.csv')
    diagnostic=df(OUT/'tables/diagnostic_258_all_splits.csv')
    policies=df(OUT/'tables/model_policy_metrics_all.csv')
    groups=df(OUT/'tables/four_method_group_distributions.csv')
    test=diagnostic[diagnostic.split=='test']

    fig,axes=plt.subplots(1,2,figsize=(12.2,4.6))
    for split,color in [('train','#2874A6'),('validation','#CA6F1E'),('test','#148F77')]:
        own=common[common.split_original==split]
        axes[0].scatter(own.E_remizov,own.E_monte_carlo,s=15,alpha=.6,color=color,label=f'{split}, n={len(own)}')
    lim=[1e-7,1e3]
    axes[0].plot(lim,lim,'k--',lw=.8)
    axes[0].axvline(.05,color='gray',ls=':');axes[0].axhline(.05,color='gray',ls=':')
    axes[0].set(xscale='log',yscale='log',xlabel='E0 Remizov',ylabel='E0 MC mean4',title='997 пар; диагональ — равная ошибка')
    axes[0].legend()
    for method in ['remizov','monte_carlo']:
        ecdf(axes[1],common['E_'+method],LABELS[method],COLORS[method])
    axes[1].axvline(.05,color='gray',ls=':',label='ε=0.05')
    axes[1].set(xscale='log',xlabel='Абсолютная ошибка E0',ylabel='Эмпирическая доля ≤ E0',title='Распределения ошибок на тех же 997 задачах')
    axes[1].legend();axes[1].grid(alpha=.2)
    save(fig,'01_R_MC_error_comparison','Ремизов и MC: сохранённые ошибки в x=0',
         ['results/four_method_accuracy_1000_2026-11-01/release/ml_four_methods.csv'],
         'Исходный split. Все точки сохранены, выбросы не удалялись. Не является новым расчётом.')

    labels=['baseline253 new','selected253 new','diagnostic258 new','always R']
    names=['253 / t=.5','253 / t=.25','258 / t=.200667','Всегда R']
    fig,axes=plt.subplots(1,3,figsize=(14.5,4.5))
    for j,(split,color) in enumerate([('train','#5D6D7E'),('validation','#CA6F1E'),('test','#2874A6')]):
        data=comparison[(comparison.scenario=='new')&(comparison.split==split)].set_index('label')
        for ax,metric,title in zip(axes,['macro_f1','mc_recall','mean_error'],['Macro-F1','Recall MC','Средняя выбранная E0']):
            ax.bar(np.arange(4)+(j-1)*.24,[float(data.loc[name,metric]) for name in labels],width=.24,color=color,label=split)
            ax.set_xticks(np.arange(4),names,rotation=15,ha='right');ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylim(0,1);axes[1].set_ylim(0,1);axes[2].set_yscale('log');axes[0].legend()
    save(fig,'02_ML_split_metrics','Фиксированные политики: Train / Validation / Test',
         ['tables/main_model_comparisons.csv'],
         'Train — in-sample предсказание сохранённых CBM. Политика 258 диагностическая, не выбранная глобально по Validation.')

    fig,axes=plt.subplots(1,3,figsize=(11.5,3.9))
    own=comparison[comparison.label=='diagnostic258 new'].set_index('split')
    for ax,split in zip(axes,['train','validation','test']):
        cm=np.asarray(json.loads(own.loc[split,'confusion_matrix']))
        ax.imshow(cm,cmap='Blues',vmin=0,vmax=cm.max())
        for i in range(2):
            for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=17,color='white' if cm[i,j]>.5*cm.max() else 'black')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['R','MC'],yticklabels=['R','MC'],xlabel='Выбранный метод',ylabel='Истинный winner',title=f'{split}, n={int(own.loc[split,"n"])}')
    save(fig,'03_258_confusion','258 признаков: матрицы ошибок при t=0.2006671072130637',
         ['tables/main_model_comparisons.csv'], 'Строки — истинный метод минимальной ошибки; столбцы — выбор. Train не является OOF.')

    ordered=pd.concat([cases[cases.category==c].sort_values('equation_id') for c in ['TP','FP','FN']])
    fig,ax=plt.subplots(figsize=(12.5,8.5))
    positions=np.arange(len(ordered))
    ax.barh(positions,ordered.delta_R_minus_MC,color=[COLORS[c] for c in ordered.category])
    ax.set_yticks(positions,[f'{r.category}: {r.equation_id}' for r in ordered.itertuples()],fontsize=8.5)
    ax.invert_yaxis();ax.set_xscale('symlog',linthresh=.001);ax.axvline(0,color='black',lw=.6)
    ax.set_xlabel('Δ = E_R − E_MC; положительное значение означает преимущество MC')
    ax.legend(handles=[Patch(color=COLORS[k],label=v) for k,v in [('TP','6 найденных MC'),('FP','14 ложных MC'),('FN','3 пропущенных MC')]],loc='lower right')
    ax.grid(axis='x',alpha=.2)
    save(fig,'04_258_all_23_cases','Все 23 Test-случая: найденные, ложные и пропущенные MC',
         ['tables/diagnostic_258_cases.csv'], 'Симметричная логарифмическая шкала сохраняет знак и малые Δ; крупный выброс не удалён.')

    wrong=cases[cases.category.isin(['FP','FN'])].sort_values('regret',ascending=False)
    fig,axes=plt.subplots(1,2,figsize=(14.2,7.1),sharey=True)
    pos=np.arange(len(wrong))
    axes[0].barh(pos,wrong.regret,color=[COLORS[c] for c in wrong.category])
    axes[0].set_yticks(pos,[f'{r.category}: {r.equation_id}' for r in wrong.itertuples()],fontsize=8.5)
    axes[0].invert_yaxis();axes[0].set_xscale('log');axes[0].set_xlabel('Regret = E_selected − min(E_R,E_MC)')
    for i,row in enumerate(wrong.itertuples()):axes[1].plot([row.E_R,row.E_MC],[i,i],color='#BDC3C7',lw=1)
    axes[1].scatter(wrong.E_R,pos,label='E_R',color=COLORS['remizov'],s=30)
    axes[1].scatter(wrong.E_MC,pos,label='E_MC',color=COLORS['monte_carlo'],s=30)
    axes[1].axvline(.05,color='#B03A2E',ls='--',label='ε=.05')
    axes[1].set_xscale('log');axes[1].set_xlabel('Ошибки обоих методов');axes[1].legend()
    for ax in axes:ax.grid(axis='x',alpha=.2)
    save(fig,'05_wrong_choice_severity','17 неправильных выборов 258-модели: величина ущерба и порог ε',
         ['tables/diagnostic_258_cases.csv'], 'Два FP вызывают дополнительное нарушение ε; остальные неправильные winner-метки не равны нарушениям точности.')

    test_predictions=df(HYP/'test_predictions.csv')
    rowsets={
        'Всегда R':test.copy().assign(chosen=test.E_R.to_numpy()),
        'Всегда MC':test.copy().assign(chosen=test.E_MC.to_numpy()),
        '258, diagnostic':test.copy().assign(chosen=test.E_selected.to_numpy()),
        'Oracle':test.copy().assign(chosen=np.minimum(test.E_R,test.E_MC))}
    chosen=test_predictions[(test_predictions.candidate_id=='new__X__classifier__w1__baseline')&(test_predictions.policy=='grid_0.25')]
    rowsets['253, selected']=chosen.rename(columns={'E_selected':'chosen'})
    order=['Всегда R','Всегда MC','253, selected','258, diagnostic','Oracle']
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.4));plot_rows=[]
    for ax,exclude in zip(axes,[False,True]):
        values=[]
        for name in order:
            data=rowsets[name];data=data[data.equation_id!='AMV1_F0180_S04'] if exclude else data
            value=float(data.chosen.mean());values.append(value)
            plot_rows.append(dict(policy=name,excluded_largest=exclude,n=len(data),mean_error=value))
        ax.bar(np.arange(5),values,color=['#2874A6','#148F77','#7F8C8D','#8E44AD','#273746'])
        ax.set_xticks(np.arange(5),order,rotation=20,ha='right');ax.set_yscale('log')
        ax.set_title('Чувствительность: 149 без F0180/S04' if exclude else 'Основной результат: все 150 Test')
        ax.set_ylabel('Средняя выбранная E0');ax.grid(axis='y',alpha=.2)
    pd.DataFrame(plot_rows).to_csv(OUT/'tables/figure06_sensitivity.csv',index=False)
    save(fig,'06_outlier_sensitivity','Влияние одного большого R-выброса на среднюю ошибку',
         ['tables/diagnostic_258_all_splits.csv','results/ml_selector_hypotheses_2026-10-31/test_predictions.csv'],
         'Постфактум анализ чувствительности. F0180/S04 остаётся во всех основных метриках; исключение не используется для выбора модели.')

    vm=policies[policies.split=='validation'];tm=policies[policies.split=='test']
    joined=vm.merge(tm,on=['candidate_id','scenario','policy'],suffixes=('_val','_test'))
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for ax,scenario in zip(axes,['old','new']):
        own=joined[joined.scenario==scenario]
        for fs,color in [('X','#2874A6'),('augmented','#CA6F1E')]:
            data=own[own.feature_set_val==fs]
            ax.scatter(data.mean_regret_val,data.mean_regret_test,s=16,alpha=.45,color=color,label='253' if fs=='X' else '258')
        highlight=own[(own.candidate_id=='new__augmented__classifier__w1__baseline')&(own.policy=='validation_optimal')]
        if len(highlight):ax.scatter(highlight.mean_regret_val,highlight.mean_regret_test,marker='*',s=180,color='#148F77',edgecolor='black',label='258 diagnostic')
        selected=own[(own.candidate_id==('new__X__classifier__w1__baseline' if scenario=='new' else 'old__augmented__classifier__w12__baseline'))&(own.policy==('grid_0.25' if scenario=='new' else 'validation_optimal'))]
        ax.scatter(selected.mean_regret_val,selected.mean_regret_test,marker='X',s=90,color='#B03A2E',label='выбран по Val')
        ax.set(xscale='log',yscale='log',xlabel='Validation mean regret',ylabel='Test mean regret',title=f'{scenario}: все сохранённые политики')
        ax.legend();ax.grid(alpha=.2)
    save(fig,'07_all_policy_results','Разведочное сопоставление 844 политик: Validation и раскрытый Test',
         ['tables/model_policy_metrics_all.csv'], 'График не является новым выбором по Test; показывает множественность просмотренных вариантов.')

    chosen_rows=[]
    for weight in [1,2,3,5,8,12]:
        data=vm[(vm.scenario=='new')&(vm.feature_set=='X')&(vm.kind=='classifier')&(vm.weight==weight)&(vm.policy=='validation_optimal')]
        row=data.sort_values(['mean_regret','success_rate','macro_f1','mc_predictions','candidate_id'],ascending=[True,False,False,True,True]).iloc[0]
        testrow=tm[(tm.candidate_id==row.candidate_id)&(tm.policy==row.policy)].iloc[0]
        chosen_rows.append(dict(weight=weight,candidate_id=row.candidate_id,threshold=row.threshold,val_regret=row.mean_regret,test_regret=testrow.mean_regret,TP=testrow.mc_correct,FP=testrow.mc_false))
    weights=pd.DataFrame(chosen_rows);weights.to_csv(OUT/'tables/weights_new253_val_selected.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.3))
    axes[0].plot(range(6),weights.val_regret,'o-',color='#2874A6');axes[0].set_xticks(range(6),weights.weight);axes[0].set(xlabel='Вес класса MC',ylabel='Validation mean regret',title='253 признака: лучшая сохранённая конфигурация по Val')
    axes[1].bar(np.arange(6)-.17,weights.TP,width=.34,color='#148F77',label='верно выбрано MC')
    axes[1].bar(np.arange(6)+.17,weights.FP,width=.34,color='#B03A2E',label='ложно выбрано MC')
    axes[1].set_xticks(range(6),weights.weight);axes[1].set(xlabel='Вес класса MC',ylabel='Число Test-случаев',title='Результат того же Val-выбора на Test');axes[1].legend()
    for ax in axes:ax.grid(axis='y',alpha=.2)
    save(fig,'08_class_weights','Веса классов: снижение Validation loss не гарантирует Test-перенос',
         ['tables/weights_new253_val_selected.csv'], 'Никаких новых fit. Показанные строки извлечены из ранее сохранённых Val-optimal политик; Test не использован для их выбора.')

    imp=df(HYP/'feature_importance.csv');imp=imp[imp.candidate_id=='new__augmented__classifier__w1__baseline'].sort_values('importance',ascending=False)
    imp=imp.assign(rank=np.arange(1,len(imp)+1));imp.to_csv(OUT/'tables/diagnostic_258_feature_importance.csv',index=False)
    top=imp.head(25).iloc[::-1]
    fig,ax=plt.subplots(figsize=(11.5,8.3))
    ax.barh(np.arange(len(top)),top.importance,color=['#CA6F1E' if str(n).startswith('research_') else '#2874A6' for n in top.feature])
    ax.set_yticks(np.arange(len(top)),top.feature,fontsize=8.5);ax.set_xlabel('Сохранённая CatBoost feature importance, %')
    ax.legend(handles=[Patch(color='#CA6F1E',label='добавленные признаки'),Patch(color='#2874A6',label='исходные 253')]);ax.grid(axis='x',alpha=.2)
    save(fig,'09_feature_importance','258-модель: top-25 сохранённых важностей',
         ['results/ml_selector_hypotheses_2026-10-31/feature_importance.csv'], 'PredictionValuesChange — описательная важность модели, не причинный эффект и не доказательство независимого вклада признака.')

    fig,axes=plt.subplots(1,2,figsize=(12.5,4.5))
    for method in LABELS:
        values=common['E_'+method].to_numpy(float)
        ecdf(axes[0],values,LABELS[method],COLORS[method])
        axes[1].boxplot([np.log10(values)],positions=[list(LABELS).index(method)],widths=.55,
                         patch_artist=True,boxprops=dict(facecolor=COLORS[method],alpha=.6),
                         flierprops=dict(markersize=2,markeredgecolor=COLORS[method],alpha=.45),medianprops=dict(color='black'))
    axes[0].set(xscale='log',xlabel='E0',ylabel='Эмпирическая доля ≤ E0',title='ECDF, общее пересечение 997');axes[0].axvline(.05,color='gray',ls=':',label='ε=.05');axes[0].legend()
    axes[1].set_xticks(range(4),list(LABELS.values()),rotation=15,ha='right');axes[1].set(ylabel='log10(E0)',title='Медиана, IQR и выбросы');axes[1].axhline(np.log10(.05),color='gray',ls=':')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    save(fig,'10_four_method_distributions','Четыре метода: распределения E0 на одних и тех же 997 задачах',
         ['results/four_method_accuracy_1000_2026-11-01/release/ml_four_methods.csv'], 'Данные положительны; логарифм используется только для отображения. Все выбросы сохранены.')

    fig,ax=plt.subplots(figsize=(10.5,5))
    levels=['L1','L2','L3','L4']
    for method in LABELS:
        data=groups[(groups.group_type=='complexity_level')&(groups.method==method)].set_index('group').loc[levels]
        ax.plot(levels,data['median'],'o-',color=COLORS[method],label=LABELS[method])
        ax.fill_between(range(4),data.q25.to_numpy(float),data.q75.to_numpy(float),alpha=.12,color=COLORS[method])
    ax.axhline(.05,color='gray',ls=':',label='ε=.05');ax.set(yscale='log',xlabel='Уровень сложности',ylabel='E0',title='Линии — медианы; заливка — межквартильный диапазон');ax.legend();ax.grid(alpha=.2)
    save(fig,'11_four_complexity','Ошибки четырёх методов по L1–L4', ['tables/four_method_group_distributions.csv'], 'IQR описывает задачи, а не доверительный интервал; группы: 250/250/247/250.')

    regimes=['constant','slow_varying','moderate_varying','oscillatory','high_contrast']
    rates=np.array([[float(groups[(groups.group_type=='regime')&(groups.group==g)&(groups.method==m)].success_rate.iloc[0]) for g in regimes] for m in LABELS])
    fig,axes=plt.subplots(1,2,figsize=(13.2,4.3),gridspec_kw={'width_ratios':[1.6,1]})
    axes[0].imshow(rates,cmap='YlGnBu',vmin=0,vmax=1,aspect='auto')
    for i in range(4):
        for j in range(5):axes[0].text(j,i,f'{100*rates[i,j]:.1f}%',ha='center',va='center',color='white' if rates[i,j]>.65 else 'black')
    axes[0].set_xticks(range(5),regimes,rotation=20,ha='right');axes[0].set_yticks(range(4),list(LABELS.values()));axes[0].set_title('Доля E0 ≤ .05, общее пересечение')
    axes[1].bar(range(4),[0,0,997,0],color=list(COLORS[m] for m in LABELS));axes[1].set_xticks(range(4),list(LABELS.values()),rotation=20,ha='right');axes[1].set_ylabel('Число winner-меток');axes[1].set_title('997 побед RK4; равенств нет')
    save(fig,'12_regime_accuracy_and_winners','Точность по режимам и отсутствие разнообразия четырёхметодных целей',
         ['tables/four_method_group_distributions.csv','results/four_method_accuracy_1000_2026-11-01/release/winners.csv'], 'В каждой клетке все методы сравниваются на одинаковых ID; oscillatory n=197, остальные n=200.')

    fig,ax=plt.subplots(figsize=(13,4.7))
    for method in LABELS:
        data=groups[(groups.group_type=='family_id')&(groups.method==method)].sort_values('group')
        ax.plot(range(200),data['median'],'.-',ms=2,lw=.7,color=COLORS[method],label=LABELS[method],alpha=.9)
    for boundary in [49.5,99.5,149.5]:ax.axvline(boundary,color='#ABB2B9',lw=.6)
    ax.axhline(.05,color='gray',ls=':');ax.set(yscale='log',xlabel='Индекс структурного семейства F0000…F0199',ylabel='Медианная E0 в семействе');ax.legend(ncol=4);ax.grid(axis='y',alpha=.2)
    save(fig,'13_family_median_errors','Все 200 структурных семейств: медианные ошибки методов', ['tables/four_method_group_distributions.csv'], 'В каждом семействе 5 параметризаций; F0134/F0136/F0139 содержат 4 общих ответа. Линии соединяют категории только для читаемости.')

    worst=[common.loc[common.E_rk4.idxmax(),'equation_id'],common.loc[common.E_product_integral.idxmax(),'equation_id']]
    fig,axes=plt.subplots(1,2,figsize=(12.6,4.6))
    for ax,pid in zip(axes,worst):
        for method in ['rk4','product_integral']:
            path=FOUR/'records'/method/(pid+'.json');raw=json.loads(Path(source(path)).read_text())
            x=np.array(raw['result']['x'])[::4];exact=np.array(raw['exact_grid33']);err=np.abs(np.array(raw['result']['y'])[::4]-exact)
            ax.plot(x,err,'o-',ms=2.5,lw=1,color=COLORS[method],label=LABELS[method])
        path=ROOT/'results/chernov_solve1000_2026-10-18/solutions'/(pid+'.json');raw=json.loads(Path(source(path)).read_text())
        err=np.abs(np.asarray(raw['result']['f'])-np.asarray(raw['y_exact']))
        ax.plot(raw['result']['x'],err,'.-',ms=3,lw=1,color=COLORS['remizov'],label=LABELS['remizov'])
        mc=float(common.loc[common.equation_id==pid,'E_monte_carlo'].iloc[0]);ax.scatter([0],[mc],color=COLORS['monte_carlo'],s=45,marker='D',label='MC mean4: только x=0')
        ax.axvline(0,color='gray',lw=.6);ax.axhline(.05,color='gray',ls=':');ax.set(yscale='symlog',ylim=(0,None),xlabel='x',ylabel='Абсолютная ошибка в сохранённых узлах',title=pid)
        ax.set_yscale('symlog',linthresh=1e-10);ax.legend(fontsize=7.5);ax.grid(alpha=.2)
    save(fig,'14_saved_worst_trajectories','Сохранённые ошибки: максимальная E0 RK4 и максимальная E0 PI',
         [f'results/four_method_accuracy_1000_2026-11-01/records/{method}/{pid}.json'
          for pid in worst for method in ('rk4','product_integral')]
         +[f'results/chernov_solve1000_2026-10-18/solutions/{pid}.json' for pid in worst]
         +['tables/four_method_common997.csv'],
         'Только существующие 33 узла и MC в нуле. Symlog с линейной областью до 1e-10 сохраняет нулевую ошибку НУ без подстановки floor. Междуузловая точность не установлена; новых exact/solver вызовов нет.')

    (OUT/'figure_manifest.json').write_text(json.dumps(figures,indent=2,ensure_ascii=False)+'\n')
    (OUT/'used_sources.json').write_text(json.dumps(used,indent=2,ensure_ascii=False)+'\n')
    from PIL import Image,ImageDraw
    for page,start in enumerate(range(0,len(figures),6),1):
        canvas=Image.new('RGB',(1500,1350),'#ECEFF1');draw=ImageDraw.Draw(canvas)
        for slot,entry in enumerate(figures[start:start+6]):
            picture=Image.open(OUT/'figures'/entry['file']).convert('RGB');picture.thumbnail((730,420))
            col,row=slot%2,slot//2
            canvas.paste(picture,(15+750*col,25+450*row));draw.text((15+750*col,5+450*row),entry['file'],fill='black')
        canvas.save(OUT/'checks'/f'contact_sheet_{page}.png')
    print(json.dumps(dict(figures=len(figures),formats=['png','pdf'],new_solver_calls=0,model_fits=0)))


if __name__=='__main__':main()
