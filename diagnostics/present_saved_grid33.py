#!/usr/bin/env python3
"""Scientific presentation from frozen CSV only; no solver, exact or ML code."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("results/saved_grid33_1000_2026-11-04")
REPORT = Path("results/scientific_report_revision_2026-11-04/report.tex")
CONTRACT = Path("reports/ml_selector_contract/CONTRACT.md")
AUDIT = Path("reports/ml_selector_readiness/AUDIT_REPORT.md")
METHODS = ("R", "RK4", "PI", "MC")
LABELS = dict(R="Ремизов", RK4="RK4", PI="Продукт-интеграл", MC="MC mean4")
COLORS = dict(R="#8b479d", RK4="#226bb0", PI="#c77522", MC="#47786a")
EPS = .05


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quantile(values, q):
    a = sorted(values)
    p = (len(a) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    return a[lo] + (a[hi] - a[lo]) * (p - lo)


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == len({r['equation_id'] for r in rows}) == 1000, "expected 1000 unique IDs")
    for r in rows:
        for method in ("R", "RK4", "PI"):
            key = method + "_E33"
            r[key] = None if r[key] == "" else float(r[key])
            v = r[key]
            require(v is None or math.isfinite(v) and v >= 0, "invalid error")
            if v is not None:
                require(r[method + "_accuracy_met"] == str(v <= EPS), "accuracy flag differs from CSV error")
        require(r['R_available'] == str(r['R_E33'] is not None), "R availability mismatch")
        require(r['RK4_E33'] is not None and r['PI_E33'] is not None, "missing IVP answer")
        require(r['MC_E33'] == r['MC_accuracy_met'] == '', "MC grid error must remain absent")
        r['MC_E0_lower_bound'] = float(r['MC_E0_lower_bound'])
        require(math.isfinite(r['MC_E0_lower_bound']) and r['MC_E0_lower_bound'] >= 0, "invalid MC bound")
        require(r['MC_threshold_evidence'] == ('fails_by_lower_bound' if r['MC_E0_lower_bound'] > EPS else 'unknown'),
                "MC threshold evidence mismatch")
    return rows


def values(rows, method):
    key = 'MC_E0_lower_bound' if method == 'MC' else method + '_E33'
    return [r[key] for r in rows if r[key] is not None]


def build_data(rows):
    common = [r for r in rows if r['R_E33'] is not None]
    comparison = []
    for scope, subset in (('all_available', rows), ('common997', common)):
        for m in METHODS:
            a = values(subset, m)
            is_mc = m == 'MC'
            passed = sum(v <= EPS for v in a)
            failed = len(a) - passed
            comparison.append(dict(scope=scope, method=m, metric='E0_lower_bound' if is_mc else 'E33',
                n_values=len(a), n_scope=len(subset), missing_E33=len(subset) if is_mc else len(subset)-len(a),
                mean=statistics.mean(a), median=statistics.median(a), q95=quantile(a,.95), maximum=max(a),
                E33_pass=None if is_mc else passed, E33_fail=None if is_mc else failed,
                pass_fraction_available=None if is_mc else passed/len(a),
                pass_fraction_scope=None if is_mc else passed/len(subset),
                E33_fail_lower_bound=failed if is_mc else None,
                E33_unknown=passed if is_mc else len(subset)-len(a)))
    exceedances = sorted((dict(equation_id=r['equation_id'], family_id=r['family_id'],
        R_E33=r['R_E33'], RK4_E33=r['RK4_E33'], PI_E33=r['PI_E33'])
        for r in rows if r['R_E33'] is not None and r['R_E33'] > EPS), key=lambda r:(-r['R_E33'],r['equation_id']))
    r_values = sorted(values(rows,'R'), reverse=True)
    shares=[]
    for m in METHODS:
        a = values(rows,m)
        mc=m=='MC'
        shares.append(dict(method=m, denominator=len(rows), measured_pass=None if mc else sum(v<=EPS for v in a),
            measured_fail=None if mc else sum(v>EPS for v in a),
            missing_answer=0 if mc else len(rows)-len(a),
            fails_by_lower_bound=sum(v>EPS for v in a) if mc else 0,
            unknown_grid_accuracy=sum(v<=EPS for v in a) if mc else 0,
            full_grid_error_missing=len(rows) if mc else len(rows)-len(a)))
    margins={m:min((r['MC_E0_lower_bound' if m=='MC' else m+'_E33']-r['RK4_E33'],r['equation_id'])
                   for r in common) for m in ('R','PI','MC')}
    for r in common:
        require(all(r['RK4_E33'] < r[k] for k in ('R_E33','PI_E33','MC_E0_lower_bound')),
                "CSV no longer supports unique RK4 winner")
    return dict(comparison=comparison, shares=shares, exceedances=exceedances,
                common_ids=[r['equation_id'] for r in common], total=len(rows), common=len(common),
                missing_ids=[r['equation_id'] for r in rows if r['R_E33'] is None],
                distributions_common={m:sorted(values(common,m)) for m in ('R','RK4','PI')},
                distributions_available={m:sorted(values(rows,m)) for m in METHODS},
                minimum_margins={m:dict(margin=v,equation_id=pid) for m,(v,pid) in margins.items()},
                R_largest_two_sum_share=sum(r_values[:2])/sum(r_values), R_above_one=sum(v>1 for v in r_values),
                new_solver_calls=0,new_MC_calls=0,new_fit_calls=0,new_model_inference_calls=0)


def draw_figures(out, data):
    with tempfile.TemporaryDirectory(prefix='grid33_mpl_') as cache:
        prior=os.environ.get('MPLCONFIGDIR')
        os.environ['MPLCONFIGDIR']=cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,
            'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':180})
        folder=out/'figures'; folder.mkdir()
        manifest=[]
        def save(fig,name,caption,source):
            fig.tight_layout()
            files=[]
            for ext in ('png','pdf'):
                path=folder/(name+'.'+ext)
                fig.savefig(path,facecolor='white',metadata={'Creator':'saved-grid33 presentation',
                            **({'CreationDate':None,'ModDate':None} if ext=='pdf' else {})})
                files.append(dict(path=str(path.relative_to(out)),sha256=sha(path)))
            manifest.append(dict(name=name,caption=caption,source=source,files=files))
            plt.close(fig)
        all_a=[v for a in data['distributions_available'].values() for v in a]
        require(min(all_a)>0,'Logarithmic plots require positive data; zero must be displayed separately, never shifted')
        lo,hi=10**math.floor(math.log10(min(all_a))),10**math.ceil(math.log10(max(all_a)))
        fig,ax=plt.subplots(figsize=(9,5.5))
        for m,a in data['distributions_common'].items():
            a=np.asarray(a); y=np.arange(1,len(a)+1)/len(a)
            ax.step(np.r_[lo,a,hi],np.r_[0,y,1],where='post',label=f'{LABELS[m]} (n={len(a)})',color=COLORS[m],lw=2)
        ax.set(xscale='log',xlim=(lo,hi),ylim=(0,1.02),xlabel='Максимальная абсолютная ошибка E₃₃ (логарифмическая шкала)',
               ylabel='Доля задач с ошибкой ≤ E₃₃',title='Эмпирическое распределение E₃₃: общие 997 задач')
        ax.axvline(EPS,color='#a1272d',ls='--',label='Допуск 0,05'); ax.grid(alpha=.22); ax.legend(loc='lower right')
        save(fig,'01_ecdf_common997','Общие 997 задач; логарифмическая ось ошибок, линейная накопленная доля. MC не показан, поскольку E33 неизвестна.','CSV: R_E33/RK4_E33/PI_E33; R_available=True')
        fig,axes=plt.subplots(2,2,figsize=(11,7.4),sharex=True,sharey=True)
        bins=np.geomspace(lo,hi,37)
        for ax,m in zip(axes.flat,METHODS):
            a=np.asarray(data['distributions_available'][m]); counts,edges=np.histogram(a,bins=bins)
            require(int(counts.sum())==len(a),'histogram dropped observations')
            ax.stairs(counts/len(a),edges,fill=True,color=COLORS[m],alpha=.75)
            ax.axvline(EPS,color='#a1272d',ls='--'); ax.set_xscale('log')
            ax.set_title(f'{LABELS[m]} · n={len(a)}'+('\nE₀: нижняя граница E₃₃' if m=='MC' else '\nизмеренная E₃₃'))
            ax.set_xlim(lo,hi);ax.grid(alpha=.18)
        fig.supxlabel('Ошибка; одинаковые логарифмические интервалы');fig.supylabel('Доля наблюдений в интервале (не плотность)')
        fig.suptitle('Распределения по всему доступному покрытию; MC показан отдельно',fontsize=14)
        save(fig,'02_histograms_available','Равные интервалы по log10 ошибки, доля в каждом интервале. MC-панель показывает только E0 как нижнюю границу E33; её нельзя читать как распределение MC E33. Все наблюдения сохранены.','CSV: all available errors, MC_E0_lower_bound')
        e=data['exceedances'];fig,ax=plt.subplots(figsize=(11,9.3))
        yy=np.arange(len(e)); xx=[r['R_E33'] for r in e]
        ax.hlines(yy,EPS,xx,color='#c9b8ce',lw=1)
        ax.scatter(xx,yy,c=['#b62e3a' if i<2 else COLORS['R'] for i in range(len(e))],s=32,zorder=3)
        for i,v in enumerate(xx): ax.annotate(f'{v:.5g}',(v,i),xytext=(5,0),textcoords='offset points',va='center',fontsize=9)
        ax.set_yticks(yy,[r['equation_id'] for r in e],fontsize=9);ax.invert_yaxis()
        ax.set(xscale='log',xlim=(EPS*.8,max(xx)*5),xlabel='E₃₃ Ремизова (логарифмическая шкала)',
               title=f'Ремизов: все {len(e)} превышений допуска 0,05 из 997 ответов')
        ax.axvline(EPS,color='#a1272d',ls='--');ax.grid(axis='x',alpha=.22)
        save(fig,'03_remizov_exceedances','Показаны все ответы Ремизова с E33>0,05, без удаления из основных статистик. Два крупнейших выделены цветом; «выброс» здесь описательное слово, не результат статистического теста.','CSV: R_E33>0.05; descending error')
        fig,ax=plt.subplots(figsize=(11,5.6))
        y=np.arange(4);left=np.zeros(4)
        categories=[('measured_pass','Измерено E₃₃ ≤ 0,05','#397963',''),('measured_fail','Измерено E₃₃ > 0,05','#bd5761',''),
                    ('missing_answer','Ответ отсутствует','#777b85','//'),('fails_by_lower_bound','Нарушение по нижней границе','#e2a260','..'),
                    ('unknown_grid_accuracy','Достижение допуска неизвестно','#dfe3e9','xx')]
        for key,label,color,hatch in categories:
            counts=np.array([r[key] or 0 for r in data['shares']]);width=counts/10
            ax.barh(y,width,left=left,label=label,color=color,hatch=hatch,edgecolor='white',height=.62)
            for i,(count,w,l) in enumerate(zip(counts,width,left)):
                if count>=60:
                    label_text=(f'{count}/1000\n{w:g}%' if count>=80 else f'{count}\n{w:g}%').replace('.',',')
                    ax.text(l+w/2,i,label_text,ha='center',va='center',fontsize=10,
                            color='white' if key in ('measured_pass','measured_fail') else '#20252a')
            left+=width
        ax.annotate('27/1000 превышений (2,7%)\n3/1000 пропуска (0,3%)',xy=(99.85,0),xytext=(76,.52),fontsize=8.5,
                    va='center',arrowprops=dict(arrowstyle='-',color='#555'))
        ax.set_yticks(y,['Ремизов','RK4','Продукт-интеграл','MC (только E₀)']);ax.invert_yaxis()
        ax.set(xlim=(0,100),xlabel='Доля всех 1000 задач, %',title='Достижение допуска: общий знаменатель 1000 задач')
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,fontsize=9,frameon=False)
        fig.subplots_adjust(bottom=.25)
        save(fig,'04_threshold_shares','Доли от всех 1000 задач: пропуски R не считаются численными нарушениями. У MC нет измерений E33: 62 нарушения следуют из E0, для 938 достижение допуска неизвестно; условие — тот же ответ MC в нуле.','CSV: availability and threshold evidence; denominator=1000')
        if prior is None: os.environ.pop('MPLCONFIGDIR',None)
        else: os.environ['MPLCONFIGDIR']=prior
        return manifest,matplotlib.__version__


def requirement_rows():
    """Traceable scope: current request, report goals, dated historical contract."""
    return [dict(id=key, requirement=req, status=status, evidence=evidence, remaining=remaining, basis=basis)
        for key,req,status,evidence,remaining,basis in [
        ('T01','Набор задач с известным exact и воспроизводимыми идентификаторами','подтверждено для сохранённого набора',
         '1000 уникальных ID в CSV; прежний verifier проверил метаданные и 7000 raw JSON; источники закреплены SHA',
         'Перенос на другие распределения задач этим не подтверждён','report.tex: Цель и задачи; saved_grid33/verification.json'),
        ('T02','Сравнительная таблица и распределения E33 при ε=0,05','закрыто в доступном покрытии',
         'comparison.csv; четыре рисунка; сравнение измеренных R/RK4/PI на common997 и полном доступном наборе',
         'Полный четырёхметодный набор измеренных E33 отсутствует','текущее задание, пункты 1–3'),
        ('T03','Полные E33 всех четырёх методов на всех 1000 задачах','частично',
         'R997, RK4/PI1000; MC E0 как нижняя граница; три R worker_exception учтены отдельно',
         'MC33 того же профиля со статистическим протоколом; причины отсутствующих R и отдельная восстановительная серия при необходимости','report.tex: grid33; текущее задание, пункт 4'),
        ('T04','Строгий победитель по E33 фиксированных ответов','подтверждено на общих997',
         'Положительные per-ID разности к R/PI и MC E0; условие — сохранение того же mean4 в нуле',
         'Для трёх missing R победитель среди четырёх неизвестен; универсальное превосходство не установлено','текущее задание; report.tex: grid33'),
        ('T05','Различия постановок и входной информации','закрыто как описание условий',
         'IVP RK4/PI с точными y(-1),y′(-1); R/MC — ограниченное решение на всей прямой; section.md/tex',
         'Для сопоставления при равном объёме информации нужен отдельный протокол доступности/погрешности НУ','текущее задание, пункт 4; report.tex: Математическая постановка'),
        ('T06','Скорость/порядок сходимости сравниваемых реализаций','не доказано этой серией',
         'Одна фиксированная конфигурация и одна контрольная сетка; распределение ошибок не является серией уточнений',
         'Не менее трёх успешных уровней одного изменяемого параметра при фиксированных остальных; MC с учётом статистического разброса','текущее задание, пункт 5; .claude/skills/experiment/SKILL.md'),
        ('T07','Контроль непрерывного максимума ошибки на [-1,1]','не сертифицировано',
         'Измерены значения только в 33 узлах',
         'Сгущение контрольной сетки как диагностика; для строгой границы — аналитический контроль между узлами','report.tex: grid33 и Заключение'),
        ('T08','Самый быстрый надёжный метод при Egrid≤0,01','не закрыто текущей серией',
         'Контракт v1 требует другой допуск и warm end-to-end timing; CSV E33/0,05 времён не содержит',
         'Согласованные профили, повторные замеры стоимости достижения 0,01 и оценка надёжности по исходному контракту','reports/ml_selector_contract/CONTRACT.md §§2,4; критерий исторического ТЗ'),
        ('T09','Полная сеточная статистика MC и разделение смещений','не установлено на1000',
         'Доступно mean4 в нуле; полная MC E33 отсутствует; нижняя граница даёт62 нарушений и938 неизвестных',
         'Сохранённые seed-ответы на всей сетке, репликации; отдельный анализ M, шага времени и горизонта','report.tex: Заключение; текущее задание, пункт 4'),
        ('T10','Обучить и проверить селекторы по точечной ошибке E0','исторический эксперимент выполнен; улучшение не подтверждено',
         'report.tex: выбранный бинарный253, диагностический258, трёхметодный эксперимент; результаты остаются E0',
         'Нельзя переносить прежние ML-метрики на E33 или заявлять полезный перенос без независимой проверки','report.tex: Цель и задачи; Результаты; Заключение'),
        ('T11','Обоснованный ML-селектор по сеточной цели E33','не проверен',
         '997 четырёхметодных winner принадлежат одному классу RK4; постоянное правило воспроизводит их все',
         'Четырёхклассовый fit на этих метках не нужен; для задачи без RK4 нужны полные сопоставимые MC33 и заранее определённая цель','report.tex: grid33; saved_grid33/summary.json'),
        ('T12','Независимая генерализация на новые семейства','не установлено',
         'Существующий отчёт прямо ограничивает выводы исторически раскрытым Test',
         'Замороженный протокол и новая независимая семейная проверка при отдельной научной гипотезе','report.tex: Заключение'),
        ('T13','Надёжность пространственной реализации R и отсутствие катастрофических ошибок','не доказано',
         '27 превышений0,05; четыре E33>1; два крупнейших дают98,84% суммы; причинный вывод из CSV невозможен',
         'Контролируемая диагностика окна/продолжения/шага и проверка исправления на независимых задачах; не подменять текущие ответы','report.tex: Ограничения реализации Ремизова; CSV'),
        ('T14','Научная интерпретация и воспроизводимость оформления','закрыто для нового комплекта',
         'Исходный CSV, machine-readable figure_data, SHA, saved-only script, проверка таблиц/рисунков/LaTeX',
         'Сборка section.tex в полном TeX-отчёте и проверка пагинации в TeX-окружении','текущее задание, пункты 3,5,6')]]


def fmt(value):
    return f'{value:.6g}'


def comparison_md(data,scope):
    lines=['| Метод | Величина | n | Среднее | Медиана | Q₀,₉₅ | Максимум | Допуск E33≤0,05 |',
           '|---|---|---:|---:|---:|---:|---:|---|']
    for r in data['comparison']:
        if r['scope'] != scope: continue
        mc=r['method']=='MC'
        outcome=f"≥{r['E33_fail_lower_bound']} нарушений; {r['E33_unknown']} неизвестных" if mc else f"{r['E33_pass']}/{r['n_values']} ({100*r['pass_fraction_available']:.4f}%)"
        lines.append('| '+' | '.join([LABELS[r['method']], 'E₀, нижняя граница E33' if mc else 'E33',str(r['n_values']),
            *[fmt(r[k]) for k in ('mean','median','q95','maximum')],outcome])+' |')
    return '\n'.join(lines)


def texnum(value):
    s=fmt(value)
    if 'e' in s:
        mantissa,exponent=s.split('e')
        return mantissa+r'\times10^{'+str(int(exponent))+'}'
    return s


def comparison_tex(data,scope='all_available'):
    lines=[r'\begin{table}[!htbp]',r'\centering',
           r'\caption{'+('Полное доступное покрытие.' if scope=='all_available' else 'Общие 997 задач.')+r' У MC приведена ошибка в нуле как нижняя граница, а не измеренная $E_{33}$.}',
           r'\label{tab:grid33final-'+scope+'}',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{llrrrrrr}',r'\toprule',
           r'Метод & Величина & $n$ & Среднее & Медиана & $Q_{0.95}$ & Максимум & Допуск $E_{33}\le0.05$\\',r'\midrule']
    for r in data['comparison']:
        if r['scope']!=scope:continue
        mc=r['method']=='MC'
        method='MC mean4' if mc else LABELS[r['method']]
        outcome=f"$\\ge {r['E33_fail_lower_bound']}$ нарушений; {r['E33_unknown']} неизвестных" if mc else f"{r['E33_pass']}/{r['n_values']}"
        lines.append(' & '.join([method,r'$E_0\le E_{33}$' if mc else '$E_{33}$',str(r['n_values']),
             *['$'+texnum(r[k])+'$' for k in ('mean','median','q95','maximum')],outcome])+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}}',r'\end{table}'])
    return '\n'.join(lines)+'\n'


def section_md(data,figures):
    top=data['exceedances'][:3]
    text=f'''# Максимальная ошибка сохранённых численных решений на общей контрольной сетке

## Методика

Рассматриваются 1000 изготовленных линейных ОДУ второго порядка проекта №2488.
Сравнение выполнено по сохранённым ответам при фиксированных конфигурациях.
Определим G33 = {{xₖ = −1+k/16: k=0,…,32}} и
**Eₘ,33 = maxₓₖ∈G33 |yₘ(xₖ) − y_exact(xₖ)|**. Допуск выполнен при Eₘ,33≤0,05,
включая равенство. Это дискретная максимальная абсолютная ошибка; она не
сертифицирует максимум на непрерывном интервале. Производные в эту метрику не входят.

R означает сохранённый массовый профиль `ChernovSolver.solve`: n=128,
513 временных узлов, dx=0,0125, кубический сплайн, продолжение taper и padding=12.
Это фиксированная реализация аппроксимации Ремизова, а не новая серия адаптивного решателя.
RK4 и продукт-интеграл используют 128 шагов h=1/64 на [−1,1]. Их 129 сохранённых
узлов содержат G33: взят каждый четвёртый, без интерполяции. Exact-массивы
уже сохранены и ранее сверены; новое оценивание точного решения не выполнялось.
MC — арифметическое среднее четырёх сохранённых MC1024-ответов только в нуле,
с запрошенным верхним шагом 1/128 и tail_eps=10⁻⁴. Реальный шаг подстраивался под
горизонт; это не гарантированно точное равенство Δt=1/128. Времена из разных
исторических серий здесь не используются для вывода об эффективности.

### Постановка и входные условия

В обычной форме уравнение записано как a y″+b y′+c y=g, где c=c_R−λ,
g=−g_R. R/MC приближают ограниченное на всей ℝ решение (λ−L)y=g_R,
L=a∂²+b∂+c_R, и не получают произвольных начальных условий.
RK4/PI решают задачу Коши на [−1,1] и получают точные y(−1), y′(−1),
извлечённые при изготовлении задачи. Благодаря компактному носителю exact,
гладкости, равномерной эллиптичности, λ>‖c_R‖∞ и согласованным начальным
условиям целевая функция общая. При этом объём предоставленной информации
различается: результат RK4/PI условен доступностью точных НУ.
Для нового уравнения такие НУ не возникают бесплатно. Здесь сравниваются
конкретные реализации с разными бюджетами и входами, а не одинаковые
информационные и вычислительные условия.

## Результаты

### Полное доступное покрытие

{comparison_md(data,'all_available')}

Статистики в строках относятся к указанной величине: для MC это **E₀**, а не E33.
Числа в таблице округлены для чтения; comparison.csv содержит полные float-значения.
Q₀,₉₅ — линейно интерполированный эмпирический квантиль.
У Ремизова точны 970/997={970/997*100:.4f}% доступных ответов. Если знаменатель —
все 1000 задач, подтверждённо точных 97%, с измеренным превышением 2,7%,
с отсутствующим ответом 0,3%. Отсутствие ответа не является измеренной ошибкой.

### Общее пересечение

{comparison_md(data,'common997')}

В первой таблице PI точен на 404/1000, на пересечении — 402/997.
Это разные покрытия. Все сравнения между измеренными распределениями на
рисунке 1 используют те же общие 997 задач.

### MC: нижняя граница и область вывода

Обозначим E_MC,0=|mean(f₀,f₁,f₂,f₃)−y_exact(0)|. Это ошибка среднего ответа,
а не среднее или медиана четырёх абсолютных ошибок.
Если дополнить **тот же** сохранённый mean4-ответ другими узлами G33,
сохраняя значение в нуле, то E_MC,33≥E_MC,0, поскольку 0∈G33.
На всех общих 997 задачах проверено строгое неокруглённое неравенство
E_RK4,33 < min(E_R,33,E_PI,33,E_MC,0). Минимальные положительные запасы:

'''
    for m in ('R','PI','MC'):
        v=data['minimum_margins'][m]
        text+=f"- {m}: {v['margin']:.17g}, `{v['equation_id']}`.\n"
    text+='''
Следовательно, RK4 — единственный победитель по E33 на этих 997 задачах
для сохранённых R/PI и любого дополнения указанного MC-ответа.
По всему покрытию RK4 строго меньше PI E33 и MC E0 на 1000 задачах;
для трёх отсутствующих R четырёхметодный вывод не установлен.
Нижняя граница MC уже превышает 0,05 на 62 задачах; у остальных 938
достижение сеточного допуска неизвестно. **938/1000 не является долей
сеточно точных MC-решений.** Среднее, медиану, максимум полной MC E33
и точную долю её выполнения допуска по этим данным определить нельзя.
Вывод не переносится на новую случайную реализацию или математическое ожидание MC.

### Большие ошибки Ремизова и пропуски

'''
    for r in top: text+=f"- `{r['equation_id']}`: E33={r['R_E33']:.12g}.\n"
    text+=f'''
Всего {len(data['exceedances'])} из 997 ответов превышают 0,05, из них
{data['R_above_one']} имеют E33>1. Два крупнейших дают
{100*data['R_largest_two_sum_share']:.4f}% суммы R E33. Поэтому среднее
39,4119 и медиана 0,00274797 описывают разные стороны распределения.
Это описательная концентрация суммы ошибок, не доказательство причины
выбросов или нарушения условий теоремы. Историческая диагностика
продолжения сплайна обсуждалась в прежнем отчёте; её ответы здесь
не подставляются вместо исходного cubic/taper. Все 27 превышений
перечислены в remizov_exceedances.csv и показаны без удаления из статистик.

Отсутствуют ответы `{data['missing_ids'][0]}`, `{data['missing_ids'][1]}`,
`{data['missing_ids'][2]}`: прежний статус worker_exception, не численный
результат с большой ошибкой. Их ошибки R и четырёхметодные winner пусты;
доступные RK4/PI и MC E0 сохранены в missing_remizov.csv.

## Графические материалы

'''
    for i,f in enumerate(figures,1):
        text+=f"![Рисунок {i}](figures/{f['name']}.png)\n\n**Рисунок {i}.** {f['caption']}\n\n"
    text+='''## Интерпретация и ограничения

Распределения подтверждают малую измеренную ошибку RK4 при данном бюджете
и точных НУ. PI существенно чаще превышает допуск на сетке, чем можно
было заключить по одной точке. У R большая часть ответов точна, но редкие
экстремальные значения резко увеличивают среднюю ошибку. Ни одна из этих
закономерностей не объясняет причин автоматически.

Контрольная сетка и параметры не уточнялись: порядок или скорость сходимости
не оценивались. Низкая ошибка в одной конфигурации не означает более быстрого
схождения по шагу или лучшей стоимости достижения заданной точности.
Увеличение плотности контрольной сетки само по себе не является теоремой о
непрерывной норме. Уравнения из одного семейства зависимы; показанные доли —
описательные доли фиксированного набора. Доверительные интервалы популяционной
успешности из предположения о 1000 независимых задачах не строились.

Исторические ML-модели и показатели относятся к E0. Этот раздел их не
переоценивает и не доказывает полезность селектора по E33. На всех 997
имеющихся четырёхметодных метках победитель один — RK4; обучение такого
селектора не добавило бы содержательной проверки выбора. Для сравнения без
RK4 нужны полные сопоставимые сеточные MC-данные. Перенос на новые семейства,
другие бюджеты, неточные НУ и иной класс ОДУ остаётся отдельным вопросом.

## Выводы

1. Сохранённых ответов достаточно для воспроизводимого сравнения E33 у R/RK4/PI
   без повторного решения ОДУ: 970/997, 1000/1000 и 404/1000 выполнений допуска.
2. Строгая победа RK4 доказана для фиксированных ответов на общих 997 задачах;
   для MC используется нижняя граница с явным условием сохранения mean4 в нуле.
3. Полная MC E33, три R-ответа, непрерывная норма и асимптотическая сходимость
   этим сравнением не установлены. Универсальное превосходство RK4 не утверждается.
4. Прежние E0/ML и историческое требование «быстрейший при 0,01» остаются
   отдельными постановками. Реестр закрытых и открытых требований — REQUIREMENTS.md.

Источники: `results/saved_grid33_1000_2026-11-04/grid33_all1000.csv`, его summary,
verification и config; `results/scientific_report_revision_2026-11-04/report.tex`,
разделы «Математическая постановка», «Отдельная проверка…», «Заключение».
Все новые численные методы/MC/fit/inference-вызовы: 0. Выполнена только
обработка сохранённых скалярных ошибок, оформление и проверка соответствия CSV.
'''
    return text


def section_tex(data,figures):
    text=r'''% Insert into the existing report after copying this package to grid33_supplement/.
% Uses the existing report's booktabs, graphicx and Russian font setup.
\subsubsection{Максимальная ошибка сохранённых решений на общей сетке}
\label{sec:grid33final}
\paragraph{Методика.}
Исследованы сохранённые ответы для 1000 изготовленных ОДУ проекта №2488.
Общая контрольная сетка и критерий определены формулами
\[
G_{33}=\{-1+k/16:\ k=0,\ldots,32\},\qquad
E_{m,33}=\max_{x_k\in G_{33}}|y_m(x_k)-y_{\mathrm{exact}}(x_k)|,
\qquad E_{m,33}\le0.05.
\]
Это дискретная максимальная абсолютная ошибка, не сертификат непрерывной
нормы. Производные в критерий не входят. У R сохранены 33 узла, у RK4/PI
выбраны каждый четвёртый из 129 узлов: интерполяция не применяется.
Новых решений ОДУ, MC-траекторий, обучения и inference не выполнялось.

Массовый профиль R использует $n=128$, 513 временных узлов,
$dx=0.0125$, кубический сплайн с продолжением \texttt{taper},
\texttt{padding}=12. RK4/PI используют 128 шагов $h=1/64$ на $[-1,1]$.
MC представлен средним четырёх MC1024-ответов только в нуле, с запрошенным
верхним шагом $1/128$ и \texttt{tail\_eps}$=10^{-4}$; фактический шаг
подстраивается под горизонт. Бюджеты методов не эквивалентны.

\paragraph{Различия входных условий.}
Обычная форма $ay''+by'+cy=g$ связана с операторной формой
$(\lambda-L)y=g_R$ соотношениями $c=c_R-\lambda$, $g=-g_R$,
$L=a\partial_x^2+b\partial_x+c_R$. R/MC приближают ограниченное решение
на всей $\mathbb R$, а RK4/PI решают задачу Коши на $[-1,1]$ с точными
$y(-1),y'(-1)$ из изготовленного решения. Его компактный носитель,
гладкость, равномерная эллиптичность и $\lambda>\|c_R\|_\infty$
обеспечивают согласованность целевой функции при этих НУ.
Однако объём входной информации различается: для нового ОДУ точные НУ
должны быть известны заранее. Сопоставление при одинаковой доступности
информации и одинаковой стоимости данным экспериментом не выполнено.

\paragraph{Результаты.}
В таблицах приведены полное доступное покрытие и общие 997 задач.
Все статистики относятся к величине в соответствующей строке: MC~$E_0$
не заменяет полную MC~$E_{33}$. Квантиль $0.95$ линейно интерполирован.
\input{grid33_supplement/comparison_table.tex}
\input{grid33_supplement/comparison_common997.tex}
Достижение допуска: R~970/997, RK4~1000/1000, PI~404/1000;
на общем пересечении PI~402/997. У R доля среди доступных ответов
равна 97.2919\%, среди всех задач --- 97\%; измеренных превышений
27 (2.7\%), отсутствующих ответов 3 (0.3\%). Пропуски не считаются
численной ошибкой. Их ID: \texttt{AMV1\_F0134\_S03},
\texttt{AMV1\_F0136\_S01}, \texttt{AMV1\_F0139\_S00}.
Сохранённый статус --- \texttt{worker\_exception}; ошибки R неизвестны.

\paragraph{Нижняя граница MC и строгий победитель.}
Для $\bar f(0)=\frac14\sum_{s=0}^{3}f_s(0)$ обозначим
$E_{\mathrm{MC},0}=|\bar f(0)-y_{\mathrm{exact}}(0)|$.
Это ошибка среднего ответа, не среднее абсолютных ошибок seed.
При дополнении именно этого сохранённого ответа остальными узлами,
без изменения $\bar f(0)$, выполняется
\[
E_{\mathrm{MC},33}\ge E_{\mathrm{MC},0},\qquad
E_{\mathrm{RK4},33}<\min\{E_{\mathrm R,33},E_{\mathrm{PI},33},E_{\mathrm{MC},0}\}
\]
на каждой из общих 997 задач. Минимальные положительные запасы составляют
'''
    for i,m in enumerate(('R','PI','MC')):
        d=data['minimum_margins'][m]
        escaped_id=d['equation_id'].replace('_',r'\_')
        text+=f"${texnum(d['margin'])}$ ({m}, \\texttt{{{escaped_id}}})"+(' и ' if i==1 else '; ' if i==0 else '.\n')
    text+=r'''Следовательно, RK4 --- единственный победитель по сеточному критерию
на этих 997 задачах для сохранённых R/PI и дополнения того же MC-ответа.
На всех 1000 задачах RK4 также строго меньше PI~$E_{33}$ и MC~$E_0$;
сравнение с R на трёх пропусках неизвестно.
Полная MC~$E_{33}$ не измерена ни на одной задаче этой массовой серии.
Нижняя граница доказывает 62 нарушения допуска, для 938 достижение
допуска неизвестно. Число 938/1000 нельзя интерпретировать как сеточную
точность MC. Среднее, медиана и максимум MC~$E_{33}$ неизвестны.
Вывод не относится к новой случайной реализации или ожиданию MC.

\paragraph{Большие ошибки Ремизова.}
Все 27 превышений допуска сохранены в статистиках.
'''
    for r in data['exceedances'][:3]:
        text+=r'\texttt{'+r['equation_id'].replace('_',r'\_')+'}: $E_{33}='+texnum(r['R_E33'])+'$.\n'
    text+=f"Четыре ответа имеют $E_{{33}}>1$; два крупнейших дают {100*data['R_largest_two_sum_share']:.4f}\\% суммы ошибок R.\n"
    text+=r'''Среднее $39.4119$ при медиане $0.00274797$ отражает тяжёлый правый хвост.
Термин <<выброс>> используется описательно, а не как результат теста.
Причину катастрофических ошибок или нарушение условий применимости нельзя
доказать по этой CSV. Исторические диагностики продолжения сплайна
не подменяют исходный профиль \texttt{cubic/taper} в текущих таблицах.

\paragraph{Распределения и доли.}
'''
    for i,f in enumerate(figures,1):
        cap=f['caption'].replace('_',r'\_').replace('₀','$ _0$').replace('₃₃','$_{33}$')
        text+='\n'+r'\begin{figure}[!htbp]'+'\n'+r'\centering'+'\n'+r'\includegraphics[width=0.95\textwidth]{grid33_supplement/figures/'+f['name']+'.pdf}\n'+r'\caption{'+cap+'}\n'+r'\label{fig:grid33final-'+str(i)+'}\n'+r'\end{figure}'+'\n'
    text+=r'''
\paragraph{Интерпретация и ограничения.}
RK4 имеет наименьшую измеренную ошибку при данном бюджете и точных НУ.
Это не доказательство порядка или скорости сходимости: контрольная сетка
и параметры не уточнялись. Для такой оценки нужны отдельные серии
как минимум из трёх успешных уровней при контроле остальных параметров.
Малая ошибка в одной конфигурации не устанавливает лучшую стоимость
достижения точности. Уточнение контрольной сетки может диагностировать
пропущенные экстремумы, но само по себе не доказывает непрерывной границы.
Задачи внутри семейства зависимы; доли описывают фиксированный набор,
а не доверительные оценки для 1000 независимых наблюдений.

Исторические ML-метрики относятся к $E_0$ и на $E_{33}$ не переносятся.
На нынешних 997 четырёхметодных метках один класс RK4; обучение такого
селектора не даёт содержательной проверки выбора. Для постановки без
RK4 нужны полные сопоставимые MC33. Независимая генерализация на новые
семейства, другие бюджеты, неточные НУ и иной класс ОДУ не установлена.

\paragraph{Выводы и оставшиеся требования.}
Сохранённые массивы позволяют без повторного решения получить E33
для R/RK4/PI и условно доказать строгую победу RK4 на общих 997 задачах.
Три отсутствующих R-ответа, полная MC33, непрерывная норма и сходимость
остаются открытыми. Универсальное превосходство RK4 не утверждается.
Историческое требование выбора быстрейшего надёжного метода при
$E_{\mathrm{grid}}\le0.01$ не закрывается серией с $\varepsilon=0.05$
без сопоставимых времён. Подробный реестр доказательств и требуемых
экспериментов вынесен в сопровождающий файл \texttt{REQUIREMENTS.md}.
Источники: сохранённая CSV \texttt{saved\_grid33\_1000\_2026-11-04},
её независимая проверка и исходный научный отчёт. Новых solver/MC/ML: 0.
'''
    return text


def build(root,out):
    root,out=Path(root).resolve(),Path(out).resolve()
    require(not out.exists(),'Choose a new output directory; historical files are never overwritten')
    paths=[SOURCE/name for name in ('grid33_all1000.csv','summary.json','verification.json','input_sha256.json','missing_remizov.csv')]+[REPORT,CONTRACT,AUDIT]
    before={str(p):sha(root/p) for p in paths}
    source_verification=json.loads((root/SOURCE/'verification.json').read_text())
    require(source_verification['passed'],'Saved release verification did not pass')
    for name,digest in source_verification['artifacts_sha256'].items():
        require(sha(root/SOURCE/name)==digest,'Saved release artifact changed: '+name)
    rows=load_rows(root/SOURCE/'grid33_all1000.csv');data=build_data(rows)
    saved=json.loads((root/SOURCE/'summary.json').read_text())
    for r in data['comparison']:
        if r['method']=='MC':continue
        expected=saved['full_available' if r['scope']=='all_available' else 'common997'][r['method']]
        require(expected['n']==r['n_values'] and expected['accurate']==r['E33_pass'],'Saved summary counts differ')
        for key in ('mean','median','q95','maximum'):
            require(math.isclose(r[key],expected[key],rel_tol=2e-13,abs_tol=1e-15),'Saved summary statistic differs')
    out.mkdir(parents=True,exist_ok=False)
    figures,mpl_version=draw_figures(out,data)
    write_csv(out/'comparison.csv',data['comparison'])
    write_csv(out/'remizov_exceedances.csv',data['exceedances'])
    write_csv(out/'threshold_shares.csv',data['shares'])
    with (root/SOURCE/'missing_remizov.csv').open(newline='') as stream:
        write_csv(out/'missing_remizov.csv',list(csv.DictReader(stream)))
    write_json(out/'figure_data.json',data)
    write_json(out/'figure_manifest.json',dict(figures=figures,logarithm='base10 error axis, linear fractions; no zero shift or trimming'))
    requirements=requirement_rows();write_csv(out/'requirements.csv',requirements)
    requirement_text='# Доказательства выполнения требований и открытые научные задачи\n\n'
    requirement_text+='Реестр сопоставляет текущие шесть пунктов задания, цели существующего научного отчёта и исторический контракт №2488 v1. Его допуск 0,01 и критерий скорости сохранены как отдельное требование; они не заменены текущим E33/0,05. Статусы планов не считаются доказательствами выполнения.\n\n'
    requirement_text+='| ID | Требование | Статус | Доказательство | Что ещё требуется | Основание |\n|---|---|---|---|---|---|\n'
    for r in requirements:requirement_text+='| '+' | '.join(str(r[k]) for k in r)+' |\n'
    for name,body in [('section.md',section_md(data,figures)),('section.tex',section_tex(data,figures)),
        ('comparison_table.tex',comparison_tex(data)),('comparison_common997.tex',comparison_tex(data,'common997')),
        ('comparison.md','# Сравнительная таблица\n\n'+comparison_md(data,'all_available')+'\n\nMC: статистики E0 — только нижней границы E33.\n\n'+comparison_md(data,'common997')+'\n'),
        ('REQUIREMENTS.md',requirement_text)]:
        with (out/name).open('x',encoding='utf-8') as handle:handle.write(body)
    readme='''# Научное оформление E33 · проект №2488

Основной раздел: [section.md](section.md), LaTeX-вставка [section.tex](section.tex).
Сравнительные таблицы: [comparison.md](comparison.md), [comparison.csv](comparison.csv)
и две готовые LaTeX-таблицы. [REQUIREMENTS.md](REQUIREMENTS.md) разделяет
закрытые требования, частичное покрытие и оставшиеся эксперименты.
В figures/ находятся четыре рисунка в PNG и векторном PDF; подписи и SHA
записаны в figure_manifest.json, точные данные — в figure_data.json.
remizov_exceedances.csv содержит все 27 превышений допуска; missing_remizov.csv — три пропуска.

Новый каталог не заменяет исторический отчёт. В нём section «Отдельная проверка…»
может быть заменён этой вставкой после редакторского решения. Для Overleaf
скопируйте данный каталог как grid33_supplement/ рядом с прежним report.tex,
затем используйте `\\input{grid33_supplement/section.tex}` в нужном месте.
Фигуры/таблицы используют явно этот относительный префикс и уникальные метки.
Пакеты booktabs/graphicx/русские шрифты уже есть в прежнем отчёте.
TeX-компиляция и пагинация здесь не проверены: локального движка нет.
Просмотрены готовые PNG; PDF рисунков экспортирован Matplotlib.

Воспроизведение из корня в НОВЫЙ каталог:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/present_saved_grid33.py --out results/grid33_scientific_reproduction
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/verify_grid33_presentation.py --out results/grid33_scientific_reproduction
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_grid33_presentation.py tests/test_saved_grid33.py tests/test_four_method_accuracy.py
```

Числовая/структурная проверка нового комплекта: verification.json.
При подготовке поставленного комплекта тесты, воспроизводимость и сохранность
прежних файлов дополнительно закреплены в checks.json и integrity.json.
Новых solver, MC, fit, model inference и подборов: 0.
'''
    (out/'README.md').write_text(readme)
    require(all(sha(root/p)==digest for p,digest in before.items()),'A source changed during presentation')
    write_json(out/'input_sha256.json',before)
    write_json(out/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),matplotlib=mpl_version,
              source_script_sha256=sha(Path(__file__)),new_solver_calls=0,new_MC_calls=0,new_fit_calls=0,new_model_inference_calls=0))
    print(json.dumps(dict(out=str(out),rows=len(rows),figures=len(figures),comparison_rows=len(data['comparison']),requirements=len(requirements),new_solver_calls=0),ensure_ascii=False))
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();build(args.root,args.out)


if __name__=='__main__':main()
