"""Targeted editorial integration into the frozen report; saved files only."""
from pathlib import Path
import csv
import difflib
import hashlib
import json

OUT=Path(__file__).resolve().parent
BASE=(OUT/'base_report.tex').read_text()
text=BASE
changes=[]


def replace(old,new,reason):
    global text
    if text.count(old)!=1:raise ValueError('Expected exactly one editorial anchor: '+old[:100])
    text=text.replace(old,new);changes.append(reason)


def read(name):return json.loads((OUT/'evidence'/name).read_text())
def csv_rows(name):
    with (OUT/'evidence'/name).open(newline='') as stream:return list(csv.DictReader(stream))
def num(value,digits=7):
    s=f'{float(value):.{digits}g}'
    if 'e' in s:
        a,b=s.split('e');return a+r'\cdot10^{'+str(int(b))+'}'
    return s


replace(r'''\usepackage[T2A]{fontenc}
\usepackage[utf8]{inputenc}''',r'''% Portable Cyrillic fonts: retain the original path for pdfLaTeX.
\usepackage{iftex}
\ifPDFTeX
  \usepackage[T2A]{fontenc}
  \usepackage[utf8]{inputenc}
\else
  \usepackage{fontspec}
  \IfFontExistsTF{Times New Roman}
    {\setmainfont{Times New Roman}}
    {\setmainfont{DejaVu Serif}}
  \IfFontExistsTF{Courier New}
    {\setmonofont{Courier New}}
    {\setmonofont{DejaVu Sans Mono}}
\fi''','Engine-dependent Cyrillic font setup is required by the available XeTeX engine; pdfLaTeX settings remain as in the basis.')
replace(r'\fontfamily{Tempora-TLF}\fontsize{14}{16}\selectfont',
        r'\ifPDFTeX\fontfamily{Tempora-TLF}\fi\fontsize{14}{16}\selectfont',
        'Keep the original title font under pdfLaTeX; use the selected Unicode Cyrillic font under XeTeX.')
replace(r'\begin{titlepage}',r'\hypersetup{pageanchor=false}'+'\n'+r'\begin{titlepage}',
        'Disable the invisible title-page anchor to avoid duplicating body page 1.')
replace(r'\end{titlepage}',r'\end{titlepage}'+'\n'+r'\hypersetup{pageanchor=true}',
        'Restore page anchors after the title page.')
replace(r'\newcommand{\code}[1]{\texttt{#1}}',
        r'\DeclareRobustCommand{\code}[1]{\texttt{\let\savedunderscore\_\def\_{\savedunderscore\allowbreak}#1}}',
        'Allow line breaks after underscores in long identifiers; no identifier text changes.')

replace(r'\usepackage{graphicx}',r'\usepackage{graphicx}'+'\n'+r'\usepackage{rotating}',
        'Only add rotating for legible full-page ready E33 figures.')
replace('% Исправленная редакция от 04.11.2026; исходный текст сохранён в original_report.tex.\n% Новых решений ОДУ, MC-траекторий и обучений моделей не выполнялось.',
        '% Сводная редакция от 05.11.2026; основа сохранена в base_report.tex.\n'
        '% Использованы только ранее выполненные исследования. При подготовке этой редакции новых solver/MC/ML вызовов нет.',
        'Date/provenance comment distinguishes completed experiments from this saved-only editorial task.')

method_anchor='''\item Отдельно проверена максимальная абсолютная ошибка на 33 узлах
$[-1,1]$ по сохранённым ответам Ремизова, RK4 и мультипликативного
интеграла; для Монте-Карло использована только нижняя оценка через ноль.'''.replace('\\item',r'\item')
method_new=method_anchor+r'''
\item На 60 development-задачах выполнен отдельный ограниченный пилот
минимальной стоимости при сеточном допуске $0.01$: RK4 и PI,
$N=32,64,128,256$, прогрев и пять повторов каждой конфигурации.
\item На 52 development-задачах сравнены сохранённый \code{cubic/taper}
и экспериментальный \code{cubic/zero} при неизменных остальных параметрах;
проверены выбросы, регрессии и пространственные предпосылки.'''
replace(method_anchor,method_new,'Abstract methods: add only completed cost and boundary experiments.')

abstract_results=r'''\subsubsection*{Результат 6. Стоимость достижения сеточного допуска}
На 60 задачах oracle при $E_{33}\le0.01$ выбирает только RK4:
59 раз $N=32$ и один раз $N=64$. Лучшая постоянная конфигурация
с допуском на всех задачах --- RK4($N=64$). Oracle экономит
49.1967\% в среднем по относительным стоимостям, или 48.2849\%
по суммарной стоимости, но полностью совпадает с RK4-only выбором $N$.
PI не выигрывает ни разу; решение для дальнейшего ML выбора метода ---
NO-GO. Это оценка oracle-потенциала, а не измеренный выигрыш модели.

\subsubsection*{Результат 7. Граничное продолжение Ремизова}
Все три исходные ошибки $E_{33}>1$ на выбранной development-части
устранены при \code{cubic/zero}; контролей в допуске осталось 30/30,
проблемных ответов в допуске стало 1/22 вместо 0/22. Формальное
NO-GO по заранее заданному строгому условию отсутствия регрессий
вызвано тремя увеличениями порядка $10^{-9}$. Практически заметного
ухудшения они не устанавливают. Производственный профиль не заменён;
устойчивость пространственной аппроксимации полностью не сертифицирована.

'''
replace(r'\subsubsection*{Выводы}'+'\n'+r'\begin{enumerate}',abstract_results+r'\subsubsection*{Выводы}'+'\n'+r'\begin{enumerate}',
        'Abstract: two concise completed results, with correct NO-GO scopes.')

goals_anchor='''\\item Отдельно проверить ошибку на 33 узлах по сохранённым ответам
и установить, сохраняется ли победитель без полного сеточного MC-расчёта.'''
replace(goals_anchor,goals_anchor+r'''
\item Проверить на ограниченной development-выборке, даёт ли выбор
между RK4 и PI выигрыш в стоимости при $E_{33}\le0.01$ относительно
лучшей постоянной конфигурации и выбора числа шагов только у RK4.
\item Проверить на development-задачах, устраняет ли замена
\code{cubic/taper} на \code{cubic/zero} крупные ошибки без потерь
допуска, сохраняя остальные параметры и математическую формулу метода.''',
        'Goals: add the two research questions actually completed.')

boundary_begin=text.index(r'\paragraph{Ограничения реализации Ремизова\\}')
boundary_end=text.index(r'\subsubsection{Выбор из трёх методов}',boundary_begin)
short_boundary=r'''\paragraph{Пространственная реализация Ремизова\\}
Редкие крупные значения существенно влияют на среднюю ошибку R.
Контролируемое сравнение граничных продолжений и расширение прежней
диагностики до development-выборки рассмотрены отдельно
в разделе~\ref{sec:boundarydev}. Исходные \code{cubic/taper}-ответы
не подменяются экспериментальными в таблицах этого раздела.

Сравнение на одной контрольной сетке не является исследованием
скорости сходимости или сертификацией непрерывной нормы.

'''
text=text[:boundary_begin]+short_boundary+text[boundary_end:]
changes.append('Move duplicated old two-case boundary discussion into the expanded boundary development section; retain historical Test result there.')

full=csv_rows('grid33_figures/comparison.csv')
full=[r for r in full if r['scope']=='all_available']
full_table=r'''
\begin{table}[!htbp]
\centering
\caption{Полное доступное покрытие сохранённой серии. В строке MC
статистики относятся только к нижней оценке $E_{\mathrm{MC},0}$,
а не к неизмеренной $E_{\mathrm{MC},33}$.}
\label{tab:grid33-full}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrl}
\toprule
Метод и величина & $n$ & Средняя & Медиана & Максимум & Допуск $E_{33}\le0.05$\\
\midrule
'''
labels=dict(R='Ремизов, $E_{33}$',RK4='RK4, $E_{33}$',PI='PI, $E_{33}$',MC='MC, $E_0$')
for r in full:
    last=(r'\parbox{24mm}{\raggedright $\ge62$ нарушений, 938 неизвестных}' if r['method']=='MC' else r['E33_pass']+'/'+r['n_values'])
    full_table+=' & '.join([labels[r['method']],r['n_values'],*['$'+num(r[k])+'$' for k in ('mean','median','maximum')],last])+r'\\'+'\n'
full_table+=r'''\bottomrule
\end{tabular}}
\end{table}

Для MC из нижней оценки следуют 62 нарушения допуска; для остальных
938 задач достижение сеточного допуска неизвестно. Число 938/1000
не является долей сеточно точных MC-решений. Значения полной
$E_{\mathrm{MC},33}$ и её признака точности остаются отсутствующими.
Три отсутствующих R-ответа --- \texttt{F0134/S03}, \texttt{F0136/S01},
\texttt{F0139/S00} --- учитываются отдельно. Их нельзя считать ошибками
нулевой величины или поражениями R; четырёхметодный winner для них
не установлен. Источники полных таблиц и графиков~\cite{savedgrid33release,savedgrid33figures}.

'''
anchor=r'\paragraph{Нижняя оценка для Монте-Карло и победа RK4\\}'
replace(anchor,full_table+anchor,'E33: full-coverage four-method table with MC lower-bound type and missing R explicitly preserved.')

figures=r'''\paragraph{Распределения ошибок и выполнение допуска\\}
На рисунке~\ref{fig:grid33-ecdf} сравниваются одинаковые 997 задач.
Логарифмическая ось показывает весь диапазон ошибок без удаления
выбросов. На рисунке~\ref{fig:grid33-hist} приведено полное доступное
покрытие; панель MC относится только к $E_0$ и не является измерением
его сеточного распределения. Все 27 исходных превышений R сохранены
в статистиках и отдельно показаны на рисунке~\ref{fig:grid33-outliers}.
Два крупнейших дают 98.8426\% суммы R-ошибок; это описательное свойство
выборки, а не доказательство причины. На рисунке~\ref{fig:grid33-shares}
знаменатель равен 1000: у R 97\% подтверждённо точных ответов,
2.7\% превышений и 0.3\% пропусков. Среди доступных 997 ответов
точных 97.2919\%. Ошибки внутри семейства зависимы; эти доли не
интерпретируются как независимые популяционные вероятности.

'''
specs=[('01_ecdf_common997','ecdf','Накопленное распределение измеренных $E_{33}$ на общих 997 задачах. Логарифмическая ось ошибок; пунктиром показан допуск $0.05$. Полная MC~$E_{33}$ отсутствует.'),
       ('02_histograms_available','hist','Распределения по всему доступному покрытию; одинаковые логарифмические интервалы. Ордината --- доля наблюдений в интервале, не плотность. Отдельная MC-панель показывает лишь нижнюю оценку $E_0$. Пунктир --- $0.05$.'),
       ('03_remizov_exceedances','outliers','Все 27 исходных ответов R с $E_{33}>0.05$. Два крупнейших выделены цветом. Выбросы не удалены из исходной статистики; термин используется описательно.'),
       ('04_threshold_shares','shares','Доли от всех 1000 задач. Пропуски R отделены от измеренных превышений. Для MC 62 нарушения следуют только из нижней оценки, в 938 случаях выполнение допуска неизвестно; это не 938 сеточно точных ответов.')]
for name,label,caption in specs:
    figures+=r'\begin{sidewaysfigure}[p]'+'\n'+r'\centering'+'\n'
    # rotating resets textwidth; paperwidth keeps the height limit physical.
    figures+=r'\includegraphics[width=0.94\textheight,height=0.64\paperwidth,keepaspectratio]{grid33_'+name+'.pdf}\n'
    figures+=r'\caption{'+caption+'}\n'+r'\label{fig:grid33-'+label+'}\n'+r'\end{sidewaysfigure}'+'\n\n'
figures+=r'\clearpage'+'\n'
replace(short_boundary,short_boundary+figures,'Insert four ready byte-identical E33 graphics after the related text, with denominator/MC qualifiers; rotate full-page for legibility.')

fragment=(OUT/'fragments/recent_sections.tex').read_text()
replace(r'\subsubsection{Выбор из трёх методов}\label{sec:ml3-res}',fragment+'\n\n'+r'\subsubsection{Выбор из трёх методов}\label{sec:ml3-res}',
        'Add completed cost60 and boundary development studies before unchanged ML3 results.')

end_abstract='''Это отдельная сеточная проверка; показатели ML остаются точечными.
\\end{enumerate}'''
replace(end_abstract,'''Это отдельная сеточная проверка; показатели ML остаются точечными.
\\item В cost60-пилоте добавление PI не улучшает RK4-only oracle;
в граничной проверке zero устраняет крупные выбросы, но строгий
критерий отсутствия регрессий не выполнен. Оба NO-GO имеют разные основания.
\\end{enumerate}''','Abstract conclusions: concise distinction between the two completed NO-GO decisions.')

conclusion_end='''и те же 98\\%: модель почти всегда выбирает метод Ремизова (146 раз
из 150) и не находит ни одного уравнения, где точнее другой метод.
\\end{enumerate}'''
conclusion_add=r'''
\item На 60 development-задачах при $E_{33}\le0.01$ oracle выбирает
только RK4 (59 раз $N=32$, один раз $N=64$). Экономия относительно
постоянного RK4($N=64$) равна 49.1967\% в среднем по относительным
стоимостям и 48.2849\% по сумме, но дополнительный выигрыш относительно
RK4-only выбора $N$ равен нулю. Для ML выбора метода получено NO-GO.
\item В граничной development-проверке все три исходные $E_{33}>1$
устранены, контролей в допуске осталось 30/30, проблемных стало
1/22 вместо 0/22. Формальный NO-GO вызван тремя увеличениями
порядка $10^{-9}$ сверх заранее заданного строгого порога; практически
заметного ухудшения они не устанавливают. В этой серии сохраняются
21 неточный случай и теоретические ограничения пространственной схемы.
Производственный профиль не заменён.
'''
replace(conclusion_end,conclusion_end.removesuffix(r'\end{enumerate}')+conclusion_add+r'\end{enumerate}',
        'Conclusion: append completed study results without rewriting earlier findings.')

old_completed='''RK4 с использованием нижней оценки для MC. Новые solver-расчёты
и переобучение для этого дополнения не проводились.'''
new_completed=r'''RK4 с использованием нижней оценки для MC. Отдельно завершены
ограниченный cost60-пилот и граничная development-проверка; их результаты
приведены в разделах~\ref{sec:costpilot} и~\ref{sec:boundarydev}.
При подготовке настоящей сводной редакции использовались только
сохранённые результаты: новых solver-, MC-, fit- и inference-вызовов нет.
Ранее выполненные численные серии не смешиваются с редакторской проверкой.'''
replace(old_completed,new_completed,'Completeness: distinguish saved-only report preparation from the earlier completed numerical experiments.')
replace('''\\item Ошибка между контрольными узлами, сходимость и стоимость
достижения допуска при согласованных ресурсных бюджетах.''',r'''\item Ошибка между контрольными узлами и контролируемые серии
уточнения. Стоимость достижения допуска за пределами выполненного
пилота, включая стоимость выбора/пробных решений, не установлена.''',
        'Open tasks: limited cost60 completed; general cost and convergence still open.')
replace('''\\item Надёжная пространственная реализация Ремизова и разделение
оставшихся источников ошибки; причины трёх исторических выходов worker.''',r'''\item Пространственная устойчивость Ремизова, строгий контроль окна
и разделение остаточной ошибки после zero-проверки; причины трёх
исторических выходов worker остаются неизвестными.''',
        'Open tasks: boundary experiment does not resolve all residual/spatial errors.')

old_bib='''Проверка 997 общих уравнений, выполненная без новых решений,
зафиксирована в \\path{grid33_verification.json} и \\path{grid33_per_equation.csv}
сопроводительного каталога этой редакции.'''
replace(old_bib,'''Проверка 997 общих уравнений без новых решений:
\\path{results/scientific_report_revision_2026-11-04/grid33_verification.json}
и \\path{results/scientific_report_revision_2026-11-04/grid33_per_equation.csv}.
Самодостаточная новая таблица и её проверка приведены в источнике~\\cite{savedgrid33release}.''',
        'Bibliography: remove stale relative references to old package-only filenames.')
new_bib=r'''
\bibitem{savedgrid33release} Проверенные сеточные результаты проекта:
\path{results/saved_grid33_1000_2026-11-04/}.
В комплекте этой редакции: \path{evidence/grid33/grid33_all1000.csv},
\path{evidence/grid33/summary.json}, \path{evidence/grid33/verification.json}.
\bibitem{savedgrid33figures} Научное оформление сохранённых E33:
\path{results/grid33_scientific_presentation_2026-11-04/}.
Исходные данные рисунков и таблиц: \path{evidence/grid33_figures/figure_data.json},
\path{evidence/grid33_figures/comparison.csv}; изображения сохранены побайтово.
\bibitem{savedcost60} Ограниченный development-пилот стоимости RK4/PI:
\path{results/cost60_rk4_pi_2026-11-04_v2/}.
План, профили, winner-таблица и независимая проверка:
\path{evidence/cost60/plan.json}, \path{evidence/cost60/profiles.csv},
\path{evidence/cost60/winners.csv}, \path{evidence/cost60/verification.json}.
\bibitem{savedboundarydev} Development-проверка граничного продолжения:
\path{results/remizov_boundary_dev_2026-11-05/}.
Таблица до/после и регрессии:
\path{evidence/boundary_dev/before_after.csv},
\path{evidence/boundary_dev/material_regressions.csv}; пространственный разбор:
\path{evidence/boundary_dev/SPATIAL_ASSESSMENT.md}.
\bibitem{savedthreeML} Проверенные результаты трёхметодного ML-селектора:
\path{results/ml_three_method_selector_2026-11-03/}.
Сохранённые Test-метрики и независимая сверка:
\path{evidence/ml_three/test_metrics.json}, \path{evidence/ml_three/verification.json}.
'''
replace(r'\end{thebibliography}',new_bib+r'\end{thebibliography}',
        'Add only existing completed-study/data bibliography items; no invented external citations.')
replace(r'\begin{thebibliography}{9}',r'\begin{thebibliography}{9}'+'\n'+r'\raggedright',
        'Set the path-heavy bibliography ragged right to avoid stretched lines.')

replace(r'\subsubsection{Выбор из трёх методов}\label{sec:ml3-res}',
        r'\subsubsection{Выбор из трёх методов}\label{sec:ml3-res}'+'\n'+
        r'Сохранённые метрики и их проверка приведены в источнике~\cite{savedthreeML}.',
        'Add provenance citation for the preserved three-method ML results.')

(OUT/'report.tex').write_text(text)
(OUT/'report_changes.diff').write_text(''.join(difflib.unified_diff(BASE.splitlines(True),text.splitlines(True),fromfile='base_report.tex',tofile='report.tex')))
(OUT/'editorial_changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(target=str(OUT/'report.tex'),targeted_changes=len(changes),new_solver_calls=0,new_MC_calls=0,new_fit_calls=0,new_model_inference_calls=0),ensure_ascii=False))
