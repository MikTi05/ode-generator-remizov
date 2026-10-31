"""Typeset saved research evidence only. No solver, fit, prediction or new experiments."""
from pathlib import Path
import json,csv,re
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ml_research_report_part3_2026-10-27'
MAT=ROOT/'results/ml_report_illustrations_2026-10-27'
read=lambda p:json.loads(Path(p).read_text())
num=lambda v:repr(float(v))

def equation(s,label=None):
 s=s.strip().removesuffix('.')
 return '\\begin{equation}'+('\\label{'+label+'}' if label else '')+'\n'+s+'\n\\end{equation}\n'

def split_equation(s):
 # Readability edits only. Every original component remains in this report.
 s=s.strip().removesuffix('.')
 if s.startswith('a(x)='):
  return ''.join(equation(t) for t in re.split(r',\\quad\s*',s))
 if s.startswith(r'\underline a='):
  parts=re.split(r',\\quad\s*',s)
  return equation(r'\begin{aligned}'+r',\quad '.join(parts[:2])+r',\\'+r',\quad '.join(parts[2:4])+r',\\'+parts[4]+r'.\end{aligned}')
 if s.startswith(r'\lambda='):
  parts=re.split(r',\\quad\s*|,\\qquad\s*',s)
  return equation(r'\begin{aligned}'+r',\quad '.join(parts[:2])+r',\\'+r',\quad '.join(parts[2:])+r'.\end{aligned}')
 if s.startswith(r'z='):
  return equation(s.replace(r',\qquad s=',r',\\s=').replace(r',\qquad D=',r',\quad D=').join(['\\begin{aligned}','\\end{aligned}']))
 if s.startswith(r'\ell(x)='):
  return equation(r'\begin{aligned}'+s.replace(r',\qquad \ell_1',r',\\\ell_1')+r'.\end{aligned}')
 return equation(s)

def build_a():
 trace=read(MAT/'internal/provenance.json');csvrows=list(csv.DictReader((MAT/'example_results.csv').open()))
 out=[r'\renewcommand{\theHequation}{A.\arabic{equation}}\renewcommand{\theHtable}{A.\arabic{table}}',r'\section*{Приложение А. Полные дифференциальные задачи}\label{app:a}',r'\renewcommand{\theequation}{А.\arabic{equation}}\setcounter{equation}{0}',r'\renewcommand{\thetable}{А.\arabic{table}}\setcounter{table}{0}',r'\small\setlength{\abovedisplayskip}{3pt}\setlength{\belowdisplayskip}{3pt}\setlength{\belowdisplayshortskip}{3pt}',
 'Ниже приведены все семь реальных примеров. Числа сохранены с точностью исходного представления. Факторизация не оставляет неопределённых функций: каждая вспомогательная функция задана численно через ранее определённые выражения. Аргумент $x$ у них опущен. Символы $h_k,d_k,e_k$ означают функцию и её первую и вторую производные; сами производные также выписаны явно. Все задачи заданы на $\\R$; численные ошибки относятся только к 33 узлам $[-1,1]$. Примеры 1--6 входят в финальные данные, пример 7 относится к отдельному историческому опыту. Графики каждого примера собраны в одноимённом пункте приложения Б. Причинная роль свойств коэффициентов в наблюдаемых ошибках не установлена.']
 for ex in trace:
  i=ex['example'];m=ex['metadata'];text=(MAT/'equations'/f'example_{i:02d}_factored.tex').read_text();blocks=re.findall(r'\\\[(.*?)\\\]',text,re.S)
  out += ([r'\clearpage'] if i>1 else [])+[r'\subsection*{А.'+str(i)+r'. Пример '+str(i)+r'}\label{app:ex'+str(i)+'}',ex['selection_reason']+'.']
  out += [r'Исходная задача имеет вид \eqref{eq:original}. Полная численная запись её коэффициентов и правой части задаётся следующими равенствами.']
  coeff=next(s for s in blocks if s.startswith('a(x)='));parts=re.split(r',\\quad\s*',coeff.strip().removesuffix('.'));a,b,c=[s.split('=',1)[1] for s in parts]
  substitutions=[]
  for block in blocks[1:]:
   original=block
   for full,short in substitutions:block=block.replace(full,short)
   match=re.match(r'(h_\{\d+\})=(.*)\.?$',original,re.S)
   if match and len(match.group(2).strip().removesuffix('.'))>25:
    rhs=match.group(2).strip().removesuffix('.')
    substitutions.append((rhs,match.group(1)))
    if rhs.startswith('- '):
     substitutions.append((rhs[2:],r'(-'+match.group(1)+')'))
   if block.startswith('g(x)='):
    # The fully specified forcing has no uncomputed y or H derivatives.
    roots=re.search(r'a\(x\)(e_\{\d+\})\+.*?\](d_\{\d+\})\+.*?\](h_\{\d+\})',block)
    assert roots,block
    e,d,h=roots.groups()
    out += [equation(r'\Gamma_2(x)='+a),equation(r'\Gamma_1(x)=2\Gamma_2(x)\ell(x)+\bigl('+b+r'\bigr)'),equation(r'\begin{aligned}\Gamma_0(x)&=\Gamma_2(x)[\ell(x)^2+\ell_1(x)]\\&\quad+\bigl('+b+r'\bigr)\ell(x)\\&\quad+\bigl('+c+r'\bigr).\end{aligned}')]
    forcing=r'g(x)=\begin{cases}s e^{1-D^{-1}}[\Gamma_2(x)'+e+r'+\Gamma_1(x)'+d+r'+\Gamma_0(x)'+h+r'],&|z|<1,\\0,&|z|\ge1.\end{cases}'
    out += [equation(forcing,'eq:ex'+str(i)+'-forcing')]
    continue
   if block.startswith('W(x)='):
    w,y=block.split(r'\qquad y(x)=')
    out += [equation(w),equation('y(x)='+y,'eq:ex'+str(i)+'-solution')];continue
   out.append(split_equation(block))
  out += [r'\noindent\textbf{Уравнение с числовыми коэффициентами.} После всех определений выше полная подстановка в исходную форму даёт',
    r'\begin{multline}\label{eq:ex'+str(i)+r'-ode}\bigl('+a+r'\bigr)y^{\prime\prime}(x)\\+\bigl('+b+r'\bigr)y^\prime(x)\\+\bigl('+c+r'\bigr)y(x)=g(x),\qquad x\in\R.\end{multline}',
    'Правая часть задана замкнутой формулой~\\eqref{eq:ex'+str(i)+'-forcing}; все её числовые параметры и функции определены в этом пункте, обращения к программным данным не требуется. На границе носителя и вне него $y=g=0$; гладкость обеспечивается функцией $W$. Границы носителя: $'+num(m['support_left'])+'$ и $'+num(m['support_right'])+'$.',
    'Глобальные оценки $\\underline a,\\overline a,B,C$ не обязательно являются точными экстремумами. Резольвентное равенство задаёт ограниченное решение на всей прямой. Начальные данные, выписанные выше с точностью хранения, совместимы с этим решением, но не заменяют постановку: произвольные другие данные Коши дали бы другое решение.',
    r'\begin{center}\captionof{table}{Результаты примера '+str(i)+r'}\label{tab:ex'+str(i)+r'}\begin{tabular}{lrrr}\toprule Приближение&$E_{\min}$--$E_{\max}$&Точных&Время, с\\\midrule']
  for method in ['Ремизов','Монте-Карло']:
   for n in (['128','256'] if method=='Ремизов' else ['']):
    rr=[r for r in csvrows if int(r['example'])==i and r['method']==method and r['n']==n]
    if not rr:continue
    vals=[float(r['E_grid']) for r in rr];ts=sorted(float(r['seconds']) for r in rr);med=(ts[len(ts)//2]+ts[(len(ts)-1)//2])/2
    name=(r'Ремизов, $n='+n+'$') if method=='Ремизов' else r'Монте-Карло, $M=4096$'
    out.append(name+f'&{min(vals):.8g}--{max(vals):.8g}&{sum(r["accurate"]=="True" for r in rr)}/{len(rr)}&{med:.6g}'+r'\\')
  out += [r'\bottomrule\end{tabular}\end{center}', 'Время в таблице --- '+('медиана четырёх сопоставимых измерений внутри прежнего блока; реализации Монте-Карло не усреднялись. Один из четырёх MC-ответов точен, но это не оценка высокой надёжности нового запуска.' if i==7 else 'один сохранённый замер; повторный разброс по нему не известен.')]
 out.append(r'\normalsize\setlength{\abovedisplayskip}{6pt}\setlength{\belowdisplayskip}{6pt}')
 (OUT/'sections/appendix_a.tex').write_text('\n\n'.join(out)+'\n')

# Each cropped figure is a verbatim rectangle of a prepared PDF page.
def build_graphics_index():
 figs=read(MAT/'internal/figures_manifest.json')
 index=[]
 for f in figs:
  # Bottom caption and old page counter are excluded; axes and all data remain.
  name=f['name'];frac=.12
  if name.startswith('summary_'):frac=.13
  index.append(dict(name=name,source=str(MAT/f['atlas']),page=f['atlas_page']-1,bottom_fraction=frac,dest=str(OUT/'figures'/(name+'.pdf')),top_fraction=.06 if name.startswith('summary_') else 0.))
 (OUT/'internal/graphics_index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2))

def build_b():
 figs=read(MAT/'internal/figures_manifest.json');by={r['name']:r for r in figs}
 out=[r'\renewcommand{\theHfigure}{B.\arabic{figure}}',r'\section*{Приложение Б. Графическое исследование семи уравнений}\label{app:b}',r'\renewcommand{\thefigure}{Б.\arabic{figure}}\setcounter{figure}{0}',
 'Рисунки используют только сохранённые численные ответы. Густые линии коэффициентов и точного решения вычислены по известным аналитическим формулам; численные решения существуют лишь в 33 узлах. Отрезки между их маркерами служат для чтения. Везде показан коэффициент $c$ исходного уравнения, а не $c_R$.']
 def graphic(name,width='.94'):
  r=by[name]
  return r'\begin{center}\begin{minipage}{\linewidth}\centering\includegraphics[width='+width+r'\linewidth]{figures/'+name+r'.pdf}\captionof{figure}{'+r['caption'].replace('MC-','Монте-Карло-').replace('[-1,1]',r'$[-1,1]$')+r'}\label{fig:'+name+r'}\end{minipage}\end{center}'
 for i in range(1,8):
  if i>1:out +=[r'\clearpage']
  out +=[r'\subsection*{Б.'+str(i)+r'. Пример '+str(i)+r'}\label{app:graphs'+str(i)+'}',
   'Полная дифференциальная постановка и числовые результаты приведены в п.~А.'+str(i)+'. Сначала сопоставлены коэффициенты и точное решение на носителе.',graphic(f'example_{i:02d}_coefficients','.80' if i==1 else '.94'),graphic(f'example_{i:02d}_exact','.80' if i==1 else '.94')]
  out +=[r'\clearpage',r'\noindent\textbf{Пример '+str(i)+r': численное приближение и его ошибка.}',
   'Кривые решений на общем масштабе могут казаться совпадающими, поэтому ниже отдельно показана абсолютная ошибка; критерием является её максимум, а не видимое совпадение отдельных участков.',graphic(f'example_{i:02d}_solutions'),graphic(f'example_{i:02d}_error')]
  if i==7:
   out +=[r'\clearpage',r'\noindent\textbf{Исторический опыт: четыре реализации на одном уравнении.}',graphic('example_07_repeats'),
   'На этом уравнении все четыре повтора Ремизова при $n=128$ точны. Монте-Карло быстрее в данном временном блоке, но достигает допуска лишь в одном из четырёх ответов: $E=0{,}00804326$ против $0{,}01155969$, $0{,}01030280$ и $0{,}01399365$ в остальных. Среднее этих ответов не подставлено вместо каждой отдельной реализации.',
   'Четыре реализации относятся к одному уравнению, а не к четырём независимым семействам. Этот пример сознательно выбран для обсуждения изменчивости и не является новой независимой проверкой модели. Его единственный точный стохастический ответ не добавляется к финальным $0/20$.']
 (OUT/'sections/appendix_b.tex').write_text('\n\n'.join(out)+'\n')

if __name__=='__main__':
 build_a();build_graphics_index();build_b()
 print('Appendices A/B and immutable figure index generated; no numerical calculations')
