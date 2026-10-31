"""Saved-results scientific materials. No solver calls, training or inference."""
from __future__ import annotations
import csv,hashlib,io,json,math,sys,textwrap
from pathlib import Path
import numpy as np
import sympy as sp
from sympy.printing.latex import LatexPrinter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from all_methods_dataset_loader import load_all_methods_equations
from src.all_methods_expressions import SafeExpressionNode
OUT=ROOT/'results/ml_report_illustrations_2026-10-27'
FINAL=ROOT/'results/ml_selector_final140_protocol_2026-10-27'
HISTORY=ROOT/'results/f0047_three_profile_2026-10-26'
SELECTION=[('AMV1_F0035_S00','Достаточность малого числа итераций; неточный результат Монте-Карло у порога'),('AMV1_F0102_S00','Достаточность малого числа итераций при переменных коэффициентах'),('AMV1_F0271_S00','Достижение точности после увеличения числа итераций'),('AMV1_F0385_S00','Достижение точности после уточнения при контрастных коэффициентах'),('AMV1_F0521_S00','Нарушение порога обоими приближениями'),('AMV1_F0738_S00','Нарушение порога при осциллирующих коэффициентах'),('AMV1_F0047_S00','Изменчивость точности четырёх реализаций Монте-Карло')]
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
x=sp.Symbol('x',real=True)
N=lambda v:sp.Float(repr(float(v)),17)
class Printer(LatexPrinter):
 def _print_Float(self,e):
  s=repr(float(e))
  if 'e' in s:
   a,b=s.split('e');return a+r'\times 10^{'+str(int(b))+'}'
  return s
printer=Printer()
L=lambda e:printer.doprint(e)

def op_expr(node,ch):
 p={k:N(v) for k,v in node.params.items()};o=node.op
 if o=='constant':return p['value']
 if o=='sin_affine':return sp.sin(p['omega']*x+p['phi'])
 if o=='cos_affine':return sp.cos(p['omega']*x+p['phi'])
 if o=='tanh_affine':return sp.tanh(p['omega']*x+p['phi'])
 if o=='gaussian':return sp.exp(-p['alpha']*(x-p['mu'])**2)
 if o=='lorentz':return 1/(1+p['alpha']*(x-p['mu'])**2)
 if o=='saturating':return (x-p['mu'])/sp.sqrt(1+(x-p['mu'])**2)
 if o=='lincomb':return p['alpha']*ch[0]+p['beta']*ch[1]
 if o=='mul':return ch[0]*ch[1]
 u=ch[0]
 if o=='sin':return sp.sin(u)
 if o=='cos':return sp.cos(u)
 if o=='tanh':return sp.tanh(u)
 if o=='exp_neg_square':return sp.exp(-u**2)
 if o=='rational_saturate':return u/(1+u**2)
 raise ValueError(o)

def symbolic(node):return op_expr(node,[symbolic(c) for c in node.children])

def dag(node,prefix,defs,jets=False):
 children=[dag(c,prefix,defs,jets) for c in node.children]
 k=1+sum(1 for a,b in defs if a.name.startswith(prefix+'_'))
 u=sp.Symbol(f'{prefix}_{k}');expr=op_expr(node,[c[0] for c in children]);defs.append((u,expr))
 if not jets:return u,None,None
 d=sp.Symbol(f'd_{k}');e=sp.Symbol(f'e_{k}')
 d1=sp.diff(expr,x)+sum(sp.diff(expr,c[0])*c[1] for c in children)
 d2=sp.diff(expr,x,2)+sum(2*sp.diff(expr,x,c[0])*c[1]+sp.diff(expr,c[0])*c[2] for c in children)
 d2+=sum(sp.diff(expr,a[0],b[0])*a[1]*b[1] for a in children for b in children)
 defs.extend([(d,d1),(e,d2)])
 return u,d,e

def model_math(m):
 ast=m['serialized_ast'];nodes={k:SafeExpressionNode.from_dict(ast[k]) for k in ('a_raw','b_raw','c_raw','y_shape')};c=ast['construction']
 A,B,C,H=[symbolic(nodes[k]) for k in ('a_raw','b_raw','c_raw','y_shape')]
 a=N(c['a_mid'])+N(c['a_half'])*sp.tanh(A);b=N(c['b_scale'])*sp.tanh(B);cr=N(c['c_scale'])*sp.tanh(C);co=cr-N(c['lambda'])
 H1=sp.diff(H,x);H2=sp.diff(H,x,2);z=(x-N(c['bump_mu']))/N(c['bump_radius']);D=1-z**2;W=sp.exp(1-1/D)
 ell=-2*z/(N(c['bump_radius'])*D**2);ell1=-2/(N(c['bump_radius'])**2*D**2)-8*z**2/(N(c['bump_radius'])**2*D**3)
 y=N(c['solution_scale'])*W*H;g=N(c['solution_scale'])*W*(a*H2+(2*a*ell+b)*H1+(a*(ell**2+ell1)+b*ell+co)*H)
 assert all(not e.has(sp.Derivative) and e.free_symbols<={x} for e in (a,b,co,y,g,H1,H2))
 defs=[];roots=[dag(nodes[k],p,defs,k=='y_shape') for k,p in [('a_raw','p'),('b_raw','q'),('c_raw','r'),('y_shape','h')]]
 return dict(a=a,b=b,c=co,c_R=cr,H=H,H1=H1,H2=H2,y=y,g=g,ell=ell,ell1=ell1,defs=defs,roots=roots,construction=c)

def evaluate_closed(mathobj,points):
 p=np.asarray(points);c=mathobj['construction'];mask=abs(p-c['bump_mu'])<c['bump_radius'];v={}
 for k in ('a','b','c','c_R'):
  a=np.asarray(sp.lambdify(x,mathobj[k],'numpy')(p),float);v[k]=np.broadcast_to(a,p.shape).copy()
 for k in ('y','g'):
  a=np.zeros_like(p);a[mask]=sp.lambdify(x,mathobj[k],'numpy')(p[mask]);v[k]=a
 return v

NAMES={'R128':'Ремизов, $n=128$','R256':'Ремизов, $n=256$','MC1':'Монте-Карло, $M=4096$'}
COLORS={'R128':'#2468A0','R256':'#7B4D9C','MC1':'#D47719','selector':'#117A65'}
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.titlesize':12,'axes.labelsize':11,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
figures=[]
def savefig(fig,name,caption,atlas):
 fig.subplots_adjust(bottom=getattr(fig,'_bottom',.22),top=.87,left=getattr(fig,'_left',.11),right=.97)
 fig.text(.055,.045,'\n'.join(textwrap.wrap(caption,110)),fontsize=8.5,va='bottom')
 fig.text(.97,.015,str(len(figures)+1),ha='right',fontsize=8,color='#555555')
 fig.savefig(OUT/'figures'/f'{name}.png',dpi=300)
 atlas.savefig(fig)
 figures.append(dict(name=name,caption=caption,atlas_page=atlas.get_pagecount(),atlas=atlas._bundle_name,figure=len(figures)+1))
 plt.close(fig)

def plot_example(i,e,rs,atlas):
 cp=e.cauchy_problem;xx=np.linspace(-1,1,1201);title=f'Пример {i}'
 fig,axs=plt.subplots(2,2,figsize=(8.8,7.3));fig.suptitle(title+': коэффициенты исходного уравнения')
 for ax,k in zip(axs.flat,('a','b','c','g')):
  rp=e.remizov_problem;vals={'a':rp.a,'b':rp.b,'c':lambda z:rp.c_remizov(z)-rp.lambda_value,'g':lambda z:-rp.g_remizov(z)}[k](xx);ax.plot(xx,vals,color='#2468A0',lw=1.4);ax.set(xlabel='$x$',ylabel=f'${k}(x)$');ax.grid(alpha=.2);ax.ticklabel_format(useOffset=False)
 fig.subplots_adjust(hspace=.4,wspace=.35)
 savefig(fig,f'example_{i:02d}_coefficients',f'{title}. Коэффициенты исходного уравнения на отрезке сравнения. Показан c(x), а не коэффициент $c_R(x)$ оператора. Все функции вычислены по сохранённым аналитическим определениям.',atlas)
 left=e.metadata['support_left'];right=e.metadata['support_right'];xxglobal=np.linspace(left-.25,right+.25,1601)
 fig,ax=plt.subplots(figsize=(8.8,4.9));ax.plot(xxglobal,e.y_exact(xxglobal),color='black',lw=1.5);ax.axvspan(-1,1,color='#2468A0',alpha=.08);ax.axvline(left,color='gray',ls=':');ax.axvline(right,color='gray',ls=':');ax.set(xlabel='$x$',ylabel='$y(x)$',title=title+': известное точное решение');ax.grid(alpha=.2)
 savefig(fig,f'example_{i:02d}_exact',f'{title}. Аналитическое решение на носителе и прилегающих участках прямой. Закрашен отрезок [-1,1], на котором сохранены численные решения; пунктиром отмечены границы носителя.',atlas)
 grid=np.asarray(rs['R128'][0]['result']['x']);exact=np.asarray(rs['R128'][0]['y_exact'])
 fig,ax=plt.subplots(figsize=(8.8,4.9));ax.plot(xx,e.y_exact(xx),color='black',lw=1.6,label='Точное решение')
 for profile in ('R128','R256','MC1'):
  for j,r in enumerate(rs.get(profile,[])):
   ax.plot(grid,r['result']['f'],marker={'R128':'o','R256':'s','MC1':'^'}[profile],ms=3,mfc='none',lw=.8,color=COLORS[profile],alpha=1 if j==0 else .3,label=NAMES[profile] if j==0 else None)
 ax.set(xlabel='$x$',ylabel='$y(x)$',title=title+': точное и численные решения');ax.grid(alpha=.2);ax.legend()
 savefig(fig,f'example_{i:02d}_solutions',f'{title}. Точная кривая и сохранённые значения численных решений в 33 узлах. Отрезки между маркерами служат только для чтения. '+('Тонкие линии показывают остальные реализации Монте-Карло; усреднение не выполнено.' if i==7 else 'Новые численные значения между узлами не рассчитывались.'),atlas)
 fig,ax=plt.subplots(figsize=(8.8,6.4 if i==7 else 4.9))
 if i==7:fig._bottom=.4
 for profile in ('R128','R256','MC1'):
  for j,r in enumerate(rs.get(profile,[])):
   err=abs(np.asarray(r['result']['f'])-exact)
   assert np.all(err>0),'log plot must not fabricate zero errors'
   label=NAMES[profile]+(f', реализация {j+1}' if profile=='MC1' and len(rs[profile])>1 else '')
   ax.semilogy(grid,err,marker='o',ms=2.6,lw=1,color=COLORS[profile],alpha=1 if j==0 else .5,label=label if profile=='MC1' or j==0 else None)
 ax.axhline(.01,color='#B22222',ls='--',lw=1.4,label='Допуск 0,01');ax.set(xlabel='$x$',ylabel='Абсолютная ошибка',title=title+': ошибка в сохранённых узлах');ax.grid(alpha=.2,which='both');ax.legend(ncol=2,**({'loc':'upper center','bbox_to_anchor':(.5,-.24)} if i==7 else {}))
 savefig(fig,f'example_{i:02d}_error',f'{title}. Абсолютная ошибка по сохранённым численным значениям. Порог относится к максимуму по 33 узлам; поведение ошибки между узлами не установлено.',atlas)
 if i==7:
  fig,axs=plt.subplots(1,2,figsize=(8.8,4.9));rr=rs['MC1'];idx=np.arange(1,5)
  axs[0].bar(idx,[r['E_grid'] for r in rr],color=[COLORS['selector'] if r['accuracy_met'] else COLORS['MC1'] for r in rr]);axs[0].axhline(.01,color='#B22222',ls='--');axs[0].set(xlabel='Независимая реализация',ylabel='$E_{\\mathrm{grid}}$',xticks=idx)
  for p in ('R128','R256','MC1'):axs[1].plot(idx,[r['method_wall_seconds'] for r in rs[p]],'o-',label=NAMES[p],color=COLORS[p])
  axs[1].set(xlabel='Повтор измерения',ylabel='Время, с',xticks=idx);axs[1].legend();fig.suptitle('Пример 7: наблюдаемый успех и изменчивость результата')
  savefig(fig,'example_07_repeats','Один из четырёх MC-ответов точен и быстрее метода Ремизова при n=128; три других нарушают допуск. Четыре реализации не подтверждают высокую надёжность следующего запуска. Опыт не включён в финальные 60 семейств.',atlas)

def math_document(i,m,obj):
 c=obj['construction'];roots=obj['roots'];ar,br,cr=[r[0] for r in roots[:3]];H,H1,H2=roots[3]
 A=N(c['a_mid'])+N(c['a_half'])*sp.tanh(ar);B=N(c['b_scale'])*sp.tanh(br);C=N(c['c_scale'])*sp.tanh(cr)-N(c['lambda'])
 lines=[f'## Пример {i}',f'Исходное уравнение на всей действительной прямой:',r'\[a(x)y^{\prime\prime}(x)+b(x)y^\prime(x)+c(x)y(x)=g(x),\qquad x\in\mathbb R.\]',
 'Все числа ниже сохраняют числовые параметры исходной задачи. Вспомогательные функции полностью определены последовательно; неопределённых коэффициентов или подбираемых параметров нет.',
 'Для компактной записи аргумент x у вспомогательных функций опущен.']
 for sym,expr in obj['defs']:lines.append(r'\['+L(sym)+'='+L(expr)+r'.\]')
 lines.extend([r'\[a(x)='+L(A)+r',\quad b(x)='+L(B)+r',\quad c(x)='+L(C)+r'.\]',
 r'\[z=\frac{x-('+repr(c['bump_mu'])+')}{'+repr(c['bump_radius'])+r'},\qquad D=1-z^2,\qquad s='+repr(c['solution_scale'])+r'.\]',
 r'\[W(x)=\begin{cases}\exp(1-D^{-1}),&|z|<1,\\0,&|z|\ge1,\end{cases}\qquad y(x)=sW(x)'+L(H)+r'.\]',
 r'\[\ell(x)=-\frac{2z}{'+repr(c['bump_radius'])+r'D^2},\qquad \ell_1(x)=-\frac{2}{('+repr(c['bump_radius'])+r')^2D^2}-\frac{8z^2}{('+repr(c['bump_radius'])+r')^2D^3}.\]',
 r'\[g(x)=\begin{cases}s e^{1-D^{-1}}\left\{a(x)'+L(H2)+r'+[2a(x)\ell(x)+b(x)]'+L(H1)+r'+[a(x)(\ell(x)^2+\ell_1(x))+b(x)\ell(x)+c(x)]'+L(H)+r'\right\},&|z|<1,\\0,&|z|\ge1.\end{cases}\]',
 r'Здесь $'+L(H1)+r'$ и $'+L(H2)+r'$ заданы явными формулами выше и равны первой и второй производным $'+L(H)+r'$. Таким образом, в правой части нет невычисленных производных.',
 'Границы носителя: '+repr(m['support_left'])+' и '+repr(m['support_right'])+'. Гладкое продолжение решения и правой части за носителем равно нулю; коэффициенты определены на всей прямой.',
 r'Резольвентная постановка: $L=a(x)\partial_x^2+b(x)\partial_x+c_R(x)$, где',
 r'\[\lambda='+repr(c['lambda'])+r',\quad c_R(x)=c(x)+'+repr(c['lambda'])+r',\quad g_R(x)=-g(x),\qquad(\lambda-L)y=g_R.\]',
 r'\[\underline a='+repr(m['a_lower_cert'])+r',\quad\overline a='+repr(m['a_upper_cert'])+r',\quad |b|\le'+repr(m['b_abs_cert'])+r',\quad |c_R(x)|\le C='+repr(m['remizov_c_abs_cert'])+r',\quad\lambda-C\approx'+repr(m['remizov_gap_cert'])+r'>0.\]',
 r'Оценки $\underline a\le a(x)\le\overline a$, $|b(x)|$ и $C$ глобальны и могут быть неточными; они не названы точными экстремумами.',
 'Ищется ограниченное решение на всей прямой. Отрезок [-1,1] - область сравнения сохранённых численных значений, а не граница определения коэффициентов.',
 r'Отдельная совместимая задача Коши на $[-1,1]$ имела бы дополнительные условия',
 r'\[y(-1)\approx'+repr(m['y0'])+r',\qquad y^\prime(-1)\approx'+repr(m['v0'])+r'.\]',
 'Эти значения следуют из указанного точного решения. Они не заменяют условие ограниченности в резольвентной задаче; произвольные другие начальные данные определяли бы другую задачу Коши.'])
 markdown='\n\n'.join(lines)+'\n'
 (OUT/'equations'/f'example_{i:02d}.md').write_text(markdown)
 # Valid LaTeX fragments; long substituted expressions are source material for an appendix.
 tex=markdown.replace(f'## Пример {i}',r'\subsection*{Пример '+str(i)+'}')
 (OUT/'equations'/f'example_{i:02d}_factored.tex').write_text(tex)
 expanded=[r'\subsection*{Пример '+str(i)+r': форма с полной подстановкой}', 'Числовые определения совпадают с факторизованной формой; x является единственной свободной переменной.']
 for k in ('a','b','c'):
  expanded.append(r'\['+k+'(x)='+L(obj[k])+r'.\]')
 for k in ('y','g'):
  expanded.append(r'\['+k+r'(x)=\begin{cases}'+L(obj[k])+r',&'+repr(m['support_left'])+'<x<'+repr(m['support_right'])+r',\\0,&\text{вне этого интервала}.\end{cases}\]')
 (OUT/'equations'/f'example_{i:02d}_expanded.tex').write_text('\n\n'.join(expanded)+'\n')
 return markdown

def aggregate(a,atlas):
 cases=a['cases'];labels={'constant_R128':NAMES['R128'],'constant_R256':NAMES['R256'],'selector':'Выбор модели'}
 fig,axs=plt.subplots(1,2,figsize=(9,5.3))
 for policy in labels:
  errors=np.sort([c['policies'][policy]['error'] for c in cases]);axs[0].step(errors,np.arange(1,61)/60,where='post',label=labels[policy],color=COLORS.get(policy,COLORS.get(policy.replace('constant_',''))))
 errors=np.sort([c['profiles']['MC1']['error'] for c in cases if c['profiles']['MC1']['correct'] is not None]);axs[1].step(errors,np.arange(1,21)/20,where='post',color=COLORS['MC1'],label='Монте-Карло: 20 задач')
 for ax in axs:
  ax.axvline(.01,color='#B22222',ls='--');ax.set(xscale='log',xlabel='$E_{\\mathrm{grid}}$',ylabel='Эмпирическая доля',ylim=(0,1.05));ax.legend();ax.grid(alpha=.2)
 axs[0].set_title('Ремизов и выбор модели: 60 задач');axs[1].set_title('Отдельная подвыборка Монте-Карло')
 savefig(fig,'summary_01_error_distributions','Эмпирические распределения полной сеточной ошибки. Знаменатели различаются: 60 и 20 задач. Доли достигших допуска для модели и n=256 совпадают; распределения ошибок различаются. Равенство будущих рисков не установлено.',atlas)
 fig,axs=plt.subplots(1,2,figsize=(9,5.3));names=list(labels);cost=[a['primary'][p]['total_warm_cost'] for p in names];good=[a['primary'][p]['confirmed_correct'] for p in names]
 colors=[COLORS['R128'],COLORS['R256'],COLORS['selector']]
 axs[0].bar(range(3),cost,color=colors);axs[0].set(ylabel='Суммарная стоимость, с',xticks=range(3),xticklabels=['Ремизов\n$n=128$','Ремизов\n$n=256$','Выбор\nмодели'])
 for j,t in enumerate(cost):axs[0].text(j,t+25,f'{t:.1f}',ha='center',fontsize=9)
 axs[0].set_ylim(0,max(cost)*1.18)
 axs[1].bar(range(3),good,color=colors);axs[1].set(ylabel='Точных доставок из 60',ylim=(0,66),xticks=range(3),xticklabels=['Ремизов\n$n=128$','Ремизов\n$n=256$','Выбор\nмодели'])
 for j,t in enumerate(good):axs[1].text(j,t+1,str(t),ha='center')
 savefig(fig,'summary_02_cost_accuracy','Цена и точность на одних 60 задачах. Учтены неточные вызовы; для модели добавлено ранее измеренное время извлечения признаков и прогноза. Ремизов при n=128 дешевле, но даёт меньше точных решений.',atlas)
 fig,axs=plt.subplots(1,2,figsize=(9,5.3));t128=[c['profiles']['R128']['seconds'] for c in cases];t256=[c['profiles']['R256']['seconds'] for c in cases]
 axs[0].scatter(t128,t256,s=22,color=COLORS['R256'],alpha=.6);axs[0].set(xlabel='Ремизов n=128, с',ylabel='Ремизов n=256, с',title='Парные измерения времени');lims=[min(t128)*.97,max(t128)*1.03];axs[0].plot(lims,[2*t for t in lims],ls='--',color='gray',label='Отношение 2:1');axs[0].legend()
 for p in names:axs[1].plot(np.arange(1,61),np.cumsum([c['policies'][p]['cost'] for c in cases]),label=labels[p],color=COLORS.get(p,COLORS.get(p.replace('constant_',''))))
 axs[1].set(xlabel='Номер задачи в фиксированном порядке',ylabel='Накопленная стоимость, с');axs[1].legend(fontsize=7)
 for ax in axs:ax.grid(alpha=.2)
 savefig(fig,'summary_03_time_comparison','Каждая точка слева соответствует одной семье; справа показана накопленная стоимость по заранее фиксированному порядку. Один замер на профиль не оценивает разброс времени повторного исполнения.',atlas)
 matrix=np.zeros((3,2),int)
 for c in cases:
  row=0 if c['profiles']['R128']['correct'] else (1 if c['profiles']['R256']['correct'] else 2);col=int(c['policies']['selector']['profile']=='R256');matrix[row,col]+=1
 fig,ax=plt.subplots(figsize=(8.8,5.3));ax.imshow(matrix,cmap='Blues',vmin=0,vmax=34,aspect='auto')
 ax.set(xticks=[0,1],xticklabels=['Модель: n=128','Модель: n=256'],yticks=[0,1,2],yticklabels=['Оба приближения точны','Точен только $n=256$','Оба приближения неточны'],title='Выбор модели относительно двух приближений Ремизова')
 for i in range(3):
  for j in range(2):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>17 else 'black',fontsize=18)
 fig._left=.33
 savefig(fig,'summary_04_selection_outcomes','Из 57 задач с точным ответом Ремизова самое быстрое точное приближение выбрано в 23; в 34 случаях выбран более дорогой n=256. На трёх задачах оба приближения неточны. Полный минимум с учётом неизмеренного Монте-Карло неизвестен.',atlas)
 mc=[c for c in cases if c['profiles']['MC1']['correct'] is not None];fig,ax=plt.subplots(figsize=(8.8,4.9));ax.scatter([c['profiles']['MC1']['seconds'] for c in mc],[c['profiles']['MC1']['error'] for c in mc],color=COLORS['MC1'],s=40);ax.axhline(.01,color='#B22222',ls='--',label='Допуск 0,01');ax.set(yscale='log',xlabel='Время Монте-Карло, с',ylabel='$E_{\\mathrm{grid}}$',title='Монте-Карло: время и ошибка на 20 заранее выбранных задачах');ax.legend();ax.grid(alpha=.2,which='both')
 savefig(fig,'summary_05_mc20','Все 20 сохранённых реализаций Монте-Карло нарушили допуск. Отсутствие точного ответа в этих реализациях не доказывает невозможность успеха при другом случайном наборе траекторий; по одной реализации надёжность нового запуска не оценивается.',atlas)
 return matrix

def build():
 from src.all_methods_expressions import evaluate_jet2
 for name in ('equations','figures'):(OUT/name).mkdir(exist_ok=True)
 protected={}
 for root,manifest in [(FINAL,'final_artifact_sha256.json'),(ROOT/'results/ml_selector_final_2026-10-27','artifact_sha256.json'),(HISTORY,'artifact_sha256.json')]:
  for name,h in read(root/manifest).items():
   p=root/name;assert sha(p)==h,p;protected[str(p.relative_to(ROOT))]=h
 write(OUT/'internal/source_hashes.before.json',protected)
 plan=read(FINAL/'plan.json');data=ROOT/plan['data_dir'];finaleq={e.equation_id:e for e in load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl')}
 original=ROOT/read(HISTORY/'plan.json')['data_dir'];histdir=OUT/'internal/historical_input';histdir.mkdir(exist_ok=True)
 with (original/'all_methods_equations_v1.csv').open(newline='') as f:
  reader=csv.DictReader(f);fields=reader.fieldnames;row=next(r for r in reader if r['equation_id']==SELECTION[-1][0])
 with (histdir/'equations.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow(row)
 line=next(l for l in (original/'all_methods_equations_v1_metadata.jsonl').open() if json.loads(l)['equation_id']==SELECTION[-1][0]);(histdir/'metadata.jsonl').write_text(line)
 historical=load_all_methods_equations(histdir/'equations.csv',histdir/'metadata.jsonl')[0]
 a=read(FINAL/'analysis.json');records_final={};records_hist={};trace=[];rows=[];checks=[];materials=[];maths=[];notes=[];all_math=[]
 for i,(pid,why) in enumerate(SELECTION,1):
  e=historical if i==7 else finaleq[pid];rs={}
  paths=sorted((HISTORY if i==7 else FINAL).joinpath('records').glob(pid+'_*.json'))
  for path in paths:
   r=read(path);profile=r['profile'];assert r['status']=='completed' and r['result']['success']
   assert r['expression_hash']==e.metadata['expression_hash'] and r['config']==r['job']['config']
   assert r['seed']==r['config'].get('seed');np.testing.assert_array_equal(r['y_exact'],e.y_exact(np.asarray(r['result']['x'])))
   err=float(np.max(abs(np.asarray(r['result']['f'])-np.asarray(r['y_exact']))));assert err==r['E_grid'] and (err<=.01)==r['accuracy_met']
   if profile in ('R128','R256'):assert r['config']['n']==int(profile[1:])
   else:assert r['config']['M']==4096 and r['config']['dt']==1/128
   rs.setdefault(profile,[]).append(r)
  for p in rs:rs[p].sort(key=lambda r:r['job']['episode'])
  assert all(len(rs[p])==(4 if i==7 else 1) for p in ('R128','R256'))
  if i==7:assert [r['accuracy_met'] for r in rs['MC1']]==[False,False,False,True]
  obj=model_math(e.metadata);points=np.r_[np.linspace(-1,1,513),e.metadata['support_left']-.1,e.metadata['support_right']+.1]
  v=evaluate_closed(obj,points);rp=e.remizov_problem;ref=dict(a=rp.a(points),b=rp.b(points),c=rp.c_remizov(points)-rp.lambda_value,c_R=rp.c_remizov(points),g=-rp.g_remizov(points),y=e.y_exact(points))
  residuals={}
  for k in ref:
   np.testing.assert_allclose(v[k],ref[k],rtol=3e-11,atol=3e-10);residuals[k]=float(np.max(abs(v[k]-ref[k])))
  jet=evaluate_jet2(SafeExpressionNode.from_dict(e.metadata['serialized_ast']['y_shape']),points)
  for k,expected in [('H',jet.value),('H1',jet.d1),('H2',jet.d2)]:np.testing.assert_allclose(np.broadcast_to(sp.lambdify(x,obj[k],'numpy')(points),points.shape),expected,rtol=3e-11,atol=3e-10)
  # Verify every factored derivative definition, not just the expanded expression.
  vals={x:points}
  for sym,expr in obj['defs']:
   args=sorted(expr.free_symbols,key=str);vals[sym]=np.broadcast_to(sp.lambdify(args,expr,'numpy')(*[vals[t] for t in args]),points.shape)
  h,d,dd=obj['roots'][3]
  for sym,expected in [(h,jet.value),(d,jet.d1),(dd,jet.d2)]:np.testing.assert_allclose(vals[sym],expected,rtol=3e-11,atol=3e-10)
  checks.append(dict(example=i,passed=True,analytic_comparison_points=len(points),maximum_absolute_differences=residuals,factored_jet_checked=True,numerical_records=len(paths),solver_calls=0))
  trace.append(dict(example=i,equation_id=pid,expression_hash=e.metadata['expression_hash'],identity_dataset_sha256=sha(original/'all_methods_equations_v1.csv') if i==7 else next(c['dataset_sha256'] for c in plan['cases'] if c['equation_id']==pid),scope='historical_train' if i==7 else 'final_holdout',metadata=e.metadata,records={str(p.relative_to(ROOT)):sha(p) for p in paths},selection_reason=why))
  for profile,rr in rs.items():
   for r in rr:rows.append(dict(example=i,scope='Ранее выполненный опыт' if i==7 else 'Финальная выборка',method='Монте-Карло' if profile=='MC1' else 'Ремизов',n='' if profile=='MC1' else int(profile[1:]),M=4096 if profile=='MC1' else '',repeat=r['job']['episode']+1,E_grid=r['E_grid'],accurate=r['accuracy_met'],seconds=r['method_wall_seconds']))
  all_math.append(math_document(i,e.metadata,obj));materials.append((e,rs));maths.append(obj)
  local=np.linspace(-1,1,1201);obs=dict(a=rp.a(local),b=rp.b(local),c=rp.c_remizov(local)-rp.lambda_value,g=-rp.g_remizov(local),y=e.y_exact(local))
  ranges={k:(float(np.min(z)),float(np.max(z))) for k,z in obs.items()}
  notes.append(f'## Пример {i}\n\n'+why+'.\n\nНа отрезке сравнения: '+', '.join(f'${k}(x)$ от {lo:.5g} до {hi:.5g}' for k,(lo,hi) in ranges.items())+'. Это диапазоны на густой аналитической сетке для описания рисунка, не глобальные сертификаты. Коэффициент a положителен, а c отрицателен на всей прямой в силу сохранённых глобальных оценок. Видимые особенности не устанавливают причин ошибки.\n\n'+('Результаты при n=128 и n=256 оба удовлетворяют допуску.' if i<=2 else 'Уточнение с n=128 до n=256 пересекло порог точности.' if i<=4 else 'Оба изученных приближения Ремизова неточны; причина ошибки не отделена от остальных источников дискретизации.' if i<=6 else 'Один точный и три неточных MC-ответа при одинаковом числе траекторий иллюстрируют изменчивость. Нельзя переносить этот единичный успех на вероятность следующего запуска или на финальную выборку.')+'\n')
 write(OUT/'internal/provenance.json',trace);write(OUT/'internal/verification.json',dict(passed=True,examples=checks,solver_calls=0,new_training=0,new_inference=0))
 (OUT/'MATHEMATICAL_EXAMPLES.md').write_text('# Математические определения семи исследованных примеров\n\nЭто исходные материалы, а не текст научного отчёта. Все вспомогательные функции полностью определены числовыми формулами. Десятичные параметры имеют точность, достаточную для восстановления сохранённых чисел двойной точности.\n\n'+'\n'.join(all_math))
 (OUT/'EXPLANATIONS.md').write_text('# Пояснения к выбору примеров\n\nПримеры намеренно выбраны по фактически наблюдаемым сценариям, а не случайно. Их нельзя использовать для оценки частоты успеха; для этого отдельно рассматриваются все 60 финальных семейств.\n\n'+'\n'.join(notes))
 with (OUT/'example_results.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 table=['# Результаты отдельных примеров','', '| Пример | Метод | Число повторов | Сеточная ошибка: минимум - максимум | Точных ответов | Время: медиана [минимум; максимум], с |','|---|---|---:|---:|---:|---:|']
 for i,(e,rs) in enumerate(materials,1):
  for p,rr in rs.items():
   errs=[r['E_grid'] for r in rr];ts=[r['method_wall_seconds'] for r in rr];name='Монте-Карло, M=4096' if p=='MC1' else 'Ремизов, n='+p[1:]
   table.append(f'| {i} | {name} | {len(rr)} | {min(errs):.8g} - {max(errs):.8g} | {sum(r["accuracy_met"] for r in rr)}/{len(rr)} | {np.median(ts):.6g} [{min(ts):.6g}; {max(ts):.6g}] |')
 table+=['','В примерах 1-6 один замер на профиль, поэтому минимум, медиана и максимум времени совпадают. Пример 7 - отдельный ранее выполненный опыт с четырьмя повторами на профиль; он не включён в статистику финальных 60 задач. Число реализаций Монте-Карло не является числом независимых уравнений.','', 'Неизмеренные значения Монте-Карло отсутствуют у примеров 2, 3, 5 и 6. Численные решения вне 33 сохранённых узлов отсутствуют для всех примеров; между ними проведены только направляющие отрезки. Вариация повторного времени неизвестна для финальных задач.']
 (OUT/'TABLES.md').write_text('\n'.join(table)+'\n')
 with PdfPages(OUT/'examples_atlas.pdf',metadata={'Title':'Математические примеры: коэффициенты, решения и ошибки','Author':'Исследовательские материалы'}) as atlas:
  atlas._bundle_name='examples_atlas.pdf'
  for i,(e,rs) in enumerate(materials,1):plot_example(i,e,rs,atlas)
 with PdfPages(OUT/'final_sample_atlas.pdf',metadata={'Title':'Финальная выборка: точность, стоимость и выбор метода'}) as atlas:
  atlas._bundle_name='final_sample_atlas.pdf';matrix=aggregate(a,atlas)
 assert matrix.tolist()==[[19,34],[0,4],[2,1]]
 write(OUT/'internal/figures_manifest.json',figures)
 (OUT/'CAPTIONS.md').write_text('# Научные подписи к рисункам\n\n'+'\n\n'.join(f'**Рисунок {j}.** '+r['caption'] for j,r in enumerate(figures,1))+'\n')
 for name,h in protected.items():assert sha(ROOT/name)==h,name
 write(OUT/'internal/source_hashes.after.json',{n:sha(ROOT/n) for n in protected})
 write(OUT/'internal/build_summary.json',dict(passed=True,examples=7,numerical_records=26,PNG_figures=len(figures),PDF_atlases=2,example_atlas_pages=29,summary_atlas_pages=5,all_previous_artifacts_unchanged=len(protected),solver_calls=0,new_training=0,new_inference=0))
 print(json.dumps(read(OUT/'internal/build_summary.json')))

if __name__=='__main__':build()
