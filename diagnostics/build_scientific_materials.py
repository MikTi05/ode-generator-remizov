"""Verified report MATERIALS from saved data; no ODE/MC solve and no ML fit."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/ode-scientific-materials-mpl')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/scientific_report_materials_2026-10-25'
DATA=ROOT/'data/all_methods_v1_1000_2026-10-18'
R=ROOT/'results/remizov_ml_v1_2026-10-25'
C=ROOT/'results/mc1000_x0_2026-10-25'
J=ROOT/'results/joint_grid100_2026-10-25'
F=ROOT/'results/final_grid5_2026-10-25'
ML=ROOT/'results/ml_research_closure_2026-10-25'
O=ROOT/'results/mc_optimized_backend_2026-10-25'
TEMPLATES=ROOT/'diagnostics/scientific_report_templates'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n')
def f(value):return format(float(value),'.17g')
def mdtable(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join('---' for _ in headers)+' |\n'+''.join('| '+' | '.join(str(v) for v in row)+' |\n' for row in rows)


def latex_ast(node):
    p=node['params'];op=node['op'];ch=node['children']
    if op=='constant':return f(p['value'])
    if op in ('sin_affine','cos_affine','tanh_affine'):
        return '\\'+op.split('_')[0]+r'\!\left('+f(p['omega'])+'x+('+f(p['phi'])+r')\right)'
    if op=='lorentz':return r'\frac{1}{1+'+f(p['alpha'])+r'\left(x-('+f(p['mu'])+r')\right)^2}'
    if op=='saturating':
        u='x-('+f(p['mu'])+')'
        return r'\frac{'+u+r'}{\sqrt{1+('+u+r')^2}}'
    if op=='mul':return r'\left('+latex_ast(ch[0])+r'\right)\left('+latex_ast(ch[1])+r'\right)'
    if op=='lincomb':return f(p['alpha'])+r'\left('+latex_ast(ch[0])+r'\right)+('+f(p['beta'])+r')\left('+latex_ast(ch[1])+r'\right)'
    raise ValueError('Unimplemented example display operation: '+op)


def example_checks(meta):
    """Independent factored formulas vs safe AST jets; coefficient evaluation only."""
    import numpy as np
    import sympy as sp
    from src.all_methods_expressions import SafeExpressionNode,evaluate_jet2,evaluate_solution
    z=sp.symbols('z',real=True);q=1-z*z;ell=1-1/q
    assert sp.simplify(sp.diff(ell,z)+2*z/q**2)==0
    assert sp.simplify(sp.diff(ell,z,2)+sp.diff(ell,z)**2-(6*z**4-2)/q**4)==0
    result=[]
    for pid in ('AMV1_F0005_S02','AMV1_F0047_S00'):
        m=meta[pid];ast=m['serialized_ast'];p=ast['construction'];mu=p['bump_mu'];radius=p['bump_radius']
        x=np.unique(np.r_[np.linspace(-10,10,2049),np.linspace(mu-radius+.02,mu+radius-.02,1025),
            mu-radius,mu+radius,np.nextafter(mu-radius,-np.inf),np.nextafter(mu+radius,np.inf)])
        if pid=='AMV1_F0005_S02':
            assert ast['y_shape']['op']=='mul' and [v['op'] for v in ast['y_shape']['children']]==['lorentz','tanh_affine']
            lp,tp=[v['params'] for v in ast['y_shape']['children']]
            alpha,nu,omega,phi=lp['alpha'],lp['mu'],tp['omega'],tp['phi']
            d=x-nu;den=1+alpha*d*d;L=1/den;L1=-2*alpha*d/den**2;L2=2*alpha*(3*alpha*d*d-1)/den**3
            U=np.tanh(omega*x+phi);U1=omega*(1-U*U);U2=-2*omega*omega*U*(1-U*U)
            H=L*U;H1=L1*U+L*U1;H2=L2*U+2*L1*U1+L*U2
            a=np.full_like(x,p['a_mid']+p['a_half']*np.tanh(ast['a_raw']['params']['value']))
            b=np.full_like(x,p['b_scale']*np.tanh(ast['b_raw']['params']['value']))
            c=np.full_like(x,p['c_scale']*np.tanh(ast['c_raw']['params']['value'])-p['lambda'])
        else:
            # Check exact expected topology before using the independent compact form.
            assert ast['a_raw']['op']=='mul' and [v['op'] for v in ast['a_raw']['children']]==['lorentz','cos_affine']
            assert ast['b_raw']['op']=='lincomb' and [v['op'] for v in ast['b_raw']['children']]==['constant','saturating']
            assert ast['c_raw']['op']=='mul' and [v['op'] for v in ast['c_raw']['children']]==['lorentz','constant']
            assert [v['op'] for v in ast['y_shape']['children']]==['cos_affine','sin_affine']
            al,ac=[v['params'] for v in ast['a_raw']['children']]
            a=p['a_mid']+p['a_half']*np.tanh(np.cos(ac['omega']*x+ac['phi'])/(1+al['alpha']*(x-al['mu'])**2))
            br=ast['b_raw'];d=x-br['children'][1]['params']['mu']
            b=p['b_scale']*np.tanh(br['params']['alpha']*br['children'][0]['params']['value']+br['params']['beta']*d/np.sqrt(1+d*d))
            cl,cc=[v['params'] for v in ast['c_raw']['children']]
            c=p['c_scale']*np.tanh(cc['value']/(1+cl['alpha']*(x-cl['mu'])**2))-p['lambda']
            cp,sp_=[v['params'] for v in ast['y_shape']['children']]
            w,v=cp['omega'],sp_['omega'];u=w*x+cp['phi'];t=v*x+sp_['phi']
            H=np.cos(u)*np.sin(t);H1=-w*np.sin(u)*np.sin(t)+v*np.cos(u)*np.cos(t)
            H2=-(w*w+v*v)*np.cos(u)*np.sin(t)-2*w*v*np.sin(u)*np.cos(t)
        inside=np.abs((x-mu)/radius)<1
        B=np.zeros_like(x);D=np.zeros_like(x);E=np.zeros_like(x)
        z=(x[inside]-mu)/radius;q=1-z*z
        B[inside]=np.exp(1-1/q);D[inside]=-2*z/(radius*q*q);E[inside]=(6*z**4-2)/(radius*radius*q**4)
        y=p['solution_scale']*B*H
        dy=p['solution_scale']*B*(H1+D*H)
        d2=p['solution_scale']*B*(H2+2*D*H1+E*H)
        g=p['solution_scale']*B*(a*H2+(2*a*D+b)*H1+(a*E+b*D+c)*H)
        node={k:SafeExpressionNode.from_dict(ast[k]) for k in ('a_raw','b_raw','c_raw','y_shape')}
        aa=p['a_mid']+p['a_half']*np.tanh(evaluate_jet2(node['a_raw'],x).value)
        bb=p['b_scale']*np.tanh(evaluate_jet2(node['b_raw'],x).value)
        cr=p['c_scale']*np.tanh(evaluate_jet2(node['c_raw'],x).value);cc=cr-p['lambda']
        oracle=evaluate_solution(node['y_shape'],mu,radius,p['solution_scale'],x)
        gg=aa*oracle.d2+bb*oracle.d1+cc*oracle.value
        diffs={key:float(np.max(np.abs(v-u))) for key,v,u in [
            ('a',a,aa),('b',b,bb),('c',c,cc),('y',y,oracle.value),('dy',dy,oracle.d1),('d2y',d2,oracle.d2),('g',g,gg)]}
        assert max(diffs.values())<1e-10,diffs
        result.append(dict(equation_id=pid,expression_hash=m['expression_hash'],points=len(x),
            max_abs_differences=diffs,topology_checked=True,bump_log_derivatives_symbolically_verified=True,
            global_identity_basis='algebraic substitution of exact stored AST parameters; finite checks are supplementary, not global certificates'))
    write(OUT/'SUPPORT/example_formula_checks.json',result)
    write(OUT/'SUPPORT/examples_metadata.json',{pid:meta[pid] for pid in ('AMV1_F0005_S02','AMV1_F0047_S00')})
    return result


def equations(meta):
    text=r'''# Формулы для отбора в научный отчёт (не текст отчёта)

Все формулы согласованы с кодом проекта. Нумерация ниже служит для ссылок.
Проверена реализованная схема; первоисточник теоремы Ремизова/Чернова в
репозитории отсутствует и литературная атрибуция в этом комплекте не проверялась.
Не следует объявлять вычислительную проверку доказательством теоремы.

## M1. ОДУ и выбор ограниченного решения

$$a(x)y''(x)+b(x)y'(x)+c(x)y(x)=g(x),\qquad x\in\mathbb R.$$

Заданы вещественные коэффициенты a,b,c и правая часть g; y — искомое решение.
В эксперименте измерения только на [-1,1], но коэффициенты и постановка заданы
на всей прямой. Ищется ограниченное решение, а не произвольная задача Коши.
y(-1),y'(-1) в датасете существуют как справочные значения, эти методы их не используют.

## M2. Резольвента и достаточные условия в исследуемом классе

$$c_R=c+\lambda,\qquad g_R=-g,\qquad
L=a(x)\partial_x^2+b(x)\partial_x+c_R(x),\qquad(\lambda-L)y=g_R.$$

$$0<a_-\le a(x)\le a_+,\quad |b(x)|\le B,\quad
|c_R(x)|\le C,\quad \delta=\lambda-C>0,\qquad x\in\mathbb R.$$

Границы a_-,a_+,B,C глобальные аналитические. В all_methods_v1 коэффициенты
гладкие ограниченные; безопасная AST-конструкция даёт необходимую регулярность,
равномерную эллиптичность и корректную диффузию. Для произвольных callbacks
самого сертификата bounds недостаточно, регулярность отдельно предполагается.
g_R гладкая с компактным носителем. Условия обеспечивают затухание интеграла
резольвенты и выделяют ограниченное решение; проверка на конечной сетке не
доказывает эти условия. Здесь используется достаточное λ>C, а не оптимальная оценка sup c_R.

## M3. Ремизов / произведения Чернова

$$y(x)=\int_0^\infty e^{-\lambda t}(e^{tL}g_R)(x)\,dt,
\qquad e^{tL}u=\lim_{n\to\infty}S(t/n)^n u.$$

$$\begin{aligned}
(S(\tau)u)(x)={}&\tfrac14u(x+2\sqrt{a(x)\tau})
+\tfrac14u(x-2\sqrt{a(x)\tau})\\
&+\tfrac12u(x+2b(x)\tau)+\tau c_R(x)u(x).
\end{aligned}$$

$$y_{n,T}(x)=\int_0^T e^{-\lambda t}\,[S(t/n)^n g_R](x)\,dt.$$

u — тестовая функция/состояние, τ — шаг произведения; n — число применений S
для каждого t. При гладкой u: S(τ)u=u+τLu+O(τ²), ||S(τ)||∞≤1+τC.
В S входит c_R, не c; знак дрейфа плюс; λ находится в весе и выборе T.
Нельзя автоматически считать все веса неотрицательными при c_R<0.

## M4. Фактический ChernovSolver.solve в массовых и сравнительных сериях

$$t_j=T\left(\frac{j}{Q-1}\right)^2,\quad j=0,\ldots,Q-1,\qquad Q=513,$$
$$\widehat y_R(x_i)=I_{\Delta x}\!\left[
\sum_{j=0}^{Q-1}w_j e^{-\lambda t_j}
\bigl(S_{\Delta x}(t_j/n)\bigr)^n g_R\right](x_i).$$

w_j — веса составной **неравномерной формулы Симпсона**, I — кубический
сплайн с той же политикой продолжения при применении S. Для каждой тройки
(t_{2k},t_{2k+1},t_{2k+2}), h_0=t_{2k+1}-t_{2k}, h_1=t_{2k+2}-t_{2k+1},
H=h_0+h_1, вклады в общие веса:

$$w_{2k}\mathrel{+}=\frac H6(2-h_1/h_0),\quad
w_{2k+1}\mathrel{+}=\frac{H^3}{6h_0h_1},\quad
w_{2k+2}\mathrel{+}=\frac H6(2-h_0/h_1).$$

На квадратичной сетке отдельные веса могут быть отрицательными.
Фиксированное рабочее окно [-13,13], Δx=.0125, padding12, 2081 узел.
Исторический n128; заключительный n256. Это не адаптивный remizov_solver
с квадратурой Гаусса и не streaming-ветвь. Эти другие реализации в таблицы не включены.

`zero`: сплайн внутри окна, ноль за окном. `taper`: за краем продолжение
краевого значения и производной сплайна, умноженное на кубическое затухание;
при d≥0 вне края, s=clip(1-d/w,0,1), u_ext=(u_edge+σu'_edge d)s²(3-2s),
σ=-1 слева,+1 справа,w=.1 длины окна=2.6 при default. Это численные
продолжения, **не две разные краевые задачи исходного ОДУ**. Сплайн/окно
не наследуют автоматически теоретическую оценку нормы S.

## M5. Представление Фейнмана—Каца

$$dX_t=b(X_t)\,dt+\sqrt{2a(X_t)}\,dW_t,\qquad X_0=x,$$
$$y(x)=\mathbb E_x\!\left[\int_0^\infty
g_R(X_t)\exp\!\left(\int_0^t[c_R(X_s)-\lambda]\,ds\right)dt\right].$$

W_t — стандартное одномерное броуновское движение. Множитель √2 обязателен:
генератор этой диффузии a∂²+b∂, а не a∂²/2+b∂. Представление имеет ту же
резольвентную постановку и знак g_R=-g, что метод Чернова.

## M6. Реальный MC: Эйлер—Маруяма, две трапеции, усреднение

$$N=\lceil T/h_{\rm req}\rceil,\quad \Delta t=T/N\le h_{\rm req},\quad
X_{k+1}=X_k+b(X_k)\Delta t+\sqrt{2a(X_k)\Delta t}\,\xi_k,
\quad\xi_k\sim N(0,1).$$
$$A_0=0,\qquad A_{k+1}=A_k+\frac{\Delta t}{2}
\bigl[c_R(X_k)+c_R(X_{k+1})-2\lambda\bigr],$$
$$Z^{(m)}=\Delta t\left[\tfrac12g_R(X_0)
+\sum_{k=1}^{N-1}e^{A_k}g_R(X_k)+\tfrac12e^{A_N}g_R(X_N)\right],
\qquad\widehat y_{MC}(x)=\frac1M\sum_{m=1}^M Z^{(m)}.$$

M — пути **на одну пространственную точку и один seed**. PCG64/SeedSequence
адресует независимые точки/seed; chunk1024 — вычислительная порция, не M.
Backend Numba сохраняет именно эти формулы: scalar AST, без пространственной
интерполяции коэффициентов, без fastmath и параллельных редукций.
Повтор с новым seed — новое стохастическое решение; усреднение4 seed увеличивает бюджет в4 раза.

## M7. Аналитический хвост и выбор T

$$G\ge\sup_{x\in\mathbb R}|g_R(x)|,\qquad
\|y-y_T\|_\infty\le\frac G\delta e^{-\delta T},\qquad
T_{\rm cert}=\max\!\left(0,\frac1\delta\log\frac{G}{\delta\varepsilon_{tail}}\right).$$

При G=0 берётся T=0. Реализация округляет безопасно наружу; в общих сериях
использован ε_tail=1e-4. T контролирует только непрерывное усечение, не
временную схему и не статистический шум.
G получается аналитически из AST и производных manufactured y:

$$G=a_+Y_2+B Y_1+(C+\lambda)Y_0,\quad Y_j\ge\|y^{(j)}\|_\infty,$$
$$Y_0\le s\Phi_0H_0,\quad
Y_1\le s(\Phi_1H_0+\Phi_0H_1),\quad
Y_2\le s(\Phi_2H_0+2\Phi_1H_1+\Phi_0H_2).$$

Φ_j — глобальные границы производных bump, H_j — y_shape из рекурсивной AST,
s — solution_scale. Сертификат использует аналитическую конструкцию заданной
правой части, не известную ошибку solver; сеточный максимум G не подменяет доказательство.

Исторический Chernov autoT был иным:
$$T_{legacy}=\max\!\left(0,\frac{\log(2/(\delta\varepsilon))}{\delta}\right),
\qquad\text{получающаяся оценка хвоста }\le G\varepsilon/2.$$
При ε=1e-8 это **не** универсальная абсолютная граница ε без множителя G.
Новый общий T в grid100/grid5 выбран по G, затем явно передан ChernovSolver.

## M8. Метрики и статистическая неопределённость

$$x_i=-1+i/16,\quad i=0,\ldots,32,\qquad
E_{\infty,33}=\max_i|\widehat y(x_i)-y_{exact}(x_i)|,\qquad
E_{\infty,33}\le0.01.$$
$$E_0=|\widehat y(0)-y_{exact}(0)|\quad\text{(отдельная одноточечная метрика).}$$

Это дискретная максимум-норма на33 узлах, не доказанный sup на всём интервале.
В x0 используется только E0, не E∞33; соответствующие success нельзя смешивать.
Точность абсолютная, не относительная; переменные без заданных физических единиц.
Время — wall seconds в сохранённом окружении, не оптимальное время достижения допуска.

$$s_Z^2=\frac1{M-1}\sum_m(Z^{(m)}-\overline Z)^2,\qquad
SE=\frac{s_Z}{\sqrt M},\qquad
\overline Z\pm z_{1-\alpha/2}SE.$$

В коде объединяемые центральные моменты вместо E[Z²]-E[Z]²; при M=1
статистическая оценка не заменяется ложным нулевым CI. Интервал асимптотический
для дискретного усечённого среднего μ_{h,T}, без h-bias. Для одной фиксированной
сетки33 Bonferroni берёт z_(1-α/(2·33)); сохранённые grid100/grid5 также
поправляют число промежуточных оценок/задач согласно их протоколам.
Нельзя показывать33 отдельных95% CI как одновременный95% band.

$$\widehat y-y=(\widehat y-\mu_{h,T})+(\mu_{h,T}-y_T)+(y_T-y).$$

Три слагаемых: статистика, временная дискретизация (EM+трапеции), непрерывный
хвост. В исследованных MC-профилях отдельная граница второго не установлена.
Наблюдаемая ошибка≤.01 не доказывает p(success)≥.95. Даже4/4 успеха дают
одностороннюю95% Clopper–Pearson нижнюю границу .05^(1/4)=.47287.
Для Ремизова отдельно остаются конечное n, пространство/окно, квадратура и хвост.

## Реальные примеры production (числа без подгонки)

Ниже параметры печатаются с17 значащими цифрами из JSON, без округлённой
замены исходных функций. Это факторизация настоящих выражений, а не условные
коэффициенты. Полные исходные строки g_expr и AST лежат в
`SUPPORT/examples_metadata.json`. Проверки: `SUPPORT/example_formula_checks.json`.

Общий для двух примеров bump и его точные логарифмические производные:

$$z=(x-\mu)/R,\quad B_{\mu,R}(x)=
\begin{cases}\exp(1-1/(1-z^2)),&|z|<1,\\0,&|z|\ge1,\end{cases}$$
$$D_{\mu,R}(x)=\frac{-2z}{R(1-z^2)^2},\qquad
F_{\mu,R}(x)=\frac{6z^4-2}{R^2(1-z^2)^4}\quad(|z|<1).$$

D=B'/B,F=B''/B только внутри носителя; за ним y,g и производные равны0.
Нельзя вычислять D,F за носителем и затем умножать бесконечности на0.
В формуле ниже H,H',H'' **полностью заданы явными выражениями** для каждого
примера, так что g не оставлена как необъявленная производная решения.

'''
    for index,pid in enumerate(('AMV1_F0005_S02','AMV1_F0047_S00'),1):
        m=meta[pid];ast=m['serialized_ast'];p=ast['construction']
        a=f(p['a_mid'])+'+'+f(p['a_half'])+r'\tanh\!\left('+latex_ast(ast['a_raw'])+r'\right)'
        b=f(p['b_scale'])+r'\tanh\!\left('+latex_ast(ast['b_raw'])+r'\right)'
        c=f(p['c_scale'])+r'\tanh\!\left('+latex_ast(ast['c_raw'])+r'\right)-'+f(p['lambda'])
        text+=f'### Пример {index}: `{pid}`\n\nTrain; {m["complexity_level"]}, режим {m["regime"]}; production seed2488.\n\n'
        text+='$$a_'+str(index)+'(x)='+a+'.$$\n\n'
        text+='$$b_'+str(index)+'(x)='+b+'.$$\n\n'
        text+='$$c_'+str(index)+'(x)='+c+'.$$\n\n'
        text+=f'$$s_{index}={f(p["solution_scale"])},\\quad\\mu_{index}={f(p["bump_mu"])},\\quad R_{index}={f(p["bump_radius"])},\\quad\\lambda_{index}={f(p["lambda"])}.$$\n\n'
        text+='$$H_'+str(index)+'(x)='+latex_ast(ast['y_shape'])+r',\qquad y_'+str(index)+'(x)=s_'+str(index)+'B_{\\mu_'+str(index)+',R_'+str(index)+'}(x)H_'+str(index)+'(x).$$\n\n'
        if index==1:
            l,t=[v['params'] for v in ast['y_shape']['children']]
            text+=r'Для явных производных обозначим '+f'$\\alpha={f(l["alpha"])},\\nu={f(l["mu"])},\\omega={f(t["omega"])},\\varphi={f(t["phi"])}$'+r'''.
$$q=x-\nu,\quad L=(1+\alpha q^2)^{-1},\quad U=\tanh(\omega x+\varphi),$$
$$L_1=\frac{-2\alpha q}{(1+\alpha q^2)^2},\quad
L_2=\frac{2\alpha(3\alpha q^2-1)}{(1+\alpha q^2)^3},$$
$$U_1=\omega(1-U^2),\quad U_2=-2\omega^2U(1-U^2),$$
$$H_1=LU,\quad H_1'=L_1U+LU_1,\quad H_1''=L_2U+2L_1U_1+LU_2.$$
'''
        else:
            cp,sp_=[v['params'] for v in ast['y_shape']['children']]
            text+=f'Положим $p={f(cp["omega"])},q={f(cp["phi"])},r={f(sp_["omega"])},v={f(sp_["phi"])}$.\n'
            text+=r'''$$u=px+q,\quad w=rx+v,\quad H_2=\cos u\sin w,$$
$$H_2'=-p\sin u\sin w+r\cos u\cos w,$$
$$H_2''=-(p^2+r^2)\cos u\sin w-2pr\sin u\cos w.$$
'''
        i=str(index)
        text+='\nНастоящая правая часть в компактной явной факторизации (внутри носителя):\n\n'
        text+='$$g_'+i+'(x)=s_'+i+'B_{\\mu_'+i+',R_'+i+'}(x)\\Bigl\\{a_'+i+"(x)H_"+i+"''(x)+(2a_"+i+'(x)D_{\\mu_'+i+',R_'+i+'}(x)+b_'+i+"(x))H_"+i+"'(x)\n+(a_"+i+'(x)F_{\\mu_'+i+',R_'+i+'}(x)+b_'+i+'(x)D_{\\mu_'+i+',R_'+i+'}(x)+c_'+i+'(x))H_'+i+'(x)\\Bigr\\}.$$\n\n'
        text+='Вне $|x-\\mu_'+i+'|<R_'+i+'$: $g_'+i+'=y_'+i+'=0$. Для резольвенты $g_{R,'+i+'}=-g_'+i+'$, $c_{R,'+i+'}=c_'+i+'+\\lambda_'+i+'$.\n\n'
        text+=f'Идентичность: `expression_hash={m["expression_hash"]}`.\n\n'
    (OUT/'EQUATIONS.md').write_text(text)


def saved_series():
    import numpy as np
    import pandas as pd
    jt=pd.read_csv(J/'analysis/comparison.csv');ft=pd.read_csv(F/'analysis/comparison.csv')
    selected=[]
    for pid in ft.problem_id:
        records=[read(F/f'records/{pid}_MC{k}.json') for k in range(4)]
        rr=read(F/f'records/{pid}_R256.json');old=read(F/f'baseline/{pid}_R128.json')
        assert rr['expression_hash']==old['expression_hash']==records[0]['expression_hash']
        x=np.array(rr['result']['x']);y=np.array(rr['y_exact']);assert len(x)==33
        rows=[]
        for label,r in [('R128',old),('R256',rr)]+[(f'MC{k}',r) for k,r in enumerate(records)]:
            assert np.array_equal(r['result']['x'],x) and np.array_equal(r['y_exact'],y)
            values=np.array(r['result']['f']);err=float(np.max(abs(values-y)))
            assert abs(err-r['E_inf'])<1e-13
            rows.append(dict(method=label,values=values.tolist(),error=err,seconds=r['result']['runtime_seconds']))
        selected.append(dict(problem_id=pid,x=x.tolist(),exact=y.tolist(),T=rr['result']['T'],series=rows))
    write(OUT/'SUPPORT/final5_saved_series.json',selected)
    return jt,ft,selected


def figures(jt,ft,series):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve,roc_auc_score
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
        'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.savefig(OUT/f'FIGURES/{name}.png',dpi=400,bbox_inches='tight',facecolor='white')
        fig.savefig(OUT/f'FIGURES/{name}.pdf',bbox_inches='tight',facecolor='white')
        plt.close(fig)
    s=next(s for s in series if s['problem_id']=='AMV1_F0005_S02');x=np.array(s['x']);y=np.array(s['exact']);r={r['method']:r for r in s['series']}
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.9),layout='constrained')
    axes[0].plot(x,y,'k-',lw=1.8,label='Exact (stored nodes)')
    axes[0].plot(x,r['R256']['values'],'o',ms=3,mfc='none',label='Remizov n=256')
    for k in range(4):axes[0].plot(x,r[f'MC{k}']['values'],lw=.6,alpha=.5,color='#d17b21',label='MC: four fresh seeds' if k==0 else None)
    mean=np.mean([r[f'MC{k}']['values'] for k in range(4)],axis=0)
    axes[0].plot(x,mean,'--',color='#b84a00',lw=1.5,label='MC mean: 4 x 4096 paths')
    axes[0].set(xlabel='x (dimensionless)',ylabel='Solution y(x)',title='AMV1_F0005_S02: saved solutions');axes[0].legend()
    for label,key in [('Remizov n=128','R128'),('Remizov n=256','R256')]:axes[1].plot(x,np.array(r[key]['values'])-y,label=label)
    axes[1].plot(x,mean-y,'--',color='#b84a00',label='MC mean: 4 x 4096')
    axes[1].axhline(.01,color='gray',ls=':',lw=.8);axes[1].axhline(-.01,color='gray',ls=':',lw=.8)
    axes[1].set(xlabel='x (dimensionless)',ylabel='Signed error (dimensionless)',title=f"Common certified T = {s['T']:.5f}");axes[1].legend()
    for ax in axes:ax.grid(alpha=.2)
    save(fig,'01_solution_and_error')
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.9),layout='constrained')
    for ax,cohort,title in zip(axes,['representative','stress'],['Representative train sample (80)','Targeted stress sample (20)']):
        sub=jt[jt.cohort==cohort]
        for label,col in [('Remizov n=128, zero','R_error'),('MC1024: first seed','MC_primary_error')]:
            v=np.sort(sub[col]);ax.step(np.r_[min(v)*.8,v],np.r_[0,np.arange(1,len(v)+1)/len(v)],where='post',label=label)
        ax.axvline(.01,color='black',ls='--',lw=.8,label='Tolerance 0.01')
        ax.set(xscale='log',ylim=(0,1.02),xlabel=r'$E_{\infty,33}$ (dimensionless)',ylabel='Empirical fraction',title=title);ax.grid(alpha=.2);ax.legend(loc='lower right')
    save(fig,'02_grid33_error_distributions')
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.9),layout='constrained')
    for cohort,color,marker,label in [('representative','#2379a8','o','Representative 80'),('stress','#d17b21','^','Stress 20')]:
        sub=jt[jt.cohort==cohort];axes[0].scatter(sub.R_seconds,sub.MC_primary_seconds,s=14,c=color,marker=marker,alpha=.7,label=label)
    maxv=max(jt.R_seconds.max(),jt.MC_primary_seconds.max())*1.05;axes[0].plot([0,maxv],[0,maxv],'k--',lw=.7,label='Equal time')
    axes[0].set(xlabel='Remizov grid33 solve time (s)',ylabel='MC grid33 solve time (s)',title='Grid100: direct fixed-profile timing');axes[0].legend()
    for i,s in enumerate(series):
        rr={r['method']:r for r in s['series']}
        axes[1].scatter(rr['R128']['seconds'],rr['R128']['error'],c='#2379a8',marker='o',s=24,label='Remizov n=128' if i==0 else None)
        axes[1].scatter(rr['R256']['seconds'],rr['R256']['error'],c='#164259',marker='s',s=24,label='Remizov n=256' if i==0 else None)
        axes[1].plot([rr['R128']['seconds'],rr['R256']['seconds']],[rr['R128']['error'],rr['R256']['error']],c='#2379a8',alpha=.3,lw=.7)
        for k in range(4):axes[1].scatter(rr[f'MC{k}']['seconds'],rr[f'MC{k}']['error'],c='#d17b21',marker='x',s=24,label='MC4096: individual seed' if i==0 and k==0 else None)
        mean=np.mean([rr[f'MC{k}']['values'] for k in range(4)],axis=0);err=max(abs(mean-np.array(s['exact'])))
        axes[1].scatter(sum(rr[f'MC{k}']['seconds'] for k in range(4)),err,c='#ac4218',marker='D',s=24,label='MC mean4: full cost' if i==0 else None)
    axes[1].axhline(.01,color='black',ls='--',lw=.8)
    axes[1].set(xscale='log',yscale='log',xlabel='Full solution cost (s)',ylabel=r'$E_{\infty,33}$',title='Targeted five: accuracy versus cost');axes[1].legend(fontsize=6.7)
    for ax in axes:ax.grid(alpha=.2)
    save(fig,'03_accuracy_and_cost')
    rp=pd.read_csv(R/'predictions.csv');cp=pd.read_csv(C/'ml/predictions.csv');att=pd.read_csv(C/'analysis/attempts.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.9),layout='constrained')
    sub=rp[(rp.split=='test')&rp.target_known];y=sub.accuracy_met.to_numpy()
    for name,label in [('catboost','CatBoost'),('logistic','Logistic')]:
        p=sub['p_success_'+name];false,true,_=roc_curve(1-y,1-p);auc=roc_auc_score(1-y,1-p)
        axes[0].plot(false,true,label=f'{label}: AUC={auc:.3f}')
    lookup=cp.set_index('equation_id');a=att[att.split=='test'];y=a.accuracy_met.to_numpy()
    for name,label in [('catboost','CatBoost'),('logistic','Logistic')]:
        p=lookup.loc[a.equation_id,'p_mc_single_seed_accuracy_'+name].to_numpy();false,true,_=roc_curve(1-y,1-p);auc=roc_auc_score(1-y,1-p)
        axes[1].plot(false,true,label=f'{label}: AUC={auc:.3f}')
    for ax,title in zip(axes,['R taper grid33: 150 equations','MC1024 x=0: 600 seed labels']):
        ax.plot([0,1],[0,1],'k--',lw=.7,label='Constant baseline: AUC=0.5')
        ax.set(xlabel='False positive rate',ylabel='True positive rate',title=title,xlim=(0,1),ylim=(0,1.02));ax.grid(alpha=.2);ax.legend(loc='lower right',fontsize=7)
    save(fig,'04_ml_roc')


def tables(ft):
    import pandas as pd
    rt=pd.read_csv(R/'targets.csv');eq=pd.read_csv(C/'analysis/equations.csv');att=pd.read_csv(C/'analysis/attempts.csv');jt=pd.read_csv(J/'analysis/comparison.csv')
    table1=[['R1000','E∞33','n128, taper, legacy T','871/997','16620,74'],
        ['R1000: узел x=0','E0','тот же исторический solve33','930/997','не измерено для x=0'],
        ['MC1000','E0','один seed ×1024','1426/4000','1633,25 (все seed)'],
        ['MC1000','E0','среднее4×1024','535/1000','те же 1633,25'],
        ['Grid100: repr80','E∞33','R128 zero / MC1024','66/80 / 0/80','989,60 / 1060,42'],
        ['Grid100: stress20','E∞33','R128 zero / MC1024','7/20 / 0/20','248,74 / 773,46'],
        ['Grid5: целевые train','E∞33','R128 → R256 zero','2/5 → 5/5','baseline60,94 →117,78'],
        ['Grid5: целевые train','E∞33','MC: четыре свежих seed×4096','5/20 trial','505,03 (все20)']]
    assert int(rt.accuracy_met.sum())==871 and rt.target_known.sum()==997
    assert eq.remizov_accuracy0.sum()==930 and eq.mc_aggregate_accuracy.sum()==535 and att.accuracy_met.sum()==1426
    assert ((jt[jt.cohort=='representative'].R_error<=.01).sum(),(jt[jt.cohort=='stress'].R_error<=.01).sum())==(66,7)
    assert (ft.R128_pass.sum(),ft.R256_pass.sum(),ft.MC_successes.sum())==(2,5,5)
    rm=read(R/'metrics.json');cm=read(C/'ml/metrics.json')
    table2=[]
    for scope,task,models in [('R grid33 taper','accuracy',rm['classification']),('R x0 taper','remizov_point_accuracy',cm['classification']['remizov_point_accuracy']),('MC x0 single1024','mc_single_seed_accuracy',cm['classification']['mc_single_seed_accuracy'])]:
        selected='catboost';names=['dummy_prior','logistic','catboost'] if task=='accuracy' else ['dummy','logistic','catboost']
        for name in names:
            m=models[name]['test']['validation_threshold'] if task=='accuracy' else models[name]['test']
            table2.append([scope,name+(' (выбран)' if name==selected else ''),f"{m['roc_auc']:.4f}",f"{m['logloss']:.4f}",f"{m['brier']:.4f}"])
    text='''# Две компактные таблицы для основного отчёта

Это заготовки таблиц с оговорками, не текст научного отчёта. Если объём тесен,
в таблице2 оставить по одной строке CatBoost и dummy на цель. Не объединять
строки разных пространственных областей/профилей в общий рейтинг.

## Таблица1. Наблюдаемая точность и измеренная стоимость фиксированных профилей

'''+mdtable(['Серия','Метрика','Профиль','Достигли ≤0,01','Σ solver wall, с'],table1)+'''
Примечания: (1) 3 worker_exception R1000 имеют неизвестную математическую
ошибку; denominator997 не1000. (2) Историческое время R33 не делить на33
и не сопоставлять как стоимость одного x0 MC. Оно смешивает планировавшиеся
8/1 внешних workers. (3) Среднее MC — тот же raw набор с иным estimator,
затраты не складывать дважды. (4) Grid100 repr80 иstress20 различаются
отбором; смешанные73/100 не оценка успеха production1000. (5) Baseline grid5
скопирован из прежних результатов, не пересчитан. Его время не новая затрата.
(6) M указан на точку/seed. Общий grid T одинаков у R иMC и сертифицирован;
h_req1/128, Q513,dx.0125,padding12. R1000 имеет другой legacy T.
(7) Время fixed profile не является оптимальным time-to-tolerance.

## Таблица2. Классификация качества на сохранённом семейном test

'''+mdtable(['Цель / область','Модель','ROC-AUC','Log loss','Brier'],table2)+'''
Положительный класс здесь успех; ROC на рисунке4 показывает риск неуспеха
(инверсия обеих y иp даёт тот же AUC). CB выбран прежним validation, а не
наилучшим test. R-grid иR-point:150 test-уравнений из30семейств; MC600
seed-меток тех же150уравнений, не600 независимых задач. Старый test ранее
изучался численно, поэтому это описательная проверка, не pristine prospective
проверка policy. AUC не сертификат точности и не качество выбора solver.

Регрессия времени/ошибки, все splits и baseline: SUPPORT/ml_metrics.csv.
Полная таблица grid5 и точные числа: SUPPORT/final5_comparison.csv.
Сравнительная таблица с дополнительными строками: SUPPORT/all_experiments.csv.
'''
    (OUT/'TABLES.md').write_text(text)


def facts(meta,ft,checks):
    import pandas as pd
    rt=pd.read_csv(R/'targets.csv');eq=pd.read_csv(C/'analysis/equations.csv');att=pd.read_csv(C/'analysis/attempts.csv')
    timing=pd.read_csv(O/'timing_summary.csv')
    assert abs(rt.solver_seconds.sum()-16620.74014193064)<1e-8
    assert (int(rt.target_known.sum()),int(rt.accuracy_met.sum()),int(eq.remizov_accuracy0.sum()),int(eq.mc_aggregate_accuracy.sum()),int(att.accuracy_met.sum()))==(997,871,930,535,1426)
    assert abs(timing.speedup_numba.max()-8.30727197567654)<1e-12
    assert {s:sum(m['split']==s for m in meta.values()) for s in ('train','validation','test')}=={'train':700,'validation':150,'test':150}
    text=(TEMPLATES/'FACTS.md').read_text()
    text+='\n\n## F11. Точные значения финальных пяти (справочно)\n\n'
    for row in ft.itertuples():
        text+=f'- `{row.problem_id}`: T={row.T:.8f}; R128 {row.R128_error:.9f} → R256 {row.R256_error:.9f}; MC {row.MC_successes}/4; ΣMC={row.MC_total_solver_seconds:.5f} с.\n'
    (OUT/'FACTS.md').write_text(text)


def sources():
    import pandas as pd
    files=[
        'AGENTS.md','PROJECT_STATUS.md','.claude/rules/generator.md','.claude/rules/remizov.md',
        '.claude/skills/experiment/SKILL.md','chernov_solver.py','remizov_solver.py','mc_solver.py','mc_optimized.py',
        'mc_ast_kernels.py','all_methods_bounds.py','all_methods_dataset_loader.py',
        'src/all_methods_expressions.py','src/all_methods_dataset_v1.py',
        'ml/remizov_features.py','ml/quality_prototype.py',
        'diagnostics/build_scientific_materials.py',
        'data/all_methods_v1_1000_2026-10-18/all_methods_equations_v1_report.json',
        'results/dataset1000_2026-10-18/REPORT.md',
        'results/remizov_ml_v1_2026-10-25/REPORT.md',
        'results/remizov_ml_v1_2026-10-25/data_audit.json',
        'results/remizov_ml_v1_2026-10-25/models/metadata.json',
        'results/remizov_ml_v1_2026-10-25/metrics.json',
        'results/mc1000_x0_2026-10-25/REPORT.md',
        'results/mc1000_x0_2026-10-25/analysis/summary.json',
        'results/mc1000_x0_2026-10-25/ml/metrics.json',
        'results/mc1000_x0_2026-10-25/ml/models/metadata.json',
        'results/mc_optimized_backend_2026-10-25/REPORT.md',
        'results/mc_optimized_backend_2026-10-25/compilation.json',
        'results/mc_optimized_backend_2026-10-25/protocol.json',
        'results/mc_optimized_backend_2026-10-25/equivalence_summary.json',
        'results/joint_grid100_2026-10-25/REPORT.md',
        'results/joint_grid100_2026-10-25/INTERPRETATION.md',
        'results/joint_grid100_2026-10-25/analysis/summary.json',
        'results/final_grid5_2026-10-25/REPORT.md',
        'results/final_grid5_2026-10-25/INTERPRETATION.md',
        'results/final_grid5_2026-10-25/analysis/summary.json',
        'results/final_grid5_2026-10-25/analysis/statistics.json',
        'results/ml_research_closure_2026-10-25/REPORT.md',
        'results/ml_research_closure_2026-10-25/model_audit.json',
        'results/ml_research_closure_2026-10-25/verification.json',
        'results/ml_research_closure_2026-10-25/registry.json']
    files.extend(str(p.relative_to(ROOT)) for p in sorted(TEMPLATES.glob('*.md')))
    for relative in files:
        dest=OUT/'SOURCES_SNAPSHOT'/relative;dest.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(ROOT/relative,dest)
    copies=[(R/'targets.csv','remizov_targets.csv'),(R/'predictions.csv','remizov_predictions.csv'),
        (C/'analysis/equations.csv','mc1000_equations.csv'),(C/'analysis/attempts.csv','mc1000_attempts.csv'),
        (C/'ml/predictions.csv','mc1000_ml_predictions.csv'),(J/'analysis/comparison.csv','grid100_comparison.csv'),
        (F/'analysis/comparison.csv','final5_comparison.csv'),(ML/'ml_metrics.csv','ml_metrics.csv'),
        (ML/'experiments.csv','all_experiments.csv'),(O/'timing_summary.csv','optimization_timing.csv')]
    for src,name in copies:shutil.copy2(src,OUT/'SUPPORT'/name)
    evidence=[]
    for p in files:
        evidence.append(dict(project_path=p,sha256=sha(ROOT/p),archive_path='SOURCES_SNAPSHOT/'+p))
    for src,name in copies:evidence.append(dict(project_path=str(src.relative_to(ROOT)),sha256=sha(src),archive_path='SUPPORT/'+name))
    for p in sorted((F/'records').glob('*.json')):
        dest=OUT/'SUPPORT/final5_raw'/p.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(p,dest)
        evidence.append(dict(project_path=str(p.relative_to(ROOT)),sha256=sha(p),archive_path=str(dest.relative_to(OUT))))
    for p in sorted((F/'baseline').glob('*_R128.json')):
        dest=OUT/'SUPPORT/final5_raw'/p.name;shutil.copy2(p,dest)
        evidence.append(dict(project_path=str(p.relative_to(ROOT)),sha256=sha(p),archive_path=str(dest.relative_to(OUT))))
    write(OUT/'SUPPORT/source_manifest.json',evidence)
    for item in evidence:
        assert sha(ROOT/item['project_path'])==item['sha256']==sha(OUT/item['archive_path'])
    (OUT/'SOURCES.md').write_text((TEMPLATES/'SOURCES.md').read_text())



def main():
    import numpy as np
    import pandas as pd
    import importlib.metadata
    assert (OUT/'SUPPORT/protected_before.json').exists(),'create fresh output and snapshot first'
    meta={m['equation_id']:m for m in map(json.loads,(DATA/'all_methods_equations_v1_metadata.jsonl').read_text().splitlines())}
    checks=example_checks(meta);equations(meta)
    jt,ft,series=saved_series();figures(jt,ft,series);tables(ft);facts(meta,ft,checks);sources()
    write(OUT/'SUPPORT/environment.json',dict(python=sys.version,versions={k:importlib.metadata.version(k)
        for k in ('numpy','pandas','scipy','sympy','matplotlib','Pillow')},new_ode_solves=0,new_mc_runs=0,new_fits=0))
    shutil.copy2(TEMPLATES/'CAPTIONS.md',OUT/'FIGURES/CAPTIONS.md')
    shutil.copy2(TEMPLATES/'README.md',OUT/'README.md')
    before=read(OUT/'SUPPORT/protected_before.json');changed=[p for p,h in before.items() if not (ROOT/p).exists() or sha(ROOT/p)!=h]
    assert not changed,changed
    write(OUT/'SUPPORT/verification.json',dict(protected_paths=len(before),changed=changed,new_ode_solves=0,new_mc_runs=0,new_model_fits=0,
        formula_examples=checks,figures_png=4,figures_pdf=4,main_tables=2,source_copy_hashes_verified=True,
        known_secondary_summary_error='closure speedup21.93 unsupported; primary maximum8.30727197567654 used'))
    print(json.dumps(read(OUT/'SUPPORT/verification.json'),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
