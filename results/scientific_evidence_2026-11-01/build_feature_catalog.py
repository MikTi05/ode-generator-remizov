"""Exact 258-name catalogue from saved feature order and inspected source."""
from pathlib import Path
from collections import Counter
import csv
import json
import re
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
with np.load(ROOT / "results/ml_selector_hypotheses_2026-10-31/features.npz", allow_pickle=False) as z:
    names = z["augmented_names"].tolist()
with np.load(ROOT / "data/all_methods_v1_1000_2026-10-18/all_methods_equations_v1_ml.npz", allow_pickle=False) as z:
    x = z["x_grid"].copy()
indices = np.linspace(0, 255, 32, dtype=int)
certs = {
    "cert_a_lower_cert": ("A_minus", "Аналитический глобальный нижний сертификат a(x), не sampled min и не guard a_min"),
    "cert_a_upper_cert": ("A_plus", "Аналитический глобальный верхний сертификат a(x)"),
    "cert_b_abs_cert": ("B", "Глобальный сертификат abs(b(x)) <= B"),
    "cert_remizov_c_abs_cert": ("C", "Глобальный сертификат abs(c_R(x)) <= C"),
    "cert_remizov_lambda": ("lambda", "Заданный параметр резольвенты, c=c_R-lambda"),
}
pres = {
    "pre_gap": ("gap=lambda-C", "Сертифицированный положительный реакционный зазор"),
    "pre_T": ("T=max(0,(ln(2)-ln(gap)-ln(1e-8))/gap)", "Предвычисленный исторический горизонт R128; не полная погрешность и не G-нормированная граница"),
    "pre_T_over_n": ("T/128", "Шаг искусственного времени для конечной аппроксимации Чернова n=128; не шаг RK4 по x"),
    "pre_C_T": ("C*T", "Сертификат потенциала, умноженный на исторический горизонт"),
    "pre_lambda_T": ("lambda*T", "Масштаб затухания резольвентного веса на горизонте T"),
    "pre_a_contrast": ("A_plus/A_minus", "Контраст аналитических сертификатов диффузии"),
    "pre_diffusive_shift": ("2*sqrt(A_plus*T/128)", "Предвычисленная максимальная диффузионная трансляция оператора при tau=T/128"),
    "pre_drift_shift": ("2*B*T/128", "Предвычисленная максимальная дрейфовая трансляция при tau=T/128"),
    "pre_diffusive_length": ("sqrt(2*A_plus*T)", "Описательный диффузионный масштаб на горизонте T"),
    "pre_drift_length": ("B*T", "Описательный дрейфовый масштаб на горизонте T"),
}
stats = {
    "min": ("min_i(v_i)", "Минимум отсчётов", 52),
    "max": ("max_i(v_i)", "Максимум отсчётов", 52),
    "mean": ("mean_i(v_i)", "Невзвешенное арифметическое среднее", 52),
    "std": ("sqrt(mean_i((v_i-mean(v))^2))", "Стандартное отклонение ddof=0", 52),
    "rms": ("sqrt(mean_i(v_i^2))", "Среднеквадратическая амплитуда отсчётов", 53),
    "absmax": ("max_i(abs(v_i))", "Максимальный модуль отсчётов", 53),
    "median": ("median_i(v_i)", "Медиана; NumPy усредняет два средних значения при чётной длине", 53),
    "total_variation": ("sum_i(abs(v[i+1]-v[i]))", "Дискретная полная вариация без деления на длину интервала", 54),
    "d1_rms": ("sqrt(mean_i(d1_i^2))", "RMS первых разделённых разностей d1, 255 значений", 54),
    "d1_absmax": ("max_i(abs(d1_i))", "Максимальный модуль первых разделённых разностей", 55),
    "d2_rms": ("sqrt(mean_i(d2_i^2))", "RMS вторых разделённых разностей по серединам соседних интервалов, 254 значения", 55),
    "negative_fraction": ("mean_i(v_i<0)", "Доля строго отрицательных отсчётов; нулевые не входят", 56),
    "zero_crossings": ("count_i(v[i]*v[i+1]<0)", "Число строгих смен знака; переход через точный нулевой отсчёт не учитывается", 56),
    "spectral_high_fraction": ("sum(P_k,k>=8)/sum(P_k); 0 if sum(P_k)=0", "Доля мощности rFFT по индексам от 8; это индексный, не универсальный физический cutoff", 57),
}
channels = {"a": "a", "b": "b", "cR": "c_R=c+lambda", "gR": "g_R=-g"}
operations = {
    "constant": "константный лист value",
    "sin_affine": "лист sin(omega*x+phi)",
    "cos_affine": "лист cos(omega*x+phi)",
    "tanh_affine": "лист tanh(omega*x+phi)",
    "gaussian": "лист exp(-alpha*(x-mu)^2)",
    "lorentz": "лист 1/(1+alpha*(x-mu)^2)",
    "saturating": "лист (x-mu)/sqrt(1+(x-mu)^2)",
    "sin": "унарный sin(u)", "cos": "унарный cos(u)", "tanh": "унарный tanh(u)",
    "exp_neg_square": "унарный exp(-u^2)",
    "rational_saturate": "унарный u/(1+u^2)",
    "lincomb": "бинарный alpha*u+beta*v", "mul": "бинарный u*v",
}
research = {
    "research_local_peclet": ("abs(b0)/sqrt(a0*q0)", "Локальный дрейф относительно диффузионно-реакционного масштаба; не стандартный Pe с заданной длиной L", 63),
    "research_forcing_curvature_ratio": ("a0*RMS(G2(F))/(q0*RMS(F)); 0 if F=0", "Кривизна заданного forcing относительно реакционного масштаба; в коде F предварительно нормирован на RMS", 68),
    "research_diffusion_gradient_ratio": ("RMS(G(a))/sqrt(a0*q0)", "Изменчивость диффузии относительно локального масштаба", 65),
    "research_forcing_cancellation": ("abs(trapz(F))/trapz(abs(F)); 0 if F=0", "Компенсация положительной и отрицательной частей forcing на входной сетке", 74),
    "research_reaction_proxy_defect": ("RMS(a*G2(F/q)+b*G(F/q))/RMS(F); 0 if F=0", "Дифференциальный дефект реакционного proxy u=F/q; решение ОДУ не вычисляется", 79),
}
rows = []
for ordinal, name in enumerate(names, 1):
    channel = ""
    if name in certs:
        formula, meaning = certs[name]
        group, source, symbol, line = "certificate", "ml/remizov_features.py", "coefficient_features", 38
    elif name in pres:
        formula, meaning = pres[name]
        group, source, symbol, line = "precomputed", "ml/remizov_features.py", "coefficient_features", 39
    elif name in research:
        formula, meaning, line = research[name]
        group, source, symbol = "research", "ml/selector_research_features.py", "additional_coefficient_features"
    elif name.startswith("ast_"):
        match = re.fullmatch(r"ast_(a_raw|b_raw|c_raw)_(nodes|depth|op_.+)", name)
        assert match, name
        tree, suffix = match.groups()
        channel = tree
        group, source, symbol, line = "AST", "ml/remizov_features.py", "ast_features", 21
        if suffix == "nodes":
            formula, meaning = f"number_of_nodes({tree})", "Общее число узлов сырого дерева коэффициента до tanh-преобразования"
        elif suffix == "depth":
            formula, meaning = f"max_node_depth({tree}), root_depth=1", "Максимальная глубина сырого коэффициентного AST, корень имеет глубину 1"
        else:
            op = suffix[3:]
            formula, meaning = f"count_nodes(op='{op}', tree={tree})", "Число узлов операции: " + operations[op] + "; численные параметры узлов в счётчик не входят"
    elif name.startswith("corr_"):
        a, b = name[5:].split("_")
        formula = f"sum(center({a})*center({b}))/sqrt(sum(center({a})^2)*sum(center({b})^2)); 0 if denominator=0"
        meaning = f"Корреляция отсчётов каналов {channels[a]} и {channels[b]}; ноль для константного канала — соглашение кода"
        group, source, symbol, line = "correlation", "ml/remizov_features.py", "coefficient_features", 60
    else:
        match = re.fullmatch(r"(a|b|cR|gR)_(.+)", name)
        assert match, name
        channel, suffix = match.groups()
        source, symbol = "ml/remizov_features.py", "coefficient_features"
        if suffix.startswith("sample_"):
            k = int(suffix[7:]); j = int(indices[k])
            formula = f"{channels[channel]} at j={j}, x_j=-1+2*{j}/255={x[j]:.17g}"
            meaning = f"Отсчёт #{k:02d} канала {channels[channel]} в точном выбранном индексе входной сетки; не интерполяция"
            group, line = "sample", 59
        else:
            formula, meaning, line = stats[suffix]
            meaning += "; v=" + channels[channel]
            group = "channel_statistic"
    rows.append(dict(index=ordinal, feature_name=name, group=group, channel=channel,
                     mathematical_meaning=meaning, computation=formula,
                     source=source, function=symbol, line=line))
assert len(rows) == 258 and len({r['feature_name'] for r in rows}) == 258
expected = dict(certificate=5, precomputed=10, channel_statistic=56, sample=128,
                correlation=6, AST=48, research=5)
assert dict(Counter(r['group'] for r in rows)) == expected
with (OUT/'tables/all_258_features.csv').open('w',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
lines=['| № | Точное имя | Математический смысл | Вычисление | Код |', '|---:|---|---|---|---|']
for row in rows:
    citation=f"[{row['source']}:{row['line']}, {row['function']}](../../{row['source']}#L{row['line']})"
    lines.append('| '+ ' | '.join([str(row['index']), '`'+row['feature_name']+'`', row['mathematical_meaning'], '`'+row['computation']+'`',citation])+' |')
(OUT/'feature_catalog.md').write_text('\n'.join(lines)+'\n')
(OUT/'checks/feature_catalog.json').write_text(json.dumps(dict(features=258,groups=expected,exact_order_matches_saved_matrix=True,omitted_names=[]),indent=2)+'\n')
print('All 258 feature names catalogued with exact computation and code location')
