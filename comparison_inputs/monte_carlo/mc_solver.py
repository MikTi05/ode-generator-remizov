"""Метод Монте-Карло для уравнения a f'' + b f' + (c − λ) f = −g на прямой.

Решение записывается через формулу Фейнмана–Каца

    f(x0) = E ∫_0^∞ exp(−λt + ∫_0^t c(X_s) ds) g(X_t) dt,
    dX = b(X) dt + sqrt(2 a(X)) dW,   X_0 = x0,

и считается так: M траекторий по схеме Эйлера–Маруямы с шагом dt до
времени T, вдоль каждой — интеграл по t трапецией, затем среднее по
траекториям. Схема работы по шагам — в README.md.

Пример:
    from mc_solver import solve
    res = solve(a=lambda x: 1 + 0*x, b=lambda x: 0*x, c=lambda x: 0*x,
                g=my_g, lam=2.0, x0=0.0, M=100_000)
    print(res.value, "±", res.ci95)
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class MCResult:
    """Оценка f(x0) и её статистическая погрешность."""
    x0: float
    value: float      # среднее по траекториям
    stderr: float     # стандартная ошибка среднего, σ/sqrt(M)
    ci95: float       # полуширина 95-процентного интервала, 1.96·stderr
    M: int
    dt: float         # фактический шаг (T делится на целое число шагов)
    T: float
    seconds: float


def _vectorize(func):
    """Приводит f(x) к массиву той же формы, что x (на случай констант)."""
    def wrapped(xs):
        return np.broadcast_to(np.asarray(func(xs), dtype=float), xs.shape)
    return wrapped


def auto_T(lam: float, c_sup: float, eps_T: float) -> float:
    """Верхний предел по t: отбрасываемый хвост интеграла не больше eps_T.

    То же правило, что в черновской схеме:
    T = max(0, ln(2/(δ·eps_T))/δ), δ = λ − sup|c|.
    """
    delta = lam - c_sup
    if delta <= 0:
        raise ValueError(f"нужно λ > sup|c|, а λ − sup|c| = {delta:.3g}")
    return max(0.0, float(np.log(2.0 / (delta * eps_T)) / delta))


def path_contributions(a, b, c, g, lam: float, x0: float, T: float,
                       dt: float, m_paths: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Вклады m_paths траекторий в оценку f(x0); их среднее и есть оценка.

    Интеграл по t считается трапецией на той же сетке, что и траектория.
    """
    a, b, c, g = map(_vectorize, (a, b, c, g))
    n_steps = max(int(round(T / dt)), 1)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    xs = np.full(m_paths, float(x0))
    int_c = np.zeros(m_paths)                 # ∫_0^t c(X_s) ds
    c_prev = c(xs).copy()
    total = 0.5 * dt * g(xs)                  # узел t = 0, вес трапеции dt/2

    for step in range(1, n_steps + 1):
        xs = xs + b(xs) * dt + np.sqrt(2.0 * a(xs)) * sqrt_dt * \
            rng.standard_normal(m_paths)
        c_now = c(xs)
        int_c += 0.5 * (c_prev + c_now) * dt
        c_prev = c_now.copy()
        weight = 0.5 * dt if step == n_steps else dt
        total += weight * np.exp(-lam * step * dt + int_c) * g(xs)
    return total


def solve(a, b, c, g, lam: float, x0, *, M: int = 100_000,
          dt: float = 0.005, T: float | None = None, eps_T: float = 1e-4,
          c_sup: float | None = None, seed: int | None = None,
          chunk: int = 200_000):
    """Значение f в точке x0 (или в каждой точке массива x0).

    a, b, c, g -- функции numpy-массива; lam -- число, lam > sup|c|.
    M          -- число траекторий: статистическая ошибка ~ 1/sqrt(M).
    dt         -- шаг Эйлера–Маруямы: смещение ~ dt (при постоянном a
                  и c = 0 -- ~ dt²).
    T          -- верхний предел по t; если None, берётся auto_T.
    c_sup      -- sup|c| на прямой; если None, оценивается на [x0−20, x0+20].
    chunk      -- траекторий за один проход (ограничивает память).

    Возвращает MCResult для числа x0 и список MCResult для массива.
    """
    points = np.atleast_1d(np.asarray(x0, dtype=float))
    rng = np.random.default_rng(seed)
    results = []
    for p in points:
        if T is None:
            if c_sup is None:
                probe = np.linspace(p - 20.0, p + 20.0, 40001)
                c_sup_p = float(np.max(np.abs(_vectorize(c)(probe))))
            else:
                c_sup_p = c_sup
            T_p = auto_T(lam, c_sup_p, eps_T)
        else:
            T_p = T
        n_steps = max(int(round(T_p / dt)), 1)

        t0 = time.perf_counter()
        s1 = s2 = 0.0                          # сумма вкладов и их квадратов
        done = 0
        while done < M:
            m = min(chunk, M - done)
            v = path_contributions(a, b, c, g, lam, p, T_p, dt, m, rng)
            s1 += float(v.sum())
            s2 += float((v * v).sum())
            done += m
        mean = s1 / M
        var = max(s2 / M - mean * mean, 0.0) * M / max(M - 1, 1)
        se = float(np.sqrt(var / M))
        results.append(MCResult(x0=float(p), value=mean, stderr=se,
                                ci95=1.96 * se, M=int(M), dt=T_p / n_steps,
                                T=float(T_p),
                                seconds=time.perf_counter() - t0))
    return results[0] if np.ndim(x0) == 0 else results


def paths_for_tolerance(stderr: float, M: int, tol: float) -> int:
    """Сколько траекторий нужно, чтобы полуширина 95%-интервала стала ≤ tol.

    Берётся по пробному прогону: σ = stderr·sqrt(M), нужно 1.96·σ/sqrt(M') ≤ tol.
    """
    sigma = stderr * np.sqrt(M)
    return int(np.ceil((1.96 * sigma / tol) ** 2))
