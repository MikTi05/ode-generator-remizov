import os
import multiprocessing as mp

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline


# Воркер для solve(n_jobs>1): каждый (S(t/n))^n g при разных t считается
# независимо (формула (3) — интеграл по t от независимых слагаемых), поэтому
# solver кладётся в глобальную переменную процесса через initializer, а не
# передаётся с каждой задачей — так избегаем pickle'а a/b/c/g (это могут
# быть замыкания sympy.lambdify) на каждый вызов, только один раз при
# форке пула.
_pool_solver = None


def _init_pool_worker(solver):
    global _pool_solver
    _pool_solver = solver


def _pool_eval_t(t):
    s = _pool_solver
    u = s._chernoff_on_grid(t)
    return np.exp(-s.lam * t) * u


class ChernovSolver:
    """
    Решение ОДУ вида
        a(x) f''(x) + b(x) f'(x) + (c(x) - lam) f(x) = -g(x),   x in R,
    методом черновских приближений по Теореме 6 из статьи Ремизова (2025).

    Формула оператора Чернова (формула (6) в статье):
        (S(t)f)(x) = 1/4 f(x + 2 sqrt(a(x) t))
                   + 1/4 f(x - 2 sqrt(a(x) t))
                   + 1/2 f(x + 2 b(x) t)
                   + t c(x) f(x).

    Приближение решения:
        f(x) ≈ ∫_0^T e^{-lam t} (S(t/n))^n g (x) dt.

    Условия применимости:
        • lam > sup_x |c(x)|;
        • a(x) >= a0 > 0;
        • g ∈ C_0(R) (g непрерывна и стремится к нулю на бесконечности).
    Область [x_min, x_max] следует брать с запасом относительно носителя g,
    чтобы продолжение g нулём вне [x_min, x_max] было корректным.

    Параметры
    ---------
    a, b, c, g : callable
        Коэффициенты уравнения и правая часть (векторизованные).
    lam : float
        Параметр (lambda) в уравнении. Должно быть lam > ‖c‖_∞.
    x_min, x_max : float
        Границы пространственной сетки усечения R.
    dx : float
        Шаг пространственной сетки.
    T : float, optional
        Верхний предел интегрирования по t. Если None, выбирается автоматически
        по item 5 Теоремы 6.
    n : int, optional
        Число шагов в композиции (S(t/n))^n. По умолчанию 100. Согласно item 4,
        ошибка усечения ведёт себя как 2·C_g / (n · (λ − ‖c‖)^3), так что для
        более точных расчётов n следует увеличивать.
    num_t : int, optional
        Число узлов временной сетки (должно быть нечётным для Симпсона).
    eps : float, optional
        Целевая точность усечения интеграла по t (используется в _auto_T).
    c_sup : float, optional
        Верхняя граница |c(x)|. Если None, оценивается по тонкой подсетке c(x).
    t_grading : str, optional
        Тип сетки по t: 'uniform' или 'quadratic' (сгущение у 0).

    Два способа счёта
    -----------------
    solve() — формула (3) как есть: для каждого узла t квадратуры строится
    своя композиция (S(t/n))^n g, O(num_t * n) применений S; цикл по t
    распараллеливается (n_jobs).
    solve_streaming() — фиксированный шаг tau = T/n_steps и одна цепочка
    u_{k+1} = S(tau) u_k, O(n_steps) применений S. Порядок тот же O(1/n),
    при равной точности в 5-10 раз дешевле, но сетка по t обязательно
    равномерная, поэтому на задачах с резким изменением подынтегральной
    функции у t=0 (высокие частоты) он проигрывает solve(). Подробное
    сравнение — в streaming.ipynb.

    Замечания о реализации
    ----------------------
    Применение S(τ) по умолчанию использует кубический сплайн
    (scipy.interpolate.CubicSpline) с продолжением нулём вне [x_min, x_max].
    Это важно: линейная интерполяция (interp='linear') портит сходимость,
    поскольку оценка скорости из item 3 теоремы 6 содержит производные g до
    4-го порядка, и грубая интерполяция теряет гладкость — см. сравнение
    точность/время двух вариантов в efficiency.ipynb.

    Продолжение нулём при interp='cubic' сделано гладким (C^1): за пределами
    [x_min, x_max] берётся линейное продолжение сплайна по касательной в
    граничной точке, умноженное на кубический smoothstep-множитель,
    спадающий с 1 до 0 на полосе ширины taper_width. Это устраняет скачок
    значения/производной, который давала бы жёсткая замена NaN -> 0
    (актуально, если решение ещё не пренебрежимо мало у края окна).
    При interp='linear' используется простое (не гладкое) продолжение нулём
    через np.interp(..., left=0.0, right=0.0) — это самый дешёвый вариант,
    им нарочно не подмешивается gладкий taper, чтобы сравнение "сплайн vs
    линейная интерполяция" не путало два независимых улучшения.

    Параметры
    ---------
    interp : str, optional
        'cubic' (по умолчанию) или 'linear' — метод интерполяции u при
        применении S(τ).
    taper_width : float, optional
        Ширина приграничной полосы гладкого спада до нуля (используется
        только при interp='cubic'). По умолчанию 0.1*(x_max - x_min).
    """

    def __init__(self, a, b, c, g, lam, x_min, x_max, dx, T=None, n=100,
                 num_t=151, eps=1e-6, c_sup=None, t_grading='quadratic',
                 taper_width=None, interp='cubic'):
        self.a = a
        self.b = b
        self.c = c
        self.g = g
        self.lam = lam
        self.x_min = x_min
        self.x_max = x_max
        self.dx = dx
        self.T = T
        self.n = int(n)
        self.num_t = num_t
        self.eps = eps
        self.t_grading = t_grading
        self.taper_width = (0.1 * (x_max - x_min)
                             if taper_width is None else float(taper_width))
        if interp not in ('cubic', 'linear'):
            raise ValueError(f"Неизвестный interp: '{interp}'. "
                             "Используйте 'cubic' или 'linear'.")
        self.interp = interp

        # Пространственная сетка
        self.x_grid = np.arange(x_min, x_max + dx, dx)
        self.n_points = len(self.x_grid)

        # Кэширование коэффициентов
        self.a_vals = np.asarray(a(self.x_grid), dtype=float)
        self.b_vals = np.asarray(b(self.x_grid), dtype=float)
        self.c_vals = np.asarray(c(self.x_grid), dtype=float)
        self.g_vals = np.asarray(g(self.x_grid), dtype=float)

        # Оценка ‖c‖_∞ на тонкой подсетке (а не на основной)
        if c_sup is None:
            fine = np.linspace(x_min, x_max, 10 * self.n_points)
            self.c_sup = float(np.max(np.abs(c(fine))))
        else:
            self.c_sup = float(c_sup)

        # Проверка условия применимости
        if self.lam <= self.c_sup:
            raise ValueError(
                f"Условие λ > ‖c‖_∞ нарушено: λ = {self.lam}, "
                f"‖c‖_∞ ≈ {self.c_sup}. Интеграл может расходиться."
            )

        # Проверка эллиптичности
        a_min = float(np.min(self.a_vals))
        if a_min <= 0:
            raise ValueError(f"Условие a(x) > 0 нарушено: min a(x) = {a_min}.")
        self.a0 = a_min

    def _tapered_eval(self, cs, x_query):
        """Значения сплайна cs в точках x_query с гладким (C^1) спадом до нуля
        вне [x_min, x_max] вместо жёсткой замены NaN -> 0.

        За краем берётся линейное продолжение по касательной в граничной
        точке (значение и производная сплайна там), умноженное на
        smoothstep-окно 3s^2-2s^3, которое равно 1 при выходе за край и 0
        на расстоянии taper_width — обе стыковки (с интерполянтом внутри
        окна и с нулём за полосой) непрерывны по значению и производной.
        """
        vals = cs(x_query)
        mask = np.isnan(vals)
        if not np.any(mask):
            return vals

        # Реальные узлы сетки, а не self.x_min/self.x_max: np.arange копит
        # ошибку округления, и x_grid[-1] может быть на ~1e-13 меньше
        # x_max, из-за чего cs(x_max) сам оказывается вне домена сплайна
        # (extrapolate=False) и возвращает NaN.
        x_lo = self.x_grid[0]
        x_hi = self.x_grid[-1]

        m = self.taper_width
        xo = x_query[mask]
        out = np.zeros_like(xo)

        left = xo < x_lo
        if np.any(left):
            d = x_lo - xo[left]
            edge_val = float(cs(x_lo))
            edge_der = float(cs(x_lo, 1))
            s = np.clip(1.0 - d / m, 0.0, 1.0)
            w = s * s * (3.0 - 2.0 * s)
            out[left] = (edge_val - edge_der * d) * w

        right = ~left
        if np.any(right):
            d = xo[right] - x_hi
            edge_val = float(cs(x_hi))
            edge_der = float(cs(x_hi, 1))
            s = np.clip(1.0 - d / m, 0.0, 1.0)
            w = s * s * (3.0 - 2.0 * s)
            out[right] = (edge_val + edge_der * d) * w

        vals[mask] = out
        return vals

    def _apply_S_on_grid(self, u_vals, tau):
        """Векторизованное применение оператора S(tau) к u, заданной в узлах сетки.

        interp='cubic' (по умолчанию): кубический сплайн с гладким
        продолжением нулём вне [x_min, x_max] (см. _tapered_eval) — критично
        для сохранения порядка сходимости из теоремы 6 (линейная интерполяция
        накапливает ошибку через композиции).
        interp='linear': np.interp с жёстким (не гладким) продолжением нулём
        — дешевле, но грубее; см. efficiency.ipynb для количественного
        сравнения.
        """
        shift = 2.0 * np.sqrt(np.maximum(self.a_vals * tau, 0.0))
        x1 = self.x_grid + shift
        x2 = self.x_grid - shift
        x3 = self.x_grid + 2.0 * self.b_vals * tau

        if self.interp == 'cubic':
            cs = CubicSpline(self.x_grid, u_vals, extrapolate=False)
            u1 = self._tapered_eval(cs, x1)
            u2 = self._tapered_eval(cs, x2)
            u3 = self._tapered_eval(cs, x3)
        else:
            u1 = np.interp(x1, self.x_grid, u_vals, left=0.0, right=0.0)
            u2 = np.interp(x2, self.x_grid, u_vals, left=0.0, right=0.0)
            u3 = np.interp(x3, self.x_grid, u_vals, left=0.0, right=0.0)

        return 0.25 * u1 + 0.25 * u2 + 0.5 * u3 + tau * self.c_vals * u_vals

    def _chernoff_on_grid(self, t):
        """Возвращает (S(t/n))^n g на сетке x_grid."""
        if t <= 0.0:
            return self.g_vals.copy()
        tau = t / self.n
        u = self.g_vals.copy()
        for _ in range(self.n):
            u = self._apply_S_on_grid(u, tau)
        return u

    def _auto_T(self):
        """Автоматический выбор T по item 5 Теоремы 6:
              T = max(0, ln(2 / ((λ − ‖c‖) eps)) / (λ − ‖c‖)).
        """
        delta = self.lam - self.c_sup
        T = np.log(2.0 / (delta * self.eps)) / delta
        return max(0.0, float(T))

    def _build_t_grid(self):
        """Сетка по t. 'quadratic' даёт сгущение у нуля, что эффективнее
        для подынтегральной функции exp(-λt) u(t,x).
        """
        num_t = self.num_t if self.num_t % 2 == 1 else self.num_t + 1
        if self.t_grading == 'uniform':
            return np.linspace(0.0, self.T, num_t)
        elif self.t_grading == 'quadratic':
            s = np.linspace(0.0, 1.0, num_t)
            return self.T * s**2
        else:
            raise ValueError(f"Неизвестный t_grading: {self.t_grading}")

    def solve(self, n_jobs=1):
        """Вычисляет черновское приближение решения.

        Параметры
        ---------
        n_jobs : int, optional
            Число процессов. По умолчанию 1 (последовательно). n_jobs=-1 —
            все доступные ядра (os.cpu_count()). Распараллеливается цикл по
            t (формула (3)): (S(t/n))^n g при разных t считается полностью
            независимо, поэтому декомпозиция точная, без halo-обмена и без
            изменения результата (см. тест на совпадение с n_jobs=1).

            Это НЕ распараллеливание по x из Замечания 9 статьи (там речь о
            вычислении f(x) в одной точке независимо от остальных x). Здесь
            все x_grid всё так же считаются одним векторизованным
            numpy-проходом внутри _apply_S_on_grid — распараллеливание по x
            потребовало бы halo-обмена между процессами, так как сдвиги в
            S(τ) могут "дотягиваться" за пределы локального куска сетки.
            Параллелизм по t — независимая ось, уже присутствующая в
            формуле (3), без этой сложности.

        Returns
        -------
        f_chern : ndarray of shape (n_points,)
            Значения приближённого решения на пространственной сетке.
        """
        if self.T is None:
            self.T = self._auto_T()

        t_arr = self._build_t_grid()
        num_t = len(t_arr)

        # Подынтегральная функция: e^{-λt} (S(t/n))^n g
        if n_jobs == 1:
            F = np.empty((num_t, self.n_points))
            for i, t in enumerate(t_arr):
                u_t = self._chernoff_on_grid(t)
                F[i, :] = np.exp(-self.lam * t) * u_t
        else:
            workers = os.cpu_count() if n_jobs == -1 else int(n_jobs)
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=workers, initializer=_init_pool_worker,
                          initargs=(self,)) as pool:
                F = np.array(pool.map(_pool_eval_t, t_arr))

        # Интегрирование по t (Симпсон корректно работает и на неравномерной сетке).
        f_chern = simpson(F, x=t_arr, axis=0)
        return f_chern

    def solve_streaming(self, n_steps=None):
        """Потоковый вариант формулы (3) с фиксированным шагом tau = T/n_steps.

        В solve() для каждого узла t квадратуры композиция (S(t/n))^n g
        строится заново — это O(num_t * n) применений S. Здесь одна цепочка
        u_{k+1} = S(tau) u_k сразу даёт значения подынтегральной функции во
        всех узлах t_k = k*tau, то есть O(n_steps) применений S.

        Цена — другая математика: u_k = (S(t_k/k))^k g, то есть число
        сомножителей n = k растёт вместе с t. Это сходящийся вариант (по
        теореме Чернова при t фиксированном и k -> oo), но он НЕ совпадает
        строго с п.4 Теоремы 6, где n фиксировано для всех t, и оценка
        2*C_g/(n*(lam-||c||)^3) к нему напрямую не относится: у малых t
        композиция короткая (при k=1 это просто S(tau) g), и именно там
        точность хуже.

        Сетка по t здесь обязательно равномерная (шаг tau фиксирован), так
        что t_grading игнорируется; узлов при этом много (n_steps+1), и
        отсутствие сгущения у нуля компенсируется их числом.

        Параметры
        ---------
        n_steps : int, optional
            Число шагов по t. По умолчанию self.n. Нечётное значение
            увеличивается на 1: для Симпсона нужно чётное число интервалов.

        Returns
        -------
        f_chern : ndarray of shape (n_points,)
            Значения приближённого решения на пространственной сетке.
        """
        if self.T is None:
            self.T = self._auto_T()

        n_steps = int(self.n if n_steps is None else n_steps)
        if n_steps < 2:
            raise ValueError(f"n_steps должно быть >= 2, получено {n_steps}.")
        if n_steps % 2:
            n_steps += 1
        tau = self.T / n_steps

        F = np.empty((n_steps + 1, self.n_points))
        u = self.g_vals.copy()
        F[0, :] = u                      # t=0: e^{-lam*0} * g
        for k in range(1, n_steps + 1):
            u = self._apply_S_on_grid(u, tau)
            F[k, :] = np.exp(-self.lam * k * tau) * u

        t_arr = np.arange(n_steps + 1) * tau
        return simpson(F, x=t_arr, axis=0)
