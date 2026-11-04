"""Five pre-solve interactions of the given coefficient grid on [-1, 1].

These are finite-grid descriptive features, not global certificates or error
bounds. They neither evaluate the exact solution nor execute a numerical ODE
method. The historical 253-feature implementation is left unchanged.

Write q = lambda - c_R = -c > 0, f = g_R = -g and let RMS denote the
unweighted root mean square on the input grid. Values at zero use linear
interpolation. Derivatives use numpy.gradient(..., edge_order=2) on the given
grid, applied twice for the second derivative. The five quantities are:

* |b(0)| / sqrt(a(0) q(0)): local drift versus diffusion/reaction scale;
* a(0) RMS(f'') / (q(0) RMS(f)): forcing curvature on that scale;
* RMS(a') / sqrt(a(0) q(0)): diffusion-coefficient variation;
* |integral f| / integral |f|: forcing cancellation, using trapezoids;
* RMS(a u'' + b u') / RMS(f), u=f/q: differential defect of the
  pointwise reaction-only proxy. No equation is solved to compute this proxy.

The zero forcing gives zero for the three forcing-derived features. The
interactions re-express information already available before solver execution;
they are not an additional independent source of evidence.
"""
from __future__ import annotations

import math

import numpy as np


FEATURE_NAMES = (
    "research_local_peclet",
    "research_forcing_curvature_ratio",
    "research_diffusion_gradient_ratio",
    "research_forcing_cancellation",
    "research_reaction_proxy_defect",
)


def _rms(values):
    """A scaled RMS avoids squaring large finite inputs unnecessarily."""
    scale = float(np.max(np.abs(values)))
    return 0.0 if scale == 0 else scale * float(np.sqrt(np.mean((values / scale) ** 2)))


def additional_coefficient_features(values, x, meta=None):
    """Return exactly five finite features from X[a,b,c,g] on [-1, 1].

    ``meta`` is accepted for compatibility with coefficient_features, but no
    metadata fields are read: q=-c already follows from the input channels.
    """
    coefficients = np.asarray(values, dtype=float)
    x = np.asarray(x, dtype=float)
    if (x.ndim != 1 or len(x) < 5 or coefficients.shape != (4, len(x))
            or not np.isfinite(x).all() or not np.isfinite(coefficients).all()
            or not np.all(np.diff(x) > 0)):
        raise ValueError("four finite coefficient channels on an increasing grid required")
    if not np.allclose([x[0], x[-1]], [-1.0, 1.0], atol=1e-12, rtol=0):
        raise ValueError("research features are defined on the input interval [-1, 1]")
    a, b, c, g = coefficients
    q, forcing = -c, -g
    if np.any(a <= 0) or np.any(q <= 0):
        raise ValueError("positive sampled diffusion a and reaction q=-c required")
    a0, b0, q0 = (float(np.interp(0.0, x, v)) for v in (a, b, q))
    scale = math.sqrt(a0) * math.sqrt(q0)
    derivative_a = np.gradient(a, x, edge_order=2)
    forcing_rms = _rms(forcing)
    curvature, cancellation, defect = 0.0, 0.0, 0.0
    if forcing_rms > 0:
        # Normalizing first keeps the ratios invariant to forcing amplitude.
        normalized = forcing / forcing_rms
        derivative_f = np.gradient(normalized, x, edge_order=2)
        second_f = np.gradient(derivative_f, x, edge_order=2)
        curvature = a0 * _rms(second_f) / q0
        widths = np.diff(x)
        absolute = np.abs(normalized)
        denominator = float(np.sum((absolute[:-1] + absolute[1:]) * widths) / 2)
        integral = float(np.sum((normalized[:-1] + normalized[1:]) * widths) / 2)
        cancellation = abs(integral) / denominator
        proxy = normalized / q
        derivative_proxy = np.gradient(proxy, x, edge_order=2)
        second_proxy = np.gradient(derivative_proxy, x, edge_order=2)
        defect = _rms(a * second_proxy + b * derivative_proxy)
    result = dict(zip(FEATURE_NAMES, (
        abs(b0) / scale, curvature, _rms(derivative_a) / scale,
        cancellation, defect,
    )))
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("nonfinite research feature")
    return result
