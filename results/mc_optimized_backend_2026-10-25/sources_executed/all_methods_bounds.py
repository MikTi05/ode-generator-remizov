"""Analytical whole-line forcing certificates for unchanged all_methods_v1.

No coordinate samples, formula eval, or numerical maximization. See
docs/ALL_METHODS_FORCING_BOUNDS.md for all primitive/bump inequalities.
Positive arithmetic is rounded upward using exact rational comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from numbers import Real
from typing import Any, Mapping

from all_methods_dataset_loader import (_decode_bundle, _validate_construction,
                                        _validate_metadata_shape)
from mc_solver import MCForcingBound
from remizov_solver import RemizovProblem, _problem_conditions_valid
from src.all_methods_expressions import SafeExpressionNode, validate_expression

CERTIFICATE_VERSION = "all_methods_forcing_bounds_v1"


class BoundCertificationError(ValueError):
    """The construction or its finite float64 upper bound cannot be certified."""


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise BoundCertificationError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise BoundCertificationError(f"invalid {name}")
    return value


def _upper(value: Fraction) -> float:
    """Smallest nearby float >= an exact nonnegative rational; preserve zero."""
    try:
        result = float(value)
    except OverflowError as exc:
        raise BoundCertificationError("global bound exceeds float64 range") from exc
    if not math.isfinite(result):
        raise BoundCertificationError("global bound exceeds float64 range")
    if Fraction(result) < value:
        result = math.nextafter(result, math.inf)
    if not math.isfinite(result):
        raise BoundCertificationError("upward global bound exceeds float64 range")
    return result


def _add(*values: float) -> float:
    return _upper(sum((Fraction(v) for v in values), Fraction(0)))


def _mul(*values: float) -> float:
    product = Fraction(1)
    for value in values:
        product *= Fraction(value)
    return _upper(product)


def _divide(value: float, denominator: float, power: int = 1) -> float:
    # Never round a denominator upward before division.
    return _upper(Fraction(value) / Fraction(denominator)**power)


def _sqrt_upper(value: float) -> float:
    result = math.sqrt(value)
    if Fraction(result)**2 < Fraction(value):
        result = math.nextafter(result, math.inf)
    return result


@dataclass(frozen=True)
class DerivativeBounds:
    """Nonnegative global upper bounds on |u|, |u'| and |u''|."""
    value: float
    d1: float
    d2: float

    def __post_init__(self):
        for name in ("value", "d1", "d2"):
            _number(getattr(self, name), name)


def _propagate(node: SafeExpressionNode, order: int) -> tuple[float, ...]:
    op, p = node.op, node.params
    children = [_propagate(child, order) for child in node.children]
    if op == "constant":
        value = abs(p['value'])
    elif op in {"sin_affine", "cos_affine", "tanh_affine", "gaussian", "lorentz", "saturating", "cos", "exp_neg_square"}:
        value = 1.
    elif op in {"sin", "tanh"}:
        value = min(1., children[0][0])
    elif op == "rational_saturate":
        value = min(.5, children[0][0])
    elif op == "lincomb":
        value = _add(_mul(abs(p['alpha']), children[0][0]), _mul(abs(p['beta']), children[1][0]))
    elif op == "mul":
        value = _mul(children[0][0], children[1][0])
    else:
        raise BoundCertificationError(f"unsupported AST operation: {op}")
    if order == 0:
        return (value,)
    if op == "constant":
        d1, d2 = 0., 0.
    elif op in {"sin_affine", "cos_affine", "tanh_affine"}:
        d1 = abs(p['omega'])
        d2 = _mul(d1, d1)  # also valid for tanh: sup|tanh''| < 1
    elif op in {"gaussian", "lorentz"}:
        d1, d2 = _sqrt_upper(p['alpha']), _mul(2., p['alpha'])
    elif op == "saturating":
        d1, d2 = 1., 1.
    elif op == "lincomb":
        d1, d2 = (_add(_mul(abs(p['alpha']), children[0][k]),
                       _mul(abs(p['beta']), children[1][k])) for k in (1, 2))
    elif op == "mul":
        u, v = children
        d1 = _add(_mul(u[1], v[0]), _mul(u[0], v[1]))
        d2 = _add(_mul(u[2], v[0]), _mul(2., u[1], v[1]), _mul(u[0], v[2]))
    else:
        u = children[0]
        if op == 'sin':
            first, second = 1., min(1., u[0])
        elif op == 'cos':
            first, second = min(1., u[0]), 1.
        elif op == 'tanh':
            first, second = 1., 1. if u[0] >= .5 else _mul(2., u[0])
        elif op == 'exp_neg_square':
            first, second = (1. if u[0] >= .5 else _mul(2., u[0])), 2.
        else:  # rational_saturate
            first, second = 1., 1.5 if u[0] >= .25 else _mul(6., u[0])
        d1 = _mul(first, u[1])
        d2 = _add(_mul(second, u[1], u[1]), _mul(first, u[2]))
    return value, d1, d2


def expression_bounds(node: SafeExpressionNode) -> DerivativeBounds:
    """Recursively bound every operation in the v1 whitelist on all real x."""
    validate_expression(node)
    return DerivativeBounds(*_propagate(node, 2))


def bump_bounds(radius: float = 1.) -> DerivativeBounds:
    """For phi((x-mu)/R): (1, 3/R, (1809/64)/R^2), any real mu."""
    radius = _number(radius, 'radius', positive=True)
    return DerivativeBounds(1., _divide(3., radius), _divide(1809/64, radius, 2))


def solution_bounds(shape: DerivativeBounds, scale: float, radius: float) -> DerivativeBounds:
    """Product rule for scale*bump*shape, including the two cross derivatives."""
    if isinstance(scale, bool) or not isinstance(scale, Real) or not math.isfinite(scale):
        raise BoundCertificationError('scale must be finite real')
    s, b = abs(float(scale)), bump_bounds(radius)
    return DerivativeBounds(_mul(s, shape.value),
        _mul(s, _add(_mul(b.d1, shape.value), shape.d1)),
        _mul(s, _add(_mul(b.d2, shape.value), _mul(2., b.d1, shape.d1), shape.d2)))


def _forcing_terms(y: DerivativeBounds, a_upper: float, b_abs: float,
                   c_abs: float, lambda_value: float) -> dict[str, float]:
    for name, value in [('a_upper', a_upper), ('b_abs', b_abs), ('c_abs', c_abs)]:
        _number(value, name)
    if isinstance(lambda_value, bool) or not isinstance(lambda_value, Real) or not math.isfinite(lambda_value):
        raise BoundCertificationError('lambda must be finite real')
    return dict(diffusion=_mul(a_upper, y.d2), drift=_mul(b_abs, y.d1),
                potential=_mul(_add(abs(lambda_value), c_abs), y.value))


def forcing_bound_from_solution_bounds(y: DerivativeBounds, *, a_upper: float,
                                      b_abs: float, c_abs: float,
                                      lambda_value: float) -> MCForcingBound:
    """G for g_R=-[a*y''+b*y'+(c_R-lambda)*y]; no compactness needed here."""
    terms = _forcing_terms(y, a_upper, b_abs, c_abs, lambda_value)
    return MCForcingBound(_add(*terms.values()), 'analytical derivative envelopes: A*Y2+B*Y1+(abs(lambda)+C)*Y0')


@dataclass(frozen=True)
class AllMethodsForcingCertificate:
    problem_id: str
    version: str
    serialized_ast_sha256: str
    forcing_bound: MCForcingBound
    shape_bounds: DerivativeBounds
    bump_bounds: DerivativeBounds
    solution_bounds: DerivativeBounds
    coefficient_bounds: dict[str, float]
    term_bounds: dict[str, float]
    support: tuple[float, float]
    rounding: str = "exact_rational_upward_binary64_v1"


def certify_all_methods_forcing(problem: RemizovProblem,
                                metadata: Mapping[str, Any]) -> AllMethodsForcingCertificate:
    """Certify a *matching unmodified* problem/metadata from the safe v1 loader.

    Reuses the loader's AST decoder and construction checks. No sampling and
    no reconstruction/evaluation of formula strings. The theorem concerns the
    AST-defined functions; replacing callables invalidates that association.
    """
    pid = problem.problem_id
    if metadata.get('equation_id') != pid:
        raise BoundCertificationError('problem/metadata equation_id mismatch')
    _validate_metadata_shape(metadata, pid)
    bundle = _decode_bundle(metadata, pid)
    _validate_construction(metadata, bundle, pid)
    if not _problem_conditions_valid(problem):
        raise BoundCertificationError('invalid global Remizov coefficient certificates')
    for field, key in [('a_lower_bound','a_lower_cert'), ('a_upper_bound','a_upper_cert'),
                       ('b_abs_upper_bound','b_abs_cert'), ('c_abs_upper_bound','remizov_c_abs_cert'),
                       ('lambda_value','remizov_lambda'), ('x_target_min','x0'), ('x_target_max','x1')]:
        if getattr(problem, field) != metadata[key]:
            raise BoundCertificationError(f'problem/metadata mismatch: {field}')
    p = bundle.parameters
    if (p.lambda_value != problem.lambda_value or p.b_scale != problem.b_abs_upper_bound
            or p.c_scale != problem.c_abs_upper_bound):
        raise BoundCertificationError('AST/problem coefficient certificate mismatch')
    shape = expression_bounds(bundle.y_shape)
    y = solution_bounds(shape, p.solution_scale, p.bump_radius)
    # |tanh(u)| <= min(1, sup|u|); coefficient derivatives are not needed.
    a = _add(p.a_mid, _mul(p.a_half, min(1., _propagate(bundle.a_raw, 0)[0])))
    b = _mul(p.b_scale, min(1., _propagate(bundle.b_raw, 0)[0]))
    c = _mul(p.c_scale, min(1., _propagate(bundle.c_raw, 0)[0]))
    coefficients = dict(a_upper=a, b_abs=b, c_R_abs=c, q_abs=_add(abs(p.lambda_value), c))
    terms = _forcing_terms(y, a, b, c, p.lambda_value)
    payload = json.dumps(metadata['serialized_ast'], sort_keys=True, separators=(',', ':'), allow_nan=False)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    bound = MCForcingBound(_add(*terms.values()), f'{CERTIFICATE_VERSION}:{pid}:sha256={digest}')
    return AllMethodsForcingCertificate(pid, CERTIFICATE_VERSION, digest, bound, shape,
        bump_bounds(p.bump_radius), y, coefficients, terms,
        (metadata['support_left'], metadata['support_right']))
