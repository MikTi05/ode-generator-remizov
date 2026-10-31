"""Equation data shared by the generator and safe dataset loader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CauchyProblem:
    equation_id: str
    a: Callable[[float], float]
    b: Callable[[float], float]
    c: Callable[[float], float]
    g: Callable[[float], float]
    x0: float
    x1: float
    y0: float
    v0: float

