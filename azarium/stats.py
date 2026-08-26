"""Distribuciones y tests estadisticos implementados sobre stdlib + numpy.

No se usa scipy a proposito: el proyecto tiene que correr sin instalar nada.
Las implementaciones siguen Numerical Recipes (gamma incompleta) y son exactas
hasta ~1e-12, muy por debajo de cualquier umbral de decision que usemos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_ITMAX = 500
_EPS = 3.0e-16
_FPMIN = 1e-300


def _gser(a: float, x: float) -> float:
    """Serie para la gamma incompleta regularizada inferior P(a, x)."""
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(_ITMAX):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * _EPS:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gcf(a: float, x: float) -> float:
    """Fraccion continua para la gamma incompleta regularizada superior Q(a, x)."""
    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, _ITMAX):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def gamma_q(a: float, x: float) -> float:
    """Gamma incompleta regularizada superior Q(a, x) = 1 - P(a, x)."""
    if x < 0 or a <= 0:
        raise ValueError("gamma_q requiere a > 0 y x >= 0")
    if x == 0:
        return 1.0
    if x < a + 1.0:
        return 1.0 - _gser(a, x)
    return _gcf(a, x)


def chi2_sf(x: float, df: int) -> float:
    """P(X > x) para una chi-cuadrado con df grados de libertad."""
    if df <= 0:
        raise ValueError("df debe ser >= 1")
    if x <= 0:
        return 1.0
    return gamma_q(df / 2.0, x / 2.0)


def norm_sf(z: float) -> float:
    """P(Z > z) para la normal estandar."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def norm_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def norm_ppf(p: float) -> float:
    """Inversa de la normal estandar (Acklam, error < 1.15e-9)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p debe estar en (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def binom_sf_ge(k: int, n: int, p: float) -> float:
    """P(X >= k) exacta para una binomial(n, p). Suma directa en log-espacio."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return _binom_sum(range(k, n + 1), n, p)


def _binom_sum(indices, n: int, p: float) -> float:
    lp, lq = math.log(p), math.log1p(-p)
    total = 0.0
    for i in indices:
        logterm = (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                   + i * lp + (n - i) * lq)
        total += math.exp(logterm)
    return min(1.0, total)


@dataclass
class Resultado:
    """Salida uniforme de todos los tests: estadistico, p-valor y lectura."""
    nombre: str
    estadistico: float
    p_valor: float
    df: int | None = None
    detalle: str = ""

    @property
    def significativo(self) -> bool:
        return self.p_valor < 0.05

    def lectura(self, alfa: float = 0.05) -> str:
        if self.p_valor < alfa:
            return "Se rechaza la uniformidad (p < %.3g)" % alfa
        return "Compatible con azar puro"


def benjamini_hochberg(p_valores: np.ndarray, fdr: float = 0.05) -> np.ndarray:
    """Devuelve mascara booleana de hipotesis rechazadas controlando FDR.

    Sin esto, testear 90 numeros a alfa=0.05 produce ~4-5 'hallazgos' que son
    puro ruido. Es la correccion que casi nunca aplican los sistemas de loteria.
    """
    p = np.asarray(p_valores, dtype=float)
    n = p.size
    orden = np.argsort(p)
    umbrales = fdr * (np.arange(1, n + 1) / n)
    pasa = p[orden] <= umbrales
    rechazadas = np.zeros(n, dtype=bool)
    if pasa.any():
        corte = np.max(np.nonzero(pasa)[0])
        rechazadas[orden[:corte + 1]] = True
    return rechazadas


def ic_wilson(exitos: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Intervalo de confianza de Wilson para una proporcion (mejor que Wald)."""
    if n == 0:
        return (0.0, 1.0)
    z = norm_ppf(1 - (1 - conf) / 2)
    ph = exitos / n
    denom = 1 + z * z / n
    centro = (ph + z * z / (2 * n)) / denom
    margen = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centro - margen), min(1.0, centro + margen))
