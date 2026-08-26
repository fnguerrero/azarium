"""Bateria de tests de aleatoriedad sobre historicos de sorteos.

Cada funcion devuelve un `Resultado` con estadistico, p-valor y lectura. La
hipotesis nula siempre es la misma: los sorteos son independientes y uniformes.
Rechazarla significaria que hay estructura explotable; no rechazarla significa
que no la hay, o que es mas chica que lo que la muestra permite ver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..stats import (Resultado, benjamini_hochberg, binom_sf_ge, chi2_sf,
                     ic_wilson, norm_sf)
from .modelo import Historico


def chi2_uniformidad(hist: Historico) -> Resultado:
    """Bondad de ajuste global: todos los numeros equiprobables."""
    obs = hist.frecuencias().astype(float)
    n = obs.sum()
    k = hist.juego.cardinalidad
    esp = n / k
    if esp < 5:
        detalle = f"ADVERTENCIA: frecuencia esperada {esp:.1f} < 5, chi2 poco fiable"
    else:
        detalle = f"n={int(n)} extracciones, esperado {esp:.1f} por numero"
    stat = float(((obs - esp) ** 2 / esp).sum())
    df = k - 1
    return Resultado("Chi-cuadrado de uniformidad", stat, chi2_sf(stat, df), df, detalle)


@dataclass
class FilaNumero:
    numero: int
    observado: int
    esperado: float
    desvio_z: float
    p_valor: float
    ic_bajo: float
    ic_alto: float
    significativo_crudo: bool
    significativo_fdr: bool


def por_numero(hist: Historico, fdr: float = 0.05) -> list[FilaNumero]:
    """Test individual por numero, con y sin correccion por multiplicidad.

    Esta es la tabla que todo sistema de loteria muestra ("estos numeros salen
    mas") y donde casi todos se equivocan: al correr 90-100 tests simultaneos,
    unos 5 dan p menor a 0.05 por puro azar. La columna con FDR es la honesta.
    """
    obs = hist.frecuencias()
    n = int(obs.sum())
    k = hist.juego.cardinalidad
    p = 1.0 / k
    esp = n * p
    sigma = np.sqrt(n * p * (1 - p))

    p_vals = np.empty(k)
    for i, o in enumerate(obs):
        # Test bilateral exacto: doblamos la cola mas extrema.
        if o >= esp:
            cola = binom_sf_ge(int(o), n, p)
        else:
            cola = 1.0 - binom_sf_ge(int(o) + 1, n, p)
        p_vals[i] = min(1.0, 2 * cola)

    rechazadas = benjamini_hochberg(p_vals, fdr=fdr)
    filas = []
    for i in range(k):
        bajo, alto = ic_wilson(int(obs[i]), n)
        filas.append(FilaNumero(
            numero=hist.juego.minimo + i,
            observado=int(obs[i]),
            esperado=esp,
            desvio_z=float((obs[i] - esp) / sigma),
            p_valor=float(p_vals[i]),
            ic_bajo=bajo, ic_alto=alto,
            significativo_crudo=bool(p_vals[i] < 0.05),
            significativo_fdr=bool(rechazadas[i]),
        ))
    return filas


def persistencia_calientes(hist: Historico, corte: float = 0.5) -> Resultado:
    """El test decisivo: los numeros calientes de la primera mitad, siguen calientes?

    Si existiera un sesgo real y estable, el ranking de frecuencias de la
    primera mitad correlacionaria con el de la segunda. Si los sorteos son
    independientes, la correlacion es ~0 y jugar los calientes no sirve.

    Devuelve la correlacion de Spearman entre ambos rankings.
    """
    n = hist.n_sorteos
    if n < 20:
        raise ValueError("hacen falta al menos 20 sorteos")
    mitad = int(n * corte)
    k = hist.juego.cardinalidad
    mn = hist.juego.minimo
    f1 = np.bincount(hist.resultados[:mitad].reshape(-1) - mn, minlength=k)
    f2 = np.bincount(hist.resultados[mitad:].reshape(-1) - mn, minlength=k)

    r1, r2 = _rankear(f1), _rankear(f2)
    rho = float(np.corrcoef(r1, r2)[0, 1])
    # Bajo H0, rho * sqrt(k - 1) se distribuye aprox. N(0, 1)
    z = rho * np.sqrt(k - 1)
    p = 2 * norm_sf(abs(z))
    detalle = (f"Spearman rho={rho:+.4f} entre las frecuencias de los primeros "
               f"{mitad} y los ultimos {n - mitad} sorteos")
    return Resultado("Persistencia de numeros calientes", float(z), float(p), None, detalle)


def _rankear(v: np.ndarray) -> np.ndarray:
    """Rangos, promediando empates."""
    orden = np.argsort(v, kind="mergesort")
    rangos = np.empty(len(v), dtype=float)
    rangos[orden] = np.arange(1, len(v) + 1)
    valores, cuenta = np.unique(v, return_counts=True)
    for val, c in zip(valores, cuenta):
        if c > 1:
            mask = v == val
            rangos[mask] = rangos[mask].mean()
    return rangos


def rachas_paridad(hist: Historico) -> Resultado:
    """Wald-Wolfowitz sobre la secuencia par/impar de todos los numeros extraidos.

    Detecta dependencia serial: si un numero par tendiera a seguir a otro par,
    habria menos rachas de las esperadas.
    """
    serie = hist.plano()
    bits = (serie % 2 == 0).astype(int)
    n1 = int(bits.sum())
    n0 = int(len(bits) - n1)
    if n0 == 0 or n1 == 0:
        raise ValueError("la serie no tiene ambas paridades")
    rachas = int(1 + (bits[1:] != bits[:-1]).sum())
    n = n0 + n1
    mu = 2 * n0 * n1 / n + 1
    var = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n * n * (n - 1))
    z = (rachas - mu) / np.sqrt(var)
    p = 2 * norm_sf(abs(z))
    detalle = f"{rachas} rachas observadas, {mu:.1f} esperadas ({n0} impares / {n1} pares)"
    return Resultado("Test de rachas (paridad)", float(z), float(p), None, detalle)


def autocorrelacion(hist: Historico, max_lag: int = 20) -> tuple[np.ndarray, float]:
    """ACF de la serie de numeros extraidos y su banda de confianza al 95 por ciento.

    Cualquier barra dentro de la banda es ruido. Se espera ~1 de cada 20 barras
    afuera aunque la serie sea perfectamente aleatoria.
    """
    x = hist.plano().astype(float)
    x = x - x.mean()
    n = len(x)
    denom = float((x * x).sum())
    acf = np.array([float((x[:n - h] * x[h:]).sum()) / denom
                    for h in range(1, max_lag + 1)])
    banda = 1.959964 / np.sqrt(n)
    return acf, float(banda)


def test_gaps(hist: Historico, numero: int, max_bin: int | None = None) -> Resultado:
    """Test de huecos: la distancia entre apariciones sigue una geometrica?

    Es la refutacion formal de "hace 40 sorteos que no sale, ya tiene que salir".
    Si los gaps son geometricos, el proceso no tiene memoria.
    """
    juego = hist.juego
    if not (juego.minimo <= numero <= juego.maximo):
        raise ValueError("numero fuera del rango del juego")
    presente = (hist.resultados == numero).any(axis=1)
    idx = np.nonzero(presente)[0]
    if len(idx) < 20:
        raise ValueError(f"el numero {numero} aparece en solo {len(idx)} sorteos; "
                         "hacen falta al menos 20")
    gaps = np.diff(idx)
    p_apar = float(presente.mean())
    if max_bin is None:
        max_bin = max(3, int(np.ceil(np.log(0.02) / np.log(1 - p_apar))))

    obs = np.zeros(max_bin, dtype=float)
    for g in gaps:
        obs[min(int(g), max_bin) - 1] += 1
    total = obs.sum()
    probs = np.array([p_apar * (1 - p_apar) ** i for i in range(max_bin)])
    probs[-1] = 1.0 - probs[:-1].sum()  # el ultimo bin absorbe la cola
    esp = total * probs

    validos = esp >= 5
    if int(validos.sum()) < 2:
        raise ValueError("muestra insuficiente para el test de huecos")
    stat = float((((obs - esp) ** 2 / esp)[validos]).sum())
    df = int(validos.sum()) - 1
    detalle = (f"numero {numero}: {len(idx)} apariciones, gap medio {gaps.mean():.1f} "
               f"(esperado {1 / p_apar:.1f}), gap maximo {gaps.max()}")
    return Resultado("Test de huecos", stat, chi2_sf(stat, df), df, detalle)


def test_serial(hist: Historico, grupos: int = 10) -> Resultado:
    """Independencia entre extracciones consecutivas, agrupando en bloques.

    Testear la tabla completa 100x100 dejaria celdas vacias; agrupar en decenas
    mantiene la frecuencia esperada por celda por encima de 5.
    """
    serie = hist.plano()
    k = hist.juego.cardinalidad
    ancho = int(np.ceil(k / grupos))
    g = (serie - hist.juego.minimo) // ancho
    g = np.clip(g, 0, grupos - 1)
    tabla = np.zeros((grupos, grupos), dtype=float)
    np.add.at(tabla, (g[:-1], g[1:]), 1)
    n = tabla.sum()
    esp = np.outer(tabla.sum(axis=1), tabla.sum(axis=0)) / n
    if (esp < 5).any():
        return Resultado("Test serial", float("nan"), float("nan"), None,
                         "muestra insuficiente: hay celdas con esperado menor a 5")
    stat = float(((tabla - esp) ** 2 / esp).sum())
    df = (grupos - 1) ** 2
    detalle = f"tabla {grupos}x{grupos} sobre {int(n)} pares consecutivos"
    return Resultado("Test serial (independencia entre sorteos)", stat,
                     chi2_sf(stat, df), df, detalle)


def bateria(hist: Historico) -> list[Resultado]:
    """Corre todos los tests globales y devuelve los resultados en orden."""
    salida = [chi2_uniformidad(hist), persistencia_calientes(hist),
              rachas_paridad(hist), test_serial(hist)]
    # Huecos sobre el numero mas frecuente, que es el candidato mas favorable
    frec = hist.frecuencias()
    candidato = int(hist.juego.minimo + int(np.argmax(frec)))
    try:
        salida.append(test_gaps(hist, candidato))
    except ValueError as exc:
        salida.append(Resultado("Test de huecos", float("nan"), float("nan"), None,
                                f"no aplicable: {exc}"))
    return salida
