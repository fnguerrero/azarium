"""Backtest de estrategias de seleccion de numeros.

Regla estricta: para elegir los numeros del sorteo t solo se usan los sorteos
anteriores a t. Sin esa disciplina cualquier estrategia parece ganadora, porque
se la esta evaluando con los datos que uso para elegir.

Lo que se compara no es "cual gana" (ninguna gana: todas tienen la misma
esperanza negativa) sino si alguna se despega de las demas mas alla del ruido.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..stats import norm_sf
from .modelo import Historico

# Pagos tipicos de la quiniela argentina. Son pagos TOTALES por unidad
# apostada, no ganancia neta: quien acierta a la cabeza con $1 cobra $70 en
# total, no $71. Es la convencion de la banca y hay que respetarla, porque
# confundirla con "N a 1" subestima la ventaja de la casa en un punto entero.
PAGO_CABEZA = 70.0     # acertar la primera bolilla
PAGO_A_LOS_20 = 3.5    # que el numero aparezca en cualquiera de las 20


@dataclass
class ResultadoBacktest:
    estrategia: str
    apuestas: int
    aciertos: int
    unidades_apostadas: float
    unidades_cobradas: float
    modalidad: str

    @property
    def neto(self) -> float:
        return self.unidades_cobradas - self.unidades_apostadas

    @property
    def retorno(self) -> float:
        """Resultado neto por unidad apostada."""
        if self.unidades_apostadas == 0:
            return 0.0
        return self.neto / self.unidades_apostadas

    @property
    def tasa_acierto(self) -> float:
        return self.aciertos / self.apuestas if self.apuestas else 0.0


Selector = Callable[[np.ndarray, "Historico", int], list[int]]


def _historial_frecuencias(previos: np.ndarray, hist: Historico) -> np.ndarray:
    return np.bincount(previos.reshape(-1) - hist.juego.minimo,
                       minlength=hist.juego.cardinalidad)


def sel_calientes(previos, hist, k):
    """Los k numeros que mas salieron hasta ahora."""
    f = _historial_frecuencias(previos, hist)
    return [int(hist.juego.minimo + i) for i in np.argsort(f)[::-1][:k]]


def sel_frios(previos, hist, k):
    """Los k numeros que menos salieron ('ya les toca')."""
    f = _historial_frecuencias(previos, hist)
    return [int(hist.juego.minimo + i) for i in np.argsort(f)[:k]]


def sel_atrasados(previos, hist, k):
    """Los k numeros con mayor cantidad de sorteos sin aparecer."""
    juego = hist.juego
    ultima = np.full(juego.cardinalidad, -1, dtype=int)
    for t in range(len(previos)):
        for v in previos[t]:
            ultima[v - juego.minimo] = t
    atraso = len(previos) - ultima
    return [int(juego.minimo + i) for i in np.argsort(atraso)[::-1][:k]]


def sel_ultimo(previos, hist, k):
    """Repetir los numeros del sorteo anterior."""
    ultimos = list(dict.fromkeys(int(v) for v in previos[-1]))
    return ultimos[:k]


def _fabricar_aleatorio(semilla: int) -> Selector:
    rng = np.random.default_rng(semilla)

    def sel(previos, hist, k):
        return [int(v) for v in rng.choice(hist.juego.valores, size=k, replace=False)]

    return sel


def _fabricar_fijos(numeros: list[int]) -> Selector:
    def sel(previos, hist, k):
        return numeros[:k]

    return sel


def correr(hist: Historico, selector: Selector, nombre: str, k: int = 1,
           calentamiento: int = 365, modalidad: str = "cabeza") -> ResultadoBacktest:
    """Evalua una estrategia sobre todo el historico.

    `modalidad` define contra que se compara el numero elegido:
      - "cabeza": solo la primera bolilla del sorteo (paga 70 a 1)
      - "a20": cualquiera de las bolillas del sorteo (paga 3.5 a 1)
    """
    if modalidad not in ("cabeza", "a20"):
        raise ValueError("modalidad debe ser 'cabeza' o 'a20'")
    pago = PAGO_CABEZA if modalidad == "cabeza" else PAGO_A_LOS_20
    n = hist.n_sorteos
    if n <= calentamiento:
        raise ValueError(f"el historico tiene {n} sorteos, insuficiente para un "
                         f"calentamiento de {calentamiento}")

    apuestas = aciertos = 0
    apostado = cobrado = 0.0
    for t in range(calentamiento, n):
        previos = hist.resultados[:t]
        elegidos = selector(previos, hist, k)
        sorteo = hist.resultados[t]
        objetivo = {int(sorteo[0])} if modalidad == "cabeza" else set(int(v) for v in sorteo)
        for numero in elegidos:
            apuestas += 1
            apostado += 1.0
            if numero in objetivo:
                aciertos += 1
                cobrado += pago
    return ResultadoBacktest(nombre, apuestas, aciertos, apostado, cobrado, modalidad)


def comparar(hist: Historico, k: int = 1, calentamiento: int = 365,
             modalidad: str = "cabeza", semilla: int = 0) -> list[ResultadoBacktest]:
    """Corre todas las estrategias sobre el mismo historico."""
    juego = hist.juego
    fijos = [juego.minimo + i for i in range(k)]
    estrategias: list[tuple[str, Selector]] = [
        ("Calientes (los que mas salieron)", sel_calientes),
        ("Frios (los que menos salieron)", sel_frios),
        ("Atrasados (mas tiempo sin salir)", sel_atrasados),
        ("Repetir el sorteo anterior", sel_ultimo),
        ("Aleatorio", _fabricar_aleatorio(semilla)),
        ("Numeros fijos", _fabricar_fijos(fijos)),
    ]
    return [correr(hist, sel, nombre, k=k, calentamiento=calentamiento,
                   modalidad=modalidad)
            for nombre, sel in estrategias]


def esperanza_teorica(hist: Historico, modalidad: str = "cabeza") -> float:
    """Retorno esperado por unidad, si el sorteo es uniforme e independiente."""
    juego = hist.juego
    if modalidad == "cabeza":
        p = 1.0 / juego.cardinalidad
        return p * PAGO_CABEZA - 1
    p = 1 - (1 - 1.0 / juego.cardinalidad) ** juego.bolillas
    return p * PAGO_A_LOS_20 - 1


def diferencia_significativa(a: ResultadoBacktest, b: ResultadoBacktest) -> float:
    """p-valor de que dos estrategias tengan distinta tasa de acierto.

    Test de dos proporciones. Si da alto (lo normal), la diferencia de retorno
    entre las dos estrategias es ruido muestral, no habilidad.
    """
    n1, n2 = a.apuestas, b.apuestas
    x1, x2 = a.aciertos, b.aciertos
    if n1 == 0 or n2 == 0:
        return float("nan")
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (x1 / n1 - x2 / n2) / se
    return float(2 * norm_sf(abs(z)))
