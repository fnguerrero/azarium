"""Simulacion Monte Carlo de sesiones de ruleta.

El objetivo no es "encontrar el sistema que gana" sino medir con precision que
le pasa a la banca bajo cada sistema: cuanto dura, con que probabilidad se
arruina, y como se reparte el resultado final.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .motor import Apuesta, Mesa, Rueda, EUROPEA, catalogo, esperanza
from .sistemas import SISTEMAS, Sistema


@dataclass
class Config:
    """Parametros de una tanda de simulaciones."""
    banca_inicial: float = 1000.0
    apuesta_base: float = 10.0
    tiradas: int = 500
    sesiones: int = 10_000
    limite_mesa: float = 5000.0
    objetivo: float | None = None  # si se alcanza, se corta la sesion
    rueda: Rueda = EUROPEA
    apuesta: str = "rojo"
    semilla: int = 0


@dataclass
class Resumen:
    """Metricas agregadas de una tanda."""
    sistema: str
    config: Config
    bancas_finales: np.ndarray = field(repr=False)
    tiradas_jugadas: np.ndarray = field(repr=False)
    max_drawdown: np.ndarray = field(repr=False)
    total_apostado: np.ndarray = field(repr=False)
    arruinadas: int = 0
    topearon_limite: int = 0
    alcanzaron_objetivo: int = 0

    @property
    def prob_ruina(self) -> float:
        return self.arruinadas / len(self.bancas_finales)

    @property
    def prob_objetivo(self) -> float:
        return self.alcanzaron_objetivo / len(self.bancas_finales)

    @property
    def resultado_medio(self) -> float:
        return float(self.bancas_finales.mean() - self.config.banca_inicial)

    @property
    def resultado_mediano(self) -> float:
        return float(np.median(self.bancas_finales) - self.config.banca_inicial)

    @property
    def prob_terminar_ganando(self) -> float:
        return float((self.bancas_finales > self.config.banca_inicial).mean())

    @property
    def retorno_sobre_apostado(self) -> float:
        """Resultado neto dividido el volumen total apostado.

        Esta es la cifra clave: converge a la ventaja de la casa sin importar
        el sistema, porque es lo unico que la matematica fija.
        """
        apostado = self.total_apostado.sum()
        if apostado == 0:
            return 0.0
        neto = float(self.bancas_finales.sum() - self.config.banca_inicial * len(self.bancas_finales))
        return neto / apostado

    def percentiles(self, ps=(5, 25, 50, 75, 95)) -> dict[int, float]:
        return {p: float(np.percentile(self.bancas_finales, p)) for p in ps}


def _resolver_apuesta(cfg: Config) -> Apuesta:
    cat = catalogo(cfg.rueda)
    if cfg.apuesta not in cat:
        raise ValueError(f"apuesta desconocida: {cfg.apuesta!r}")
    return cat[cfg.apuesta]


def simular(nombre_sistema: str, cfg: Config,
            sesgo: dict[str, float] | None = None) -> Resumen:
    """Corre `cfg.sesiones` sesiones independientes del sistema indicado."""
    if nombre_sistema not in SISTEMAS:
        raise ValueError(f"sistema desconocido: {nombre_sistema!r}")
    clase = SISTEMAS[nombre_sistema]
    apuesta = _resolver_apuesta(cfg)
    mesa = Mesa(cfg.rueda, semilla=cfg.semilla, sesgo=sesgo)
    ganadoras = apuesta.ganadoras
    devuelve_mitad = apuesta.dinero_par and cfg.rueda.regla_cero == "partage"
    ceros = set(cfg.rueda.ceros)

    finales = np.empty(cfg.sesiones)
    jugadas = np.empty(cfg.sesiones, dtype=int)
    drawdowns = np.empty(cfg.sesiones)
    apostados = np.empty(cfg.sesiones)
    arruinadas = topes = objetivos = 0

    # Se generan todas las tiradas de una sesion de golpe: mucho mas rapido
    # que llamar al RNG tirada por tirada.
    for s in range(cfg.sesiones):
        sistema: Sistema = clase(cfg.apuesta_base)
        banca = cfg.banca_inicial
        pico = banca
        peor = 0.0
        apostado = 0.0
        resultados = mesa.girar(cfg.tiradas)
        t = 0
        for t in range(cfg.tiradas):
            monto = sistema.apuesta()
            if monto > cfg.limite_mesa:
                monto = cfg.limite_mesa
                topes += 1
            if monto > banca:
                monto = banca  # se apuesta lo que queda
            if monto <= 0:
                arruinadas += 1
                break

            res = resultados[t]
            apostado += monto
            if res in ganadoras:
                banca += monto * apuesta.pago
                gano = True
            elif devuelve_mitad and res in ceros:
                banca -= monto / 2
                gano = False
            else:
                banca -= monto
                gano = False
            sistema.resolver(gano)

            pico = max(pico, banca)
            peor = max(peor, pico - banca)

            if banca <= 0:
                arruinadas += 1
                break
            if cfg.objetivo is not None and banca >= cfg.objetivo:
                objetivos += 1
                break

        finales[s] = max(0.0, banca)
        jugadas[s] = t + 1
        drawdowns[s] = peor
        apostados[s] = apostado

    return Resumen(sistema=clase.nombre, config=cfg, bancas_finales=finales,
                   tiradas_jugadas=jugadas, max_drawdown=drawdowns,
                   total_apostado=apostados, arruinadas=arruinadas,
                   topearon_limite=topes, alcanzaron_objetivo=objetivos)


def comparar(cfg: Config, sistemas: list[str] | None = None,
             sesgo: dict[str, float] | None = None) -> list[Resumen]:
    """Corre la misma configuracion con varios sistemas para compararlos."""
    sistemas = sistemas or list(SISTEMAS)
    return [simular(s, cfg, sesgo=sesgo) for s in sistemas]


def ventaja_teorica(cfg: Config) -> float:
    """Ventaja de la casa esperada para la apuesta configurada, en porcentaje."""
    return float(-esperanza(_resolver_apuesta(cfg), cfg.rueda)) * 100
