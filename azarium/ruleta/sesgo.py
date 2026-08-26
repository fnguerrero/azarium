"""Deteccion de sesgo mecanico en una rueda concreta.

Es el unico enfoque con base historica real (Jagger en Montecarlo 1873, el
equipo de Hibbs y Walford en Reno 1947, los Pelayo en los 90): si una rueda
esta desnivelada o tiene celdas desgastadas, algunas casillas salen mas de lo
que deberian, y con suficiente muestra eso se puede medir.

Dos advertencias que el modulo hace explicitas en los resultados:

1. El umbral de explotabilidad no es "sale mas", es "sale mas del 1/36".
   Un pleno paga 35 a 1, asi que la apuesta solo tiene esperanza positiva si la
   probabilidad real de esa casilla supera 1/36 = 2.778%, contra el 1/37 =
   2.703% teorico. Es una diferencia relativa de apenas 2.8%.

2. Detectar una diferencia asi de chica exige cientos de miles de tiradas de
   LA MISMA rueda. `tiradas_necesarias()` calcula el numero exacto.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..stats import Resultado, benjamini_hochberg, binom_sf_ge, chi2_sf, ic_wilson, norm_ppf
from .motor import Rueda, EUROPEA


@dataclass
class Registro:
    """Tiradas observadas de una rueda concreta."""
    rueda: Rueda
    tiradas: list[str]
    mesa: str = "sin identificar"

    def __post_init__(self) -> None:
        validas = set(self.rueda.casillas)
        invalidas = sorted({t for t in self.tiradas if t not in validas})
        if invalidas:
            raise ValueError(f"casillas invalidas para {self.rueda.nombre}: {invalidas}")

    @property
    def n(self) -> int:
        return len(self.tiradas)

    def frecuencias(self) -> np.ndarray:
        """Conteos en orden fisico del disco."""
        idx = {c: i for i, c in enumerate(self.rueda.casillas)}
        conteo = np.zeros(self.rueda.n, dtype=int)
        for t in self.tiradas:
            conteo[idx[t]] += 1
        return conteo

    @classmethod
    def cargar_csv(cls, ruta: str | Path, rueda: Rueda = EUROPEA,
                   columna: int = 0, mesa: str = "sin identificar") -> "Registro":
        """Lee un CSV con una tirada por fila."""
        ruta = Path(ruta)
        tiradas: list[str] = []
        with ruta.open(encoding="utf-8-sig", newline="") as fh:
            for fila in csv.reader(fh):
                if not fila:
                    continue
                valor = fila[columna].strip()
                if not valor or valor.lower() in ("resultado", "tirada", "numero"):
                    continue
                tiradas.append(valor)
        if not tiradas:
            raise ValueError(f"{ruta} no contiene tiradas")
        return cls(rueda=rueda, tiradas=tiradas, mesa=mesa)


def tiradas_necesarias(rueda: Rueda = EUROPEA, potencia: float = 0.80,
                       alfa: float = 0.05, corregir_multiplicidad: bool = True,
                       p_real: float | None = None) -> int:
    """Cuantas tiradas hacen falta para detectar un sesgo explotable.

    Por defecto calcula el caso limite: distinguir una casilla que sale con
    p = 1/36 (el minimo que hace rentable el pleno) de la uniforme p = 1/n.
    """
    p0 = 1.0 / rueda.n
    p1 = p_real if p_real is not None else 1.0 / 36.0
    if p1 <= p0:
        raise ValueError("p_real debe ser mayor que la probabilidad uniforme")
    alfa_efectiva = alfa / rueda.n if corregir_multiplicidad else alfa
    z_a = norm_ppf(1 - alfa_efectiva)
    z_b = norm_ppf(potencia)
    num = z_a * math.sqrt(p0 * (1 - p0)) + z_b * math.sqrt(p1 * (1 - p1))
    return int(math.ceil((num / (p1 - p0)) ** 2))


def horas_de_mesa(tiradas: int, tiradas_por_hora: int = 40) -> float:
    """Traduce una cantidad de tiradas a horas de observacion continua."""
    return tiradas / tiradas_por_hora


def chi2_rueda(reg: Registro) -> Resultado:
    """Bondad de ajuste global sobre las casillas."""
    obs = reg.frecuencias().astype(float)
    esp = reg.n / reg.rueda.n
    stat = float(((obs - esp) ** 2 / esp).sum())
    df = reg.rueda.n - 1
    if esp < 5:
        detalle = (f"ADVERTENCIA: solo {esp:.1f} tiradas esperadas por casilla; "
                   "hacen falta al menos 5 para que el chi2 sea fiable")
    else:
        detalle = f"{reg.n} tiradas, {esp:.1f} esperadas por casilla"
    return Resultado("Chi-cuadrado de la rueda", stat, chi2_sf(stat, df), df, detalle)


@dataclass
class FilaCasilla:
    casilla: str
    observado: int
    esperado: float
    frecuencia: float
    p_valor: float
    ic_bajo: float
    ic_alto: float
    significativo_fdr: bool
    explotable: bool
    ventaja_jugador: float


def por_casilla(reg: Registro, fdr: float = 0.05) -> list[FilaCasilla]:
    """Test unilateral por casilla (buscamos exceso) con control de FDR.

    `explotable` es la columna que importa: exige que el limite INFERIOR del
    intervalo de confianza supere 1/36. Que el estimador puntual pase el umbral
    no alcanza, porque lo que se juega es dinero real contra el error muestral.
    """
    obs = reg.frecuencias()
    n = reg.n
    p0 = 1.0 / reg.rueda.n
    esp = n * p0
    umbral_rentable = 1.0 / 36.0

    p_vals = np.array([binom_sf_ge(int(o), n, p0) for o in obs])
    rechazadas = benjamini_hochberg(p_vals, fdr=fdr)

    filas = []
    for i, casilla in enumerate(reg.rueda.casillas):
        bajo, alto = ic_wilson(int(obs[i]), n)
        frec = obs[i] / n if n else 0.0
        explotable = bool(bajo > umbral_rentable and rechazadas[i])
        filas.append(FilaCasilla(
            casilla=casilla,
            observado=int(obs[i]),
            esperado=esp,
            frecuencia=frec,
            p_valor=float(p_vals[i]),
            ic_bajo=bajo, ic_alto=alto,
            significativo_fdr=bool(rechazadas[i]),
            explotable=explotable,
            ventaja_jugador=(frec * 36.0 - 1.0) * 100,
        ))
    return filas


@dataclass
class Sector:
    centro: str
    casillas: list[str]
    observado: int
    esperado: float
    z: float
    p_valor: float
    significativo_fdr: bool


def por_sector(reg: Registro, radio: int = 3, fdr: float = 0.05) -> list[Sector]:
    """Analisis por sectores fisicos contiguos del disco.

    Un defecto mecanico (desnivel, celda floja, desgaste del deflector) no
    afecta a un numero suelto sino a una zona del plato. Sumar las casillas
    vecinas concentra la senal y multiplica la potencia del test frente a
    mirar cada numero por separado.
    """
    obs = reg.frecuencias()
    n = reg.n
    ancho = 2 * radio + 1
    p_sector = ancho / reg.rueda.n
    esp = n * p_sector
    sigma = math.sqrt(n * p_sector * (1 - p_sector))

    crudos = []
    for i, centro in enumerate(reg.rueda.casillas):
        indices = [(i + d) % reg.rueda.n for d in range(-radio, radio + 1)]
        total = int(obs[indices].sum())
        z = (total - esp) / sigma if sigma else 0.0
        p = binom_sf_ge(total, n, p_sector)
        crudos.append((centro, [reg.rueda.casillas[j] for j in indices], total, z, p))

    rechazadas = benjamini_hochberg(np.array([c[4] for c in crudos]), fdr=fdr)
    return [Sector(centro=c[0], casillas=c[1], observado=c[2], esperado=esp,
                   z=float(c[3]), p_valor=float(c[4]),
                   significativo_fdr=bool(rechazadas[i]))
            for i, c in enumerate(crudos)]


@dataclass
class Veredicto:
    """Conclusion operativa del analisis de una rueda."""
    muestra_suficiente: bool
    n_observado: int
    n_necesario: int
    chi2: Resultado
    casillas_explotables: list[FilaCasilla]
    sectores_significativos: list[Sector]

    @property
    def hay_sesgo_explotable(self) -> bool:
        return bool(self.casillas_explotables)

    def texto(self) -> str:
        lineas = []
        if not self.muestra_suficiente and not self.hay_sesgo_explotable:
            # El n necesario esta calculado para el sesgo MAS CHICO que seria
            # rentable. Un sesgo grande se detecta con mucha menos muestra, por
            # eso el aviso solo aplica cuando no se encontro nada.
            faltan = self.n_necesario - self.n_observado
            lineas.append(
                f"MUESTRA INSUFICIENTE PARA DESCARTAR SESGO: {self.n_observado} tiradas "
                f"registradas. Detectar el sesgo mas chico que seria rentable (1/36) "
                f"exige ~{self.n_necesario} ({faltan} mas, "
                f"~{horas_de_mesa(faltan):.0f} horas de mesa). Un sesgo grande se veria "
                "con mucha menos muestra; no encontrar nada aca solo descarta los grandes.")
        elif not self.muestra_suficiente:
            lineas.append(
                f"Sesgo detectado con {self.n_observado} tiradas: el efecto es lo bastante "
                f"grande como para verse muy por debajo de las ~{self.n_necesario} tiradas "
                "que exigiria el sesgo marginal.")
        lineas.append(f"{self.chi2.nombre}: p={self.chi2.p_valor:.4g} -> {self.chi2.lectura()}")
        if self.hay_sesgo_explotable:
            for f in self.casillas_explotables:
                lineas.append(
                    f"Casilla {f.casilla}: {f.frecuencia * 100:.3f}% observado "
                    f"(IC95 {f.ic_bajo * 100:.3f}-{f.ic_alto * 100:.3f}%), "
                    f"ventaja estimada del jugador {f.ventaja_jugador:+.2f}%")
        else:
            lineas.append("No hay ninguna casilla con sesgo explotable: ninguna supera "
                          "el umbral de 1/36 con el limite inferior del IC 95%.")
        if self.sectores_significativos:
            nombres = ", ".join(s.centro for s in self.sectores_significativos[:5])
            lineas.append(f"Sectores fisicos con exceso significativo (centro): {nombres}")
        return "\n".join(lineas)


def analizar(reg: Registro, radio_sector: int = 3, fdr: float = 0.05) -> Veredicto:
    """Corre el analisis completo y devuelve un veredicto operativo."""
    necesario = tiradas_necesarias(reg.rueda)
    filas = por_casilla(reg, fdr=fdr)
    sectores = por_sector(reg, radio=radio_sector, fdr=fdr)
    return Veredicto(
        muestra_suficiente=reg.n >= necesario,
        n_observado=reg.n,
        n_necesario=necesario,
        chi2=chi2_rueda(reg),
        casillas_explotables=[f for f in filas if f.explotable],
        sectores_significativos=[s for s in sectores if s.significativo_fdr],
    )
