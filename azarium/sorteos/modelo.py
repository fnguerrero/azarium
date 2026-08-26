"""Modelo de datos y persistencia de historicos de sorteos."""

from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Juego:
    """Define el espacio muestral de un juego de loteria.

    minimo/maximo son inclusivos. `bolillas` es cuantos numeros salen por sorteo.
    `con_reposicion` distingue la quiniela (cada posicion es independiente y
    puede repetirse) del loto (extraccion sin reposicion).
    """
    nombre: str
    minimo: int
    maximo: int
    bolillas: int
    con_reposicion: bool

    @property
    def cardinalidad(self) -> int:
        return self.maximo - self.minimo + 1

    @property
    def valores(self) -> np.ndarray:
        return np.arange(self.minimo, self.maximo + 1)

    def prob_por_numero(self) -> float:
        """Probabilidad de que un numero dado aparezca en una posicion."""
        return 1.0 / self.cardinalidad


QUINIELA = Juego("Quiniela", 0, 99, 20, con_reposicion=True)
LOTO = Juego("Loto", 0, 41, 6, con_reposicion=False)
QUINI6 = Juego("Quini 6", 0, 45, 6, con_reposicion=False)


JUEGOS = {j.nombre.lower().replace(" ", ""): j for j in (QUINIELA, LOTO, QUINI6)}


@dataclass
class Historico:
    """Serie de sorteos: matriz (n_sorteos x bolillas) + fechas."""
    juego: Juego
    fechas: list[dt.date] = field(default_factory=list)
    resultados: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=int))
    fuente: str = "desconocida"

    def __post_init__(self) -> None:
        if self.resultados.size and self.resultados.ndim != 2:
            raise ValueError("resultados debe ser una matriz 2D")

    @property
    def n_sorteos(self) -> int:
        return int(self.resultados.shape[0]) if self.resultados.size else 0

    @property
    def n_extracciones(self) -> int:
        """Total de numeros individuales observados."""
        return int(self.resultados.size)

    def plano(self) -> np.ndarray:
        """Todos los numeros extraidos, en orden cronologico."""
        return self.resultados.reshape(-1)

    def frecuencias(self) -> np.ndarray:
        """Conteo por numero del espacio muestral (indice 0 == juego.minimo)."""
        return np.bincount(self.plano() - self.juego.minimo,
                           minlength=self.juego.cardinalidad)

    def rango_fechas(self) -> tuple[dt.date | None, dt.date | None]:
        if not self.fechas:
            return (None, None)
        return (min(self.fechas), max(self.fechas))

    def anios_cubiertos(self) -> float:
        a, b = self.rango_fechas()
        if a is None or b is None or a == b:
            return 0.0
        return (b - a).days / 365.25

    def guardar_csv(self, ruta: str | Path) -> Path:
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        with ruta.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["fecha"] + [f"n{i + 1}" for i in range(self.resultados.shape[1])])
            for i in range(self.n_sorteos):
                fecha = self.fechas[i].isoformat() if i < len(self.fechas) else ""
                w.writerow([fecha] + [int(v) for v in self.resultados[i]])
        return ruta

    @classmethod
    def cargar_csv(cls, ruta: str | Path, juego: Juego) -> "Historico":
        """Lee un CSV con columna `fecha` (opcional) y N columnas de numeros."""
        ruta = Path(ruta)
        fechas: list[dt.date] = []
        filas: list[list[int]] = []
        with ruta.open(encoding="utf-8-sig", newline="") as fh:
            lector = csv.reader(fh)
            cabecera = next(lector, None)
            if cabecera is None:
                raise ValueError(f"{ruta} esta vacio")
            tiene_fecha = cabecera[0].strip().lower() in ("fecha", "date", "sorteo_fecha")
            for nro_linea, fila in enumerate(lector, start=2):
                if not fila or not any(c.strip() for c in fila):
                    continue
                crudos = fila[1:] if tiene_fecha else fila
                try:
                    numeros = [int(c.strip()) for c in crudos if c.strip() != ""]
                except ValueError as exc:
                    raise ValueError(f"{ruta}:{nro_linea} numero invalido: {exc}") from exc
                if not numeros:
                    continue
                fuera = [n for n in numeros if not (juego.minimo <= n <= juego.maximo)]
                if fuera:
                    raise ValueError(
                        f"{ruta}:{nro_linea} numeros fuera del rango "
                        f"[{juego.minimo}, {juego.maximo}]: {fuera}")
                filas.append(numeros)
                if tiene_fecha:
                    fechas.append(_parsear_fecha(fila[0].strip(), ruta, nro_linea))
        if not filas:
            raise ValueError(f"{ruta} no contiene sorteos")
        ancho = max(len(f) for f in filas)
        if any(len(f) != ancho for f in filas):
            raise ValueError(f"{ruta}: los sorteos no tienen la misma cantidad de numeros")
        return cls(juego=juego, fechas=fechas,
                   resultados=np.array(filas, dtype=int), fuente=str(ruta))


_FORMATOS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d")


def _parsear_fecha(texto: str, ruta: Path, linea: int) -> dt.date:
    for fmt in _FORMATOS:
        try:
            return dt.datetime.strptime(texto, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"{ruta}:{linea} fecha no reconocida: {texto!r}")
