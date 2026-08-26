"""Motor de ruleta: ruedas, apuestas y esperanza matematica.

El orden fisico de los numeros en el disco no coincide con el orden numerico.
Esa distincion es esencial: un defecto mecanico afecta a un sector contiguo del
disco (numeros vecinos en la rueda), no a numeros consecutivos en la mesa.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

# Orden fisico real de las casillas, en sentido horario desde el cero.
ORDEN_EUROPEA = [
    "0", "32", "15", "19", "4", "21", "2", "25", "17", "34", "6", "27", "13",
    "36", "11", "30", "8", "23", "10", "5", "24", "16", "33", "1", "20", "14",
    "31", "9", "22", "18", "29", "7", "28", "12", "35", "3", "26",
]

ORDEN_AMERICANA = [
    "0", "28", "9", "26", "30", "11", "7", "20", "32", "17", "5", "22", "34",
    "15", "3", "24", "36", "13", "1", "00", "27", "10", "25", "29", "12", "8",
    "19", "31", "18", "6", "21", "33", "16", "4", "23", "35", "14", "2",
]

ROJOS = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}


@dataclass(frozen=True)
class Rueda:
    """Configuracion de una mesa concreta.

    `regla_cero` cambia el pago de las apuestas de dinero par cuando sale cero:
      - "ninguna": se pierde todo (europea estandar)
      - "partage": se devuelve la mitad (frances, reduce la ventaja a la mitad)
      - "prision": la apuesta queda en prision para la tirada siguiente
    """
    nombre: str
    casillas: tuple[str, ...]
    regla_cero: str = "ninguna"

    @property
    def n(self) -> int:
        return len(self.casillas)

    @property
    def ceros(self) -> tuple[str, ...]:
        return tuple(c for c in self.casillas if c in ("0", "00"))

    def indice_fisico(self, casilla: str) -> int:
        return self.casillas.index(casilla)

    def vecinos(self, casilla: str, radio: int) -> list[str]:
        """Casillas dentro de `radio` posiciones fisicas a cada lado."""
        i = self.indice_fisico(casilla)
        return [self.casillas[(i + d) % self.n]
                for d in range(-radio, radio + 1)]


EUROPEA = Rueda("Europea (simple cero)", tuple(ORDEN_EUROPEA))
AMERICANA = Rueda("Americana (doble cero)", tuple(ORDEN_AMERICANA))
FRANCESA = Rueda("Francesa (con partage)", tuple(ORDEN_EUROPEA), regla_cero="partage")

RUEDAS = {"europea": EUROPEA, "americana": AMERICANA, "francesa": FRANCESA}


@dataclass(frozen=True)
class Apuesta:
    """Una apuesta con su conjunto ganador y su pago (a 1)."""
    nombre: str
    ganadoras: frozenset[str]
    pago: int
    dinero_par: bool = False

    def gana(self, resultado: str) -> bool:
        return resultado in self.ganadoras


def _num(rango) -> frozenset[str]:
    return frozenset(str(v) for v in rango)


def catalogo(rueda: Rueda) -> dict[str, Apuesta]:
    """Apuestas estandar disponibles en la mesa."""
    presentes = set(rueda.casillas)
    ap: dict[str, Apuesta] = {}

    for c in rueda.casillas:
        ap[f"pleno-{c}"] = Apuesta(f"Pleno al {c}", frozenset({c}), 35)

    ap["rojo"] = Apuesta("Rojo", _num(ROJOS) & presentes, 1, True)
    ap["negro"] = Apuesta("Negro", (_num(range(1, 37)) - _num(ROJOS)) & presentes, 1, True)
    ap["par"] = Apuesta("Par", _num(range(2, 37, 2)) & presentes, 1, True)
    ap["impar"] = Apuesta("Impar", _num(range(1, 37, 2)) & presentes, 1, True)
    ap["falta"] = Apuesta("Falta (1-18)", _num(range(1, 19)) & presentes, 1, True)
    ap["pasa"] = Apuesta("Pasa (19-36)", _num(range(19, 37)) & presentes, 1, True)

    ap["docena1"] = Apuesta("Docena 1 (1-12)", _num(range(1, 13)) & presentes, 2)
    ap["docena2"] = Apuesta("Docena 2 (13-24)", _num(range(13, 25)) & presentes, 2)
    ap["docena3"] = Apuesta("Docena 3 (25-36)", _num(range(25, 37)) & presentes, 2)

    for i in range(3):
        col = _num(range(1 + i, 37, 3)) & presentes
        ap[f"columna{i + 1}"] = Apuesta(f"Columna {i + 1}", col, 2)

    return ap


def esperanza(apuesta: Apuesta, rueda: Rueda) -> Fraction:
    """Esperanza exacta por unidad apostada, como fraccion.

    Se calcula en aritmetica racional para que el resultado sea exacto y no
    quede sujeto a redondeo: el numero que importa no admite discusion.
    """
    n = rueda.n
    g = len(apuesta.ganadoras)
    p_gana = Fraction(g, n)
    p_pierde = 1 - p_gana

    if apuesta.dinero_par and rueda.regla_cero in ("partage", "prision"):
        # Con partage se recupera la mitad; en prision, a largo plazo, equivale.
        p_cero = Fraction(len(rueda.ceros), n)
        p_pierde_real = p_pierde - p_cero
        return p_gana * apuesta.pago - p_pierde_real + p_cero * Fraction(-1, 2)

    return p_gana * apuesta.pago - p_pierde


def ventaja_casa(apuesta: Apuesta, rueda: Rueda) -> float:
    """Ventaja de la casa en porcentaje (positiva = la casa gana)."""
    return float(-esperanza(apuesta, rueda)) * 100


class Mesa:
    """Generador de resultados de una rueda.

    `sesgo` permite simular una rueda defectuosa: un diccionario
    {casilla: factor} donde factor > 1 hace esa casilla mas probable.
    """

    def __init__(self, rueda: Rueda = EUROPEA, semilla: int | None = None,
                 sesgo: dict[str, float] | None = None) -> None:
        self.rueda = rueda
        self.rng = np.random.default_rng(semilla)
        pesos = np.ones(rueda.n, dtype=float)
        if sesgo:
            for casilla, factor in sesgo.items():
                if casilla not in rueda.casillas:
                    raise ValueError(f"la casilla {casilla!r} no existe en {rueda.nombre}")
                if factor <= 0:
                    raise ValueError("el factor de sesgo debe ser > 0")
                pesos[rueda.indice_fisico(casilla)] = factor
        self.probs = pesos / pesos.sum()
        self.sesgada = bool(sesgo)

    def girar(self, n: int = 1) -> np.ndarray:
        """Devuelve `n` resultados como array de strings."""
        idx = self.rng.choice(self.rueda.n, size=n, p=self.probs)
        return np.array(self.rueda.casillas, dtype=object)[idx]


def tabla_ventajas(rueda: Rueda) -> list[tuple[str, float, str]]:
    """Ventaja de la casa para cada tipo de apuesta representativa."""
    cat = catalogo(rueda)
    claves = ["pleno-1", "rojo", "par", "docena1", "columna1"]
    salida = []
    for k in claves:
        a = cat[k]
        ev = esperanza(a, rueda)
        salida.append((a.nombre, ventaja_casa(a, rueda), f"{ev} por unidad"))
    return salida
