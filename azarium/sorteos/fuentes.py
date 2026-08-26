"""Ingesta de historicos: CSV local, CSV remoto y generador sintetico.

No se hace scraping de HTML a proposito: las paginas de loterias cambian de
maquetado sin aviso y un scraper roto que devuelve datos parciales es peor que
no tener datos. La via soportada es CSV (local o por URL).
"""

from __future__ import annotations

import datetime as dt
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from .modelo import Historico, Juego


def generar_sintetico(juego: Juego, n_sorteos: int, semilla: int = 0,
                      sesgo: dict[int, float] | None = None,
                      desde: dt.date | None = None,
                      dias_entre_sorteos: int = 1) -> Historico:
    """Genera un historico artificial.

    Con `sesgo=None` produce azar uniforme puro: sirve como grupo de control
    para verificar que los tests NO encuentran patrones donde no los hay.

    Con `sesgo={numero: factor}` inyecta una desviacion conocida (factor 1.5 =
    ese numero sale 50% mas seguido). Sirve para medir la potencia real de los
    tests: cuantos sorteos hacen falta para detectar un sesgo de ese tamano.
    """
    rng = np.random.default_rng(semilla)
    k = juego.cardinalidad
    pesos = np.ones(k, dtype=float)
    if sesgo:
        for numero, factor in sesgo.items():
            if not (juego.minimo <= numero <= juego.maximo):
                raise ValueError(f"numero {numero} fuera del rango del juego")
            if factor <= 0:
                raise ValueError("el factor de sesgo debe ser > 0")
            pesos[numero - juego.minimo] = factor
    probs = pesos / pesos.sum()

    if juego.con_reposicion:
        resultados = rng.choice(juego.valores, size=(n_sorteos, juego.bolillas),
                                replace=True, p=probs)
    else:
        filas = [rng.choice(juego.valores, size=juego.bolillas, replace=False, p=probs)
                 for _ in range(n_sorteos)]
        resultados = np.array(filas, dtype=int)

    desde = desde or (dt.date.today() - dt.timedelta(days=dias_entre_sorteos * n_sorteos))
    fechas = [desde + dt.timedelta(days=i * dias_entre_sorteos) for i in range(n_sorteos)]
    etiqueta = "sintetico-uniforme" if not sesgo else f"sintetico-sesgado{sesgo}"
    return Historico(juego=juego, fechas=fechas, resultados=resultados, fuente=etiqueta)


def descargar_csv(url: str, juego: Juego, destino: str | Path,
                  timeout: int = 30) -> Historico:
    """Baja un CSV de historicos y lo deja cacheado en `destino`.

    El CSV debe tener una columna `fecha` opcional seguida de las columnas de
    numeros, un sorteo por fila.
    """
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    peticion = urllib.request.Request(url, headers={"User-Agent": "Azarium/1.0"})
    try:
        with urllib.request.urlopen(peticion, timeout=timeout) as resp:
            contenido = resp.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"No se pudo descargar {url}: {exc}") from exc
    destino.write_bytes(contenido)
    return Historico.cargar_csv(destino, juego)


def cargar(ruta: str | Path, juego: Juego) -> Historico:
    """Atajo para leer un CSV ya presente en disco."""
    return Historico.cargar_csv(ruta, juego)
