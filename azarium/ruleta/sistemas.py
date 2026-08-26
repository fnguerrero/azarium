"""Sistemas de progresion de apuestas.

Todos comparten la misma interfaz. Ninguno cambia la esperanza matematica: la
esperanza de una suma de apuestas es la suma de las esperanzas, y cada apuesta
individual ya tiene esperanza negativa. Lo unico que cambia un sistema es la
FORMA de la distribucion de resultados (asimetria, varianza, probabilidad de
ruina), nunca su media.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Sistema(ABC):
    """Interfaz comun de las progresiones."""

    nombre = "base"
    descripcion = ""

    def __init__(self, base: float = 1.0) -> None:
        self.base = float(base)
        self.reiniciar()

    def reiniciar(self) -> None:
        """Vuelve al estado inicial (arranque de sesion)."""
        self._estado_inicial()

    @abstractmethod
    def _estado_inicial(self) -> None:
        ...

    @abstractmethod
    def apuesta(self) -> float:
        """Monto a apostar en la proxima tirada."""

    @abstractmethod
    def resolver(self, gano: bool) -> None:
        """Actualiza el estado interno con el resultado de la tirada."""


class Plana(Sistema):
    nombre = "Plana"
    descripcion = "Siempre la misma apuesta. La referencia contra la que se miden las demas."

    def _estado_inicial(self) -> None:
        pass

    def apuesta(self) -> float:
        return self.base

    def resolver(self, gano: bool) -> None:
        pass


class Martingala(Sistema):
    nombre = "Martingala"
    descripcion = "Se dobla al perder. Gana poco y seguido; pierde todo de golpe."

    def _estado_inicial(self) -> None:
        self.actual = self.base

    def apuesta(self) -> float:
        return self.actual

    def resolver(self, gano: bool) -> None:
        self.actual = self.base if gano else self.actual * 2


class GranMartingala(Sistema):
    nombre = "Gran Martingala"
    descripcion = "Se dobla y se suma una unidad al perder. Acelera la ruina."

    def _estado_inicial(self) -> None:
        self.actual = self.base

    def apuesta(self) -> float:
        return self.actual

    def resolver(self, gano: bool) -> None:
        self.actual = self.base if gano else self.actual * 2 + self.base


class DAlembert(Sistema):
    nombre = "D'Alembert"
    descripcion = "Sube una unidad al perder, baja una al ganar. Progresion suave."

    def _estado_inicial(self) -> None:
        self.unidades = 1

    def apuesta(self) -> float:
        return self.base * self.unidades

    def resolver(self, gano: bool) -> None:
        self.unidades = max(1, self.unidades - 1) if gano else self.unidades + 1


class Fibonacci(Sistema):
    nombre = "Fibonacci"
    descripcion = "Avanza en la sucesion al perder, retrocede dos lugares al ganar."

    def _estado_inicial(self) -> None:
        self.serie = [1, 1]
        self.pos = 0

    def apuesta(self) -> float:
        while self.pos >= len(self.serie):
            self.serie.append(self.serie[-1] + self.serie[-2])
        return self.base * self.serie[self.pos]

    def resolver(self, gano: bool) -> None:
        if gano:
            self.pos = max(0, self.pos - 2)
        else:
            self.pos += 1


class Labouchere(Sistema):
    nombre = "Labouchere"
    descripcion = "Lista de objetivos: se apuesta la suma de los extremos y se tachan al ganar."

    def __init__(self, base: float = 1.0, longitud: int = 6) -> None:
        self.longitud = longitud
        super().__init__(base)

    def _estado_inicial(self) -> None:
        self.lista = list(range(1, self.longitud + 1))

    def apuesta(self) -> float:
        if not self.lista:
            self.lista = list(range(1, self.longitud + 1))
        if len(self.lista) == 1:
            return self.base * self.lista[0]
        return self.base * (self.lista[0] + self.lista[-1])

    def resolver(self, gano: bool) -> None:
        if not self.lista:
            return
        if gano:
            if len(self.lista) == 1:
                self.lista.pop()
            else:
                self.lista = self.lista[1:-1]
        else:
            monto = (self.lista[0] + self.lista[-1]) if len(self.lista) > 1 else self.lista[0]
            self.lista.append(monto)


class Paroli(Sistema):
    nombre = "Paroli"
    descripcion = "Martingala inversa: se dobla al ganar, hasta tres veces. Limita perdidas."

    def __init__(self, base: float = 1.0, tope_rachas: int = 3) -> None:
        self.tope_rachas = tope_rachas
        super().__init__(base)

    def _estado_inicial(self) -> None:
        self.actual = self.base
        self.racha = 0

    def apuesta(self) -> float:
        return self.actual

    def resolver(self, gano: bool) -> None:
        if gano:
            self.racha += 1
            if self.racha >= self.tope_rachas:
                self.actual = self.base
                self.racha = 0
            else:
                self.actual *= 2
        else:
            self.actual = self.base
            self.racha = 0


class OscarsGrind(Sistema):
    nombre = "Oscar's Grind"
    descripcion = "Busca ganar una unidad por ciclo; sube una unidad tras cada victoria."

    def _estado_inicial(self) -> None:
        self.actual = self.base
        self.ganancia_ciclo = 0.0

    def apuesta(self) -> float:
        # Nunca apostar mas de lo necesario para cerrar el ciclo con +1 unidad.
        falta = self.base - self.ganancia_ciclo
        return min(self.actual, falta) if falta > 0 else self.base

    def resolver(self, gano: bool) -> None:
        monto = self.apuesta()
        if gano:
            self.ganancia_ciclo += monto
            if self.ganancia_ciclo >= self.base:
                self.actual = self.base
                self.ganancia_ciclo = 0.0
            else:
                self.actual += self.base
        else:
            self.ganancia_ciclo -= monto


SISTEMAS: dict[str, type[Sistema]] = {
    "plana": Plana,
    "martingala": Martingala,
    "granmartingala": GranMartingala,
    "dalembert": DAlembert,
    "fibonacci": Fibonacci,
    "labouchere": Labouchere,
    "paroli": Paroli,
    "oscar": OscarsGrind,
}
