"""Tests del motor de ruleta: esperanzas exactas, catalogo, progresiones,
Monte Carlo y deteccion de sesgo."""

import csv
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

from azarium.ruleta import motor, sesgo
from azarium.ruleta.montecarlo import Config, simular, ventaja_teorica
from azarium.ruleta.sistemas import (SISTEMAS, DAlembert, Fibonacci, GranMartingala,
                                     Labouchere, Martingala, OscarsGrind, Paroli, Plana)


class RuedaTests(unittest.TestCase):
    def test_tamanos_y_ceros(self):
        self.assertEqual(motor.EUROPEA.n, 37)
        self.assertEqual(motor.AMERICANA.n, 38)
        self.assertEqual(motor.EUROPEA.ceros, ("0",))
        self.assertEqual(motor.AMERICANA.ceros, ("0", "00"))

    def test_el_orden_fisico_contiene_cada_numero_una_vez(self):
        self.assertEqual(sorted(motor.ORDEN_EUROPEA, key=int), [str(i) for i in range(37)])
        self.assertEqual(len(set(motor.ORDEN_AMERICANA)), 38)
        self.assertIn("00", motor.ORDEN_AMERICANA)

    def test_vecinos_dan_la_vuelta_al_disco(self):
        self.assertEqual(motor.EUROPEA.vecinos("0", 1), ["26", "0", "32"])
        self.assertEqual(motor.EUROPEA.vecinos("26", 2), ["35", "3", "26", "0", "32"])
        self.assertEqual(len(motor.EUROPEA.vecinos("17", 3)), 7)

    def test_rojos_son_18(self):
        self.assertEqual(len(motor.ROJOS), 18)


class CatalogoTests(unittest.TestCase):
    def test_conjuntos_ganadores(self):
        cat = motor.catalogo(motor.EUROPEA)
        for clave, cantidad in (("rojo", 18), ("negro", 18), ("par", 18), ("impar", 18),
                                ("falta", 18), ("pasa", 18), ("docena1", 12), ("docena3", 12),
                                ("columna1", 12), ("columna3", 12), ("pleno-17", 1)):
            with self.subTest(clave=clave):
                self.assertEqual(len(cat[clave].ganadoras), cantidad)
        self.assertNotIn("0", cat["par"].ganadoras)
        self.assertNotIn("0", cat["negro"].ganadoras)
        self.assertEqual(cat["columna1"].ganadoras, frozenset(str(v) for v in range(1, 37, 3)))

    def test_hay_un_pleno_por_casilla(self):
        cat = motor.catalogo(motor.AMERICANA)
        plenos = [k for k in cat if k.startswith("pleno-")]
        self.assertEqual(len(plenos), 38)
        self.assertIn("pleno-00", cat)
        self.assertTrue(cat["pleno-00"].gana("00"))
        self.assertFalse(cat["pleno-00"].gana("0"))


class EsperanzaTests(unittest.TestCase):
    def test_valores_exactos(self):
        casos = (
            (motor.EUROPEA, "rojo", Fraction(-1, 37)),
            (motor.EUROPEA, "pleno-17", Fraction(-1, 37)),
            (motor.AMERICANA, "rojo", Fraction(-1, 19)),
            (motor.AMERICANA, "docena1", Fraction(-1, 19)),
            (motor.FRANCESA, "rojo", Fraction(-1, 74)),
            (motor.FRANCESA, "par", Fraction(-1, 74)),
            (motor.FRANCESA, "docena1", Fraction(-1, 37)),   # partage solo en dinero par
            (motor.FRANCESA, "pleno-0", Fraction(-1, 37)),
        )
        for rueda, clave, esperado in casos:
            with self.subTest(rueda=rueda.nombre, apuesta=clave):
                self.assertEqual(motor.esperanza(motor.catalogo(rueda)[clave], rueda), esperado)

    def test_todas_las_apuestas_de_una_rueda_tienen_la_misma_esperanza(self):
        for rueda, esperado in ((motor.EUROPEA, Fraction(-1, 37)),
                                (motor.AMERICANA, Fraction(-1, 19))):
            for apuesta in motor.catalogo(rueda).values():
                with self.subTest(rueda=rueda.nombre, apuesta=apuesta.nombre):
                    self.assertEqual(motor.esperanza(apuesta, rueda), esperado)

    def test_ventaja_en_porcentaje(self):
        cat = motor.catalogo(motor.EUROPEA)
        self.assertAlmostEqual(motor.ventaja_casa(cat["rojo"], motor.EUROPEA), 2.7027, places=3)
        cat = motor.catalogo(motor.AMERICANA)
        self.assertAlmostEqual(motor.ventaja_casa(cat["rojo"], motor.AMERICANA), 5.2632, places=3)
        cat = motor.catalogo(motor.FRANCESA)
        self.assertAlmostEqual(motor.ventaja_casa(cat["rojo"], motor.FRANCESA), 1.3514, places=3)

    def test_tabla_ventajas(self):
        tabla = motor.tabla_ventajas(motor.EUROPEA)
        self.assertEqual(len(tabla), 5)
        for nombre, ventaja, texto in tabla:
            self.assertAlmostEqual(ventaja, 2.7027, places=3)
            self.assertIn("-1/37", texto)


class MesaTests(unittest.TestCase):
    def test_reproducible_con_semilla(self):
        a = motor.Mesa(motor.EUROPEA, semilla=7).girar(50)
        b = motor.Mesa(motor.EUROPEA, semilla=7).girar(50)
        self.assertEqual(list(a), list(b))
        self.assertTrue(set(a) <= set(motor.EUROPEA.casillas))

    def test_sesgo_mueve_la_frecuencia(self):
        mesa = motor.Mesa(motor.EUROPEA, semilla=1, sesgo={"17": 5.0})
        self.assertTrue(mesa.sesgada)
        tiradas = mesa.girar(20000)
        frec17 = (tiradas == "17").mean()
        self.assertGreater(frec17, 3 / 41 * 0.9)     # 5 / (36 + 5)
        self.assertLess(frec17, 5 / 41 * 1.1)

    def test_validaciones_del_sesgo(self):
        with self.assertRaises(ValueError):
            motor.Mesa(motor.EUROPEA, sesgo={"00": 2.0})
        with self.assertRaises(ValueError):
            motor.Mesa(motor.EUROPEA, sesgo={"5": 0.0})


class SistemasTests(unittest.TestCase):
    @staticmethod
    def _jugar(sistema, resultados):
        """Devuelve la apuesta previa a cada tirada de la racha dada."""
        montos = []
        for gano in resultados:
            montos.append(sistema.apuesta())
            sistema.resolver(gano)
        return montos

    def test_plana(self):
        self.assertEqual(self._jugar(Plana(10), [False, False, True]), [10, 10, 10])

    def test_martingala_dobla_y_vuelve(self):
        self.assertEqual(self._jugar(Martingala(1), [False, False, False, True, False]),
                         [1, 2, 4, 8, 1])

    def test_gran_martingala(self):
        self.assertEqual(self._jugar(GranMartingala(1), [False, False, True, True]),
                         [1, 3, 7, 1])

    def test_dalembert_no_baja_de_una_unidad(self):
        self.assertEqual(self._jugar(DAlembert(5), [True, False, False, True, True, True]),
                         [5, 5, 10, 15, 10, 5])

    def test_fibonacci_retrocede_dos(self):
        self.assertEqual(self._jugar(Fibonacci(1), [False] * 5 + [True, True, True]),
                         [1, 1, 2, 3, 5, 8, 3, 1])

    def test_labouchere(self):
        s = Labouchere(1)
        self.assertEqual(s.apuesta(), 7)            # 1 + 6
        s.resolver(True)
        self.assertEqual(s.lista, [2, 3, 4, 5])
        s.resolver(False)
        self.assertEqual(s.lista, [2, 3, 4, 5, 7])
        self.assertEqual(s.apuesta(), 9)
        for _ in range(3):
            s.resolver(True)
        self.assertEqual(s.lista, [])
        self.assertEqual(s.apuesta(), 7)            # arranca otra vez la lista

    def test_paroli_corta_la_racha_en_tres(self):
        self.assertEqual(self._jugar(Paroli(1), [True, True, True, True, False, True]),
                         [1, 2, 4, 1, 2, 1])

    def test_oscars_grind_no_apuesta_de_mas(self):
        s = OscarsGrind(1)
        self.assertEqual(self._jugar(s, [False, False, True, True]), [1, 1, 1, 2])
        # Lleva -2 + 1 + 2 = +1: cierra el ciclo y vuelve a la base.
        self.assertEqual(s.ganancia_ciclo, 0.0)
        self.assertEqual(s.apuesta(), 1)
        s2 = OscarsGrind(1)
        self._jugar(s2, [False, True, True])         # -1 + 1 = 0, ahora apuesta 2 pero falta 1
        self.assertEqual(s2.apuesta(), 1)

    def test_reiniciar_vuelve_al_estado_inicial(self):
        for clase in SISTEMAS.values():
            with self.subTest(sistema=clase.nombre):
                s = clase(2)
                inicial = s.apuesta()
                self._jugar(s, [False, False, True])
                s.reiniciar()
                self.assertEqual(s.apuesta(), inicial)

    def test_registro_completo(self):
        self.assertEqual(len(SISTEMAS), 8)


class MonteCarloTests(unittest.TestCase):
    def test_el_retorno_converge_a_la_ventaja_de_la_casa(self):
        cfg = Config(banca_inicial=1000, apuesta_base=1, tiradas=200, sesiones=2000, semilla=5)
        for nombre in ("plana", "dalembert"):
            with self.subTest(sistema=nombre):
                r = simular(nombre, cfg)
                self.assertAlmostEqual(r.retorno_sobre_apostado * 100, -ventaja_teorica(cfg),
                                       delta=1.0)
                self.assertEqual(len(r.bancas_finales), 2000)
                self.assertTrue((r.tiradas_jugadas <= 200).all())

    def test_la_martingala_se_arruina_mas_que_la_plana(self):
        # Con 7 derrotas seguidas la Martingala necesita 1270: se queda sin banca.
        cfg = Config(banca_inicial=1000, apuesta_base=10, tiradas=100, sesiones=500,
                     limite_mesa=100_000, semilla=9)
        plana = simular("plana", cfg)
        martingala = simular("martingala", cfg)
        self.assertGreater(martingala.prob_ruina, plana.prob_ruina)
        self.assertGreater(martingala.prob_terminar_ganando, plana.prob_terminar_ganando)
        self.assertTrue((martingala.bancas_finales >= 0).all())

    def test_objetivo_corta_la_sesion(self):
        cfg = Config(banca_inicial=100, apuesta_base=10, tiradas=1000, sesiones=200,
                     objetivo=120, semilla=2)
        r = simular("plana", cfg)
        self.assertGreater(r.alcanzaron_objetivo, 0)
        self.assertTrue((r.bancas_finales[r.bancas_finales > 100] >= 120).all())
        self.assertEqual(r.alcanzaron_objetivo + r.arruinadas +
                         int((r.tiradas_jugadas == 1000).sum()), 200)

    def test_sistema_o_apuesta_desconocidos(self):
        with self.assertRaises(ValueError):
            simular("milagro", Config())
        with self.assertRaises(ValueError):
            simular("plana", Config(apuesta="verde"))

    def test_ventaja_teorica(self):
        self.assertAlmostEqual(ventaja_teorica(Config(rueda=motor.FRANCESA)), 100 / 74, places=9)
        self.assertAlmostEqual(ventaja_teorica(Config(rueda=motor.AMERICANA, apuesta="pleno-00")),
                               100 / 19, places=9)


class SesgoTests(unittest.TestCase):
    def test_tiradas_necesarias(self):
        # Mismo numero que calcula la app web para el caso limite 1/36.
        self.assertEqual(sesgo.tiradas_necesarias(motor.EUROPEA), 692484)
        self.assertEqual(sesgo.tiradas_necesarias(motor.AMERICANA), 179713)
        self.assertLess(sesgo.tiradas_necesarias(motor.EUROPEA, p_real=0.05), 5000)
        self.assertLess(sesgo.tiradas_necesarias(motor.EUROPEA, corregir_multiplicidad=False),
                        sesgo.tiradas_necesarias(motor.EUROPEA))
        with self.assertRaises(ValueError):
            sesgo.tiradas_necesarias(motor.EUROPEA, p_real=1 / 37)
        self.assertEqual(sesgo.horas_de_mesa(400), 10.0)

    def test_registro_valida_casillas(self):
        with self.assertRaises(ValueError):
            sesgo.Registro(motor.EUROPEA, ["0", "37"])
        with self.assertRaises(ValueError):
            sesgo.Registro(motor.EUROPEA, ["00"])
        reg = sesgo.Registro(motor.EUROPEA, ["0", "32", "32", "26"])
        self.assertEqual(reg.n, 4)
        # Conteos en orden fisico: 0, 32 y 26 son vecinos en el disco.
        self.assertEqual(list(reg.frecuencias()[:2]), [1, 2])
        self.assertEqual(reg.frecuencias()[-1], 1)

    def test_cargar_csv_salta_cabecera_y_vacios(self):
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d) / "mesa.csv"
            with ruta.open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["resultado"])
                for v in ("0", "17", "", "00", "17"):
                    w.writerow([v])
            reg = sesgo.Registro.cargar_csv(ruta, motor.AMERICANA, mesa="Mesa 9")
            self.assertEqual(reg.tiradas, ["0", "17", "00", "17"])
            self.assertEqual(reg.mesa, "Mesa 9")
            ruta.write_text("tirada\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                sesgo.Registro.cargar_csv(ruta, motor.EUROPEA)

    def test_una_rueda_sana_no_muestra_sesgo_explotable(self):
        mesa = motor.Mesa(motor.EUROPEA, semilla=3)
        reg = sesgo.Registro(motor.EUROPEA, list(mesa.girar(20000)))
        ver = sesgo.analizar(reg)
        self.assertFalse(ver.hay_sesgo_explotable)
        self.assertFalse(ver.muestra_suficiente)
        self.assertGreater(ver.chi2.p_valor, 0.01)
        self.assertIn("MUESTRA INSUFICIENTE", ver.texto())

    def test_una_rueda_sesgada_se_detecta(self):
        inyectado = {c: 1.6 for c in motor.EUROPEA.vecinos("17", 1)}
        mesa = motor.Mesa(motor.EUROPEA, semilla=3, sesgo=inyectado)
        reg = sesgo.Registro(motor.EUROPEA, list(mesa.girar(50000)))
        ver = sesgo.analizar(reg)
        self.assertTrue(ver.hay_sesgo_explotable)
        self.assertEqual({f.casilla for f in ver.casillas_explotables}, set(inyectado))
        for fila in ver.casillas_explotables:
            self.assertGreater(fila.ic_bajo, 1 / 36)
            self.assertGreater(fila.ventaja_jugador, 0)
        self.assertIn("17", [s.centro for s in ver.sectores_significativos])
        self.assertLess(ver.chi2.p_valor, 1e-6)
        self.assertIn("Sesgo detectado", ver.texto())

    def test_por_sector_cubre_el_disco(self):
        reg = sesgo.Registro(motor.EUROPEA, ["0"] * 10)
        sectores = sesgo.por_sector(reg, radio=2)
        self.assertEqual(len(sectores), 37)
        self.assertEqual(sectores[0].casillas, ["3", "26", "0", "32", "15"])
        self.assertEqual(sectores[0].observado, 10)
        self.assertAlmostEqual(sectores[0].esperado, 10 * 5 / 37)


if __name__ == "__main__":
    unittest.main()
