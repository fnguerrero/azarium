"""Tests del modulo de sorteos: modelo y CSV, generador sintetico, bateria de
tests de aleatoriedad y backtest sin look-ahead."""

import datetime as dt
import tempfile
import unittest
from pathlib import Path

import numpy as np

from azarium.sorteos import JUEGOS, LOTO, QUINI6, QUINIELA, Historico
from azarium.sorteos import analisis as A
from azarium.sorteos import backtest as B
from azarium.sorteos.fuentes import cargar, generar_sintetico


class JuegoTests(unittest.TestCase):
    def test_definiciones(self):
        self.assertEqual(QUINIELA.cardinalidad, 100)
        self.assertEqual(LOTO.cardinalidad, 42)
        self.assertEqual(QUINI6.cardinalidad, 46)
        self.assertEqual(list(LOTO.valores[:3]), [0, 1, 2])
        self.assertAlmostEqual(QUINIELA.prob_por_numero(), 0.01)
        self.assertEqual(set(JUEGOS), {"quiniela", "loto", "quini6"})


class HistoricoTests(unittest.TestCase):
    def _escribir(self, carpeta, texto):
        ruta = Path(carpeta) / "h.csv"
        ruta.write_text(texto, encoding="utf-8")
        return ruta

    def test_frecuencias_y_tamanos(self):
        h = Historico(QUINIELA, resultados=np.array([[0, 5, 5], [99, 0, 5]]))
        self.assertEqual(h.n_sorteos, 2)
        self.assertEqual(h.n_extracciones, 6)
        f = h.frecuencias()
        self.assertEqual(f.size, 100)
        self.assertEqual((f[0], f[5], f[99]), (2, 3, 1))
        self.assertEqual(f.sum(), 6)
        self.assertEqual(list(h.plano()), [0, 5, 5, 99, 0, 5])

    def test_historico_vacio(self):
        h = Historico(LOTO)
        self.assertEqual(h.n_sorteos, 0)
        self.assertEqual(h.rango_fechas(), (None, None))
        self.assertEqual(h.anios_cubiertos(), 0.0)
        with self.assertRaises(ValueError):
            Historico(LOTO, resultados=np.array([1, 2, 3]))

    def test_fechas(self):
        fechas = [dt.date(2020, 1, 1), dt.date(2022, 1, 1), dt.date(2021, 6, 1)]
        h = Historico(LOTO, fechas=fechas, resultados=np.zeros((3, 6), dtype=int))
        self.assertEqual(h.rango_fechas(), (dt.date(2020, 1, 1), dt.date(2022, 1, 1)))
        self.assertAlmostEqual(h.anios_cubiertos(), 2.0, places=2)

    def test_csv_ida_y_vuelta(self):
        with tempfile.TemporaryDirectory() as d:
            original = generar_sintetico(LOTO, 30, semilla=1, desde=dt.date(2024, 3, 1))
            ruta = original.guardar_csv(Path(d) / "sub" / "loto.csv")
            self.assertTrue(ruta.exists())
            leido = cargar(ruta, LOTO)
            np.testing.assert_array_equal(leido.resultados, original.resultados)
            self.assertEqual(leido.fechas, original.fechas)
            self.assertEqual(leido.fechas[0], dt.date(2024, 3, 1))

    def test_csv_sin_fecha_y_con_bom(self):
        with tempfile.TemporaryDirectory() as d:
            ruta = self._escribir(d, "﻿n1,n2,n3\n37,12,89\n\n4,71,23\n")
            h = cargar(ruta, QUINIELA)
            self.assertEqual(h.n_sorteos, 2)
            self.assertEqual(h.fechas, [])
            self.assertEqual(list(h.resultados[1]), [4, 71, 23])

    def test_csv_formatos_de_fecha(self):
        with tempfile.TemporaryDirectory() as d:
            ruta = self._escribir(d, "fecha,n1\n2016-01-04,1\n05/01/2016,2\n06-01-2016,3\n2016/01/07,4\n")
            h = cargar(ruta, QUINIELA)
            self.assertEqual([f.day for f in h.fechas], [4, 5, 6, 7])
            ruta = self._escribir(d, "fecha,n1\nayer,1\n")
            with self.assertRaisesRegex(ValueError, "fecha no reconocida"):
                cargar(ruta, QUINIELA)

    def test_csv_invalidos(self):
        with tempfile.TemporaryDirectory() as d:
            casos = (
                ("n1,n2\n1,2\n1,200\n", "fuera del rango"),
                ("n1,n2\n1,2\n3\n", "misma cantidad"),
                ("n1,n2\n1,x\n", "numero invalido"),
                ("n1,n2\n", "no contiene sorteos"),
                ("", "esta vacio"),
            )
            for texto, mensaje in casos:
                with self.subTest(texto=texto):
                    ruta = self._escribir(d, texto)
                    with self.assertRaisesRegex(ValueError, mensaje):
                        cargar(ruta, QUINIELA)


class SinteticoTests(unittest.TestCase):
    def test_forma_y_rango(self):
        h = generar_sintetico(QUINIELA, 50, semilla=0)
        self.assertEqual(h.resultados.shape, (50, 20))
        self.assertTrue(((h.resultados >= 0) & (h.resultados <= 99)).all())
        self.assertEqual(len(h.fechas), 50)
        self.assertEqual(h.fuente, "sintetico-uniforme")

    def test_sin_reposicion_no_repite_en_un_sorteo(self):
        h = generar_sintetico(LOTO, 200, semilla=4)
        for fila in h.resultados:
            self.assertEqual(len(set(fila.tolist())), 6)

    def test_reproducible(self):
        a = generar_sintetico(QUINIELA, 10, semilla=11).resultados
        b = generar_sintetico(QUINIELA, 10, semilla=11).resultados
        np.testing.assert_array_equal(a, b)

    def test_sesgo_inyectado_se_nota(self):
        h = generar_sintetico(QUINIELA, 2000, semilla=2, sesgo={7: 3.0})
        f = h.frecuencias()
        self.assertGreater(f[7], 2 * np.median(f))
        self.assertIn("sesgado", h.fuente)
        with self.assertRaises(ValueError):
            generar_sintetico(QUINIELA, 10, sesgo={100: 2.0})
        with self.assertRaises(ValueError):
            generar_sintetico(QUINIELA, 10, sesgo={1: -1.0})


class AnalisisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.uniforme = generar_sintetico(QUINIELA, 3650, semilla=42)

    def test_bateria_sobre_azar_puro_no_rechaza(self):
        resultados = A.bateria(self.uniforme)
        self.assertEqual(len(resultados), 5)
        for r in resultados:
            with self.subTest(test=r.nombre):
                self.assertFalse(np.isnan(r.p_valor), r.detalle)
                self.assertGreater(r.p_valor, 0.001)

    def test_chi2_detecta_un_sesgo_grosero(self):
        sesgado = generar_sintetico(QUINIELA, 3650, semilla=42, sesgo={7: 1.5})
        self.assertLess(A.chi2_uniformidad(sesgado).p_valor, 1e-6)
        fila7 = A.por_numero(sesgado)[7]
        self.assertTrue(fila7.significativo_fdr)
        self.assertGreater(fila7.desvio_z, 5)

    def test_por_numero(self):
        filas = A.por_numero(self.uniforme)
        self.assertEqual(len(filas), 100)
        self.assertEqual([f.numero for f in filas][:3], [0, 1, 2])
        self.assertEqual(sum(f.observado for f in filas), 3650 * 20)
        for f in filas:
            self.assertLessEqual(f.ic_bajo, f.observado / (3650 * 20))
            self.assertTrue(0 <= f.p_valor <= 1)
        # Con FDR nunca puede haber mas rechazos que sin corregir.
        self.assertLessEqual(sum(f.significativo_fdr for f in filas),
                             sum(f.significativo_crudo for f in filas))

    def test_rankear_promedia_empates(self):
        np.testing.assert_array_equal(A._rankear(np.array([10, 30, 20, 30])),
                                      [1.0, 3.5, 2.0, 3.5])

    def test_persistencia_exige_muestra(self):
        with self.assertRaises(ValueError):
            A.persistencia_calientes(generar_sintetico(QUINIELA, 10))

    def test_rachas_exige_ambas_paridades(self):
        h = Historico(QUINIELA, resultados=np.full((5, 3), 2))
        with self.assertRaises(ValueError):
            A.rachas_paridad(h)

    def test_serial_insuficiente_devuelve_nan(self):
        r = A.test_serial(generar_sintetico(QUINIELA, 5, semilla=1))
        self.assertTrue(np.isnan(r.p_valor))
        self.assertIn("insuficiente", r.detalle)

    def test_gaps(self):
        with self.assertRaises(ValueError):
            A.test_gaps(self.uniforme, 100)
        with self.assertRaises(ValueError):
            A.test_gaps(generar_sintetico(QUINIELA, 30, semilla=1), 0)
        r = A.test_gaps(self.uniforme, 7)
        self.assertGreater(r.df, 1)
        self.assertIn("numero 7", r.detalle)

    def test_autocorrelacion(self):
        acf, banda = A.autocorrelacion(self.uniforme, max_lag=10)
        self.assertEqual(acf.size, 10)
        self.assertAlmostEqual(banda, 1.959964 / np.sqrt(3650 * 20), places=9)
        self.assertLess(np.abs(acf).max(), 3 * banda)


class BacktestTests(unittest.TestCase):
    def test_esperanza_teorica(self):
        h = Historico(QUINIELA)
        self.assertAlmostEqual(B.esperanza_teorica(h, "cabeza"), -0.30)
        self.assertAlmostEqual(B.esperanza_teorica(h, "a20"),
                               (1 - 0.99 ** 20) * 3.5 - 1)

    def test_numeros_fijos_sobre_un_historico_conocido(self):
        # Sorteos de 3 bolillas: en los impares sale 0 a la cabeza, en los pares no.
        filas = [[0, 1, 2] if t % 2 else [1, 2, 0] for t in range(20)]
        h = Historico(QUINIELA, resultados=np.array(filas))
        r = B.correr(h, B._fabricar_fijos([0]), "fijo", k=1, calentamiento=10)
        self.assertEqual((r.apuestas, r.aciertos), (10, 5))
        self.assertEqual(r.unidades_cobradas, 5 * B.PAGO_CABEZA)
        self.assertAlmostEqual(r.retorno, (5 * 70 - 10) / 10)
        self.assertAlmostEqual(r.tasa_acierto, 0.5)
        a20 = B.correr(h, B._fabricar_fijos([0]), "fijo", calentamiento=10, modalidad="a20")
        self.assertEqual(a20.aciertos, 10)
        self.assertAlmostEqual(a20.retorno, 3.5 - 1)

    def test_el_selector_no_ve_el_sorteo_actual(self):
        h = generar_sintetico(QUINIELA, 40, semilla=3)
        vistos = []

        def espia(previos, hist, k):
            vistos.append(previos.shape[0])
            return [0]

        B.correr(h, espia, "espia", calentamiento=30)
        self.assertEqual(vistos, list(range(30, 40)))

    def test_calentamiento_insuficiente(self):
        with self.assertRaises(ValueError):
            B.correr(generar_sintetico(QUINIELA, 10), B.sel_calientes, "x", calentamiento=10)
        with self.assertRaises(ValueError):
            B.correr(generar_sintetico(QUINIELA, 10), B.sel_calientes, "x",
                     calentamiento=1, modalidad="redoblona")

    def test_selectores(self):
        h = Historico(QUINIELA, resultados=np.array([[5, 5, 1], [5, 2, 2], [9, 9, 9]]))
        previos = h.resultados
        self.assertEqual(B.sel_calientes(previos, h, 2), [5, 9])
        self.assertEqual(B.sel_frios(previos, h, 1), [0])
        self.assertNotIn(B.sel_atrasados(previos, h, 1)[0], {1, 2, 5, 9})
        self.assertEqual(B.sel_ultimo(previos, h, 2), [9])
        # Tras los 96 que nunca salieron viene el 1, que salio solo en el primer sorteo.
        self.assertEqual(B.sel_atrasados(previos, h, 97)[-1], 1)

    def test_comparar_devuelve_todas_las_estrategias(self):
        h = generar_sintetico(LOTO, 120, semilla=8)
        res = B.comparar(h, k=2, calentamiento=100, modalidad="a20")
        self.assertEqual(len(res), 6)
        for r in res:
            self.assertEqual(r.apuestas, 40)
            self.assertEqual(r.modalidad, "a20")
        aleatorio = next(r for r in res if r.estrategia == "Aleatorio")
        self.assertEqual(B.diferencia_significativa(aleatorio, aleatorio), 1.0)
        self.assertTrue(np.isnan(B.diferencia_significativa(
            aleatorio, B.ResultadoBacktest("x", 0, 0, 0.0, 0.0, "a20"))))
        self.assertEqual(B.ResultadoBacktest("x", 0, 0, 0.0, 0.0, "a20").retorno, 0.0)


if __name__ == "__main__":
    unittest.main()
