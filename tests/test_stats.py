"""Tests de las distribuciones y correcciones de `azarium.stats`.

Los valores de referencia son los de tabla (chi-cuadrado, normal) o se
calculan de forma independiente (binomial por suma directa, Wilson a mano).
"""

import math
import unittest

import numpy as np

from azarium import stats


class Chi2Tests(unittest.TestCase):
    def test_valores_de_tabla_al_5_por_ciento(self):
        # Cuantiles clasicos de la tabla chi-cuadrado para alfa = 0.05.
        for df, x in ((1, 3.841), (5, 11.070), (10, 18.307), (36, 50.998), (99, 123.225)):
            with self.subTest(df=df):
                self.assertAlmostEqual(stats.chi2_sf(x, df), 0.05, places=3)

    def test_valores_de_tabla_al_1_por_ciento(self):
        for df, x in ((1, 6.635), (5, 15.086), (36, 58.619)):
            with self.subTest(df=df):
                self.assertAlmostEqual(stats.chi2_sf(x, df), 0.01, places=3)

    def test_bordes(self):
        self.assertEqual(stats.chi2_sf(0, 5), 1.0)
        self.assertEqual(stats.chi2_sf(-3, 5), 1.0)
        self.assertLess(stats.chi2_sf(1000, 5), 1e-100)
        with self.assertRaises(ValueError):
            stats.chi2_sf(1.0, 0)

    def test_es_decreciente_en_x(self):
        valores = [stats.chi2_sf(x, 7) for x in np.linspace(0, 40, 50)]
        self.assertEqual(valores, sorted(valores, reverse=True))

    def test_gamma_q_cubre_las_dos_ramas(self):
        # x < a + 1 usa la serie; x >= a + 1 la fraccion continua. Ambas deben
        # pegar con la identidad Q(1, x) = exp(-x).
        for x in (0.3, 0.9, 2.5, 7.0):
            with self.subTest(x=x):
                self.assertAlmostEqual(stats.gamma_q(1.0, x), math.exp(-x), places=12)
        self.assertEqual(stats.gamma_q(2.0, 0.0), 1.0)
        with self.assertRaises(ValueError):
            stats.gamma_q(0.0, 1.0)
        with self.assertRaises(ValueError):
            stats.gamma_q(1.0, -1.0)


class NormalTests(unittest.TestCase):
    def test_colas_conocidas(self):
        self.assertAlmostEqual(stats.norm_sf(1.959964), 0.025, places=6)
        self.assertAlmostEqual(stats.norm_sf(2.575829), 0.005, places=6)
        self.assertAlmostEqual(stats.norm_sf(0.0), 0.5, places=12)
        self.assertAlmostEqual(stats.norm_cdf(0.0), 0.5, places=12)
        self.assertAlmostEqual(stats.norm_cdf(1.644854), 0.95, places=6)

    def test_sf_y_cdf_son_complementarias(self):
        for z in (-3.2, -0.7, 0.0, 1.1, 2.9):
            with self.subTest(z=z):
                self.assertAlmostEqual(stats.norm_sf(z) + stats.norm_cdf(z), 1.0, places=14)

    def test_ppf_valores_de_tabla(self):
        self.assertAlmostEqual(stats.norm_ppf(0.975), 1.959964, places=6)
        self.assertAlmostEqual(stats.norm_ppf(0.5), 0.0, places=12)
        self.assertAlmostEqual(stats.norm_ppf(0.001), -3.090232, places=6)
        self.assertAlmostEqual(stats.norm_ppf(0.8), 0.841621, places=6)

    def test_ppf_invierte_a_cdf(self):
        # Acklam promete error < 1.15e-9 en toda la recta, incluidas las colas.
        for p in (0.0001, 0.02, 0.3, 0.5, 0.77, 0.99, 0.99999):
            with self.subTest(p=p):
                self.assertAlmostEqual(stats.norm_cdf(stats.norm_ppf(p)), p, places=9)

    def test_ppf_rechaza_fuera_de_rango(self):
        for p in (0.0, 1.0, -0.1, 1.5):
            with self.subTest(p=p), self.assertRaises(ValueError):
                stats.norm_ppf(p)


class BinomialTests(unittest.TestCase):
    @staticmethod
    def _suma_directa(k, n, p):
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))

    def test_contra_suma_directa(self):
        casos = ((5, 10, 0.5), (30, 100, 0.27), (2, 37, 1 / 37), (700, 1000, 0.7), (1, 20, 0.05))
        for k, n, p in casos:
            with self.subTest(k=k, n=n, p=p):
                self.assertAlmostEqual(stats.binom_sf_ge(k, n, p), self._suma_directa(k, n, p),
                                       places=12)

    def test_moneda_justa(self):
        self.assertAlmostEqual(stats.binom_sf_ge(5, 10, 0.5), 0.623046875, places=12)

    def test_bordes(self):
        self.assertEqual(stats.binom_sf_ge(0, 10, 0.3), 1.0)
        self.assertEqual(stats.binom_sf_ge(-2, 10, 0.3), 1.0)
        self.assertEqual(stats.binom_sf_ge(11, 10, 0.3), 0.0)
        self.assertAlmostEqual(stats.binom_sf_ge(10, 10, 0.3), 0.3 ** 10, places=14)
        self.assertLessEqual(stats.binom_sf_ge(1, 5000, 0.9), 1.0)


class BenjaminiHochbergTests(unittest.TestCase):
    def test_caso_conocido(self):
        # Con 6 tests a FDR 5%, los umbrales son 0.0083, 0.0167, 0.025, 0.033, 0.042, 0.05.
        p = np.array([0.001, 0.01, 0.03, 0.04, 0.2, 0.5])
        np.testing.assert_array_equal(stats.benjamini_hochberg(p),
                                      [True, True, False, False, False, False])

    def test_el_corte_arrastra_a_los_menores(self):
        # El tercer p-valor pasa su umbral aunque el segundo no pase el suyo:
        # BH rechaza todos los que estan por debajo del mayor indice que pasa.
        p = np.array([0.001, 0.02, 0.024, 0.9])
        np.testing.assert_array_equal(stats.benjamini_hochberg(p), [True, True, True, False])

    def test_respeta_el_orden_original(self):
        p = np.array([0.5, 0.001, 0.9, 0.002])
        np.testing.assert_array_equal(stats.benjamini_hochberg(p), [False, True, False, True])

    def test_sin_senal_no_rechaza_nada(self):
        p = np.linspace(0.1, 1.0, 50)
        self.assertFalse(stats.benjamini_hochberg(p).any())

    def test_senal_fuerte_rechaza_todo(self):
        p = np.full(37, 1e-6)
        self.assertTrue(stats.benjamini_hochberg(p).all())

    def test_vacio(self):
        self.assertEqual(stats.benjamini_hochberg(np.array([])).size, 0)


class WilsonTests(unittest.TestCase):
    def test_valor_conocido(self):
        bajo, alto = stats.ic_wilson(30, 100)
        self.assertAlmostEqual(bajo, 0.21895, places=4)
        self.assertAlmostEqual(alto, 0.39585, places=4)

    def test_contiene_la_proporcion_observada(self):
        for exitos, n in ((0, 10), (3, 10), (10, 10), (540, 20000)):
            with self.subTest(exitos=exitos, n=n):
                bajo, alto = stats.ic_wilson(exitos, n)
                self.assertLessEqual(bajo, exitos / n)
                self.assertGreaterEqual(alto, exitos / n)
                self.assertGreaterEqual(bajo, 0.0)
                self.assertLessEqual(alto, 1.0)

    def test_sin_muestra(self):
        self.assertEqual(stats.ic_wilson(0, 0), (0.0, 1.0))

    def test_mas_muestra_mas_angosto(self):
        ancho = lambda n: np.subtract(*stats.ic_wilson(n // 2, n)[::-1])
        self.assertGreater(ancho(10), ancho(100))
        self.assertGreater(ancho(100), ancho(10000))

    def test_mas_confianza_mas_ancho(self):
        self.assertLess(stats.ic_wilson(30, 100, conf=0.90)[1],
                        stats.ic_wilson(30, 100, conf=0.99)[1])


class ResultadoTests(unittest.TestCase):
    def test_significancia_y_lectura(self):
        r = stats.Resultado("t", 1.0, 0.03)
        self.assertTrue(r.significativo)
        self.assertIn("rechaza", r.lectura())
        self.assertEqual(stats.Resultado("t", 1.0, 0.03).lectura(alfa=0.01),
                         "Compatible con azar puro")
        self.assertFalse(stats.Resultado("t", 1.0, 0.5).significativo)


if __name__ == "__main__":
    unittest.main()
