# Azarium

Laboratorio estadístico de juegos de azar: quiniela, loto y ruleta.

Azarium **no predice resultados y no vende un sistema para ganar**. Hace lo
contrario: mide con rigor si existe alguna estructura explotable en los datos y
cuantifica exactamente cuánto cuesta jugar. Cuando la respuesta es "no hay
nada", lo muestra con números en vez de afirmarlo.

Corre con **numpy y la librería estándar**, nada más. Las distribuciones
estadísticas (gamma incompleta, normal, binomial exacta) están implementadas en
`azarium/stats.py` y validadas contra valores de tabla; los gráficos son SVG
