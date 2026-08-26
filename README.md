# Azarium

Laboratorio estadístico de juegos de azar: quiniela, loto y ruleta.

Azarium **no predice resultados y no vende un sistema para ganar**. Hace lo
contrario: mide con rigor si existe alguna estructura explotable en los datos y
cuantifica exactamente cuánto cuesta jugar. Cuando la respuesta es "no hay
nada", lo muestra con números en vez de afirmarlo.

Corre con **numpy y la librería estándar**, nada más. Las distribuciones
estadísticas (gamma incompleta, normal, binomial exacta) están implementadas en
`azarium/stats.py` y validadas contra valores de tabla; los gráficos son SVG
generados a mano.

## La app

`app/azarium.html` es la aplicación completa: los tres módulos en una sola página web,
sin comandos. Es un único archivo autocontenido — se abre con doble clic en cualquier
navegador, anda en el celular igual que en la computadora y no necesita Python ni
servidor. El motor estadístico está portado a JavaScript y corre en el dispositivo:
no se envía nada a ningún lado.

Tiene cuatro secciones:

- **Mesa en vivo** — tocás los números que van saliendo y el análisis se recalcula con
  cada tirada: chi-cuadrado, casillas y sectores físicos con corrección por
  multiplicidad, y un veredicto de si hay algo explotable. Incluye la barra de *poder
  de la evidencia*, que muestra qué proporción de las tiradas necesarias llevás.
- **Simulador** — los ocho sistemas de apuesta sobre las condiciones que elijas.
- **Sorteos** — batería de tests y backtest sobre un histórico de quiniela o loto,
  pegado a mano o cargado desde un CSV.
- **Referencia** — ventajas de la casa, comparación entre juegos y cómo leer los números.

Las tiradas se guardan solas. Publicada como artifact usa el almacenamiento sincronizado
de la plataforma (se comparte entre la computadora y el celular); abierta como archivo
local cae a `localStorage`. En ambos casos hay exportación por texto y descarga de CSV.

El paquete Python de abajo sigue siendo el motor de referencia: la app JavaScript
reproduce sus resultados exactamente (verificado contra chi², normal, binomial exacta,
Wilson, esperanzas de las tres ruedas y retornos de quiniela).

## Instalación (paquete Python)

No hay instalación. Requiere Python 3.11+ y numpy.

```bash
py -3 -c "import numpy; print(numpy.__version__)"
```

Se invoca por el lanzador `azarium-cli.py`, que resuelve su propia ubicación y
funciona desde cualquier consola sin importar en qué carpeta estés parado:

```bash
py -3 "W:\Working Folder Personal\Azarium\azarium-cli.py" informe
```

Estando parado dentro de la carpeta del proyecto también sirve la forma de
módulo, `py -3 -m azarium.cli <comando>`.

## Uso

Todos los comandos generan un informe HTML autocontenido en `informes/`.

### Análisis de sorteos

```bash
py -3 "W:\Working Folder Personal\Azarium\azarium-cli.py" sorteos --juego quiniela --csv datos/quiniela.csv
```

Sin `--csv` genera un histórico sintético uniforme para que puedas ver el
funcionamiento (lo avisa por consola). El CSV real debe tener una columna
`fecha` opcional y luego una columna por bolilla, un sorteo por fila:

```csv
fecha,n1,n2,n3
2016-01-04,37,12,89
2016-01-05,4,71,23
```

Juegos disponibles: `quiniela` (00-99, 20 bolillas), `loto` (00-41, 6),
`quini6` (00-45, 6).

### Simulación de ruleta

```bash
py -3 "W:\Working Folder Personal\Azarium\azarium-cli.py" ruleta --rueda europea --apuesta rojo --banca 1000 --base 10 --tiradas 500 --sesiones 10000
```

Compara ocho sistemas de progresión (plana, Martingala, Gran Martingala,
D'Alembert, Fibonacci, Labouchère, Paroli, Oscar's Grind) sobre las mismas
condiciones.

### Detección de sesgo de rueda

```bash
py -3 "W:\Working Folder Personal\Azarium\azarium-cli.py" sesgo --csv datos/mesa3.csv --rueda europea --mesa "Mesa 3"
```

El CSV es una tirada por fila (`0`, `00`, `1`..`36`). Para ver cómo se comporta
con una rueda defectuosa simulada:

```bash
py -3 "W:\Working Folder Personal\Azarium\azarium-cli.py" sesgo --demo-sesgo 17 --tiradas 50000
```

### Tabla rápida de ventajas

```bash
py -3 "W:\Working Folder Personal\Azarium\azarium-cli.py" ventajas
```

## Qué mide cada módulo

### 1. Sorteos (`azarium/sorteos/`)

| Test | Qué responde |
|---|---|
| Chi-cuadrado de uniformidad | ¿Todos los números salen igual de seguido? |
| Test por número + FDR | ¿Qué números se desvían *de verdad*, corrigiendo por multiplicidad? |
| Persistencia de calientes | Los que más salieron en la primera mitad, ¿siguen saliendo más en la segunda? |
| Rachas (Wald-Wolfowitz) | ¿Hay dependencia serial en la paridad? |
| Test serial | ¿El sorteo anterior informa sobre el siguiente? |
| Test de huecos | ¿La espera entre apariciones es geométrica (sin memoria)? |
| Autocorrelación | ¿Hay memoria a lag 1..20? |
| Backtest sin look-ahead | ¿Cuánto rinde jugar calientes, fríos, atrasados o al azar? |

La corrección por **multiplicidad** es el punto que casi todos los "sistemas"
omiten: al testear 100 números a α=0.05, cinco dan significativo por puro azar.
Azarium muestra las dos columnas, cruda y corregida.

### 2. Ruleta (`azarium/ruleta/`)

Esperanza calculada en **aritmética racional exacta** (`fractions.Fraction`),
así el resultado no depende de redondeo:

| Rueda | Ventaja de la casa |
|---|---|
| Europea (un cero) | 1/37 = 2.703% en toda apuesta |
| Americana (doble cero) | 2/38 = 5.263% |
| Francesa (con *partage*) | 1/74 = 1.351% solo en apuestas de dinero par |

El Monte Carlo mide, para cada sistema: probabilidad de ruina, resultado medio
y mediano, drawdown máximo, percentiles y **retorno sobre el volumen apostado**.
Esa última métrica converge a la ventaja de la casa en todos los sistemas — es
la demostración empírica de que ninguna progresión cambia la esperanza.

### 3. Sesgo de rueda (`azarium/ruleta/sesgo.py`)

El único enfoque con base histórica real (Jagger, 1873; Hibbs y Walford, 1947;
los Pelayo en los 90). Aporta tres cosas que los análisis ingenuos no tienen:

- **Análisis por sector físico**: usa el orden real de las casillas en el disco,
  no el numérico. Un defecto mecánico afecta a una zona contigua del plato, así
  que sumar vecinos concentra la señal y multiplica la potencia del test.
- **Umbral de explotabilidad**: no alcanza con que una casilla "salga más".
  Como el pleno paga 35 a 1, la probabilidad real tiene que superar 1/36 =
  2.778% frente al 2.703% teórico. Azarium exige que el *límite inferior* del
  IC 95% pase ese umbral antes de declarar nada explotable.
- **Cálculo de potencia**: `tiradas_necesarias()` responde cuánta observación
  hace falta. Para el sesgo marginal, con 80% de potencia y corrección por las
  37 casillas: **~692.000 tiradas de la misma rueda**, unas 17.300 horas de mesa
  continua, casi 6 años observando 8 horas por día.

## Estructura

```
app/
  azarium.html          la aplicación web completa, un solo archivo
azarium/
  stats.py              distribuciones y correcciones por multiplicidad
  reporte.py            informes HTML + gráficos SVG
  cli.py                interfaz de línea de comandos
  sorteos/
    modelo.py           Juego, Historico, carga/guardado CSV
    fuentes.py          ingesta CSV local/remota y generador sintético
    analisis.py         batería de tests de aleatoriedad
    backtest.py         estrategias evaluadas sin look-ahead
  ruleta/
    motor.py            ruedas, orden físico, apuestas, esperanza exacta
    sistemas.py         ocho progresiones de apuesta
    montecarlo.py       simulación de sesiones y métricas de banca
    sesgo.py            detección de sesgo mecánico y análisis de potencia
```

## Validación

El generador sintético permite verificar que los tests están calibrados:

- **Control negativo** (azar puro, 10 años de quiniela): ningún test rechaza la
  uniformidad; exactamente 5 de 100 números dan p<0.05 sin corregir, 0 tras FDR.
- **Control positivo** (mismo tamaño, un número con peso ×1.35): el chi-cuadrado
  lo detecta con p≈4e-5 y el FDR aísla exactamente el número sesgado.

```bash
py -3 "W:\Working Folder Personal\Azarium\azarium-cli.py" sesgo --demo-sesgo 17 --tiradas 50000
```

## Sobre datos reales

Azarium no scrapea sitios de loterías: el maquetado cambia sin aviso y un
scraper roto que devuelve datos parciales es peor que no tener datos. La vía
soportada es CSV, local (`--csv`) o por URL (`--url`). Si conseguís el histórico
en otro formato, convertirlo a CSV es el único paso previo.
