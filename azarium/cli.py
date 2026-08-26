"""Interfaz de linea de comandos de Azarium."""

from __future__ import annotations

import argparse
import sys
from html import escape as html_escape
from pathlib import Path

import numpy as np

from . import reporte
from .reporte import Informe
from .ruleta import motor, sesgo as msesgo
from .ruleta.montecarlo import Config, comparar as comparar_sistemas, ventaja_teorica
from .ruleta.sistemas import SISTEMAS
from .sorteos import JUEGOS, Historico
from .sorteos import analisis as A
from .sorteos import backtest as B
from .sorteos.fuentes import cargar, descargar_csv, generar_sintetico

RAIZ = Path(__file__).resolve().parent.parent


def _pct(v: float, invertir: bool = False) -> str:
    clase = "bien" if (v > 0) != invertir else "mal"
    return f'<span class="{clase}">{v:+.2f}%</span>'


def _obtener_historico(args) -> Historico:
    juego = JUEGOS[args.juego]
    if args.csv:
        return cargar(args.csv, juego)
    if args.url:
        destino = RAIZ / "datos" / f"{args.juego}.csv"
        return descargar_csv(args.url, juego, destino)
    print(f"[i] Sin fuente de datos real: se generan {args.sorteos} sorteos sinteticos "
          f"uniformes (semilla {args.semilla}).", file=sys.stderr)
    print("[i] Para analizar datos reales use --csv RUTA o --url URL.", file=sys.stderr)
    return generar_sintetico(juego, args.sorteos, semilla=args.semilla)


# ------------------------------------------------------------------- sorteos

def cmd_sorteos(args) -> int:
    hist = _obtener_historico(args)
    juego = hist.juego
    inf = Informe(
        f"Azarium - Analisis de {juego.nombre}",
        f"{hist.n_sorteos} sorteos, {hist.n_extracciones} numeros extraidos, "
        f"fuente: {hist.fuente}")

    inf.h2("Que se esta testeando")
    inf.p("La hipotesis nula es que cada sorteo es independiente de los anteriores y "
          "que todos los numeros son equiprobables. Si esa hipotesis se rechaza, hay "
          "estructura y podria explotarse. Si no se rechaza, no existe ninguna "
          "seleccion de numeros mejor que otra.")

    resultados = A.bateria(hist)
    filas = [[r.nombre,
              "-" if np.isnan(r.estadistico) else f"{r.estadistico:.3f}",
              "-" if r.df is None else str(r.df),
              "-" if np.isnan(r.p_valor) else f"{r.p_valor:.4g}",
              f'<span class="{"mal" if r.significativo else "bien"}">{r.lectura()}</span>']
             for r in resultados]
    inf.h2("Bateria de tests de aleatoriedad")
    inf.tabla(["Test", "Estadistico", "gl", "p-valor", "Conclusion"], filas)
    for r in resultados:
        if r.detalle:
            inf.p(f'<small>{r.nombre}: {r.detalle}</small>')

    # Frecuencias
    frec = hist.frecuencias()
    n = int(frec.sum())
    p = 1 / juego.cardinalidad
    esp = n * p
    sigma = float(np.sqrt(n * p * (1 - p)))
    inf.h2("Frecuencia de cada numero")
    inf.svg(reporte.grafico_frecuencias(frec, juego.minimo, esp, sigma,
                                        f"Apariciones por numero - {juego.nombre}"))

    fnum = A.por_numero(hist)
    crudos = sum(f.significativo_crudo for f in fnum)
    con_fdr = sum(f.significativo_fdr for f in fnum)
    inf.destacado(
        f"<strong>La trampa de los numeros calientes.</strong> "
        f"{crudos} numeros dan diferencia significativa si se los mira de a uno "
        f"(p &lt; 0.05). Es exactamente lo que el azar produce: al correr "
        f"{juego.cardinalidad} tests a la vez, ~{juego.cardinalidad * 0.05:.0f} dan "
        f"positivo por casualidad. Corrigiendo por multiplicidad (Benjamini-Hochberg), "
        f"quedan <strong>{con_fdr}</strong>.")

    top = sorted(fnum, key=lambda f: -f.observado)[:10]
    inf.h3("Los 10 mas frecuentes")
    inf.tabla(["Numero", "Salio", "Esperado", "Desvio z", "p-valor", "Significativo (FDR)"],
              [[str(f.numero), str(f.observado), f"{f.esperado:.1f}", f"{f.desvio_z:+.2f}",
                f"{f.p_valor:.3f}",
                f'<span class="{"mal" if f.significativo_fdr else "bien"}">'
                f'{"SI" if f.significativo_fdr else "no"}</span>'] for f in top])

    # Autocorrelacion
    acf, banda = A.autocorrelacion(hist, max_lag=args.lags)
    inf.h2("Memoria entre sorteos")
    inf.svg(reporte.grafico_acf(acf, banda, "Autocorrelacion por lag"))
    fuera = int((np.abs(acf) > banda).sum())
    inf.p(f"{fuera} de {len(acf)} lags caen fuera de la banda del 95%. "
          f"Bajo azar puro se esperan ~{len(acf) * 0.05:.1f}.")

    # Backtest
    inf.h2("Backtest: cuanto rinde cada estrategia")
    inf.p("Cada estrategia elige numeros usando <em>solo</em> los sorteos anteriores "
          "y se la evalua sobre el sorteo siguiente. Sin esa regla, cualquier "
          "estrategia parece ganadora.")
    for modalidad, etiqueta, pago in (("cabeza", "A la cabeza", B.PAGO_CABEZA),
                                      ("a20", "A los 20", B.PAGO_A_LOS_20)):
        try:
            res = B.comparar(hist, k=args.k, calentamiento=args.calentamiento,
                             modalidad=modalidad, semilla=args.semilla)
        except ValueError as exc:
            inf.p(f"<em>{etiqueta}: no se pudo evaluar ({exc}).</em>")
            continue
        teorico = B.esperanza_teorica(hist, modalidad) * 100
        inf.h3(f"{etiqueta} (paga {pago} por unidad) - retorno teorico {teorico:+.2f}%")
        inf.tabla(["Estrategia", "Apuestas", "Aciertos", "Retorno por unidad"],
                  [[r.estrategia, str(r.apuestas), str(r.aciertos), _pct(r.retorno * 100)]
                   for r in res])
        aleat = next(r for r in res if r.estrategia == "Aleatorio")
        comparaciones = [(r.estrategia, B.diferencia_significativa(r, aleat))
                         for r in res if r is not aleat]
        peor = min(c[1] for c in comparaciones)
        inf.p(f"Comparando cada estrategia contra elegir al azar, el p-valor mas bajo "
              f"es {peor:.3f}. " +
              ("Ninguna diferencia es estadisticamente significativa: las brechas de "
               "retorno que se ven en la tabla son ruido muestral."
               if peor >= 0.05 else
               "Hay al menos una diferencia significativa; conviene revisar los datos "
               "de origen."))

    inf.destacado(
        "<strong>Conclusion.</strong> El retorno esperado esta fijado por el pago, no "
        "por los numeros elegidos. Con un pago de 70 por unidad sobre 100 numeros "
        "posibles, la banca se queda con el 30% de todo lo apostado, juegue uno los "
        "calientes, los frios, los atrasados o la fecha de su cumpleanos.")

    destino = Path(args.salida or RAIZ / "informes" / f"sorteos-{args.juego}.html")
    inf.guardar(destino)
    print(f"Informe generado: {destino}")
    return 0


# -------------------------------------------------------------------- ruleta

def cmd_ruleta(args) -> int:
    rueda = motor.RUEDAS[args.rueda]
    cfg = Config(banca_inicial=args.banca, apuesta_base=args.base, tiradas=args.tiradas,
                 sesiones=args.sesiones, limite_mesa=args.limite, rueda=rueda,
                 apuesta=args.apuesta, semilla=args.semilla,
                 objetivo=args.objetivo)
    inf = Informe("Azarium - Simulacion de ruleta",
                  f"{rueda.nombre}, apuesta a {args.apuesta}, banca {args.banca:,.0f}, "
                  f"unidad {args.base:,.0f}, {args.tiradas} tiradas, "
                  f"{args.sesiones:,} sesiones simuladas")

    inf.h2("La ventaja de la casa no depende de la apuesta")
    inf.tabla(["Apuesta", "Ventaja de la casa", "Esperanza por unidad"],
              [[nombre, f'<span class="mal">{v:.3f}%</span>', f"<code>{ev}</code>"]
               for nombre, v, ev in motor.tabla_ventajas(rueda)])
    inf.p("Todas las apuestas de la mesa tienen la misma esperanza. Cambian la "
          "varianza (el pleno paga mucho y casi nunca; el rojo paga poco y seguido), "
          "nunca la media.")

    inf.h2("Que le pasa a la banca con cada sistema")
    resumenes = comparar_sistemas(cfg)
    teorico = ventaja_teorica(cfg)
    inf.tabla(["Sistema", "Resultado medio", "Mediana", "Prob. de ruina",
               "Termina ganando", "Retorno / apostado"],
              [[r.sistema, f"{r.resultado_medio:+,.0f}",
                f"{r.resultado_mediano:+,.0f}",
                f'<span class="mal">{r.prob_ruina * 100:.1f}%</span>',
                f"{r.prob_terminar_ganando * 100:.1f}%",
                f'<span class="mal">{r.retorno_sobre_apostado * 100:+.3f}%</span>']
               for r in resumenes])

    inf.destacado(
        f"<strong>El resultado central.</strong> La ultima columna es el resultado neto "
        f"dividido por todo el dinero que paso por la mesa. Converge a "
        f"<strong>-{teorico:.3f}%</strong> en todos los sistemas, porque es el unico "
        f"numero que la matematica fija. Lo que cambian las progresiones es el reparto: "
        f"la Martingala hace que la mayoria de las sesiones terminen en verde y una "
        f"minoria pierda absolutamente todo.")

    peor = max(resumenes, key=lambda r: r.prob_ruina)
    mejor_ruina = min(resumenes, key=lambda r: r.prob_ruina)
    inf.p(f"Probabilidad de perder la banca entera: {peor.sistema} "
          f"{peor.prob_ruina * 100:.1f}%, contra {mejor_ruina.sistema} "
          f"{mejor_ruina.prob_ruina * 100:.1f}%. La diferencia no es que uno gane: "
          f"es cuanto tarda en perder.")

    inf.h2("Distribucion de resultados")
    for r in resumenes:
        if r.sistema in ("Plana", "Martingala", "D'Alembert"):
            inf.h3(r.sistema)
            inf.svg(reporte.grafico_histograma(
                r.bancas_finales, cfg.banca_inicial,
                f"Banca final tras {cfg.tiradas} tiradas - {r.sistema}"))
            pc = r.percentiles()
            inf.p("Percentiles de la banca final: " +
                  ", ".join(f"p{k}={v:,.0f}" for k, v in pc.items()))

    destino = Path(args.salida or RAIZ / "informes" / "ruleta-sistemas.html")
    inf.guardar(destino)
    print(f"Informe generado: {destino}")
    return 0


# --------------------------------------------------------------------- sesgo

def cmd_sesgo(args) -> int:
    rueda = motor.RUEDAS[args.rueda]
    if args.csv:
        reg = msesgo.Registro.cargar_csv(args.csv, rueda, mesa=args.mesa)
    else:
        inyectado = {}
        if args.demo_sesgo:
            centro = args.demo_sesgo
            inyectado = {c: 1.6 for c in rueda.vecinos(centro, 1)}
            print(f"[i] Demo: se simula una rueda con el sector {inyectado} defectuoso.",
                  file=sys.stderr)
        else:
            print("[i] Demo: se simula una rueda sana. Use --csv para datos reales.",
                  file=sys.stderr)
        mesa = motor.Mesa(rueda, semilla=args.semilla, sesgo=inyectado or None)
        reg = msesgo.Registro(rueda, list(mesa.girar(args.tiradas)), mesa=args.mesa)

    ver = msesgo.analizar(reg, radio_sector=args.radio)
    necesarias = msesgo.tiradas_necesarias(rueda)

    inf = Informe("Azarium - Deteccion de sesgo de rueda",
                  f"{rueda.nombre}, mesa: {reg.mesa}, {reg.n:,} tiradas registradas")

    inf.h2("El umbral que hay que superar")
    inf.p(f"Un pleno paga 35 a 1 sobre {rueda.n} casillas. Para que apostar a una "
          f"casilla sea rentable, su probabilidad real tiene que superar "
          f"<strong>1/36 = 2.778%</strong>, contra el {100 / rueda.n:.3f}% teorico. "
          f"Es una diferencia relativa de apenas "
          f"{(36 ** -1 - rueda.n ** -1) / rueda.n ** -1 * 100:.1f}%.")
    inf.destacado(
        f"<strong>Cuanta observacion hace falta.</strong> Detectar ese sesgo minimo con "
        f"80% de potencia, corrigiendo por las {rueda.n} casillas que se testean a la "
        f"vez, exige <strong>{necesarias:,} tiradas de la misma rueda</strong>: "
        f"unas {msesgo.horas_de_mesa(necesarias):,.0f} horas de mesa continua, o "
        f"{msesgo.horas_de_mesa(necesarias) / 8 / 365:.1f} anios observando 8 horas por "
        f"dia sin cambiar de mesa. Y eso solo para <em>saber</em> si el sesgo existe, "
        f"antes de apostar un peso.")

    inf.h2("Resultado del analisis")
    frec = reg.frecuencias()
    esp = reg.n / rueda.n
    sigma = float(np.sqrt(reg.n * (1 / rueda.n) * (1 - 1 / rueda.n)))
    inf.svg(reporte.grafico_frecuencias(
        frec, 0, esp, sigma, "Frecuencia por casilla (en orden fisico del disco)"))
    inf.p("<small>El eje horizontal sigue el orden fisico de la rueda, no el numerico: "
          "un defecto mecanico afecta a casillas vecinas en el disco.</small>")
    inf.p(f"<strong>{ver.chi2.nombre}:</strong> chi2 = {ver.chi2.estadistico:.2f} "
          f"(gl {ver.chi2.df}), p = {ver.chi2.p_valor:.4g}")
    inf.p("<pre style='white-space:pre-wrap'>" + ver.texto() + "</pre>")

    if ver.casillas_explotables:
        inf.h3("Casillas con sesgo explotable")
        inf.tabla(["Casilla", "Salio", "Esperado", "Frecuencia", "IC 95%",
                   "Ventaja del jugador"],
                  [[f.casilla, str(f.observado), f"{f.esperado:.0f}",
                    f"{f.frecuencia * 100:.3f}%",
                    f"{f.ic_bajo * 100:.3f} - {f.ic_alto * 100:.3f}%",
                    _pct(f.ventaja_jugador)] for f in ver.casillas_explotables])
    if ver.sectores_significativos:
        inf.h3("Sectores fisicos con exceso significativo")
        inf.tabla(["Centro", "Casillas del sector", "Observado", "Esperado", "z"],
                  [[s.centro, ", ".join(s.casillas), str(s.observado),
                    f"{s.esperado:.0f}", f"{s.z:+.2f}"]
                   for s in ver.sectores_significativos[:10]])

    destino = Path(args.salida or RAIZ / "informes" / "ruleta-sesgo.html")
    inf.guardar(destino)
    print(f"Informe generado: {destino}")
    return 0


def cmd_informe(args) -> int:
    """Informe consolidado: los tres modulos en un unico dictamen."""
    juego = JUEGOS[args.juego]
    hist = cargar(args.csv, juego) if args.csv else generar_sintetico(
        juego, args.sorteos, semilla=args.semilla)
    rueda = motor.RUEDAS[args.rueda]

    inf = Informe(
        "Cuánto cuesta jugar",
        f"Estudio estadístico sobre {hist.n_sorteos:,} sorteos de {juego.nombre} y "
        f"{args.sesiones:,} sesiones de ruleta simuladas. Qué dicen los datos sobre "
        f"predecir números, y qué se puede afirmar con ellos.",
        eyebrow="Azarium · Informe consolidado")

    # ---- veredicto de entrada
    ret_quiniela = B.esperanza_teorica(hist, "cabeza") * 100
    v_euro = motor.ventaja_casa(motor.catalogo(motor.EUROPEA)["rojo"], motor.EUROPEA)
    v_ame = motor.ventaja_casa(motor.catalogo(motor.AMERICANA)["rojo"], motor.AMERICANA)

    inf.veredicto(
        f"{ret_quiniela:.0f}%",
        "retorno esperado por cada peso jugado a la quiniela",
        "No depende de qué números elijas. Está fijado por el pago: la quiniela paga "
        "70 por acertar entre 100 posibilidades. La diferencia entre esos dos números "
        "es la ganancia de la banca, y ninguna selección de números la mueve.")

    inf.h2("Dónde cae cada juego")
    inf.p("Todo juego de azar tiene un número que lo define: cuánto se queda la casa de "
          "cada peso que pasa por la mesa. Es lo primero que hay que mirar y lo único "
          "que no se puede cambiar desde la silla del jugador.")
    inf.medidor([
        ("Quiniela a la cabeza", ret_quiniela),
        ("Quiniela a los 20", B.esperanza_teorica(hist, "a20") * 100),
        ("Ruleta americana", -v_ame),
        ("Ruleta europea", -v_euro),
        ("Ruleta francesa (par)", -motor.ventaja_casa(
            motor.catalogo(motor.FRANCESA)["rojo"], motor.FRANCESA)),
    ])
    inf.p("<small>Cuanto más larga la barra, más caro el juego. La ruleta es entre cinco "
          "y veinte veces más barata que la quiniela por peso apostado — lo que no la "
          "vuelve rentable, sólo más lenta.</small>")

    # ---- modulo 1
    inf.h2("¿Se pueden predecir los sorteos?")
    resultados = A.bateria(hist)
    rechazados = [r for r in resultados if r.significativo]
    inf.p("La hipótesis a testear es que cada sorteo es independiente del anterior y que "
          "todos los números son equiprobables. Si se rechaza, hay estructura y podría "
          "explotarse. Estos son los seis tests, sobre "
          f"{hist.n_extracciones:,} números extraídos:")
    inf.tabla(["Test", "Estadístico", "gl", "p-valor", "Conclusión"],
              [[r.nombre,
                "-" if np.isnan(r.estadistico) else f"{r.estadistico:.2f}",
                "-" if r.df is None else str(r.df),
                "-" if np.isnan(r.p_valor) else f"{r.p_valor:.4f}",
                f'<span class="{"mal" if r.significativo else "bien"}">{r.lectura()}</span>']
               for r in resultados])
    inf.p(f"{len(rechazados)} de {len(resultados)} tests rechazan la aleatoriedad." +
          ("" if rechazados else " Los sorteos son indistinguibles de un generador "
           "perfectamente uniforme y sin memoria."))

    frec = hist.frecuencias()
    n = int(frec.sum())
    p = 1 / juego.cardinalidad
    esp, sigma = n * p, float(np.sqrt(n * p * (1 - p)))
    inf.svg(reporte.grafico_frecuencias(frec, juego.minimo, esp, sigma,
                                        f"Apariciones por número — {juego.nombre}"))

    fnum = A.por_numero(hist)
    crudos = sum(f.significativo_crudo for f in fnum)
    con_fdr = sum(f.significativo_fdr for f in fnum)
    inf.h3("La trampa de los números calientes")
    inf.p(f"Mirando cada número por separado, <strong>{crudos}</strong> dan diferencia "
          f"significativa. Parece un hallazgo. No lo es: al correr "
          f"{juego.cardinalidad} tests simultáneos con umbral 0.05, el azar solo produce "
          f"~{juego.cardinalidad * 0.05:.0f} positivos falsos. Corrigiendo por "
          f"multiplicidad quedan <strong>{con_fdr}</strong>. "
          "Ese es el error que comete todo sistema que vende números calientes: "
          "confunde el ruido de mirar cien cosas a la vez con una señal.")

    inf.h3("Qué rinde cada estrategia")
    res = B.comparar(hist, k=args.k, calentamiento=args.calentamiento,
                     modalidad="cabeza", semilla=args.semilla)
    inf.tabla(["Estrategia", "Apuestas", "Aciertos", "Retorno"],
              [[r.estrategia, f"{r.apuestas:,}", str(r.aciertos), _pct(r.retorno * 100)]
               for r in res])
    aleat = next(r for r in res if r.estrategia == "Aleatorio")
    peor_p = min(B.diferencia_significativa(r, aleat) for r in res if r is not aleat)
    inf.p("Cada estrategia elige usando <em>sólo</em> los sorteos anteriores, y se la "
          "evalúa sobre el siguiente. Los retornos parecen muy distintos entre sí, pero "
          f"comparados contra elegir al azar el p-valor más bajo es {peor_p:.2f}: "
          "ninguna diferencia supera el ruido muestral. Con unos pocos miles de apuestas "
          "y una probabilidad de acierto del 1%, la dispersión aparente es enorme y no "
          "significa nada.")

    # ---- modulo 2
    inf.h2("¿Se le puede ganar a la ruleta con un sistema?")
    cfg = Config(banca_inicial=args.banca, apuesta_base=args.base, tiradas=args.tiradas,
                 sesiones=args.sesiones, rueda=rueda, apuesta="rojo", semilla=args.semilla)
    resumenes = comparar_sistemas(cfg)
    teorico = ventaja_teorica(cfg)
    inf.p(f"Ocho progresiones, mismas condiciones: banca {args.banca:,.0f}, unidad "
          f"{args.base:,.0f}, {args.tiradas} tiradas, {args.sesiones:,} sesiones "
          f"simuladas por sistema sobre {rueda.nombre.lower()}.")
    inf.tabla(["Sistema", "Resultado medio", "Mediana", "Ruina", "Termina ganando",
               "Retorno / apostado"],
              [[r.sistema, f"{r.resultado_medio:+,.0f}", f"{r.resultado_mediano:+,.0f}",
                f'<span class="mal">{r.prob_ruina * 100:.1f}%</span>',
                f"{r.prob_terminar_ganando * 100:.1f}%",
                f'<span class="mal">{r.retorno_sobre_apostado * 100:+.2f}%</span>']
               for r in resumenes])
    inf.destacado(
        f"<strong>La última columna es el experimento.</strong> Es el resultado neto "
        f"dividido por todo el dinero que pasó por la mesa, y converge a "
        f"<strong>−{teorico:.2f}%</strong> en los ocho sistemas. Ninguna progresión "
        f"cambia la esperanza, porque la esperanza de una suma de apuestas es la suma "
        f"de las esperanzas — y cada apuesta individual ya es negativa.")

    peor = max(resumenes, key=lambda r: r.prob_ruina)
    mejor = min(resumenes, key=lambda r: r.prob_ruina)
    marti = next((r for r in resumenes if r.sistema == "Martingala"), None)
    inf.p(f"Lo que sí cambian los sistemas es el reparto de las pérdidas. "
          f"{peor.sistema} arruina la banca en el {peor.prob_ruina * 100:.0f}% de las "
          f"sesiones; {mejor.sistema}, en el {mejor.prob_ruina * 100:.0f}%. "
          + (f"La Martingala termina en verde el "
             f"{marti.prob_terminar_ganando * 100:.0f}% de las veces y aun así pierde "
             f"{abs(marti.resultado_medio):,.0f} en promedio: gana poco y seguido, "
             f"pierde todo de golpe. Esa asimetría es exactamente lo que la hace sentir "
             f"un sistema ganador." if marti else ""))

    for r in resumenes:
        if r.sistema in ("Plana", "Martingala"):
            inf.h3(f"Distribución de resultados — {r.sistema}")
            inf.svg(reporte.grafico_histograma(
                r.bancas_finales, cfg.banca_inicial,
                f"Banca final tras {cfg.tiradas} tiradas"))

    # ---- modulo 3
    inf.h2("La única grieta real: el sesgo mecánico")
    necesarias = msesgo.tiradas_necesarias(rueda)
    horas = msesgo.horas_de_mesa(necesarias)
    inf.p("Hay un caso histórico de gente que le ganó a la ruleta: Joseph Jagger en "
          "Montecarlo (1873), Hibbs y Walford en Reno (1947), los Pelayo en los 90. "
          "Ninguno usó un sistema de apuestas. Todos hicieron lo mismo: encontrar una "
          "rueda físicamente defectuosa y apostar a las casillas que salían de más.")
    inf.p(f"Para que eso sea rentable no alcanza con que una casilla “salga más”. Como el "
          f"pleno paga 35 a 1, su probabilidad real tiene que superar "
          f"<strong>1/36 = 2,778%</strong> contra el {100 / rueda.n:.3f}% teórico: una "
          f"diferencia relativa de apenas "
          f"{(1 / 36 - 1 / rueda.n) / (1 / rueda.n) * 100:.1f}%.")
    inf.veredicto(
        f"{necesarias:,}".replace(",", "."),
        "tiradas de la misma rueda para detectar ese sesgo",
        f"Con 80% de potencia y corrigiendo por las {rueda.n} casillas que se testean a "
        f"la vez. Son ~{horas:,.0f} horas de mesa continua: "
        f"{horas / 8 / 365:.1f} años observando ocho horas por día, la misma rueda, "
        "anotando cada tirada. Y eso sólo para <em>saber</em> si el sesgo existe, antes "
        "de apostar un peso. Los casinos modernos balancean y rotan las ruedas "
        "precisamente porque esta grieta fue real.")
    inf.p("Azarium implementa el análisis completo igual — con corrección por "
          "multiplicidad y agrupando por sectores físicos del disco, no por números "
          "consecutivos, porque un defecto mecánico afecta a una zona contigua del plato. "
          "Si alguna vez conseguís miles de tiradas registradas de una mesa concreta, "
          "el módulo te dice si hay algo y con cuánta confianza.")

    inf.h2("Cómo usar la aplicación")
    inf.p("Los tres módulos se corren por línea de comandos y cada uno genera su propio "
          "informe. El lanzador resuelve su propia ubicación, así que funciona desde "
          "cualquier consola sin importar en qué carpeta estés parado:")
    lanzador = str(RAIZ / "azarium-cli.py")
    inf.partes.append(
        f'<pre>py -3 "{html_escape(lanzador)}" informe\n'
        f'py -3 "{html_escape(lanzador)}" sorteos --juego quiniela --csv datos\\quiniela.csv\n'
        f'py -3 "{html_escape(lanzador)}" ruleta  --rueda europea --banca 1000 --base 10\n'
        f'py -3 "{html_escape(lanzador)}" sesgo   --csv datos\\mesa3.csv --mesa "Mesa 3"\n'
        f'py -3 "{html_escape(lanzador)}" ventajas</pre>')
    if not args.csv:
        inf.p("<small>Este informe se generó con un histórico sintético uniforme de "
              f"{hist.n_sorteos:,} sorteos (semilla {args.semilla}), porque no se le pasó "
              "un CSV real. Los resultados de la ruleta son exactos e independientes de "
              "eso: la esperanza está calculada en aritmética racional. Para analizar "
              "sorteos reales, pasá el histórico con <code>--csv</code>.</small>")

    destino = Path(args.salida or RAIZ / "informes" / "azarium-informe.html")
    inf.guardar(destino)
    print(f"Informe generado: {destino}")
    return 0


def cmd_ventajas(args) -> int:
    for clave, rueda in motor.RUEDAS.items():
        print(f"\n{rueda.nombre}")
        for nombre, v, ev in motor.tabla_ventajas(rueda):
            print(f"  {nombre:<20s} ventaja casa {v:6.3f}%   EV = {ev}")
    return 0


# ----------------------------------------------------------------------- cli

def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="azarium",
        description="Analisis estadistico de juegos de azar: sorteos y ruleta.")
    sub = p.add_subparsers(dest="comando", required=True)

    s = sub.add_parser("sorteos", help="analizar un historico de quiniela/loto")
    s.add_argument("--juego", choices=sorted(JUEGOS), default="quiniela")
    s.add_argument("--csv", help="ruta a un CSV de historicos")
    s.add_argument("--url", help="URL de un CSV de historicos")
    s.add_argument("--sorteos", type=int, default=3650,
                   help="sorteos a generar si no hay datos reales (default: 10 anios)")
    s.add_argument("--k", type=int, default=1, help="numeros a jugar por sorteo")
    s.add_argument("--calentamiento", type=int, default=365,
                   help="sorteos iniciales usados solo para aprender")
    s.add_argument("--lags", type=int, default=20)
    s.add_argument("--semilla", type=int, default=42)
    s.add_argument("--salida", help="ruta del HTML de salida")
    s.set_defaults(func=cmd_sorteos)

    r = sub.add_parser("ruleta", help="simular sistemas de apuesta")
    r.add_argument("--rueda", choices=sorted(motor.RUEDAS), default="europea")
    r.add_argument("--apuesta", default="rojo")
    r.add_argument("--banca", type=float, default=1000)
    r.add_argument("--base", type=float, default=10)
    r.add_argument("--tiradas", type=int, default=500)
    r.add_argument("--sesiones", type=int, default=10000)
    r.add_argument("--limite", type=float, default=5000, help="limite maximo de la mesa")
    r.add_argument("--objetivo", type=float, default=None,
                   help="retirarse al alcanzar esta banca")
    r.add_argument("--semilla", type=int, default=1)
    r.add_argument("--salida")
    r.set_defaults(func=cmd_ruleta)

    g = sub.add_parser("sesgo", help="analizar tiradas de una rueda concreta")
    g.add_argument("--rueda", choices=sorted(motor.RUEDAS), default="europea")
    g.add_argument("--csv", help="CSV con una tirada por fila")
    g.add_argument("--mesa", default="sin identificar")
    g.add_argument("--tiradas", type=int, default=20000,
                   help="tiradas a simular si no hay CSV")
    g.add_argument("--demo-sesgo", metavar="CASILLA",
                   help="simular una rueda defectuosa centrada en esa casilla")
    g.add_argument("--radio", type=int, default=3, help="radio del sector fisico")
    g.add_argument("--semilla", type=int, default=3)
    g.add_argument("--salida")
    g.set_defaults(func=cmd_sesgo)

    i = sub.add_parser("informe", help="informe consolidado con los tres modulos")
    i.add_argument("--juego", choices=sorted(JUEGOS), default="quiniela")
    i.add_argument("--csv", help="ruta a un CSV de historicos reales")
    i.add_argument("--sorteos", type=int, default=3650)
    i.add_argument("--rueda", choices=sorted(motor.RUEDAS), default="europea")
    i.add_argument("--banca", type=float, default=1000)
    i.add_argument("--base", type=float, default=10)
    i.add_argument("--tiradas", type=int, default=500)
    i.add_argument("--sesiones", type=int, default=10000)
    i.add_argument("--k", type=int, default=1)
    i.add_argument("--calentamiento", type=int, default=365)
    i.add_argument("--semilla", type=int, default=42)
    i.add_argument("--salida")
    i.set_defaults(func=cmd_informe)

    v = sub.add_parser("ventajas", help="tabla de ventaja de la casa por apuesta")
    v.set_defaults(func=cmd_ventajas)

    return p


def main(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
