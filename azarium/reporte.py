"""Generacion de informes HTML autocontenidos, con graficos en SVG puro.

No se usa matplotlib: los graficos son unas pocas formas y generarlos a mano
evita una dependencia pesada y produce un HTML que se abre en cualquier lado
sin assets externos.
"""

from __future__ import annotations

import datetime as dt
import html
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- primitivas


@dataclass
class Ejes:
    """Mapeo de coordenadas de datos a coordenadas de pantalla."""
    x0: float
    y0: float
    ancho: float
    alto: float
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def px(self, x: float) -> float:
        if self.xmax == self.xmin:
            return self.x0
        return self.x0 + (x - self.xmin) / (self.xmax - self.xmin) * self.ancho

    def py(self, y: float) -> float:
        if self.ymax == self.ymin:
            return self.y0 + self.alto
        return self.y0 + self.alto - (y - self.ymin) / (self.ymax - self.ymin) * self.alto


def _texto(x, y, s, clase="lbl", ancla="middle") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" class="{clase}" '
            f'text-anchor="{ancla}">{html.escape(str(s))}</text>')


def _marco(ejes: Ejes, titulo: str, ylab: str = "") -> str:
    p = [f'<rect x="{ejes.x0:.1f}" y="{ejes.y0:.1f}" width="{ejes.ancho:.1f}" '
         f'height="{ejes.alto:.1f}" class="panel"/>']
    p.append(_texto(ejes.x0, ejes.y0 - 14, titulo, "titulo", "start"))
    if ylab:
        p.append(f'<text x="{ejes.x0 - 44:.1f}" y="{ejes.y0 + ejes.alto / 2:.1f}" '
                 f'class="lbl" text-anchor="middle" '
                 f'transform="rotate(-90 {ejes.x0 - 44:.1f} {ejes.y0 + ejes.alto / 2:.1f})">'
                 f'{html.escape(ylab)}</text>')
    return "".join(p)


def grafico_frecuencias(frecuencias: np.ndarray, minimo: int, esperado: float,
                        sigma: float, titulo: str) -> str:
    """Barras por numero con la banda de +-2 sigma esperada bajo azar puro."""
    k = len(frecuencias)
    W, H = 900, 300
    ejes = Ejes(60, 30, W - 90, H - 80, -0.5, k - 0.5,
                0, max(float(frecuencias.max()) * 1.12, esperado + 4 * sigma))
    partes = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(titulo)}">',
              _marco(ejes, titulo, "veces que salio")]

    # Banda de tolerancia: dentro de aca, la desviacion es ruido esperable.
    y_alto, y_bajo = ejes.py(esperado + 2 * sigma), ejes.py(esperado - 2 * sigma)
    partes.append(f'<rect x="{ejes.x0:.1f}" y="{y_alto:.1f}" width="{ejes.ancho:.1f}" '
                  f'height="{max(0, y_bajo - y_alto):.1f}" class="banda"/>')

    ancho_barra = max(1.0, ejes.ancho / k * 0.78)
    for i, v in enumerate(frecuencias):
        x = ejes.px(i) - ancho_barra / 2
        y = ejes.py(float(v))
        fuera = abs(v - esperado) > 2 * sigma
        clase = "barra-fuera" if fuera else "barra"
        partes.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{ancho_barra:.1f}" '
                      f'height="{max(0, ejes.py(0) - y):.1f}" class="{clase}">'
                      f'<title>{minimo + i}: {int(v)} veces</title></rect>')

    y_esp = ejes.py(esperado)
    partes.append(f'<line x1="{ejes.x0:.1f}" y1="{y_esp:.1f}" x2="{ejes.x0 + ejes.ancho:.1f}" '
                  f'y2="{y_esp:.1f}" class="ref"/>')
    partes.append(_texto(ejes.x0 + ejes.ancho + 4, y_esp + 4,
                         f"esperado {esperado:.0f}", "lbl", "start"))

    paso = max(1, k // 20)
    for i in range(0, k, paso):
        partes.append(_texto(ejes.px(i), ejes.y0 + ejes.alto + 16, minimo + i))
    partes.append(_texto(ejes.x0, ejes.y0 + ejes.alto + 34,
                         "la franja clara es el margen de +-2 sigma que el azar puro produce",
                         "nota", "start"))
    partes.append("</svg>")
    return "".join(partes)


def grafico_acf(acf: np.ndarray, banda: float, titulo: str) -> str:
    """Autocorrelacion por lag con su banda de significancia."""
    n = len(acf)
    W, H = 900, 240
    tope = max(float(np.abs(acf).max()) * 1.3, banda * 2.2)
    ejes = Ejes(60, 30, W - 90, H - 70, 0.5, n + 0.5, -tope, tope)
    partes = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(titulo)}">',
              _marco(ejes, titulo, "correlacion")]
    partes.append(f'<rect x="{ejes.x0:.1f}" y="{ejes.py(banda):.1f}" '
                  f'width="{ejes.ancho:.1f}" '
                  f'height="{ejes.py(-banda) - ejes.py(banda):.1f}" class="banda"/>')
    y0 = ejes.py(0)
    partes.append(f'<line x1="{ejes.x0:.1f}" y1="{y0:.1f}" '
                  f'x2="{ejes.x0 + ejes.ancho:.1f}" y2="{y0:.1f}" class="ref"/>')
    for i, v in enumerate(acf, start=1):
        x = ejes.px(i)
        clase = "barra-fuera" if abs(v) > banda else "barra"
        partes.append(f'<line x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{ejes.py(float(v)):.1f}" '
                      f'class="{clase}" stroke-width="6"><title>lag {i}: {v:+.4f}</title></line>')
        if i % max(1, n // 20) == 0 or i == 1:
            partes.append(_texto(x, ejes.y0 + ejes.alto + 16, i))
    partes.append(_texto(ejes.x0, ejes.y0 + ejes.alto + 34,
                         "barras dentro de la franja = sin memoria entre sorteos", "nota", "start"))
    partes.append("</svg>")
    return "".join(partes)


def grafico_histograma(valores: np.ndarray, referencia: float, titulo: str,
                       bins: int = 46) -> str:
    """Distribucion de resultados finales, con la banca inicial marcada."""
    W, H = 900, 280
    v = np.asarray(valores, dtype=float)
    lo, hi = float(v.min()), float(max(v.max(), referencia * 1.05))
    if hi <= lo:
        hi = lo + 1
    conteo, bordes = np.histogram(v, bins=bins, range=(lo, hi))
    ejes = Ejes(60, 30, W - 90, H - 70, lo, hi, 0, float(conteo.max()) * 1.12 or 1)
    partes = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(titulo)}">',
              _marco(ejes, titulo, "sesiones")]
    for i, c in enumerate(conteo):
        x1, x2 = ejes.px(bordes[i]), ejes.px(bordes[i + 1])
        y = ejes.py(float(c))
        gana = bordes[i] >= referencia
        partes.append(f'<rect x="{x1:.1f}" y="{y:.1f}" width="{max(1, x2 - x1 - 1):.1f}" '
                      f'height="{max(0, ejes.py(0) - y):.1f}" '
                      f'class="{"barra-gana" if gana else "barra"}">'
                      f'<title>{bordes[i]:.0f} a {bordes[i + 1]:.0f}: {int(c)} sesiones</title>'
                      f'</rect>')
    xr = ejes.px(referencia)
    partes.append(f'<line x1="{xr:.1f}" y1="{ejes.y0:.1f}" x2="{xr:.1f}" '
                  f'y2="{ejes.y0 + ejes.alto:.1f}" class="ref"/>')
    partes.append(_texto(xr, ejes.y0 - 4, "banca inicial", "nota"))
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        val = lo + (hi - lo) * frac
        partes.append(_texto(ejes.px(val), ejes.y0 + ejes.alto + 16, f"{val:,.0f}"))
    partes.append("</svg>")
    return "".join(partes)


# -------------------------------------------------------------------- pagina

_CSS = """/* Azarium - identidad visual del informe.
   Neutros de papel tecnico con sesgo azul; acento azul acero forense;
   rojo ladrillo reservado para lo que cae fuera de banda. */
:root{
  --papel:#FBFAF8; --superficie:#F2F0EB; --hundido:#E8E5DE;
  --tinta:#191920; --tinta-suave:#63636F; --tinta-tenue:#8C8C97;
  --linea:#DCD8D0; --linea-fuerte:#C3BEB4;
  --acento:#3A5A8C; --acento-suave:#6B87B4;
  --alerta:#B23A28; --ok:#2E7057; --banda:#E4E7EE;
  --serif:Georgia,"Iowan Old Style","Times New Roman",serif;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono:ui-monospace,"Cascadia Code","Consolas","DejaVu Sans Mono",monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --papel:#131318; --superficie:#1C1C23; --hundido:#24242D;
    --tinta:#EAE8E4; --tinta-suave:#9E9EAA; --tinta-tenue:#75757F;
    --linea:#2E2E38; --linea-fuerte:#3E3E4A;
    --acento:#89A9D6; --acento-suave:#5E7CA8;
    --alerta:#DE7458; --ok:#5CB88E; --banda:#232936;
  }
}
:root[data-theme="dark"]{
  --papel:#131318; --superficie:#1C1C23; --hundido:#24242D;
  --tinta:#EAE8E4; --tinta-suave:#9E9EAA; --tinta-tenue:#75757F;
  --linea:#2E2E38; --linea-fuerte:#3E3E4A;
  --acento:#89A9D6; --acento-suave:#5E7CA8;
  --alerta:#DE7458; --ok:#5CB88E; --banda:#232936;
}
*{box-sizing:border-box}
body{margin:0;background:var(--papel);color:var(--tinta);
  font:400 16.5px/1.65 var(--sans);-webkit-font-smoothing:antialiased}
.wrap{max-width:74ch;margin:0 auto;padding:56px 24px 96px;
  display:flex;flex-direction:column;gap:0}

/* --- cabecera --- */
.eyebrow{font:600 11px/1 var(--mono);letter-spacing:.18em;text-transform:uppercase;
  color:var(--acento);margin:0 0 18px}
h1{font:400 clamp(2.1rem,5vw,3rem)/1.1 var(--serif);margin:0 0 10px;
  letter-spacing:-.015em;text-wrap:balance}
.sub{color:var(--tinta-suave);margin:0 0 8px;font-size:1.02rem;max-width:60ch}

/* --- jerarquia --- */
h2{font:400 1.55rem/1.25 var(--serif);margin:64px 0 4px;letter-spacing:-.01em;
  text-wrap:balance}
h2::before{content:"";display:block;width:38px;height:2px;background:var(--acento);
  margin-bottom:20px}
h3{font:600 11.5px/1.3 var(--mono);letter-spacing:.14em;text-transform:uppercase;
  color:var(--tinta-suave);margin:34px 0 6px}
p{margin:13px 0;max-width:66ch}
small{font-size:.86em;color:var(--tinta-suave)}
em{font-style:italic;color:var(--tinta-suave)}

/* --- ficha de veredicto: la conclusion antes del detalle --- */
.veredicto{background:var(--superficie);border:1px solid var(--linea);
  border-radius:2px;padding:26px 28px;margin:28px 0;
  display:flex;flex-direction:column;gap:10px}
.veredicto .cifra{font:400 clamp(2.4rem,7vw,3.6rem)/1 var(--serif);
  color:var(--alerta);letter-spacing:-.03em;font-variant-numeric:tabular-nums}
.veredicto .que{font:600 11px/1.4 var(--mono);letter-spacing:.14em;
  text-transform:uppercase;color:var(--tinta-suave)}
.veredicto p{margin:0;max-width:none}

/* --- medidor: ubica cada juego sobre el eje de perdida --- */
.medidor{display:flex;flex-direction:column;gap:14px;margin:26px 0}
.medidor .fila{display:grid;grid-template-columns:minmax(90px,1fr) 3fr auto;
  gap:14px;align-items:center}
.medidor .nom{font-size:.88rem;color:var(--tinta-suave)}
.medidor .pista{position:relative;height:9px;background:var(--hundido);
  border-radius:1px;overflow:hidden}
.medidor .relleno{position:absolute;inset:0 auto 0 0;background:var(--alerta);
  opacity:.82}
.medidor .val{font:600 .85rem/1 var(--mono);color:var(--alerta);
  font-variant-numeric:tabular-nums;min-width:5.5ch;text-align:right}

/* --- destacado --- */
.destacado{border-left:2px solid var(--acento);padding:4px 0 4px 20px;margin:26px 0;
  color:var(--tinta)}
.destacado strong{color:var(--acento);font-weight:600}

/* --- tablas: el vernaculo del tema es la pizarra monoespaciada --- */
.tabla-wrap{overflow-x:auto;margin:18px 0;border-block:1px solid var(--linea)}
table{border-collapse:collapse;width:100%;font:400 .88rem/1.5 var(--mono);
  font-variant-numeric:tabular-nums}
th,td{padding:9px 12px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:normal;
  font-family:var(--sans);min-width:14ch}
th{font:600 10px/1.4 var(--mono);letter-spacing:.1em;text-transform:uppercase;
  color:var(--tinta-tenue);border-bottom:1px solid var(--linea-fuerte);
  padding-top:12px;padding-bottom:10px}
tbody tr{border-bottom:1px solid var(--linea)}
tbody tr:last-child{border-bottom:none}
tbody tr:hover{background:var(--superficie)}
.mal{color:var(--alerta);font-weight:600}
.bien{color:var(--ok);font-weight:600}
code{font:.88em var(--mono);background:var(--superficie);padding:2px 6px;
  border-radius:2px;color:var(--acento)}
pre{font:.85rem/1.6 var(--mono);background:var(--superficie);padding:16px 18px;
  border-radius:2px;overflow-x:auto;border:1px solid var(--linea);
  white-space:pre-wrap;color:var(--tinta)}

/* --- graficos --- */
svg{width:100%;height:auto;margin:16px 0 4px;display:block}
.panel{fill:var(--superficie);stroke:var(--linea)}
.banda{fill:var(--banda)}
.barra{fill:var(--acento-suave);stroke:none}
.barra-fuera{fill:var(--alerta);stroke:var(--alerta)}
.barra-gana{fill:var(--ok)}
.ref{stroke:var(--acento);stroke-width:1.5;stroke-dasharray:4 3}
text{fill:var(--tinta-suave);font-family:var(--mono)}
.lbl{font-size:10.5px}
.nota{font-size:10.5px;font-family:var(--sans);font-style:italic;fill:var(--tinta-tenue)}
.titulo{font-size:12px;font-weight:600;fill:var(--tinta);font-family:var(--sans)}

footer{margin-top:72px;padding-top:20px;border-top:1px solid var(--linea);
  color:var(--tinta-tenue);font-size:.84rem;font-family:var(--mono);line-height:1.7}
a{color:var(--acento)}
a:focus-visible,summary:focus-visible{outline:2px solid var(--acento);outline-offset:3px}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
@media (max-width:560px){
  .wrap{padding:36px 16px 64px}
  .medidor .fila{grid-template-columns:1fr 2fr auto;gap:10px}
}
"""


class Informe:
    """Acumula secciones y escribe el HTML final."""

    def __init__(self, titulo: str, subtitulo: str = "", eyebrow: str = "Azarium") -> None:
        self.titulo = titulo
        self.subtitulo = subtitulo
        self.eyebrow = eyebrow
        self.partes: list[str] = []

    def veredicto(self, cifra: str, que: str, texto: str) -> "Informe":
        """Ficha con la conclusion de la seccion, antes de cualquier detalle."""
        self.partes.append(
            f'<div class="veredicto"><span class="que">{html.escape(que)}</span>'
            f'<span class="cifra">{html.escape(cifra)}</span>'
            f"<p>{texto}</p></div>")
        return self

    def medidor(self, filas: list[tuple[str, float]], maximo: float | None = None,
                sufijo: str = "%") -> "Informe":
        """Barras comparativas: ubica cada juego sobre el mismo eje de perdida."""
        tope = maximo or max(abs(v) for _, v in filas) or 1.0
        html_filas = []
        for nombre, valor in filas:
            ancho = min(100.0, abs(valor) / tope * 100)
            html_filas.append(
                f'<div class="fila"><span class="nom">{html.escape(nombre)}</span>'
                f'<span class="pista"><span class="relleno" style="width:{ancho:.1f}%">'
                f'</span></span>'
                f'<span class="val">{valor:.2f}{sufijo}</span></div>')
        self.partes.append('<div class="medidor">' + "".join(html_filas) + "</div>")
        return self

    def h2(self, texto: str) -> "Informe":
        self.partes.append(f"<h2>{html.escape(texto)}</h2>")
        return self

    def h3(self, texto: str) -> "Informe":
        self.partes.append(f"<h3>{html.escape(texto)}</h3>")
        return self

    def p(self, texto: str) -> "Informe":
        self.partes.append(f"<p>{texto}</p>")
        return self

    def destacado(self, texto: str) -> "Informe":
        self.partes.append(f'<div class="destacado">{texto}</div>')
        return self

    def svg(self, markup: str) -> "Informe":
        self.partes.append(markup)
        return self

    def tabla(self, cabeceras: list[str], filas: list[list[str]]) -> "Informe":
        th = "".join(f"<th>{html.escape(c)}</th>" for c in cabeceras)
        cuerpo = []
        for fila in filas:
            # Las celdas pueden traer markup propio (clases de color), no se escapan.
            cuerpo.append("<tr>" + "".join(f"<td>{c}</td>" for c in fila) + "</tr>")
        self.partes.append('<div class="tabla-wrap"><table><thead><tr>' + th +
                           "</tr></thead><tbody>" + "".join(cuerpo) +
                           "</tbody></table></div>")
        return self

    def html(self) -> str:
        hoy = dt.date.today().isoformat()
        return (f"<title>{html.escape(self.titulo)}</title>"
                f"<style>{_CSS}</style>"
                f'<div class="wrap"><p class="eyebrow">{html.escape(self.eyebrow)}</p>'
                f'<h1>{html.escape(self.titulo)}</h1>'
                f'<p class="sub">{html.escape(self.subtitulo)}</p>'
                + "".join(self.partes) +
                f'<footer>Generado por Azarium el {hoy}. '
                "Todos los calculos son reproducibles con la semilla indicada en cada seccion."
                "</footer></div>")

    def guardar(self, ruta: str | Path) -> Path:
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(self.html(), encoding="utf-8")
        return ruta
