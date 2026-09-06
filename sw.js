// Service worker de Azarium.
// Existe sobre todo para que Chrome ofrezca instalar la app: sin un worker
// con handler de fetch, el navegador no muestra el prompt de instalacion.
// De paso deja la app usable sin conexion.

const CACHE = 'azarium-v1';

// Se cachea al vuelo en vez de precargar una lista fija: la app tiene muchos
// archivos en js/ y css/, y una lista escrita a mano se desactualiza sola.
const ESENCIAL = ['./', './index.html', './manifest.json'];

self.addEventListener('install', evento => {
  evento.waitUntil(
    caches.open(CACHE).then(c => c.addAll(ESENCIAL)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', evento => {
  evento.waitUntil(
    caches.keys()
      .then(claves => Promise.all(
        claves.filter(k => k !== CACHE).map(k => caches.delete(k))
      ))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', evento => {
  const pedido = evento.request;
  if (pedido.method !== 'GET' || !pedido.url.startsWith(self.location.origin)) return;

  // Primero la red, para que una version nueva llegue sin tener que
  // desinstalar. El cache es el respaldo cuando no hay señal.
  evento.respondWith(
    fetch(pedido)
      .then(respuesta => {
        if (respuesta.ok) {
          const copia = respuesta.clone();
          caches.open(CACHE).then(c => c.put(pedido, copia));
        }
        return respuesta;
      })
      .catch(() => caches.match(pedido).then(hit => hit || caches.match('./index.html')))
  );
});
