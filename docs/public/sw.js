/**
 * DIY FlashAttention - Service Worker v3.0
 * Cache-first strategy for offline docs
 */

const CACHE_NAME = 'diy-flash-attention-v3';

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll([
        '/diy-flash-attention/',
        '/diy-flash-attention/manifest.json',
      ]))
      .then(() => self.skipWaiting())
      .catch(() => {})
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((names) => Promise.all(
        names.filter((n) => n !== CACHE_NAME).map((n) => caches.delete(n))
      ))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;
  if (!new URL(request.url).pathname.startsWith('/diy-flash-attention/')) return;

  event.respondWith(
    caches.match(request).then((cached) => {
      if (cached) return cached;
      return fetch(request).then((res) => {
        if (!res || res.status !== 200) return res;
        const clone = res.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(request, clone));
        return res;
      }).catch(() => {
        if (request.mode === 'navigate') {
          return caches.match('/diy-flash-attention/offline.html');
        }
        return new Response('Offline', { status: 408 });
      });
    })
  );
});
