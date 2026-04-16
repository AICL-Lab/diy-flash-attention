/**
 * DIY FlashAttention - Service Worker
 * Enables offline access and faster loading through caching
 * Version: 2.0.0
 */

const CACHE_NAME = 'diy-flash-attention-v2';
const STATIC_ASSETS = [
  '/diy-flash-attention/',
  '/diy-flash-attention/index.html',
  '/diy-flash-attention/en/index.html',
  '/diy-flash-attention/zh/index.html',
  '/diy-flash-attention/manifest.json',
];

// 安装阶段 - 预缓存核心资源
self.addEventListener('install', (event) => {
  console.log('[SW] Installing...');
  
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('[SW] Caching static assets');
        return cache.addAll(STATIC_ASSETS);
      })
      .then(() => self.skipWaiting())
      .catch((err) => console.error('[SW] Install failed:', err))
  );
});

// 激活阶段 - 清理旧缓存
self.addEventListener('activate', (event) => {
  console.log('[SW] Activating...');
  
  event.waitUntil(
    caches.keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames
            .filter((name) => name !== CACHE_NAME)
            .map((name) => {
              console.log('[SW] Deleting old cache:', name);
              return caches.delete(name);
            })
        );
      })
      .then(() => self.clients.claim())
  );
});

// 拦截请求 - 缓存优先策略
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  
  // 仅处理同源请求
  if (!url.pathname.startsWith('/diy-flash-attention/')) {
    return;
  }
  
  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }
  
  // 策略: 缓存优先，网络回退
  event.respondWith(
    caches.match(request)
      .then((cachedResponse) => {
        if (cachedResponse) {
          // 后台更新缓存
          fetch(request)
            .then((networkResponse) => {
              if (networkResponse.ok) {
                caches.open(CACHE_NAME)
                  .then((cache) => cache.put(request, networkResponse.clone()));
              }
            })
            .catch(() => {});
          
          return cachedResponse;
        }
        
        // 缓存未命中，从网络获取
        return fetch(request)
          .then((networkResponse) => {
            if (!networkResponse || networkResponse.status !== 200) {
              return networkResponse;
            }
            
            // 缓存新资源
            const responseToCache = networkResponse.clone();
            caches.open(CACHE_NAME)
              .then((cache) => {
                cache.put(request, responseToCache);
              });
            
            return networkResponse;
          })
          .catch(() => {
            // 网络失败，返回离线页面
            if (request.mode === 'navigate') {
              return caches.match('/diy-flash-attention/offline.html');
            }
            return new Response('Network error', { status: 408 });
          });
      })
  );
});

// 后台同步 - 用于离线表单提交等
self.addEventListener('sync', (event) => {
  if (event.tag === 'background-sync') {
    console.log('[SW] Background sync triggered');
  }
});

// 推送通知支持
self.addEventListener('push', (event) => {
  const options = {
    body: event.data?.text() || 'New update available!',
    icon: '/diy-flash-attention/icons/icon-192x192.png',
    badge: '/diy-flash-attention/icons/badge-72x72.png',
    tag: 'update',
    requireInteraction: true
  };
  
  event.waitUntil(
    self.registration.showNotification('DIY FlashAttention', options)
  );
});

// 通知点击
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  event.waitUntil(
    clients.openWindow('/diy-flash-attention/')
  );
});

console.log('[SW] Service Worker loaded');
