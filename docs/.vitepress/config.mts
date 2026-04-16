import { defineConfig } from 'vitepress'

// ============================================
// DIY FlashAttention - Ultimate VitePress Config
// Version: 3.0 - Maximum Aggression Mode
// ============================================

export default defineConfig({
  // ==========================================
  // Core Settings
  // ==========================================
  lang: 'en-US',
  title: 'DIY FlashAttention',
  titleTemplate: ':title | DIY FlashAttention',
  description: 'Master GPU programming by implementing FlashAttention from scratch. Learn Triton CUDA kernels with interactive tutorials and benchmarks.',
  
  base: '/diy-flash-attention/',
  cleanUrls: true,
  lastUpdated: true,
  ignoreDeadLinks: [
    /^https?:\/\/localhost/,
    /^#/
  ],

  // ==========================================
  // Head - Maximum SEO
  // ==========================================
  head: [
    // Charset & Viewport
    ['meta', { charset: 'UTF-8' }],
    ['meta', { name: 'viewport', content: 'width=device-width, initial-scale=1.0, maximum-scale=5.0' }],
    ['meta', { 'http-equiv': 'X-UA-Compatible', content: 'IE=edge,chrome=1' }],
    
    // PWA
    ['link', { rel: 'manifest', href: '/diy-flash-attention/manifest.json' }],
    ['meta', { name: 'theme-color', content: '#10b981' }],
    ['meta', { name: 'msapplication-TileColor', content: '#10b981' }],
    ['meta', { name: 'apple-mobile-web-app-capable', content: 'yes' }],
    ['meta', { name: 'apple-mobile-web-app-status-bar-style', content: 'black-translucent' }],
    ['meta', { name: 'apple-mobile-web-app-title', content: 'FlashAttention' }],
    
    // SEO
    ['link', { rel: 'canonical', href: 'https://lessup.github.io/diy-flash-attention/' }],
    ['meta', { name: 'author', content: 'LessUp' }],
    ['meta', { name: 'robots', content: 'index, follow, max-image-preview:large, max-snippet:-1, max-video-preview:-1' }],
    
    // Keywords
    ['meta', { name: 'keywords', content: 'FlashAttention,Triton,CUDA,GPU,GPU Programming,Attention Mechanism,LLM,Large Language Model,Online Softmax,Tiling,Memory-efficient Attention,Transformer,Deep Learning,Machine Learning,PyTorch,Python,Kernel Optimization,H100,A100,RTX 4090,Matrix Multiplication,Parallel Computing,CUDA Kernel,Triton Kernel,GPU Optimization,Flash Attention,AI Acceleration' }],
    
    // Open Graph
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:site_name', content: 'DIY FlashAttention' }],
    ['meta', { property: 'og:title', content: 'DIY FlashAttention - Master GPU Programming' }],
    ['meta', { property: 'og:description', content: 'Implement FlashAttention from scratch. Learn Triton GPU programming with hands-on tutorials and production-ready code.' }],
    ['meta', { property: 'og:url', content: 'https://lessup.github.io/diy-flash-attention/' }],
    ['meta', { property: 'og:image', content: 'https://lessup.github.io/diy-flash-attention/og-image.png' }],
    ['meta', { property: 'og:image:width', content: '1200' }],
    ['meta', { property: 'og:image:height', content: '630' }],
    ['meta', { property: 'og:locale', content: 'en_US' }],
    ['meta', { property: 'og:locale:alternate', content: 'zh_CN' }],
    
    // Twitter
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
    ['meta', { name: 'twitter:site', content: '@lessup' }],
    ['meta', { name: 'twitter:creator', content: '@lessup' }],
    ['meta', { name: 'twitter:title', content: 'DIY FlashAttention - Master GPU Programming' }],
    ['meta', { name: 'twitter:description', content: 'Implement FlashAttention from scratch. Learn Triton GPU programming with hands-on tutorials.' }],
    
    // Fonts
    ['link', { rel: 'preconnect', href: 'https://fonts.googleapis.com' }],
    ['link', { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' }],
    
    // Icons
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/diy-flash-attention/logo.svg' }],
    
    // JSON-LD Structured Data
    ['script', { type: 'application/ld+json' }, JSON.stringify({
      '@context': 'https://schema.org',
      '@graph': [
        {
          '@type': 'WebSite',
          name: 'DIY FlashAttention',
          url: 'https://lessup.github.io/diy-flash-attention/',
          potentialAction: {
            '@type': 'SearchAction',
            target: 'https://lessup.github.io/diy-flash-attention/search?q={search_term_string}',
            'query-input': 'required name=search_term_string'
          }
        },
        {
          '@type': 'TechArticle',
          headline: 'DIY FlashAttention Tutorial',
          description: 'Master GPU programming by implementing FlashAttention from scratch using Python and Triton.',
          author: { '@type': 'Organization', name: 'LessUp' },
          publisher: {
            '@type': 'Organization',
            name: 'LessUp',
            logo: { '@type': 'ImageObject', url: 'https://lessup.github.io/diy-flash-attention/logo.png' }
          },
          mainEntityOfPage: { '@type': 'WebPage', '@id': 'https://lessup.github.io/diy-flash-attention/' },
          about: [
            { '@type': 'Thing', name: 'GPU Programming' },
            { '@type': 'Thing', name: 'CUDA' },
            { '@type': 'Thing', name: 'Triton' },
            { '@type': 'Thing', name: 'FlashAttention' }
          ]
        }
      ]
    })]
  ],

  // ==========================================
  // Sitemap
  // ==========================================
  sitemap: {
    hostname: 'https://lessup.github.io',
    transformItems(items) {
      return items.map((item) => {
        item.changefreq = 'weekly'
        item.priority = item.url === '/diy-flash-attention/' ? 1.0 : 
                       item.url.includes('tutorial') ? 0.9 : 0.8
        return item
      })
    }
  },

  // ==========================================
  // Markdown
  // ==========================================
  markdown: {
    lineNumbers: true,
    languageAlias: {
      cuda: 'cpp',
      triton: 'python',
      py: 'python',
    },
    theme: {
      light: 'github-light',
      dark: 'github-dark'
    }
  },

  // ==========================================
  // Vite Config
  // ==========================================
  vite: {
    build: {
      chunkSizeWarningLimit: 1000,
    },
    optimizeDeps: {
      include: ['vue']
    },
    ssr: {
      noExternal: ['@vueuse/core']
    }
  },

  // ==========================================
  // Theme Config
  // ==========================================
  themeConfig: {
    logo: { src: '/logo.svg', width: 24, height: 24 },
    siteTitle: 'DIY FlashAttention',
    
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Tutorial', link: '/en/tutorial' },
      { text: 'API', link: '/en/api' },
      {
        text: 'Guides',
        items: [
          { text: 'Performance', link: '/en/performance' },
          { text: 'Cheatsheet', link: '/en/cheatsheet' },
          { text: 'FAQ', link: '/en/faq' },
        ]
      },
      { text: 'Changelog', link: '/en/changelog' },
      {
        text: '🌐',
        items: [
          { text: 'English', link: '/en/' },
          { text: '中文', link: '/zh/' },
        ]
      },
    ],

    sidebar: {
      '/en/': [
        {
          text: '🚀 Getting Started',
          collapsed: false,
          items: [
            { text: 'Introduction', link: '/en/' },
            { text: 'Quick Start', link: '/en/tutorial#quick-start' },
          ]
        },
        {
          text: '📚 Learning',
          collapsed: false,
          items: [
            { text: 'Tutorial', link: '/en/tutorial' },
            { text: 'API Reference', link: '/en/api' },
          ]
        },
        {
          text: '⚡ Advanced',
          collapsed: false,
          items: [
            { text: 'Performance Guide', link: '/en/performance' },
            { text: 'Cheatsheet', link: '/en/cheatsheet' },
            { text: 'FAQ', link: '/en/faq' },
          ]
        },
        {
          text: '📖 Reference',
          collapsed: false,
          items: [
            { text: 'Changelog', link: '/en/changelog' },
            { text: 'GitHub', link: 'https://github.com/LessUp/diy-flash-attention' },
          ]
        },
      ],
      '/zh/': [
        {
          text: '🚀 快速开始',
          collapsed: false,
          items: [
            { text: '简介', link: '/zh/' },
            { text: '快速上手', link: '/zh/tutorial' },
          ]
        },
        {
          text: '📚 学习',
          collapsed: false,
          items: [
            { text: '教程', link: '/zh/tutorial' },
            { text: 'API 参考', link: '/zh/api' },
          ]
        },
        {
          text: '⚡ 进阶',
          collapsed: false,
          items: [
            { text: '性能指南', link: '/zh/performance' },
            { text: '速查表', link: '/zh/cheatsheet' },
            { text: '常见问题', link: '/zh/faq' },
          ]
        },
        {
          text: '📖 参考',
          collapsed: false,
          items: [
            { text: '更新日志', link: '/zh/changelog' },
          ]
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/LessUp/diy-flash-attention' },
    ],

    editLink: {
      pattern: 'https://github.com/LessUp/diy-flash-attention/edit/master/docs/:path',
      text: '✏️ Edit this page',
    },

    footer: {
      message: '<a href="/diy-flash-attention/en/changelog">Changelog</a> • <a href="https://github.com/LessUp/diy-flash-attention/blob/master/LICENSE">MIT License</a>',
      copyright: 'Copyright © 2024-2026 <a href="https://github.com/LessUp">LessUp</a>'
    },

    search: {
      provider: 'local',
      options: {
        detailedView: true,
        miniSearch: {
          options: {
            boost: { title: 10, titles: 8, text: 1 },
            fuzzy: 0.2,
            prefix: true
          }
        },
        locales: {
          zh: {
            translations: {
              button: {
                buttonText: '搜索文档',
                buttonAriaLabel: '搜索文档',
              },
              modal: {
                noResultsText: '无法找到相关结果',
                resetButtonTitle: '清除查询条件',
                footer: {
                  selectText: '选择',
                  navigateText: '切换',
                  closeText: '关闭',
                },
              },
            },
          },
        },
      },
    },

    outline: { level: [2, 3], label: 'On this page' },
    docFooter: { prev: '← Previous', next: 'Next →' },
    
    lastUpdated: {
      text: '🕐 Updated on',
      formatOptions: { dateStyle: 'medium', timeStyle: 'short' }
    },

    returnToTopLabel: '↑ Return to top',
    sidebarMenuLabel: 'Menu',
    darkModeSwitchLabel: '🌓',
    darkModeSwitchTitle: 'Switch to dark mode',
    lightModeSwitchTitle: 'Switch to light mode',
    codeCopyButtonTitle: '📋 Copy',
    externalLinkIcon: true,
  },

  // ==========================================
  // Locales
  // ==========================================
  locales: {
    root: { label: 'English', lang: 'en-US', link: '/en/' },
    zh: {
      label: '中文',
      lang: 'zh-CN',
      link: '/zh/',
      themeConfig: {
        nav: [
          { text: '首页', link: '/zh/' },
          { text: '教程', link: '/zh/tutorial' },
          { text: 'API', link: '/zh/api' },
          {
            text: '指南',
            items: [
              { text: '性能指南', link: '/zh/performance' },
              { text: '速查表', link: '/zh/cheatsheet' },
              { text: '常见问题', link: '/zh/faq' },
            ]
          },
          { text: '更新日志', link: '/zh/changelog' },
          {
            text: '🌐',
            items: [
              { text: 'English', link: '/en/' },
              { text: '中文', link: '/zh/' },
            ]
          },
        ],
        editLink: {
          pattern: 'https://github.com/LessUp/diy-flash-attention/edit/master/docs/:path',
          text: '✏️ 在 GitHub 上编辑此页',
        },
        outline: { level: [2, 3], label: '本页导航' },
        docFooter: { prev: '← 上一页', next: '下一页 →' },
        lastUpdated: {
          text: '🕐 最后更新于',
          formatOptions: { dateStyle: 'medium', timeStyle: 'short' }
        },
        returnToTopLabel: '↑ 返回顶部',
        sidebarMenuLabel: '菜单',
        darkModeSwitchLabel: '🌓',
        darkModeSwitchTitle: '切换到深色模式',
        lightModeSwitchTitle: '切换到浅色模式',
      }
    }
  }
})
