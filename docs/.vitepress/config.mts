import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import llmstxt from 'vitepress-plugin-llms'

// ============================================
// DIY FlashAttention - VitePress Config
// ============================================

// 动态 base 路径：支持本地开发 (/) 和 GitHub Pages (/diy-flash-attention/)
const rawBase = process.env.VITEPRESS_BASE
const base = rawBase
  ? rawBase.startsWith('/')
    ? rawBase.endsWith('/') ? rawBase : `${rawBase}/`
    : `/${rawBase}/`
  : '/diy-flash-attention/'

export default withMermaid(defineConfig({
  // Core Settings
  lang: 'en-US',
  title: 'DIY FlashAttention',
  titleTemplate: ':title | DIY FlashAttention',
  description: 'Forward-only educational FlashAttention in Triton, with benchmarks, architecture-aware helpers, and bilingual docs.',

  base: '/diy-flash-attention/',
  cleanUrls: true,
  lastUpdated: true,
  appearance: true, // 启用主题切换，自动检测浏览器偏好

  // Head Configuration
  head: [
    // PWA
    ['link', { rel: 'manifest', href: '/diy-flash-attention/manifest.json' }],
    ['meta', { name: 'theme-color', content: '#06b6d4' }],

    // SEO
    ['meta', { name: 'author', content: 'LessUp' }],
    ['meta', { name: 'keywords', content: 'DIY FlashAttention, Triton, CUDA, FlashAttention, GPU programming, attention kernels, educational project' }],

    // Open Graph
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:site_name', content: 'DIY FlashAttention' }],
    ['meta', { property: 'og:title', content: 'DIY FlashAttention | Learn Triton by Building Attention Kernels' }],
    ['meta', { property: 'og:description', content: 'Forward-only educational FlashAttention in Triton, with benchmarks, architecture-aware helpers, and bilingual docs.' }],
    ['meta', { property: 'og:url', content: 'https://lessup.github.io/diy-flash-attention/' }],
    ['meta', { property: 'og:image', content: 'https://lessup.github.io/diy-flash-attention/og-image.svg' }],

    // Twitter
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
    ['meta', { name: 'twitter:title', content: 'DIY FlashAttention' }],
    ['meta', { name: 'twitter:description', content: 'Forward-only educational FlashAttention in Triton, with benchmarks and bilingual docs.' }],
    ['meta', { name: 'twitter:image', content: 'https://lessup.github.io/diy-flash-attention/og-image.svg' }],

    // Fonts
    ['link', { rel: 'preconnect', href: 'https://fonts.googleapis.com' }],
    ['link', { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' }],
    ['link', { rel: 'stylesheet', href: 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap' }],

    // Icons
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/diy-flash-attention/logo.svg' }],

    // JSON-LD Structured Data
    ['script', { type: 'application/ld+json' }, JSON.stringify({
      '@context': 'https://schema.org',
      '@type': 'TechArticle',
      headline: 'DIY FlashAttention',
      description: 'Learn Triton by implementing FlashAttention from scratch using Python and CUDA GPUs.',
      author: { '@type': 'Organization', name: 'LessUp' },
      publisher: { '@type': 'Organization', name: 'LessUp' },
      mainEntityOfPage: { '@type': 'WebPage', '@id': 'https://lessup.github.io/diy-flash-attention/' },
      about: [
        { '@type': 'Thing', name: 'DIY FlashAttention' },
        { '@type': 'Thing', name: 'CUDA' },
        { '@type': 'Thing', name: 'Triton' },
        { '@type': 'Thing', name: 'FlashAttention' }
      ]
    })]
  ],

  // Sitemap
  sitemap: {
    hostname: 'https://lessup.github.io/diy-flash-attention/',
  },

  // Markdown
  markdown: {
    lineNumbers: true,
    theme: {
      light: 'github-light',
      dark: 'github-dark'
    }
  },

  // Theme Configuration
  themeConfig: {
    logo: { src: '/logo.svg', width: 28, height: 28 },
    siteTitle: 'DIY FlashAttention',

    nav: [
      { text: 'Home', link: '/' },
      { text: 'Tutorial', link: '/en/tutorial' },
      { text: 'API', link: '/en/api' },
      {
        text: 'Resources',
        items: [
          { text: 'Performance Guide', link: '/en/performance' },
          { text: 'Cheatsheet', link: '/en/cheatsheet' },
          { text: 'FAQ', link: '/en/faq' },
        ]
      },
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
          text: 'Getting Started',
          collapsed: false,
          items: [
            { text: 'Introduction', link: '/en/' },
            { text: 'Tutorial', link: '/en/tutorial' },
            { text: 'API Reference', link: '/en/api' },
            { text: 'Tensor Layout Guide', link: '/en/tensor-layout' },
          ]
        },
        {
          text: 'Whitepaper',
          collapsed: false,
          items: [
            { text: 'Architecture', link: '/en/architecture' },
            { text: 'Algorithm', link: '/en/algorithm' },
          ]
        },
        {
          text: 'Resources',
          collapsed: false,
          items: [
            { text: 'Performance Guide', link: '/en/performance' },
            { text: 'Cheatsheet', link: '/en/cheatsheet' },
            { text: 'FAQ', link: '/en/faq' },
          ]
        },
        {
          text: 'Reference',
          collapsed: false,
          items: [
            { text: 'Changelog', link: '/en/changelog' },
            { text: 'GitHub →', link: 'https://github.com/LessUp/diy-flash-attention' },
          ]
        },
      ],
      '/zh/': [
        {
          text: '开始学习',
          collapsed: false,
          items: [
            { text: '简介', link: '/zh/' },
            { text: '教程', link: '/zh/tutorial' },
            { text: 'API 参考', link: '/zh/api' },
            { text: '张量布局指南', link: '/zh/tensor-layout' },
          ]
        },
        {
          text: '白皮书',
          collapsed: false,
          items: [
            { text: '架构设计', link: '/zh/architecture' },
            { text: '算法详解', link: '/zh/algorithm' },
          ]
        },
        {
          text: '资源',
          collapsed: false,
          items: [
            { text: '性能指南', link: '/zh/performance' },
            { text: '速查表', link: '/zh/cheatsheet' },
            { text: '常见问题', link: '/zh/faq' },
          ]
        },
        {
          text: '参考',
          collapsed: false,
          items: [
            { text: '更新日志', link: '/zh/changelog' },
            { text: 'GitHub →', link: 'https://github.com/LessUp/diy-flash-attention' },
          ]
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/LessUp/diy-flash-attention' },
    ],

    editLink: {
      pattern: 'https://github.com/LessUp/diy-flash-attention/edit/master/docs/:path',
      text: 'Edit this page',
    },

    footer: {
      message: 'Forward-only educational Triton FlashAttention project · MIT License',
      copyright: 'Copyright © 2024-2026 LessUp',
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
    docFooter: { prev: 'Previous', next: 'Next' },

    lastUpdated: {
      text: 'Updated',
      formatOptions: { dateStyle: 'medium' }
    },

    returnToTopLabel: 'Top',
    sidebarMenuLabel: 'Menu',
    darkModeSwitchLabel: 'Theme',
  },

  // Locales
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
            text: '资源',
            items: [
              { text: '性能指南', link: '/zh/performance' },
              { text: '速查表', link: '/zh/cheatsheet' },
              { text: '常见问题', link: '/zh/faq' },
            ]
          },
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
          text: '在 GitHub 上编辑此页',
        },
        outline: { level: [2, 3], label: '本页导航' },
        docFooter: { prev: '上一页', next: '下一页' },
        lastUpdated: {
          text: '更新于',
          formatOptions: { dateStyle: 'medium' }
        },
        returnToTopLabel: '返回顶部',
        sidebarMenuLabel: '菜单',
        darkModeSwitchLabel: '主题',
      }
    }
  },

  // Vite Configuration
  vite: {
    plugins: [llmstxt()],
    build: {
      chunkSizeWarningLimit: 1000,
    },
    optimizeDeps: {
      include: ['vue']
    }
  }
}))
