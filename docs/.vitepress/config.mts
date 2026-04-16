import { defineConfig } from 'vitepress'

export default defineConfig({
  // Site metadata
  title: 'DIY FlashAttention',
  description: 'Implement FlashAttention from Scratch using Python + OpenAI Triton',
  
  // GitHub Pages deployment
  base: '/diy-flash-attention/',

  // Clean URLs (no .html extension)
  cleanUrls: true,
  lastUpdated: true,

  // Head metadata
  head: [
    ['link', { rel: 'canonical', href: 'https://lessup.github.io/diy-flash-attention/' }],
    ['meta', { name: 'theme-color', content: '#3b82f6' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'DIY FlashAttention' }],
    ['meta', { property: 'og:description', content: 'Implement FlashAttention from Scratch using Python + OpenAI Triton' }],
    ['meta', { property: 'og:url', content: 'https://lessup.github.io/diy-flash-attention/' }],
    ['meta', { property: 'og:image', content: 'https://lessup.github.io/diy-flash-attention/og-image.png' }],
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
    ['meta', { name: 'keywords', content: 'FlashAttention,Triton,CUDA,GPU,Attention,LLM,Online Softmax,Tiling,Deep Learning,Transformer,GPU Programming' }],
  ],

  // Markdown config
  markdown: {
    lineNumbers: true,
    languageAlias: {
      cuda: 'cpp',
      triton: 'python',
    },
  },

  // Theme config
  themeConfig: {
    // Logo
    logo: '/logo.svg',

    // Navigation
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
        ],
      },
      { text: 'Changelog', link: '/en/changelog' },
      {
        text: '🌐 Language',
        items: [
          { text: 'English', link: '/en/' },
          { text: '中文', link: '/zh/' },
        ],
      },
    ],

    // Sidebar
    sidebar: {
      '/en/': [
        {
          text: 'Getting Started',
          items: [
            { text: 'Home', link: '/en/' },
          ],
        },
        {
          text: 'Learn',
          collapsed: false,
          items: [
            { text: 'Tutorial', link: '/en/tutorial' },
            { text: 'API Reference', link: '/en/api' },
          ],
        },
        {
          text: 'Guides',
          collapsed: false,
          items: [
            { text: 'Performance Guide', link: '/en/performance' },
            { text: 'Cheatsheet', link: '/en/cheatsheet' },
            { text: 'FAQ', link: '/en/faq' },
          ],
        },
        {
          text: 'Reference',
          collapsed: false,
          items: [
            { text: 'Changelog', link: '/en/changelog' },
          ],
        },
      ],
      '/zh/': [
        {
          text: '开始',
          items: [
            { text: '首页', link: '/zh/' },
          ],
        },
        {
          text: '学习',
          collapsed: false,
          items: [
            { text: '教程', link: '/zh/tutorial' },
            { text: 'API 参考', link: '/zh/api' },
          ],
        },
        {
          text: '指南',
          collapsed: false,
          items: [
            { text: '性能指南', link: '/zh/performance' },
            { text: '速查表', link: '/zh/cheatsheet' },
            { text: '常见问题', link: '/zh/faq' },
          ],
        },
        {
          text: '参考',
          collapsed: false,
          items: [
            { text: '更新日志', link: '/zh/changelog' },
          ],
        },
      ],
    },

    // Social links
    socialLinks: [
      { icon: 'github', link: 'https://github.com/LessUp/diy-flash-attention' },
    ],

    // Edit link
    editLink: {
      pattern: 'https://github.com/LessUp/diy-flash-attention/edit/master/docs/:path',
      text: 'Edit this page on GitHub',
    },

    // Footer
    footer: {
      message: 'Released under the MIT License',
      copyright: 'Copyright © 2024-2026 LessUp',
    },

    // Local search
    search: {
      provider: 'local',
      options: {
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
                },
              },
            },
          },
        },
      },
    },

    // Outline config
    outline: {
      level: [2, 3],
      label: 'On this page',
    },

    // Document footer
    docFooter: {
      prev: 'Previous',
      next: 'Next',
    },

    // Last updated
    lastUpdated: {
      text: 'Last updated on',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'short',
      },
    },

    // Return to top
    returnToTopLabel: 'Return to top',

    // Sidebar menu
    sidebarMenuLabel: 'Menu',

    // Dark mode
    darkModeSwitchLabel: 'Theme',
    darkModeSwitchTitle: 'Switch to dark mode',
    lightModeSwitchTitle: 'Switch to light mode',

    // External link icon
    externalLinkIcon: true,
  },

  // i18n locales
  locales: {
    root: {
      label: 'English',
      lang: 'en',
      link: '/en/',
    },
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
            ],
          },
          { text: '更新日志', link: '/zh/changelog' },
          {
            text: '🌐 语言',
            items: [
              { text: 'English', link: '/en/' },
              { text: '中文', link: '/zh/' },
            ],
          },
        ],
        editLink: {
          pattern: 'https://github.com/LessUp/diy-flash-attention/edit/master/docs/:path',
          text: '在 GitHub 上编辑此页',
        },
        footer: {
          message: '基于 MIT 许可发布',
          copyright: 'Copyright © 2024-2026 LessUp',
        },
        outline: {
          level: [2, 3],
          label: '页面导航',
        },
        docFooter: {
          prev: '上一页',
          next: '下一页',
        },
        lastUpdated: {
          text: '最后更新于',
          formatOptions: {
            dateStyle: 'short',
            timeStyle: 'short',
          },
        },
        returnToTopLabel: '返回顶部',
        sidebarMenuLabel: '菜单',
        darkModeSwitchLabel: '主题',
        darkModeSwitchTitle: '切换到深色模式',
        lightModeSwitchTitle: '切换到浅色模式',
      },
    },
  },
})
