import { defineConfig } from 'vitepress'

export default defineConfig({
  lang: 'zh-CN',
  title: 'DIY FlashAttention',
  description: '使用 Python + OpenAI Triton 从零实现 FlashAttention 算法',

  // GitHub Pages 部署：base 需要与仓库名一致
  base: '/diy-flash-attention/',

  cleanUrls: true,

  head: [
    ['link', { rel: 'canonical', href: 'https://lessup.github.io/diy-flash-attention/' }],
    ['meta', { name: 'theme-color', content: '#f97316' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'DIY FlashAttention' }],
    ['meta', { property: 'og:description', content: '使用 Python + OpenAI Triton 从零实现 FlashAttention 算法' }],
    ['meta', { property: 'og:url', content: 'https://lessup.github.io/diy-flash-attention/' }],
    ['meta', { name: 'twitter:card', content: 'summary' }],
    ['meta', { name: 'twitter:title', content: 'DIY FlashAttention' }],
    ['meta', { name: 'twitter:description', content: '使用 Python + OpenAI Triton 从零实现 FlashAttention 算法' }],
    ['meta', { name: 'keywords', content: 'FlashAttention,Triton,CUDA,GPU,Attention,LLM,Online Softmax,Tiling' }],
  ],

  markdown: {
    lineNumbers: true,
    languageAlias: {
      cuda: 'cpp',
    },
  },

  lastUpdated: true,

  themeConfig: {
    nav: [
      { text: '教程', link: '/tutorial' },
      { text: 'API', link: '/api' },
      {
        text: '参考',
        items: [
          { text: '性能指南', link: '/performance' },
          { text: '速查表', link: '/cheatsheet' },
          { text: 'FAQ', link: '/faq' },
        ],
      },
      { text: '变更日志', link: '/changelog' },
    ],

    sidebar: [
      {
        text: '入门',
        items: [
          { text: '教程', link: '/tutorial' },
          { text: 'API 参考', link: '/api' },
        ],
      },
      {
        text: '进阶',
        items: [
          { text: '性能指南', link: '/performance' },
          { text: '速查表', link: '/cheatsheet' },
          { text: 'FAQ', link: '/faq' },
        ],
      },
      {
        text: '变更日志',
        items: [
          { text: '总览', link: '/changelog' },
        ],
      },
    ],

    editLink: {
      pattern: 'https://github.com/LessUp/diy-flash-attention/edit/master/docs/:path',
      text: '在 GitHub 上编辑此页',
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/LessUp/diy-flash-attention' },
    ],

    footer: {
      message: '基于 MIT 许可发布',
      copyright: 'Copyright © 2025-2026 LessUp',
    },

    search: {
      provider: 'local',
    },

    outline: {
      level: [2, 3],
      label: '目录',
    },

    lastUpdated: {
      text: '最后更新',
    },

    docFooter: {
      prev: '上一页',
      next: '下一页',
    },

    returnToTopLabel: '返回顶部',
    sidebarMenuLabel: '菜单',
    darkModeSwitchLabel: '主题',
    externalLinkIcon: true,
  },
})
