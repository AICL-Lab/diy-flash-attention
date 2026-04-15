import { defineConfig } from 'vitepress'

export default defineConfig({
  lang: 'zh-CN',
  title: 'DIY FlashAttention',
  description: '使用 Python + OpenAI Triton 从零实现 FlashAttention 算法',

  // GitHub Pages 部署：base 需要与仓库名一致
  base: '/diy-flash-attention/',

  cleanUrls: true,
  lastUpdated: true,

  head: [
    ['link', { rel: 'canonical', href: 'https://lessup.github.io/diy-flash-attention/' }],
    ['meta', { name: 'theme-color', content: '#3b82f6' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'DIY FlashAttention' }],
    ['meta', { property: 'og:description', content: '使用 Python + OpenAI Triton 从零实现 FlashAttention 算法' }],
    ['meta', { property: 'og:url', content: 'https://lessup.github.io/diy-flash-attention/' }],
    ['meta', { property: 'og:image', content: 'https://lessup.github.io/diy-flash-attention/og-image.png' }],
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
    ['meta', { name: 'twitter:title', content: 'DIY FlashAttention' }],
    ['meta', { name: 'twitter:description', content: '使用 Python + OpenAI Triton 从零实现 FlashAttention 算法' }],
    ['meta', { name: 'keywords', content: 'FlashAttention,Triton,CUDA,GPU,Attention,LLM,Online Softmax,Tiling,Deep Learning,Transformer' }],
  ],

  markdown: {
    lineNumbers: true,
    languageAlias: {
      cuda: 'cpp',
      triton: 'python',
    },
  },

  themeConfig: {
    // 导航栏
    nav: [
      { text: '首页', link: '/' },
      { text: '教程', link: '/tutorial' },
      { text: 'API', link: '/api' },
      {
        text: '指南',
        items: [
          { text: '性能指南', link: '/performance' },
          { text: '速查表', link: '/cheatsheet' },
          { text: 'FAQ', link: '/faq' },
        ],
      },
      { text: '更新日志', link: '/changelog' },
    ],

    // 侧边栏
    sidebar: {
      '/': [
        {
          text: '开始',
          items: [
            { text: '首页', link: '/' },
          ],
        },
        {
          text: '学习',
          collapsed: false,
          items: [
            { text: '教程', link: '/tutorial' },
            { text: 'API 参考', link: '/api' },
          ],
        },
        {
          text: '指南',
          collapsed: false,
          items: [
            { text: '性能指南', link: '/performance' },
            { text: '速查表', link: '/cheatsheet' },
            { text: '常见问题', link: '/faq' },
          ],
        },
        {
          text: '参考',
          collapsed: false,
          items: [
            { text: '更新日志', link: '/changelog' },
          ],
        },
      ],
    },

    // 社交链接
    socialLinks: [
      { icon: 'github', link: 'https://github.com/LessUp/diy-flash-attention' },
    ],

    // 编辑链接
    editLink: {
      pattern: 'https://github.com/LessUp/diy-flash-attention/edit/master/docs/:path',
      text: '在 GitHub 上编辑此页',
    },

    // 页脚
    footer: {
      message: '基于 MIT 许可发布',
      copyright: 'Copyright © 2024-2026 LessUp',
    },

    // 本地搜索
    search: {
      provider: 'local',
      options: {
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

    // 大纲配置
    outline: {
      level: [2, 3],
      label: '页面导航',
    },

    // 文档页脚
    docFooter: {
      prev: '上一页',
      next: '下一页',
    },

    // 最后更新时间
    lastUpdated: {
      text: '最后更新于',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'short',
      },
    },

    // 返回顶部
    returnToTopLabel: '返回顶部',

    // 侧边栏菜单
    sidebarMenuLabel: '菜单',

    // 深色模式
    darkModeSwitchLabel: '主题',
    darkModeSwitchTitle: '切换到深色模式',
    lightModeSwitchTitle: '切换到浅色模式',

    // 外部链接图标
    externalLinkIcon: true,
  },
})
