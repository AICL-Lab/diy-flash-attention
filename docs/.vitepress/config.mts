import { defineConfig } from 'vitepress'

export default defineConfig({
  lang: 'zh-CN',
  title: 'DIY FlashAttention',
  description: '使用 Python + OpenAI Triton 从零实现 FlashAttention 算法',

  // GitHub Pages 部署：base 需要与仓库名一致
  base: '/diy-flash-attention/',

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
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/LessUp/diy-flash-attention' },
    ],

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
  },
})
