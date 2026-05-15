// ============================================
// DIY FlashAttention - Theme v3.0
// ============================================

import Theme from 'vitepress/theme'
import type { EnhanceAppContext } from 'vitepress'
import { watch } from 'vue'
import { useData } from 'vitepress'
import './style.css'
import './custom.css'

// Import custom interactive components
import GpuArchitectureVisualizer from './components/GpuArchitectureVisualizer.vue'
import FlashAttentionVisualizer from './components/FlashAttentionVisualizer.vue'
import BenchmarkChart from './components/BenchmarkChart.vue'
import ArchitectureDiagram from './components/ArchitectureDiagram.vue'
import ThemeAwareFigure from './components/ThemeAwareFigure.vue'

export default {
  extends: Theme,
  enhanceApp({ app, router }: EnhanceAppContext) {
    // Register global components
    app.component('GpuArchitectureVisualizer', GpuArchitectureVisualizer)
    app.component('FlashAttentionVisualizer', FlashAttentionVisualizer)
    app.component('BenchmarkChart', BenchmarkChart)
    app.component('ArchitectureDiagram', ArchitectureDiagram)
    app.component('ThemeAwareFigure', ThemeAwareFigure)

    // Client-side only code
    if (typeof window !== 'undefined') {
      // ============================================
      // 语言自动检测与偏好记忆
      // ============================================
      const LOCALE_KEY = 'vitepress-locale-preference'
      const basePath = (import.meta.env.BASE_URL || '/').replace(/\/$/, '')

      const stripBase = (path: string) => {
        if (!basePath) return path
        if (!path.startsWith(basePath)) return path

        const stripped = path.slice(basePath.length)
        return stripped || '/'
      }

      const withBasePath = (path: string) => `${basePath}${path}`

      // 首次访问：检测浏览器语言并重定向
      const initAutoLocale = () => {
        const stored = localStorage.getItem(LOCALE_KEY)
        if (stored) return // 已有偏好，不干预

        const browserLang = navigator.language
        const isZh = browserLang === 'zh-CN' || browserLang === 'zh'
        if (!isZh) return

        const path = window.location.pathname
        const relPath = stripBase(path).replace(/\/$/, '')

        // 当前在英文路径 → 重定向到中文对应页面
        if (!relPath.startsWith('/zh')) {
          const newPath = withBasePath(`/zh${relPath}`)
          window.location.replace(newPath)
        }
      }

      // Smooth scroll transitions + 记录语言偏好
      router.onBeforeRouteChange = () => {
        document.documentElement.style.scrollBehavior = 'auto'
      }
      router.onAfterRouteChanged = (to) => {
        document.documentElement.style.scrollBehavior = 'smooth'

        // 记录用户语言偏好
        const normalizedTo = stripBase(to)
        const isZh = normalizedTo === '/zh' || normalizedTo === '/zh/' || normalizedTo.startsWith('/zh/')
        localStorage.setItem(LOCALE_KEY, isZh ? 'zh' : 'en')
      }

      // 初始化语言检测
      initAutoLocale()

      // Keyboard shortcuts
      document.addEventListener('keydown', (e) => {
        // Cmd/Ctrl + K for search
        if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
          e.preventDefault()
          const searchBtn = document.querySelector('.VPNavBarSearch button')
          searchBtn?.click()
        }
      })
    }
  },

  setup() {
    // Mermaid 主题动态切换
    const { isDark } = useData()

    watch(isDark, (dark) => {
      // Mermaid 运行时主题切换
      if (typeof window !== 'undefined' && (window as any).mermaid) {
        (window as any).mermaid.initialize({
          startOnLoad: false,
          theme: dark ? 'dark' : 'default'
        })
      }
    }, { immediate: true })
  }
}
