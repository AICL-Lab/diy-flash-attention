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

export default {
  extends: Theme,
  enhanceApp({ app, router }: EnhanceAppContext) {
    // Register global components
    app.component('GpuArchitectureVisualizer', GpuArchitectureVisualizer)
    app.component('FlashAttentionVisualizer', FlashAttentionVisualizer)
    app.component('BenchmarkChart', BenchmarkChart)
    app.component('ArchitectureDiagram', ArchitectureDiagram)

    // Client-side only code
    if (typeof window !== 'undefined') {
      // Smooth scroll transitions
      router.onBeforeRouteChange = () => {
        document.documentElement.style.scrollBehavior = 'auto'
      }
      router.onAfterRouteChanged = () => {
        document.documentElement.style.scrollBehavior = 'smooth'
      }

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
