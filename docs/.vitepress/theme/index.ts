// ============================================
// DIY FlashAttention - Theme v3.0
// ============================================

import Theme from 'vitepress/theme'
import type { EnhanceAppContext } from 'vitepress'
import './style.css'
import './custom.css'

// Import custom interactive components
import GpuArchitectureVisualizer from './components/GpuArchitectureVisualizer.vue'
import FlashAttentionVisualizer from './components/FlashAttentionVisualizer.vue'
import BenchmarkChart from './components/BenchmarkChart.vue'

export default {
  extends: Theme,
  enhanceApp({ app, router }: EnhanceAppContext) {
    // Register global components
    app.component('GpuArchitectureVisualizer', GpuArchitectureVisualizer)
    app.component('FlashAttentionVisualizer', FlashAttentionVisualizer)
    app.component('BenchmarkChart', BenchmarkChart)

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
  }
}
