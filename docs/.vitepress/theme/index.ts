// ============================================
// DIY FlashAttention - Ultra Modern Theme
// Version: 2.0 - Aggressive Modernization
// ============================================

import { h } from 'vue'
import Theme from 'vitepress/theme'
import type { EnhanceAppContext } from 'vitepress'
import './style.css'
import './custom.css'

// Import custom components
import GpuArchitectureVisualizer from './components/GpuArchitectureVisualizer.vue'
import FlashAttentionVisualizer from './components/FlashAttentionVisualizer.vue'
import CodeCopy from './components/CodeCopy.vue'
import BenchmarkChart from './components/BenchmarkChart.vue'

export default {
  extends: Theme,
  enhanceApp({ app, router, siteData }: EnhanceAppContext) {
    // Register global components
    app.component('GpuArchitectureVisualizer', GpuArchitectureVisualizer)
    app.component('FlashAttentionVisualizer', FlashAttentionVisualizer)
    app.component('CodeCopy', CodeCopy)
    app.component('BenchmarkChart', BenchmarkChart)

    // Router guards for smooth transitions
    router.onBeforeRouteChange = () => {
      document.documentElement.style.scrollBehavior = 'auto'
    }
    router.onAfterRouteChanged = () => {
      document.documentElement.style.scrollBehavior = 'smooth'
      
      // Add copy buttons to code blocks
      if (typeof window !== 'undefined') {
        setTimeout(() => {
          addCopyButtons()
        }, 100)
      }
    }

    // Initialize PWA
    if (typeof window !== 'undefined' && 'serviceWorker' in navigator) {
      window.addEventListener('load', () => {
        navigator.serviceWorker.register('/diy-flash-attention/sw.js')
          .then(reg => console.log('SW registered:', reg))
          .catch(err => console.log('SW registration failed:', err))
      })
    }

    // Keyboard shortcuts
    if (typeof window !== 'undefined') {
      document.addEventListener('keydown', (e) => {
        // Cmd/Ctrl + K for search
        if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
          e.preventDefault()
          const searchBtn = document.querySelector('.DocSearch-Button') || 
                           document.querySelector('.VPNavBarSearch button')
          searchBtn?.click()
        }
        // Cmd/Ctrl + / for theme toggle
        if ((e.metaKey || e.ctrlKey) && e.key === '/') {
          e.preventDefault()
          const themeToggle = document.querySelector('.VPSwitch')
          themeToggle?.click()
        }
      })
    }
  }
}

// Helper function to add copy buttons to code blocks
function addCopyButtons() {
  const codeBlocks = document.querySelectorAll('.vp-doc div[class*="language-"]')
  codeBlocks.forEach(block => {
    if (block.querySelector('.copy-btn')) return // Already has button
    
    const pre = block.querySelector('pre')
    if (!pre) return
    
    const code = pre.textContent || ''
    const wrapper = document.createElement('div')
    wrapper.className = 'code-copy-wrapper'
    wrapper.innerHTML = `
      <button class="copy-btn" title="Copy to clipboard">
        <span class="icon">📋</span>
        <span class="text">Copy</span>
      </button>
    `
    
    block.style.position = 'relative'
    block.appendChild(wrapper)
    
    const btn = wrapper.querySelector('.copy-btn')
    btn?.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(code)
        btn.classList.add('copied')
        btn.querySelector('.icon').textContent = '✓'
        btn.querySelector('.text').textContent = 'Copied!'
        setTimeout(() => {
          btn.classList.remove('copied')
          btn.querySelector('.icon').textContent = '📋'
          btn.querySelector('.text').textContent = 'Copy'
        }, 2000)
      } catch (err) {
        console.error('Copy failed:', err)
      }
    })
  })
}
