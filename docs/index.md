---
layout: home

hero:
  name: "DIY FlashAttention"
  text: "Master GPU Programming"
  tagline: |
    Implement FlashAttention from scratch using Python & OpenAI Triton

    <div class="hero-badges">
      <span class="badge">⚡ 99% Memory Reduction</span>
      <span class="badge">🚀 1.6x Speedup</span>
      <span class="badge">🎯 Production Ready</span>
    </div>

  actions:
    - theme: brand
      text: 🚀 Get Started
      link: /en/tutorial
    - theme: alt
      text: 💻 GitHub
      link: https://github.com/LessUp/diy-flash-attention

features:
  - icon: ⚡
    title: Learn Triton
    details: Master block pointer arithmetic, tiling strategies, and autotune optimization through hands-on matrix multiplication kernels.
    link: /en/tutorial

  - icon: 🧠
    title: FlashAttention
    details: Implement the core attention acceleration algorithm from scratch with O(N) memory complexity and causal masking support.
    link: /en/tutorial#part-4

  - icon: 📊
    title: Performance
    details: Benchmark against PyTorch SDPA, quantify optimization effects, and explore how block size impacts GPU performance.
    link: /en/performance

  - icon: 🖥️
    title: Modern GPU Support
    details: Auto-detect GPU architectures from Volta to Blackwell. Support for Hopper features like TMA and FP8.
    link: /en/api#gpu-detection

  - icon: ✅
    title: Tested & Verified
    details: Comprehensive test suite with property-based testing using Hypothesis. Validated on V100 through H100.
    link: https://github.com/LessUp/diy-flash-attention/tree/master/tests

  - icon: 🌍
    title: Bilingual Docs
    details: Complete documentation in English and Chinese (中文). Tutorial, API reference, guides, and FAQ in both languages.
    link: /zh/
---

<style>
.hero-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  justify-content: center;
  margin-top: 1.5rem;
}

.badge {
  display: inline-flex;
  align-items: center;
  padding: 0.5rem 1rem;
  background: rgba(16, 185, 129, 0.1);
  border: 1px solid rgba(16, 185, 129, 0.2);
  border-radius: 9999px;
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--vp-c-brand-1);
  backdrop-filter: blur(10px);
  transition: all 0.3s;
}

.badge:hover {
  background: rgba(16, 185, 129, 0.2);
  transform: translateY(-2px);
}

@media (max-width: 768px) {
  .hero-badges { flex-direction: column; align-items: center; }
}
</style>

## 🌐 Choose Your Language

<div class="language-selector">
  <a href="/diy-flash-attention/en/" class="lang-card">
    <span class="lang-icon">🇺🇸</span>
    <span class="lang-name">English</span>
    <span class="lang-desc">Complete documentation in English</span>
  </a>
  <a href="/diy-flash-attention/zh/" class="lang-card">
    <span class="lang-icon">🇨🇳</span>
    <span class="lang-name">中文</span>
    <span class="lang-desc">完整中文文档</span>
  </a>
</div>

<style>
.language-selector {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin: 2rem 0;
}

.lang-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 2rem;
  background: var(--vp-c-bg-soft);
  border: 2px solid var(--vp-c-divider);
  border-radius: 1rem;
  text-decoration: none;
  transition: all 0.3s;
}

.lang-card:hover {
  border-color: var(--vp-c-brand-1);
  transform: translateY(-4px);
  box-shadow: 0 10px 40px rgba(16, 185, 129, 0.15);
}

.lang-icon { font-size: 3rem; margin-bottom: 1rem; }
.lang-name { font-size: 1.5rem; font-weight: 700; color: var(--vp-c-text-1); margin-bottom: 0.5rem; }
.lang-desc { font-size: 0.875rem; color: var(--vp-c-text-2); text-align: center; }
</style>
