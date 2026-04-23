---
layout: home

hero:
  name: "DIY"
  text: "FlashAttention"
  tagline: |
    A compact Triton project for learning attention kernels, benchmarks, and GPU-aware configuration
    
    <div class="badge-group">
      <span class="badge">⚡ 99% Memory Reduction</span>
      <span class="badge purple">🚀 1.6x Speedup</span>
      <span class="badge yellow">🎯 Hands-on Learning</span>
    </div>

  actions:
    - theme: brand
      text: 🚀 Start Tutorial
      link: /en/tutorial
    - theme: alt
      text: 📊 Benchmark Guide
      link: /en/performance
    - theme: alt
      text: 💻 GitHub
      link: https://github.com/LessUp/diy-flash-attention

features:
  - icon: 🔷
    title: Read Real Triton Code
    details: Study the actual matmul and FlashAttention kernels instead of abstract slides.
    link: /en/tutorial

  - icon: ⚡
    title: Follow the Attention Algorithm
    details: Trace online softmax, SRAM tiling, and causal masking through a compact forward implementation.
    link: /en/tutorial

  - icon: 📊
    title: Benchmark Against PyTorch
    details: Compare speed and memory behavior against PyTorch SDPA with repository-native scripts.
    link: /en/performance

  - icon: 🖥️
    title: Inspect Architecture Helpers
    details: Explore Volta → Blackwell feature detection, Hopper TMA flags, and FP8-aware helpers.
    link: /en/api
---

## Why This Page Exists

This page is the **project overview** for readers who want to decide where to go next without reading the whole README first.

| Guide | Description |
|-------|-------------|
| [Tutorial](/en/tutorial) | Best entry if you want to understand the implementation line by line |
| [API Reference](/en/api) | Best entry if you want the supported kernel and helper contract |
| [Performance](/en/performance) | Best entry if you care about benchmark evidence and trade-offs |
| [Cheatsheet](/en/cheatsheet) | Best entry if you already know Triton and want a quick refresher |
| [FAQ](/en/faq) | Best entry if you want environment and troubleshooting guidance |

## What This Repository Covers

<div class="highlight-box">
  <p><strong>Scope:</strong> Triton matmul, forward-only FlashAttention, GPU capability helpers, tests, and benchmark scripts.</p>
  <p><strong>Approach:</strong> Keep the code compact enough to read end-to-end while still being real enough to benchmark and validate.</p>
  <p><strong>Why it helps:</strong> You get a tractable way to learn how attention kernels are structured without jumping into a huge framework.</p>
</div>

## Project strengths

- **Compact implementation**: small enough to finish reading, not a toy sketch
- **Benchmark evidence**: compare directly against PyTorch SDPA
- **Architecture awareness**: inspect Volta → Blackwell helper logic
- **Bilingual docs**: English and Chinese pages stay aligned

<div class="cta-section">
  <div class="cta-title">Start where your curiosity is strongest</div>
  <div class="cta-desc">Tutorial for understanding, performance pages for evidence, API docs for precise contracts.</div>
  <div class="cta-buttons">
    <a href="/diy-flash-attention/en/tutorial" class="cta-btn primary">
      <span>📚</span> Read Tutorial
    </a>
    <a href="https://github.com/LessUp/diy-flash-attention" class="cta-btn secondary">
      <span>⭐</span> Star on GitHub
    </a>
  </div>
</div>
