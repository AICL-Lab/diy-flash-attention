---
layout: home

hero:
  name: "DIY"
  text: "FlashAttention"
  tagline: |
    Learn Triton through a forward-only educational FlashAttention implementation

    <div class="badge-group">
      <span class="badge">⚡ 99% Memory Reduction</span>
      <span class="badge purple">🚀 1.6x Speedup</span>
      <span class="badge yellow">🎯 Hands-on Learning</span>
    </div>

  actions:
    - theme: brand
      text: 🚀 Start with the Tutorial
      link: /en/tutorial
    - theme: alt
      text: 📊 Explore Benchmarks
      link: /en/performance
    - theme: alt
      text: 💻 GitHub
      link: https://github.com/LessUp/diy-flash-attention

features:
  - icon: 🔷
    title: Read Real Kernels
    details: Study the actual Triton matmul and FlashAttention kernels instead of toy pseudocode.
    link: /en/tutorial

  - icon: ⚡
    title: Learn the O(N) Trick
    details: Understand online softmax, causal masking, and SRAM tiling without materializing the full attention matrix.
    link: /en/tutorial

  - icon: 📊
    title: Benchmark Against PyTorch
    details: Compare throughput and memory behavior against PyTorch SDPA with repository-native benchmark scripts.
    link: /en/performance

  - icon: 🖥️
    title: Inspect Architecture Helpers
    details: See how the project handles Volta → Blackwell feature detection, Hopper TMA, and FP8-aware configuration.
    link: /en/api
---

## Why This Repository Exists

This site is the **project landing page**, not just a copied README. It helps you decide where to jump in:

| If you want to... | Start here |
| --- | --- |
| understand the kernel design | [Tutorial](/en/tutorial) |
| inspect the public API and helpers | [API Reference](/en/api) |
| compare performance vs PyTorch SDPA | [Performance Guide](/en/performance) |
| scan the code from GitHub first | [Repository](https://github.com/LessUp/diy-flash-attention) |

## What Makes It Useful

<div class="feature-highlight">
  <div class="feature-content">
    <h3>Compact enough to finish, real enough to matter</h3>
    <p>This repository keeps the scope narrow on purpose: Triton matmul, forward-only FlashAttention, architecture-aware helpers, tests, and benchmarks.</p>
    <p>That makes it easier to understand the full stack end-to-end without getting lost in a giant training framework.</p>
  </div>
  <div class="feature-visual">
    <div class="code-preview">
      <div class="code-preview-header">
        <span class="code-dot red"></span>
        <span class="code-dot yellow"></span>
        <span class="code-dot green"></span>
        <span class="code-preview-filename">flash_attention.py</span>
      </div>
      <pre><code class="language-python">@triton.jit
    def flash_attn_kernel(...):
        # load a Q tile into SRAM
        pid_m = tl.program_id(0)
        # iterate over K/V tiles
        # maintain online softmax state
        # write the normalized output tile</code></pre>
    </div>
  </div>
</div>

## What You Can Inspect Here

<div class="stats-grid">
  <div class="stat-card">
    <div class="stat-value">2</div>
    <div class="stat-label">Core Triton Kernels</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">O(N)</div>
    <div class="stat-label">Attention Memory Path</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">Volta→Blackwell</div>
    <div class="stat-label">Architecture Coverage</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">EN / 中文</div>
    <div class="stat-label">Docs Languages</div>
  </div>
</div>

### Included in this project

- **Triton matmul** — autotune plus manual block-size control
- **FlashAttention forward** — causal masking, 3D/4D inputs, variable sequence lengths
- **Architecture helpers** — feature detection, FP8 conversion helpers, configuration selection
- **Benchmarks and docs** — PyTorch comparisons, walkthroughs, cheatsheets, bilingual pages

## Choose Your Starting Point

<div class="audience-grid">
  <div class="audience-card">
    <div class="audience-avatar">🧑‍💻</div>
    <div class="audience-title">Read the kernels</div>
    <div class="audience-benefit">Start with the tutorial if you want to trace the implementation line by line</div>
    <span class="audience-skill">Best path: Tutorial</span>
  </div>
  <div class="audience-card">
    <div class="audience-avatar">🔬</div>
    <div class="audience-title">Check the contract</div>
    <div class="audience-benefit">Use the API and OpenSpec docs if you want a precise view of supported behavior</div>
    <span class="audience-skill">Best path: API + OpenSpec</span>
  </div>
  <div class="audience-card">
    <div class="audience-avatar">🚀</div>
    <div class="audience-title">Benchmark the trade-offs</div>
    <div class="audience-benefit">Use the performance guide if you care about timing, memory, and block-size tuning</div>
    <span class="audience-skill">Best path: Performance</span>
  </div>
  <div class="audience-card">
    <div class="audience-avatar">📚</div>
    <div class="audience-title">Browse in your language</div>
    <div class="audience-benefit">Use English or Chinese docs depending on how you prefer to read technical walkthroughs</div>
    <span class="audience-skill">Best path: EN / 中文</span>
  </div>
</div>

<div class="cta-section">
  <div class="cta-title">Start with the part you care about most</div>
  <div class="cta-desc">Tutorial for understanding, API for contracts, performance pages for evidence, GitHub for code.</div>
  <div class="cta-buttons">
    <a href="/diy-flash-attention/en/tutorial" class="cta-btn primary">
      <span>🚀</span> Start Tutorial
    </a>
    <a href="https://github.com/LessUp/diy-flash-attention" class="cta-btn secondary">
      <span>⭐</span> Star on GitHub
    </a>
  </div>
</div>

## Language

<div class="lang-switcher">
  <a href="/diy-flash-attention/en/" class="lang-link active">
    <span>🇺🇸</span> English
  </a>
  <a href="/diy-flash-attention/zh/" class="lang-link">
    <span>🇨🇳</span> 中文
  </a>
</div>
