---
layout: home

hero:
  name: "DIY"
  text: "FlashAttention"
  tagline: |
    Academy portal for learning forward-only FlashAttention with Triton

  actions:
    - theme: brand
      text: 🧭 Start Learning
      link: /learning-path
    - theme: alt
      text: 📚 Read Papers
      link: /paper-guide
    - theme: alt
      text: 🗺️ Explore Concepts
      link: /knowledge-map
---

## Start Here

Use this portal to choose a study route first, then move into the reference docs that support each step.

<div class="academy-grid">

<div class="academy-card">
  <h3><a href="/diy-flash-attention/learning-path">Learning Path</a></h3>
  <p>Start from Triton basics, then move into online softmax and FlashAttention kernels.</p>
</div>

<div class="academy-card">
  <h3><a href="/diy-flash-attention/paper-guide">Paper Guide</a></h3>
  <p>Read the FlashAttention papers in an order that matches this repository.</p>
</div>

<div class="academy-card">
  <h3><a href="/diy-flash-attention/knowledge-map">Knowledge Map</a></h3>
  <p>See how concepts, source files, and docs connect.</p>
</div>

</div>

## Portal Overview

- **Learning Path** gives first-time readers a practical order: Triton basics → online softmax → attention kernels.
- **Paper Guide** keeps the reading order aligned with the repository's forward-only educational scope.
- **Knowledge Map** connects concepts, files, and docs so you can jump between theory and implementation.

## Reference Library

After the portal points you to the right entry, use these references to go deeper into implementation details.

<div class="doc-nav-grid">

<div class="doc-nav-card">
  <div class="doc-nav-icon">📖</div>
  <h3><a href="/diy-flash-attention/architecture">Architecture Design</a></h3>
  <p>System architecture, GPU memory hierarchy, kernel design, and design decisions.</p>
  <span class="doc-nav-tag">Whitepaper</span>
</div>

<div class="doc-nav-card">
  <div class="doc-nav-icon">📐</div>
  <h3><a href="/diy-flash-attention/algorithm">Algorithm Deep Dive</a></h3>
  <p>Online softmax derivation, tiling strategies, complexity analysis, and correctness proofs.</p>
  <span class="doc-nav-tag">Whitepaper</span>
</div>

<div class="doc-nav-card">
  <div class="doc-nav-icon">🚀</div>
  <h3><a href="/diy-flash-attention/tutorial">Tutorial</a></h3>
  <p>Step-by-step guide: GPU basics → Triton programming → FlashAttention implementation.</p>
  <span class="doc-nav-tag">Getting Started</span>
</div>

<div class="doc-nav-card">
  <div class="doc-nav-icon">📊</div>
  <h3><a href="/diy-flash-attention/performance">Performance Guide</a></h3>
  <p>Block size tuning, data type selection, GPU architecture adaptation, and benchmarking.</p>
  <span class="doc-nav-tag">Reference</span>
</div>

<div class="doc-nav-card">
  <div class="doc-nav-icon">🔧</div>
  <h3><a href="/diy-flash-attention/api">API Reference</a></h3>
  <p>Complete function signatures, parameters, return values, and usage examples.</p>
  <span class="doc-nav-tag">Reference</span>
</div>

<div class="doc-nav-card">
  <div class="doc-nav-icon">⚡</div>
  <h3><a href="/diy-flash-attention/cheatsheet">Cheatsheet</a></h3>
  <p>Quick reference: common APIs, commands, configuration templates, and error lookup.</p>
  <span class="doc-nav-tag">Quick Reference</span>
</div>

</div>

## Project Scope

- Educational, forward-only FlashAttention in Triton.
- Built to help readers connect papers, kernels, and performance trade-offs.
- Best used as a guided learning repo, not as a production training framework.

## Language

<div class="lang-switcher">
  <a href="/diy-flash-attention/" class="lang-link active">
    <span>🇺🇸</span> English
  </a>
  <a href="/diy-flash-attention/zh/" class="lang-link">
    <span>🇨🇳</span> 中文
  </a>
</div>
