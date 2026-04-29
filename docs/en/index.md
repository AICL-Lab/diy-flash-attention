---
layout: home

hero:
  name: "DIY"
  text: "FlashAttention"
  tagline: |
    Build FlashAttention from scratch with Triton — master GPU kernel optimization
    
    <div class="badge-group">
      <span class="badge">⚡ 99% Memory Reduction</span>
      <span class="badge purple">🚀 1.6x Speedup</span>
      <span class="badge yellow">📖 Production-Quality Code</span>
    </div>

  actions:
    - theme: brand
      text: 🚀 Start Tutorial
      link: /en/tutorial
    - theme: alt
      text: 📊 View Benchmarks
      link: /en/performance
    - theme: alt
      text: 💻 GitHub Source
      link: https://github.com/LessUp/diy-flash-attention

features:
  - icon: 🔷
    title: Read Real Triton Kernels
    details: Not toy examples — actual matmul and FlashAttention kernels you can run, benchmark, and study line by line. Compact code, detailed comments.
    link: /en/tutorial

  - icon: ⚡
    title: O(N) Memory Complexity
    details: "Understand FlashAttention's breakthrough: online softmax, SRAM tiling, causal masking — all without materializing the full attention matrix."
    link: /en/tutorial

  - icon: 📊
    title: Real Performance Data
    details: Built-in benchmark scripts comparing against PyTorch SDPA. See exactly why FlashAttention achieves 99% memory savings and 1.6x speedup.
    link: /en/performance

  - icon: 🖥️
    title: Architecture Adaptive
    details: Auto-detects Volta → Blackwell GPUs, adapts configurations automatically. Hopper+ supports TMA and FP8 feature detection.
    link: /en/api

  - icon: 🧪
    title: Comprehensive Testing
    details: 50+ unit tests, Hypothesis property-based testing covers infinite input space. Quality assured, safe for learning reference.
    link: https://github.com/LessUp/diy-flash-attention/tree/master/tests

  - icon: 🌐
    title: Bilingual Documentation
    details: All core documentation available in both English and Chinese, accessible to developers worldwide.
    link: /zh/
---

## Why This Project?

<div class="highlight-box">
  <p><strong>Compact but Real</strong>: Code small enough to read end-to-end, but not a toy. You can:</p>
  <ul>
    <li>✅ Run real benchmarks on your GPU</li>
    <li>✅ Compare performance against PyTorch SDPA</li>
    <li>✅ Understand every design decision behind each line</li>
  </ul>
</div>

### What You'll Learn

| Topic | Takeaway |
|-------|----------|
| GPU Memory Hierarchy | Data flow: HBM → L2 → SRAM → Registers |
| Triton Programming | Auto-tiling, autotune, kernel optimization techniques |
| FlashAttention Algorithm | Online softmax, causal masking, variable-length sequences |
| Performance Tuning | Block size selection, occupancy optimization, memory profiling |

### Project Stats

<div class="stats-grid">
  <div class="stat-card">
    <div class="stat-value">2+</div>
    <div class="stat-label">Core Triton Kernels</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">O(N)</div>
    <div class="stat-label">Attention Memory Complexity</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">6</div>
    <div class="stat-label">GPU Architectures Supported</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">99%</div>
    <div class="stat-label">Memory Saved (Long Sequences)</div>
  </div>
</div>

## Quick Start

```bash
# Install
pip install diy-flash-attention

# Or install from source
pip install -e ".[dev]"

# Verify
python -c "from kernels import flash_attention; print('✓ Installation successful')"
```

### Run Example

```python
import torch
from kernels import flash_attention

# FlashAttention — 99% less memory for long sequences
q = torch.randn(2, 8, 4096, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 8, 4096, 64, device="cuda", dtype=torch.float16)
v = torch.randn(2, 8, 4096, 64, device="cuda", dtype=torch.float16)

out = flash_attention(q, k, v, causal=True)  # GPT-style causal mask
print(f"Output shape: {out.shape}")  # [2, 8, 4096, 64]
```

## Learning Paths

<div class="audience-grid">
  <div class="audience-card">
    <div class="audience-avatar">🧑‍💻</div>
    <div class="audience-title">Kernel Developer</div>
    <div class="audience-benefit">Start with the tutorial, understand FlashAttention line by line</div>
    <span class="audience-skill">Path: Tutorial → API → Performance</span>
  </div>
  <div class="audience-card">
    <div class="audience-avatar">🔬</div>
    <div class="audience-title">Researcher</div>
    <div class="audience-benefit">Quick API reference lookup, reproduce and modify kernels</div>
    <span class="audience-skill">Path: API Reference → Source Code</span>
  </div>
  <div class="audience-card">
    <div class="audience-avatar">🚀</div>
    <div class="audience-title">Performance Engineer</div>
    <div class="audience-benefit">Deep dive into tuning, understand block sizes and architecture adaptation</div>
    <span class="audience-skill">Path: Performance Guide → Benchmarks</span>
  </div>
  <div class="audience-card">
    <div class="audience-avatar">📚</div>
    <div class="audience-title">Learner</div>
    <div class="audience-benefit">Systematic learning of GPU programming and attention optimization</div>
    <span class="audience-skill">Path: Tutorial → Cheatsheet → FAQ</span>
  </div>
</div>

<div class="cta-section">
  <div class="cta-title">Start Your FlashAttention Journey</div>
  <div class="cta-desc">Tutorial for understanding, API for contracts, performance guide for evidence.</div>
  <div class="cta-buttons">
    <a href="/diy-flash-attention/en/tutorial" class="cta-btn primary">
      <span>🚀</span> Read Tutorial
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
