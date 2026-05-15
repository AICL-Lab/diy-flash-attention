---
layout: home

hero:
  name: "DIY"
  text: "FlashAttention"
  tagline: |
    Build FlashAttention from scratch with Triton — master GPU kernel optimization

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

Use the academy portal first, then move into the whitepaper-style references and runnable code examples below.

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

## FlashAttention Architecture

<ArchitectureDiagram />

## Why FlashAttention?

<div class="comparison-table">

| Aspect | Traditional Attention | FlashAttention |
|--------|----------------------|----------------|
| **Memory Complexity** | O(N²) - Materializes full N×N matrix | O(N) - Never stores intermediate results |
| **HBM Accesses** | N² reads/writes for S and P matrices | ~N reads/writes, streamed in blocks |
| **Memory Savings** | ❌ 1GB+ for N=16K sequences | ✅ **99% reduction** for long sequences |
| **Speedup** | Baseline | **1.6x - 2x faster** on modern GPUs |
| **Algorithm** | Three-pass: compute → softmax → output | **Single-pass** online softmax |

</div>

## Reference Library

After the academy portal points you to the right entry, use these references to go deeper into the implementation details.

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

## Key Features

<div class="features-grid">

<div class="feature-item">
  <span class="feature-icon">🔷</span>
  <h4>Real Triton Kernels</h4>
  <p>Not toy examples — production-quality matmul and FlashAttention kernels you can run, benchmark, and study line by line.</p>
</div>

<div class="feature-item">
  <span class="feature-icon">⚡</span>
  <h4>O(N) Memory Complexity</h4>
  <p>Understand FlashAttention's breakthrough: online softmax, SRAM tiling, causal masking — all without materializing the full attention matrix.</p>
</div>

<div class="feature-item">
  <span class="feature-icon">📊</span>
  <h4>Real Performance Data</h4>
  <p>Built-in benchmark scripts comparing against PyTorch SDPA. See exactly why FlashAttention achieves 99% memory savings.</p>
</div>

<div class="feature-item">
  <span class="feature-icon">🖥️</span>
  <h4>Architecture Adaptive</h4>
  <p>Auto-detects Volta → Blackwell GPUs, adapts configurations automatically. Hopper+ supports TMA and FP8.</p>
</div>

<div class="feature-item">
  <span class="feature-icon">🧪</span>
  <h4>Comprehensive Testing</h4>
  <p>50+ unit tests, Hypothesis property-based testing covers infinite input space. Safe for learning reference.</p>
</div>

<div class="feature-item">
  <span class="feature-icon">🌐</span>
  <h4>Bilingual Documentation</h4>
  <p>All documentation available in both English and Chinese, accessible to developers worldwide.</p>
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

## GPU Support Matrix

| Architecture | GPUs | Compute Capability | Features |
|--------------|------|-------------------|----------|
| **Volta** | V100 | SM70 | ✅ Tensor Cores, FP16 |
| **Turing** | RTX 20xx | SM75 | ✅ Tensor Cores, FP16 |
| **Ampere** | A100, RTX 30xx | SM80 | ✅ Full Support, BF16 |
| **Ada** | RTX 40xx | SM89 | ✅ Full Support, BF16 |
| **Hopper** | H100 | SM90 | ✅ TMA, FP8 Features |
| **Blackwell** | B100/B200 | SM100 | ✅ Latest Features |

## Language

<div class="lang-switcher">
  <a href="/diy-flash-attention/" class="lang-link active">
    <span>🇺🇸</span> English
  </a>
  <a href="/diy-flash-attention/zh/" class="lang-link">
    <span>🇨🇳</span> 中文
  </a>
</div>
