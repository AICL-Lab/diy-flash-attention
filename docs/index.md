---
layout: home

hero:
  name: "DIY FlashAttention"
  text: ""
  tagline: |
    Master GPU Programming by Implementing FlashAttention from Scratch
    
    <div class="hero-badges">
      <span class="badge">⚡ 99% Memory Reduction</span>
      <span class="badge">🚀 1.6x Speedup</span>
      <span class="badge">🎯 Production Ready</span>
    </div>
  
  image:
    src: /illustration.svg
    alt: FlashAttention Illustration
  
  actions:
    - theme: brand
      text: 🚀 Get Started
      link: /en/tutorial
    - theme: alt
      text: 📚 View Docs
      link: /en/
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
    title: Bilingual
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
  .hero-badges {
    flex-direction: column;
    align-items: center;
  }
}
</style>

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/LessUp/diy-flash-attention.git
cd diy-flash-attention

# Install dependencies
pip install -e ".[dev]"

# Run the demo
make demo
```

## 📝 Example Usage

```python
import torch
from kernels import triton_matmul, flash_attention

# Matrix multiplication with autotune
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
c = triton_matmul(a, b)

# FlashAttention (99% less memory!)
q = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
v = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
out = flash_attention(q, k, v, causal=True)
```

## 📊 Performance Highlights

| Sequence Length | Standard Attention | FlashAttention | Memory Saved |
|-----------------|-------------------|----------------|--------------|
| 1,024 | 8 MB | 0.5 MB | **94%** |
| 4,096 | 128 MB | 2 MB | **98%** |
| 8,192 | 512 MB | 4 MB | **99%** |

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

.lang-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
}

.lang-name {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--vp-c-text-1);
  margin-bottom: 0.5rem;
}

.lang-desc {
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
  text-align: center;
}
</style>

## 🛠️ Supported GPUs

<div class="gpu-grid">
  <div class="gpu-card">
    <div class="gpu-icon">🟢</div>
    <div class="gpu-name">Volta (V100)</div>
    <div class="gpu-status">Basic</div>
  </div>
  <div class="gpu-card">
    <div class="gpu-icon">🟢</div>
    <div class="gpu-name">Turing (RTX 20xx)</div>
    <div class="gpu-status">Basic</div>
  </div>
  <div class="gpu-card">
    <div class="gpu-icon">✅</div>
    <div class="gpu-name">Ampere (A100, RTX 30xx)</div>
    <div class="gpu-status">Full</div>
  </div>
  <div class="gpu-card">
    <div class="gpu-icon">✅</div>
    <div class="gpu-name">Ada (RTX 40xx)</div>
    <div class="gpu-status">Full</div>
  </div>
  <div class="gpu-card featured">
    <div class="gpu-icon">⭐</div>
    <div class="gpu-name">Hopper (H100)</div>
    <div class="gpu-status">Advanced</div>
    <div class="gpu-features">TMA, FP8</div>
  </div>
  <div class="gpu-card featured">
    <div class="gpu-icon">🚀</div>
    <div class="gpu-name">Blackwell (B100)</div>
    <div class="gpu-status">Latest</div>
    <div class="gpu-features">Next-gen</div>
  </div>
</div>

<style>
.gpu-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
  margin: 2rem 0;
}

.gpu-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 1.5rem 1rem;
  background: var(--vp-c-bg-soft);
  border: 2px solid var(--vp-c-divider);
  border-radius: 0.75rem;
  transition: all 0.3s;
}

.gpu-card:hover {
  border-color: var(--vp-c-brand-1);
  transform: scale(1.05);
}

.gpu-card.featured {
  border-color: var(--vp-c-brand-1);
  background: linear-gradient(135deg, var(--vp-c-brand-soft), transparent);
}

.gpu-icon {
  font-size: 1.5rem;
  margin-bottom: 0.5rem;
}

.gpu-name {
  font-weight: 600;
  font-size: 0.875rem;
  text-align: center;
  color: var(--vp-c-text-1);
}

.gpu-status {
  font-size: 0.75rem;
  color: var(--vp-c-brand-1);
  font-weight: 500;
  margin-top: 0.25rem;
}

.gpu-features {
  font-size: 0.7rem;
  color: var(--vp-c-text-2);
  margin-top: 0.25rem;
}
</style>

## 📚 Documentation

| 📖 [Tutorial](/en/tutorial) | 🔧 [API Reference](/en/api) | ⚡ [Performance](/en/performance) |
|:---|:---|:---|
| Step-by-step guide from basics to FlashAttention | Complete API documentation with examples | Optimization tips and benchmarks |

| ❓ [FAQ](/en/faq) | 📋 [Cheatsheet](/en/cheatsheet) | 📝 [Changelog](/en/changelog) |
|:---|:---|:---|
| Common questions and solutions | Quick reference for common tasks | Version history and updates |

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](https://github.com/LessUp/diy-flash-attention/blob/master/CONTRIBUTING.md) for details.

<div align="center">

⭐ **Star this repo if you find it helpful!**

[🐛 Report Bug](https://github.com/LessUp/diy-flash-attention/issues) • [💡 Request Feature](https://github.com/LessUp/diy-flash-attention/discussions)

</div>

---

<div align="center" style="margin-top: 3rem; padding-top: 2rem; border-top: 1px solid var(--vp-c-divider);">

### Install as PWA

Add this documentation to your home screen for offline access!

<div style="font-size: 0.875rem; color: var(--vp-c-text-2); margin-top: 0.5rem;">
  Works offline • Fast loading • Native app experience
</div>

</div>
