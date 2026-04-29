---
layout: home

hero:
  name: "DIY"
  text: "FlashAttention"
  tagline: |
    用 Triton 从零构建 FlashAttention，掌握 GPU 内核优化的核心技术
    
    <div class="badge-group">
      <span class="badge">⚡ 内存减少 99%</span>
      <span class="badge purple">🚀 速度提升 1.6x</span>
      <span class="badge yellow">📖 教育级代码</span>
    </div>

  actions:
    - theme: brand
      text: 🚀 开始学习
      link: /zh/tutorial
    - theme: alt
      text: 📊 查看性能数据
      link: /zh/performance
    - theme: alt
      text: 💻 GitHub 源码
      link: https://github.com/LessUp/diy-flash-attention

features:
  - icon: 🔷
    title: 读得懂的 Triton 代码
    details: 不是玩具示例，而是真实可运行的 matmul 和 FlashAttention 内核。代码紧凑，注释详尽，适合逐行研读。
    link: /zh/tutorial

  - icon: ⚡
    title: O(N) 内存复杂度
    details: 理解 FlashAttention 的核心创新：在线 softmax、SRAM 分块、因果掩码——所有这一切都不需要实例化完整的注意力矩阵。
    link: /zh/tutorial

  - icon: 📊
    title: 真实性能数据
    details: 内置基准测试脚本，直接对比 PyTorch SDPA。看懂为什么 FlashAttention 能做到 99% 的内存节省和 1.6x 的速度提升。
    link: /zh/performance

  - icon: 🖥️
    title: 架构自适应
    details: 自动检测 Volta → Blackwell GPU 架构，适配最优配置。Hopper+ 支持 TMA 和 FP8 特性检测。
    link: /zh/api

  - icon: 🧪
    title: 完整测试覆盖
    details: 50+ 单元测试，Hypothesis 属性测试覆盖无限输入空间。代码质量有保障，学习参考更放心。
    link: https://github.com/LessUp/diy-flash-attention/tree/master/tests

  - icon: 🌐
    title: 中英双语文档
    details: 所有核心文档提供中英文版本，方便不同背景的开发者学习。
    link: /en/
---

## 为什么选择这个项目？

<div class="highlight-box">
  <p><strong>紧凑但真实</strong>：代码量控制在可完整阅读的范围内，但不是玩具示例。你可以：</p>
  <ul>
    <li>✅ 在 GPU 上运行真实基准测试</li>
    <li>✅ 对比 PyTorch SDPA 的性能差异</li>
    <li>✅ 理解每一行代码背后的设计决策</li>
  </ul>
</div>

### 你将学到什么

| 主题 | 收获 |
|------|------|
| GPU 内存层级 | HBM → L2 → SRAM → 寄存器的数据流动 |
| Triton 编程 | 自动分块、autotune、内核优化技巧 |
| FlashAttention 算法 | 在线 softmax、因果掩码、变长序列处理 |
| 性能调优 | 块大小选择、occupancy 优化、内存分析 |

### 项目数据

<div class="stats-grid">
  <div class="stat-card">
    <div class="stat-value">2+</div>
    <div class="stat-label">核心 Triton 内核</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">O(N)</div>
    <div class="stat-label">注意力内存复杂度</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">6</div>
    <div class="stat-label">GPU 架构支持</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">99%</div>
    <div class="stat-label">内存节省（长序列）</div>
  </div>
</div>

## 快速开始

```bash
# 安装
pip install diy-flash-attention

# 或者从源码安装
pip install -e ".[dev]"

# 验证安装
python -c "from kernels import flash_attention; print('✓ 安装成功')"
```

### 运行示例

```python
import torch
from kernels import flash_attention

# FlashAttention — 长序列内存减少 99%
q = torch.randn(2, 8, 4096, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 8, 4096, 64, device="cuda", dtype=torch.float16)
v = torch.randn(2, 8, 4096, 64, device="cuda", dtype=torch.float16)

out = flash_attention(q, k, v, causal=True)  # GPT 风格因果掩码
print(f"输出形状: {out.shape}")  # [2, 8, 4096, 64]
```

## 学习路径

<div class="audience-grid">
  <div class="audience-card">
    <div class="audience-avatar">🧑‍💻</div>
    <div class="audience-title">内核开发者</div>
    <div class="audience-benefit">从教程开始，逐行理解 FlashAttention 实现</div>
    <span class="audience-skill">推荐：教程 → API → 性能指南</span>
  </div>
  <div class="audience-card">
    <div class="audience-avatar">🔬</div>
    <div class="audience-title">研究人员</div>
    <div class="audience-benefit">快速查阅 API 契约，复现和修改内核</div>
    <span class="audience-skill">推荐：API 参考 → 源码</span>
  </div>
  <div class="audience-card">
    <div class="audience-avatar">🚀</div>
    <div class="audience-title">性能工程师</div>
    <div class="audience-benefit">深入性能调优，理解块大小和架构适配</div>
    <span class="audience-skill">推荐：性能指南 → 基准测试</span>
  </div>
  <div class="audience-card">
    <div class="audience-avatar">📚</div>
    <div class="audience-title">学习者</div>
    <div class="audience-benefit">系统学习 GPU 编程和注意力优化</div>
    <span class="audience-skill">推荐：教程 → 速查表 → FAQ</span>
  </div>
</div>

<div class="cta-section">
  <div class="cta-title">开始你的 FlashAttention 学习之旅</div>
  <div class="cta-desc">从教程入手，理解实现；用 API 参考，确认契约；看性能指南，获取证据。</div>
  <div class="cta-buttons">
    <a href="/diy-flash-attention/zh/tutorial" class="cta-btn primary">
      <span>🚀</span> 阅读教程
    </a>
    <a href="https://github.com/LessUp/diy-flash-attention" class="cta-btn secondary">
      <span>⭐</span> Star on GitHub
    </a>
  </div>
</div>

## 语言切换

<div class="lang-switcher">
  <a href="/diy-flash-attention/zh/" class="lang-link active">
    <span>🇨🇳</span> 中文
  </a>
  <a href="/diy-flash-attention/en/" class="lang-link">
    <span>🇺🇸</span> English
  </a>
</div>
