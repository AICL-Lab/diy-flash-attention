---
layout: home

hero:
  name: "DIY"
  text: "FlashAttention"
  tagline: |
    从零构建 FlashAttention — 掌握 GPU 内核优化

  actions:
    - theme: brand
      text: 📖 阅读白皮书
      link: /zh/architecture
    - theme: alt
      text: 🚀 开始教程
      link: /zh/tutorial
    - theme: alt
      text: 📊 查看基准测试
      link: /zh/performance
---

## FlashAttention 架构

<ArchitectureDiagram />

## 为什么选择 FlashAttention？

<div class="comparison-table">

| 方面 | 传统注意力 | FlashAttention |
|------|-----------|----------------|
| **内存复杂度** | O(N²) - 物化完整 N×N 矩阵 | O(N) - 从不存储中间结果 |
| **HBM 访问** | S 和 P 矩阵的 N² 次读写 | ~N 次读写，流式块处理 |
| **内存节省** | ❌ N=16K 序列需 1GB+ | ✅ **长序列节省 99%** |
| **加速** | 基线 | **现代 GPU 上快 1.6x - 2x** |
| **算法** | 三阶段：计算 → softmax → 输出 | **单阶段**在线 softmax |

</div>

## 你将学到什么

<div class="doc-nav-grid">

<div class="doc-nav-card">
  <div class="doc-nav-icon">📖</div>
  <h3><a href="/diy-flash-attention/zh/architecture">架构设计</a></h3>
  <p>系统架构、GPU 内存层次结构、内核设计、设计决策。</p>
  <span class="doc-nav-tag">白皮书</span>
</div>

<div class="doc-nav-card">
  <div class="doc-nav-icon">📐</div>
  <h3><a href="/diy-flash-attention/zh/algorithm">算法详解</a></h3>
  <p>在线 softmax 推导、分块策略、复杂度分析、正确性证明。</p>
  <span class="doc-nav-tag">白皮书</span>
</div>

<div class="doc-nav-card">
  <div class="doc-nav-icon">🚀</div>
  <h3><a href="/diy-flash-attention/zh/tutorial">教程</a></h3>
  <p>循序渐进：GPU 基础 → Triton 编程 → FlashAttention 实现。</p>
  <span class="doc-nav-tag">入门</span>
</div>

<div class="doc-nav-card">
  <div class="doc-nav-icon">📊</div>
  <h3><a href="/diy-flash-attention/zh/performance">性能指南</a></h3>
  <p>块大小调优、数据类型选择、GPU 架构适配、基准测试。</p>
  <span class="doc-nav-tag">参考</span>
</div>

<div class="doc-nav-card">
  <div class="doc-nav-icon">🔧</div>
  <h3><a href="/diy-flash-attention/zh/api">API 参考</a></h3>
  <p>完整函数签名、参数、返回值、使用示例。</p>
  <span class="doc-nav-tag">参考</span>
</div>

<div class="doc-nav-card">
  <div class="doc-nav-icon">⚡</div>
  <h3><a href="/diy-flash-attention/zh/cheatsheet">速查表</a></h3>
  <p>快速参考：常用 API、命令、配置模板、错误速查。</p>
  <span class="doc-nav-tag">速查</span>
</div>

</div>

## 核心特性

<div class="features-grid">

<div class="feature-item">
  <span class="feature-icon">🔷</span>
  <h4>真实 Triton 内核</h4>
  <p>非玩具示例 — 生产级 matmul 和 FlashAttention 内核，可运行、基准测试、逐行学习。</p>
</div>

<div class="feature-item">
  <span class="feature-icon">⚡</span>
  <h4>O(N) 内存复杂度</h4>
  <p>理解 FlashAttention 的突破：在线 softmax、SRAM 分块、因果掩码 — 无需物化完整注意力矩阵。</p>
</div>

<div class="feature-item">
  <span class="feature-icon">📊</span>
  <h4>真实性能数据</h4>
  <p>内置基准测试脚本，与 PyTorch SDPA 对比。直观展示 99% 内存节省的原因。</p>
</div>

<div class="feature-item">
  <span class="feature-icon">🖥️</span>
  <h4>架构自适应</h4>
  <p>自动检测 Volta → Blackwell GPU，适配配置。Hopper+ 支持 TMA 和 FP8。</p>
</div>

<div class="feature-item">
  <span class="feature-icon">🧪</span>
  <h4>全面测试</h4>
  <p>50+ 单元测试，Hypothesis 属性测试覆盖无限输入空间。学习参考安全可靠。</p>
</div>

<div class="feature-item">
  <span class="feature-icon">🌐</span>
  <h4>双语文档</h4>
  <p>所有文档中英双语，全球开发者均可访问。</p>
</div>

</div>

## 快速开始

```bash
# 安装
pip install diy-flash-attention

# 或从源码安装
pip install -e ".[dev]"

# 验证
python -c "from kernels import flash_attention; print('✓ 安装成功')"
```

### 运行示例

```python
import torch
from kernels import flash_attention

# FlashAttention — 长序列节省 99% 内存
q = torch.randn(2, 8, 4096, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 8, 4096, 64, device="cuda", dtype=torch.float16)
v = torch.randn(2, 8, 4096, 64, device="cuda", dtype=torch.float16)

out = flash_attention(q, k, v, causal=True)  # GPT 风格因果掩码
print(f"输出形状: {out.shape}")  # [2, 8, 4096, 64]
```

## GPU 支持矩阵

| 架构 | GPU | 计算能力 | 特性 |
|------|-----|----------|------|
| **Volta** | V100 | SM70 | ✅ Tensor Cores, FP16 |
| **Turing** | RTX 20xx | SM75 | ✅ Tensor Cores, FP16 |
| **Ampere** | A100, RTX 30xx | SM80 | ✅ 完整支持, BF16 |
| **Ada** | RTX 40xx | SM89 | ✅ 完整支持, BF16 |
| **Hopper** | H100 | SM90 | ✅ TMA, FP8 特性 |
| **Blackwell** | B100/B200 | SM100 | ✅ 最新特性 |

## 语言

<div class="lang-switcher">
  <a href="/diy-flash-attention/" class="lang-link">
    <span>🇺🇸</span> English
  </a>
  <a href="/diy-flash-attention/zh/" class="lang-link active">
    <span>🇨🇳</span> 中文
  </a>
</div>
