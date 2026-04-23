---
layout: home

hero:
  name: "DIY"
  text: "FlashAttention"
  tagline: |
    用一个紧凑但真实的 Triton 项目学习 attention kernel、benchmark 和 GPU 自适应配置
    
    <div class="badge-group">
      <span class="badge">⚡ 内存减少 99%</span>
      <span class="badge purple">🚀 1.6倍加速</span>
      <span class="badge yellow">🎯 动手实践</span>
    </div>

  actions:
    - theme: brand
      text: 🚀 开始教程
      link: /zh/tutorial
    - theme: alt
      text: 📊 Benchmark 指南
      link: /zh/performance
    - theme: alt
      text: 💻 GitHub
      link: https://github.com/LessUp/diy-flash-attention

features:
  - icon: 🔷
    title: 阅读真实 Triton 代码
    details: 直接学习仓库中的 matmul 与 FlashAttention 核函数，而不是抽象幻灯片。
    link: /zh/tutorial

  - icon: ⚡
    title: 跟踪注意力算法
    details: 通过紧凑的前向实现理解 online softmax、SRAM 分块与因果掩码。
    link: /zh/tutorial

  - icon: 📊
    title: 对比 PyTorch Benchmark
    details: 使用仓库自带脚本对比 PyTorch SDPA 的速度与内存表现。
    link: /zh/performance

  - icon: 🖥️
    title: 查看架构辅助逻辑
    details: 浏览 Volta → Blackwell 的特性检测、Hopper TMA 标志和 FP8 辅助函数。
    link: /zh/api
---

## 这个页面的作用

这是面向读者的**项目总览页**，帮助你先判断从哪条路径进入，而不是先去读完整 README。

| 指南 | 说明 |
|------|------|
| [教程](/zh/tutorial) | 想按实现细节逐步理解时，从这里开始 |
| [API 参考](/zh/api) | 想确认支持的 kernel 与 helper 契约时，从这里开始 |
| [性能指南](/zh/performance) | 想看 benchmark 证据与性能取舍时，从这里开始 |
| [速查表](/zh/cheatsheet) | 已经了解 Triton、只想快速回顾时，从这里开始 |
| [常见问题](/zh/faq) | 想看环境与排障建议时，从这里开始 |

## 这个仓库覆盖什么

<div class="highlight-box">
  <p><strong>范围：</strong>Triton matmul、仅前向 FlashAttention、GPU 能力探测、测试与 benchmark 脚本。</p>
  <p><strong>方法：</strong>把代码规模控制在可完整阅读的范围内，但仍保留真实 benchmark 和验证价值。</p>
  <p><strong>价值：</strong>你可以在不跳进超大训练框架的前提下，建立 attention kernel 的整体直觉。</p>
</div>

## 仓库的优势

- **实现紧凑**：能读完，而不是只有概念图
- **性能有证据**：直接对比 PyTorch SDPA
- **架构自适应可见**：可以检查 Volta → Blackwell 辅助逻辑
- **双语文档**：英文与中文页面保持对应

<div class="cta-section">
  <div class="cta-title">从你最关心的部分开始</div>
  <div class="cta-desc">教程适合理解实现，性能页适合看证据，API 参考适合看精确定义。</div>
  <div class="cta-buttons">
    <a href="/diy-flash-attention/zh/tutorial" class="cta-btn primary">
      <span>📚</span> 阅读教程
    </a>
    <a href="https://github.com/LessUp/diy-flash-attention" class="cta-btn secondary">
      <span>⭐</span> Star on GitHub
    </a>
  </div>
</div>
