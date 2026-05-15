---
layout: home

hero:
  name: "DIY"
  text: "FlashAttention"
  tagline: |
    面向学习者的 Triton FlashAttention 学院入口

  actions:
    - theme: brand
      text: 🧭 开始学习
      link: /zh/learning-path
    - theme: alt
      text: 📚 阅读论文
      link: /zh/paper-guide
    - theme: alt
      text: 🗺️ 浏览知识图谱
      link: /zh/knowledge-map
---

## 从这里开始

先用这个门户选择学习路线，再进入支撑各阶段的参考文档。

<div class="academy-grid">

<div class="academy-card">
  <h3><a href="./learning-path">学习路径</a></h3>
  <p>从 Triton 基础开始，再进入在线 softmax 与 FlashAttention 内核实现。</p>
</div>

<div class="academy-card">
  <h3><a href="./paper-guide">论文导读</a></h3>
  <p>按与本仓库相匹配的顺序阅读 FlashAttention 论文。</p>
</div>

<div class="academy-card">
  <h3><a href="./knowledge-map">知识图谱</a></h3>
  <p>查看核心概念、源码文件与文档之间的连接关系。</p>
</div>

</div>

## 门户导览

- **学习路径**：为第一次阅读本仓库的读者安排 Triton 基础 → 在线 softmax → 注意力内核的学习顺序。
- **论文导读**：按仓库的 educational、forward-only 范围组织论文阅读顺序。
- **知识图谱**：把概念、源码文件和文档页面串起来，方便在理论与实现之间切换。

## 参考资料库

学院门户负责导学，下面这些页面负责继续深入实现细节与实践路径。

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

## 项目范围

- 面向学习的、仅 forward pass 的 Triton FlashAttention 项目。
- 重点帮助读者建立论文、内核实现与性能权衡之间的联系。
- 适合作为导学型仓库，不以生产训练框架为目标。

## 语言

<div class="lang-switcher">
  <a href="../" class="lang-link">
    <span>🇺🇸</span> English
  </a>
  <a href="./" class="lang-link active">
    <span>🇨🇳</span> 中文
  </a>
</div>
