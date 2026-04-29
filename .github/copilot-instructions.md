# Copilot Instructions

本仓库是一个**教育性、前向传播的 Triton FlashAttention 项目**，已达到 archive-ready 状态。保持仓库小型、连贯、可信赖。

## 核心定位

**学习项目**：用 Triton 从零构建 FlashAttention，帮助开发者理解 GPU 内核优化。

## OpenSpec 优先工作流

```bash
openspec list --json          # 非平凡工作前先运行
openspec validate --specs --json  # 验证规范
```

1. 先读活跃 change 的 artifacts 和受影响的 `openspec/specs/<capability>/spec.md`
2. 无相关活跃 change 时，先创建/propose 再编辑
3. 按 `tasks.md` 实现，同步更新相关 docs/tests/specs
4. 完成后归档，不留过时的活跃 work

## 仓库优先级

- ✅ 保持**动手实践、前向传播**的学习范围
- ✅ README、双语文档、GitHub Pages、GitHub About 保持一致
- ✅ 删除过时文档、重复 changelog、工作流杂乱内容
- ✅ CI、hooks、工具指导保持轻量高信号

## 编辑规则

- ❌ 禁止通用贡献者或 AI boilerplate
- ❌ 不要在长期指导中硬编码已完成的 change 名称
- ✅ 行为或消息变化时，更新相邻的 specs/docs/tests/metadata
- ✅ 大范围或高风险清理切片在合并或归档移交前使用 review

## 验证基线

```bash
make lint        # ruff check
make typecheck   # mypy
make test-cpu    # CPU-safe 测试
make docs        # 构建 VitePress 站点
```

## 工具立场

| 工具 | 用途 |
|------|------|
| Copilot | 快速内联编辑、轻量后续修改、GitHub 原生流程 |
| Claude | 跨文件推理、文档/治理清理、OpenSpec 精化 |
| gh | README/Pages 措辞确定后的仓库元数据更新 |
| MCP | 可选；仅在明确减少本仓库重复维护工作时引入 |
