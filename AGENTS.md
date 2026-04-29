# AGENTS.md

本仓库是一个**教育性 Triton/FlashAttention 项目**，已达到 archive-ready 状态。所有变更应使仓库更清晰、更精简、更可信赖。

## 核心定位

**diy-flash-attention** = 用 Triton 从零构建 FlashAttention 的学习项目

- 🎯 **学习导向**：帮助开发者理解 FlashAttention 内部机制
- ⚡ **前向传播**：仅实现 forward pass，非生产训练框架
- 🔧 **架构感知**：自动适配 Volta → Blackwell GPU 架构

## OpenSpec 工作流

所有非平凡工作必须通过 OpenSpec 管理：

```bash
openspec list --json          # 查看活跃 change
openspec validate --specs --json  # 验证规范
```

1. 使用 `/opsx:explore` 调查模糊问题
2. 使用 `/opsx:propose` 发起新的非平凡工作
3. 使用 `/opsx:apply <change>` 执行任务
4. 使用 `/opsx:archive <change>` 归档完成的 change

**规则**：仓库级清理保持**单一活跃 change**，完成后立即归档。

## 关键文件

| 范围 | 文件 |
|------|------|
| 内核实现 | `kernels/flash_attn.py`, `kernels/flash_attn_v2.py`, `kernels/matmul.py` |
| 工具函数 | `utils/gpu_detect.py`, `utils/benchmark.py`, `utils/validation.py` |
| 测试 | `tests/` (50+ 测试，Hypothesis 属性测试) |
| 文档 | `docs/` (VitePress 中英双语站点) |
| 规范 | `openspec/specs/<capability>/spec.md` |

## 表面一致性

当以下任一表面变化时，同步更新相关表面：

- **README.md** ↔ **GitHub Pages** ↔ **GitHub About**
- **kernels/** ↔ **tests/** ↔ **docs/en/api.md**

不要让文档和实现漂移。

## 验证基线

```bash
make lint        # ruff check
make typecheck   # mypy
make test-cpu    # CPU-safe 测试
make docs        # 构建 VitePress 站点
```

## 工具立场

- **Copilot**：快速内联编辑、轻量后续修改、GitHub 原生流程
- **Claude**：跨文件推理、工作流清理、规范/设计精化、文档/架构一致性
- **gh**：GitHub 元数据更新
- **MCP**：仅在明确减少重复工作时引入

## 项目特定规则

- 保持**前向传播、教育性**范围，除非活跃 spec 明确扩展
- 删除过时/通用文档优于打磨低价值重复内容
- 工作流文档和 AI 指令必须**特定于本仓库**，禁止复制 boilerplate
- 合并或归档前运行 `/review`
