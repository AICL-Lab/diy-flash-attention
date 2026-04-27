# 贡献指南

感谢你为 DIY FlashAttention 做改进。这个仓库的目标不是持续扩张功能，而是把它维持成一个**教育性、forward-only、OpenSpec-first** 的 Triton FlashAttention 参考实现。

## 先理解项目边界

- **这是教学实现，不是完整训练框架。**
- 默认关注点是 `kernels/`、`utils/`、`tests/`、`benchmarks/`、`examples/`。
- 公开表面包括 `README.md`、`README.zh-CN.md`、`docs/`、GitHub Pages 和 GitHub About。
- 非平凡修改必须通过 OpenSpec change 承接，而不是直接改代码。

## OpenSpec-first 工作流

1. 先查看当前变更：

   ```bash
   openspec list --json
   ```

2. 选择或创建活跃 change，并阅读：

   - `openspec/changes/<change>/proposal.md`
   - `openspec/changes/<change>/design.md`
   - `openspec/changes/<change>/tasks.md`
   - `openspec/specs/<capability>/spec.md`

3. 如果没有相关活跃 change，先创建/propose，再开始实现。
4. 改动时同步更新相邻的 spec、测试、文档和配置，避免 README / Pages / CI / About 漂移。
5. 非平凡实现切片完成后先做 review，再准备 merge 或 archive。

对于仓库级清理，请保持**同一时间只有一个活跃 cleanup/finalization change**。

## 本地开发

### Python 环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 文档依赖

如果你会修改 `docs/`、`package.json`、VitePress 配置或 GitHub Pages 相关内容，再执行：

```bash
npm ci
```

## 验证基线

提交前至少运行与改动相关的现有命令；仓库级修改通常需要完整基线：

```bash
make lint
make typecheck
pytest tests/ -v -m "not cuda" --ignore=tests/test_properties.py
npm run docs:build
openspec validate
```

说明：

- GPU 相关测试在无 CUDA 环境下可以跳过，但不要伪造通过。
- 如果你修改了 README / docs / workflows / About 文案，请确保它们讲的是同一个项目故事。

## 改动风格

- 优先做**小而完整**的修改，而不是大而散的半成品。
- 优先删除过期、重复、低价值内容，而不是继续叠加样板。
- 不要引入新的基础设施，除非它能明显降低本仓库的长期维护成本。
- 不要把临时 change 名称、外部协议、不可复用的个人工作流写进长期文档。

## Pull Request 建议

- 用短生命周期分支，避免本地/云端长期分叉。
- 在 PR 描述里写清楚关联的 OpenSpec change。
- 对跨文件重构、治理文档重写、CI/工具链调整这类变更，先做 review 再合并。

## 文档结构

- `README.md` / `README.zh-CN.md` - 项目概述（中英双语）
- `AGENTS.md` - AI 代理工作流指令
- `CONTRIBUTING.md` - 贡献指南（本文件）
- `openspec/` - OpenSpec 规范管理
  - `specs/<capability>/spec.md` - 能力规范
  - `changes/<change>/` - 活跃变更
  - `changes/archive/` - 已归档变更
- `docs/` - VitePress 文档站点（中英双语）
- `CHANGELOG.md` - 版本历史

## 报告问题

请尽量附带：

- 复现步骤
- 使用的 Python / CUDA / Triton / PyTorch 版本
- GPU 型号与环境信息
- 相关日志、报错、截图或最小复现片段

## 联系方式

如有问题，可以通过以下方式联系：

- 提交 [GitHub Issue](https://github.com/LessUp/diy-flash-attention/issues)

感谢你的贡献！🎉
