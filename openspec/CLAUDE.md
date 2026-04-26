# openspec/ - 规范与变更管理

> **导航**: [← 项目根目录](../CLAUDE.md)

本目录是仓库的 **OpenSpec 单一事实源**。这里记录能力规范、活跃 change 以及归档后的变更历史。

## 结构约定

```text
openspec/
├── config.json
├── specs/<capability>/spec.md
├── changes/<change>/{proposal.md,design.md,tasks.md,specs/**}
└── changes/archive/
```

- `openspec/specs/` 保存当前生效的 capability specs。
- `openspec/changes/<change>/` 保存活跃 change 的 proposal / design / tasks / delta specs。
- `openspec/changes/archive/` 保存已完成并归档的 change。

不要再引入旧的 `product/`、`rfc/`、`api/`、`testing/` 目录叙事。

## 编辑本目录时的规则

1. 先运行 `openspec list --json`，确认当前活跃 change。
2. 非平凡工作没有相关活跃 change 时，先创建/propose，再改内容。
3. 变更顺序遵循：
   - `proposal.md` 定义为什么改
   - `design.md` 定义怎么改
   - `specs/**` 定义行为变化
   - `tasks.md` 定义执行顺序
4. broad cleanup 期间保持**同一时间只有一个活跃仓库级 change**。
5. change 完成后及时 archive，不要让完成态 change 长期挂在活跃区。

## 什么时候改哪一层

- **需求或公开行为变化** → 改 active change 下的 `specs/<capability>/spec.md`
- **技术决策变化** → 改 `design.md`
- **范围变化** → 改 `proposal.md`
- **执行项增删** → 改 `tasks.md`

## 常用命令

```bash
openspec list --json
openspec validate --specs --json
openspec status --change <change-name> --json
openspec instructions apply --change <change-name> --json
openspec validate <change-name> --json
```

## 与仓库其他表面的关系

- `project-surface` 约束 README / docs / GitHub About
- `engineering-workflow` 约束 CI / hooks / LSP / AI tooling guidance
- `project-governance` 约束 OpenSpec-first 流程和 review discipline
- `flashattention-kernels` 约束核心实现、验证和 benchmark 表面

如果这些表面发生变化，OpenSpec 也要一起更新，避免文档和实际流程再次漂移。
