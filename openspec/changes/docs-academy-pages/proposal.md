## Why

GitHub Pages 目前更像项目展示页，而不是服务于学习者的学院式导读站，难以把 README 中已经明确的教育性、forward-only FlashAttention 定位延伸成更系统的学习入口。与此同时，亮暗模式切换下的 SVG 图示缺乏统一适配策略，首页、导航、论文导读与知识地图之间也缺少成体系的入口设计。

## What Changes

- 将 docs 首页从项目展示页升级为 academy-style 学习入口，明确“从哪里开始”“先读什么论文”“核心概念如何关联文档”。
- 为站点建立双层信息架构：portal layer 负责导学入口，reference layer 保持教程、架构、算法、性能与 API 参考。
- 定义主题自适应图示策略，约束站点 logo、SVG 资产与图文插图在 light/dark 模式下的可读性。
- 为 EN/ZH 文档首页与关键导航补充学习路径、论文导读、知识地图等入口页的实现范围。

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `project-surface`: GitHub Pages 需要从通用 showcase 提升为学习学院入口，并对公开 SVG 资产提出跨主题可读性约束。

## Impact

- Affected docs surface: `docs/`, `docs/.vitepress/`, bilingual homepage and navigation content.
- Affected spec surface: `openspec/specs/project-surface/spec.md` via change delta.
- No kernel algorithm, training scope, or repository runtime dependency changes are introduced in this change.
