# 2026-03-10 GitHub Pages 优化

## Summary

优化 VitePress 文档站，增强 SEO、丰富首页内容、新增变更日志页面，优化构建流程。

## Changes

### Bug 修复
- **README.md / README.zh-CN.md** — Docs badge 修复 `docs.yml` → `pages.yml`（指向实际存在的 workflow），新增 CI badge

### 文档内容
- **`docs/index.md`** — 首页增加"API 参考"快捷入口、扩充 feature 描述细节、新增技术栈表格、快速开始代码示例和核心概念板块
- **`docs/changelog.md`** — 新增变更日志汇总页，整合 3 个版本的变更记录及未来计划

### VitePress 配置
- **`docs/.vitepress/config.mts`** — 新增 SEO `head` meta 标签（og:title/description/url、theme-color、keywords）、启用 `cleanUrls`、顶栏和侧栏新增"变更日志"导航

### 构建流程
- **`pages.yml`** — `fetch-depth: 0` 替换为 `sparse-checkout`（仅检出 docs + package 文件）、Node.js 升级至 22、paths 触发器新增 `package-lock.json`
