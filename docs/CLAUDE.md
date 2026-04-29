# docs/ - 文档站点

本目录包含 DIY FlashAttention 的 VitePress 文档站点，支持中英双语。

## 文件结构

```
docs/
├── .vitepress/
│   ├── config.mts      # VitePress 主配置
│   └── theme/          # 自定义主题和组件
├── public/             # 静态资源（logo, og-image 等）
├── en/                 # 英文文档
│   ├── index.md        # 首页
│   ├── tutorial.md     # 教程
│   ├── api.md          # API 参考
│   ├── performance.md  # 性能指南
│   ├── cheatsheet.md   # 速查表
│   ├── faq.md          # 常见问题
│   └── changelog.md    # 更新日志导航
├── zh/                 # 中文文档（同结构）
└── index.md            # 根首页（重定向到 zh/）
```

## 页面说明

| 页面 | 定位 | 内容重点 |
|------|------|----------|
| index.md | 门户首页 | 项目亮点、快速开始、学习路径 |
| tutorial.md | 教程 | GPU 基础 → Triton → FlashAttention 逐行讲解 |
| api.md | API 参考 | 所有公开函数签名、参数、返回值 |
| performance.md | 性能指南 | 块大小调优、内存分析、基准测试 |
| cheatsheet.md | 速查表 | 常用代码片段、配置模板 |
| faq.md | FAQ | 安装问题、兼容性、性能问题 |
| changelog.md | 导航 | 指向 CHANGELOG.md |

## 设计原则

1. **门户导向**：首页聚焦项目核心价值和转化
2. **内容互补**：文档与 README 不重复，各有侧重
3. **双语一致**：中英文页面内容对应
4. **SEO 优化**：Open Graph、Twitter Card、JSON-LD

## 本地开发

```bash
npm run docs:dev     # 启动开发服务器
npm run docs:build   # 构建静态站点
```

## 部署

- **触发**：push 到 master/main，或 docs/ 目录变化
- **配置**：`.github/workflows/pages.yml`
- **URL**：https://lessup.github.io/diy-flash-attention/

## 与其他表面同步

当以下任一变化时，确保一致性：
- README.md
- GitHub About description/topics
- CLAUDE.md 项目概述

---

**导航**: [← 项目根目录](../CLAUDE.md)
