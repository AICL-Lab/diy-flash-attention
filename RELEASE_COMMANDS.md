# GitHub Release 发布指南

由于网络连接问题，请使用以下命令手动创建 Release。

## 步骤 1: 推送代码 (如尚未完成)

```bash
# 推送提交和标签
git push origin master
git push origin v1.0.3
```

## 步骤 2: 使用 GitHub CLI 创建 Release

```bash
# 设置发布标题和说明
gh release create v1.0.3 \
  --title "v1.0.3: Documentation Overhaul with i18n Support / 文档国际化重构" \
  --notes-file RELEASE_NOTES.md \
  --latest
```

## 备选手动方式

如果 gh CLI 不可用，请访问: https://github.com/LessUp/diy-flash-attention/releases/new

### 填写信息:

**Choose a tag**: v1.0.3

**Release title**: 
```
v1.0.3: Documentation Overhaul with i18n Support / 文档国际化重构
```

**Release notes** (复制以下内容):

```markdown
## 🌍 Documentation Overhaul with i18n / 文档国际化重构

### ✨ New Features / 新增功能

- **Bilingual Documentation / 双语文档**: Complete English and Chinese docs
- **Enhanced VitePress / 增强的 VitePress**: Multi-language support with language switcher
- **Professional Changelog / 专业的变更日志**: Keep a Changelog format, archived history
- **Improved README / 优化的 README**: Better structure, performance highlights

### 📚 Documentation Pages / 文档页面

| English | 中文 | Description |
|---------|------|-------------|
| Tutorial | 教程 | Step-by-step GPU programming guide |
| API Reference | API 参考 | Complete API documentation |
| Performance | 性能指南 | Optimization tips and benchmarks |
| FAQ | 常见问题 | Troubleshooting and solutions |
| Cheatsheet | 速查表 | Quick reference |

### 🔄 Changes / 变更
- 22 new documentation files
- 11 modified existing files
- Full i18n support in VitePress

**Full Changelog**: https://github.com/LessUp/diy-flash-attention/compare/v1.0.2...v1.0.3
```

### 勾选选项:
- [x] Set as the latest release
- [ ] Create a discussion for this release

然后点击 **Publish release**

## 验证

创建 Release 后，请检查：
1. Release 页面正常显示
2. 标签 v1.0.3 存在
3. 文档站点构建成功: https://lessup.github.io/diy-flash-attention/
