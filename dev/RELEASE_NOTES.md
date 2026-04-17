# Release Notes for v1.0.3

## 🌍 文档全面重构与国际化 / Comprehensive Documentation Overhaul with i18n

We are excited to announce v1.0.3, featuring complete bilingual documentation and professional changelog management!

---

## ✨ What's New / 新增内容

### 📚 Bilingual Documentation / 双语文档

- **English Documentation** (`docs/en/`)
  - Complete API reference with usage examples
  - Step-by-step tutorial covering GPU basics to FlashAttention
  - Performance optimization guide with benchmarks
  - FAQ with troubleshooting steps
  - Quick reference cheatsheet

- **中文文档** (`docs/zh/`)
  - 完整 API 参考文档和使用示例
  - 从 GPU 基础到 FlashAttention 的分步教程
  - 性能优化指南和基准测试
  - 常见问题解答和问题排查
  - 快速参考速查表

### 🎨 Enhanced VitePress Site / 增强的 VitePress 站点

- Language switcher in navigation bar
- Localized search support
- Proper i18n configuration for all UI elements
- Improved navigation structure

### 📋 Professional Changelog / 专业的变更日志

- Follows [Keep a Changelog](https://keepachangelog.com/) format
- Bilingual changelog (English/Chinese)
- Archived historical changelogs
- Clear version history and roadmap

### 📝 Improved README / 优化的 README

- More engaging introduction with performance highlights
- Better organized content structure
- Quick start guide with examples
- GPU compatibility matrix
- Bilingual documentation links

---

## 🔄 Changes Summary / 变更摘要

| Category | Changes |
|----------|---------|
| Files Added | 22 new documentation files |
| Files Modified | 11 existing files |
| Languages Supported | English, Chinese (中文) |
| Documentation Pages | 14 total (7 EN + 7 ZH) |

---

## 🚀 How to Use / 使用方法

### Access Documentation / 访问文档

```bash
# Install dependencies
npm install

# Start dev server
cd docs && npm run docs:dev

# Build for production
npm run docs:build
```

### Switch Language / 切换语言

Use the 🌐 **Language** dropdown in the top navigation to switch between English and Chinese.

---

## 📊 Performance Benchmarks / 性能基准

| Operation | PyTorch | Triton (Ours) | Speedup |
|-----------|---------|---------------|---------|
| MatMul 4096² | 120 TFLOPS | 140 TFLOPS | 1.17x |
| Attention 4096 | 35.0 ms | 22.0 ms | 1.59x |

---

## 📝 Release Checklist / 发布检查清单

- [x] English documentation complete
- [x] Chinese documentation complete
- [x] VitePress i18n configured
- [x] Changelog restructured
- [x] README optimized
- [x] Git tag created
- [ ] GitHub release published

---

## 🙏 Acknowledgments / 致谢

Special thanks to the Triton team for their excellent GPU programming framework and the FlashAttention paper authors for their groundbreaking algorithm.

---

## 📞 Get Help / 获取帮助

- 📖 [Documentation](https://lessup.github.io/diy-flash-attention/)
- 🐛 [Report Issues](https://github.com/LessUp/diy-flash-attention/issues)
- 💡 [Start a Discussion](https://github.com/LessUp/diy-flash-attention/discussions)

---

**Full Changelog**: https://github.com/LessUp/diy-flash-attention/compare/v1.0.2...v1.0.3
