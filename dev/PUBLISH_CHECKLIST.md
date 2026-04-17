# 📋 发布检查清单 v1.0.3

## ✅ 已完成工作

### 1. 文档重构 (Docs Restructure)
- [x] 创建 `docs/en/` 目录，包含完整英文文档 (7 个文件)
  - [x] index.md - 首页
  - [x] tutorial.md - 教程
  - [x] api.md - API 参考
  - [x] performance.md - 性能指南
  - [x] faq.md - 常见问题
  - [x] cheatsheet.md - 速查表
  - [x] changelog.md - 更新日志
  
- [x] 创建 `docs/zh/` 目录，迁移中文文档 (7 个文件)
  - 与原 docs/ 中文文件内容一致
  
- [x] 更新 `docs/index.md` - 语言选择首页

### 2. Changelog 专业化 (Changelog Professional)
- [x] 维护根目录 `CHANGELOG.md` - 标准 Keep a Changelog 格式
- [x] 维护 `changelog/CHANGELOG.zh-CN.md` - 中文版本
- [x] 创建 `changelog/README.md` - 目录说明
- [x] 创建 `changelog/archive/` 和归档说明
- [x] 移动 7 个旧 changelog 文件到归档目录
- [x] 更新根目录 `CHANGELOG.md`

### 3. README 优化 (README Optimization)
- [x] 重写 `README.md` - 更专业的英文版
- [x] 更新 `README.zh-CN.md` - 同步中文内容
- [x] 添加双语链接和性能亮点

### 4. VitePress 配置 (VitePress Config)
- [x] 更新 `docs/.vitepress/config.mts`
- [x] 添加多语言 locale 配置
- [x] 添加语言切换下拉菜单
- [x] 配置中英文导航和侧边栏
- [x] 添加本地化搜索支持

### 5. Git 提交 (Git Commits)
- [x] 提交文档重构 (commit: d484b98)
- [x] 提交 release notes (commit: 5850f5b)
- [x] 创建本地标签 v1.0.3

---

## 🚀 待完成步骤

### 步骤 1: 推送代码到 GitHub

```bash
# 在终端执行
git push origin master
git push origin v1.0.3
```

**如果推送失败**，请检查：
- 网络连接
- GitHub 身份验证（可能需要重新登录或使用 SSH）
- 是否有冲突需要解决

### 步骤 2: 创建 GitHub Release

#### 方式 A: 使用 GitHub CLI (推荐)

```bash
# 确保已安装 gh CLI
# https://cli.github.com/

gh release create v1.0.3 \
  --title "v1.0.3: Documentation Overhaul with i18n Support / 文档国际化重构" \
  --notes-file RELEASE_NOTES.md \
  --latest
```

#### 方式 B: Web 界面手动创建

1. 访问: https://github.com/LessUp/diy-flash-attention/releases/new

2. **Choose a tag**: 选择 `v1.0.3` (或创建新标签)

3. **Release title**:
   ```
   v1.0.3: Documentation Overhaul with i18n Support / 文档国际化重构
   ```

4. **Release notes**: 复制 `RELEASE_NOTES.md` 的内容

5. **勾选选项**:
   - [x] Set as the latest release
   - [ ] Create a discussion for this release

6. 点击 **Publish release**

### 步骤 3: 验证文档站点

发布后约 2-5 分钟，检查文档站点：

1. 访问首页: https://lessup.github.io/diy-flash-attention/
   - 应该看到语言选择页面
   
2. 访问英文文档: https://lessup.github.io/diy-flash-attention/en/
   - 应该看到英文首页
   
3. 访问中文文档: https://lessup.github.io/diy-flash-attention/zh/
   - 应该看到中文首页
   
4. 测试语言切换:
   - 在英文页面点击 🌐 Language → 中文
   - 在中文页面点击 🌐 语言 → English

### 步骤 4: 验证 README

访问 GitHub 仓库主页，检查：
- README.md 显示新的专业版本
- 中英文切换链接正常工作
- 徽章和链接都正常显示

---

## 🎯 发布成功标准

- [ ] `git push origin master` 成功
- [ ] `git push origin v1.0.3` 成功
- [ ] GitHub Release v1.0.3 已发布
- [ ] 文档站点能访问且语言切换正常
- [ ] README 显示正确

---

## 🔧 故障排除

### 推送失败

```bash
# 检查远程 URL
git remote -v

# 如果需要，切换到 SSH
git remote set-url origin git@github.com:LessUp/diy-flash-attention.git

# 重试推送
git push origin master
git push origin v1.0.3
```

### 标签已存在

```bash
# 删除本地标签
git tag -d v1.0.3

# 删除远程标签
git push origin --delete tag v1.0.3

# 重新创建并推送
git tag -a v1.0.3 -m "Release v1.0.3"
git push origin v1.0.3
```

### 文档站点未更新

检查 GitHub Actions:
1. 访问: https://github.com/LessUp/diy-flash-attention/actions
2. 查看 `pages.yml` 工作流状态
3. 确保构建成功

---

## 📞 需要帮助?

- 文档问题: 检查 `docs/.vitepress/config.mts`
- Release 问题: 参考 `RELEASE_COMMANDS.md`
- Git 问题: 参考上述故障排除部分
