## Context

当前 `project-surface` 规格要求 README、GitHub Pages 与仓库 About 讲述同一个教育性 Triton/FlashAttention 故事，但 docs 首页仍偏展示页，尚未形成针对初学者的系统导学入口。此次 change 需要在不偏离“教育性、仅 forward pass、非泛化百科站点”范围的前提下，把 docs 站点升级为更像学习学院的入口，并补齐主题自适应 SVG 策略。

## Decisions

### Site Model

采用双层结构：
- **Portal layer**：homepage、learning path、paper guide、knowledge map，负责把首次访问者快速导向正确学习起点。
- **Reference layer**：tutorial、architecture、algorithm、performance、api，继续承载已有高信号参考内容。

这样可以保留现有参考文档的稳定性，同时让首页从“展示项目”转向“指导如何学习这个项目”。相比把所有内容堆在首页，双层结构更利于维护中英文镜像、导航一致性以及后续新增导学页面。

### Visual Strategy

主题适配策略分层处理：
- **Site logo**：使用单个自适应 SVG，通过 `prefers-color-scheme` 或等效主题机制保证在亮暗背景下都清晰可辨。
- **Content figures**：需要主题差异时，使用 `ThemeAwareFigure.vue` 与 light/dark 成对资源；无需差异时继续复用单图资源。
- **Mermaid / theme CSS**：继续遵循 VitePress 现有 dark mode 行为，避免另起一套颜色系统。

该策略兼顾维护成本与可读性：logo 优先单资源自适应，复杂插图才引入双资源切换。

## Risks / Trade-offs

- **入口页增加后导航复杂度上升** → 通过 portal/reference 分层和明确命名控制新增页面数量，避免首页堆叠过多内容。
- **主题适配规则不足时可能出现局部 SVG 对比度问题** → 在实施阶段补充 docs surface regression checks，并以主入口页和关键图示为优先验证对象。
- **中英文页面可能出现结构漂移** → 在实施任务中要求 EN/ZH 首页和导学页同步重构，保持表面对齐。
