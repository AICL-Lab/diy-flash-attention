# 知识图谱

这一页把核心概念、推荐文档入口与对应源码文件串起来，帮助你在“读概念”和“看实现”之间来回定位。

| 概念 | 推荐阅读 | 对应源码 |
| --- | --- | --- |
| Online softmax | [`/zh/algorithm`](/zh/algorithm) | `kernels/flash_attn.py` |
| 并行划分 | [`/zh/architecture`](/zh/architecture) | `kernels/flash_attn_v2.py`、`kernels/backend_selector.py` |
| 掩码语义 | [`/zh/tutorial`](/zh/tutorial) | `kernels/mask_dsl.py` |
| 架构自适应 | [`/zh/performance`](/zh/performance) | `utils/gpu_detect.py`、`kernels/modern_features.py` |
