# 论文导读

这一页用来回答“该先读什么”，帮助你先建立论文地图，再回到代码实现。

## 1. FlashAttention（先读这一篇）

- 它解决的问题：在不物化完整注意力矩阵的前提下做 exact attention
- 建议优先读：引言、算法概览、IO-aware 核心论证
- 与本仓库对应：[算法详解](/zh/algorithm)、`kernels/flash_attn.py`

## 2. FlashAttention-2

- 它带来的变化：更好的并行划分与工作分配
- 建议阅读前提：你已经理解 v1 的 online softmax 主循环
- 与本仓库对应：[算法详解](/zh/algorithm)、`kernels/flash_attn_v2.py`、`kernels/backend_selector.py`

## 3. 相关延伸

- PagedAttention / MQA / GQA
- 作用：扩展读者的认知地图，但不改变本仓库“教育性、仅 forward”的范围
