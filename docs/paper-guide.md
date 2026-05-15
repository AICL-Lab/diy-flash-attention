# Paper Guide

Use this page as a quick answer to what to read first before diving into the code.

## 1. FlashAttention (start here)

- Problem it solves: exact attention without materializing the full attention matrix
- Read first: introduction, algorithm overview, IO-awareness argument
- Connect back to repo: [algorithm walkthrough](/algorithm), `kernels/flash_attn.py`

## 2. FlashAttention-2

- What changes: better work partitioning and improved parallelism
- Read after: you already understand the v1 online softmax loop
- Connect back to repo: [algorithm walkthrough](/algorithm), `kernels/flash_attn_v2.py`, `kernels/backend_selector.py`

## 3. Related extensions

- PagedAttention / MQA / GQA
- Purpose: broaden the reader's map without redefining this repo's scope
