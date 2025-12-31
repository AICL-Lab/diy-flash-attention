# 贡献指南

感谢你对 DIY FlashAttention 项目的兴趣！本文档将帮助你了解如何为项目做出贡献。

## 开发环境设置

### 1. 克隆仓库

```bash
git clone <repo-url>
cd diy-flash-attention
```

### 2. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

### 3. 安装开发依赖

```bash
pip install -e ".[dev]"
# 或者
pip install -r requirements.txt
pip install hypothesis pytest-cov
```

### 4. 验证安装

```bash
make test
make gpu-info
```

## 代码规范

### Python 风格

- 遵循 PEP 8 规范
- 使用类型注解
- 函数和类需要 docstring
- 最大行长度 100 字符

### Triton Kernel 规范

- 使用描述性的变量名
- 添加注释解释关键算法步骤
- 使用 `tl.constexpr` 标记编译时常量
- 处理边界条件 (masking)

### 示例

```python
@triton.jit
def example_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,  # 编译时常量
):
    """
    示例 kernel - 简要描述功能
    
    Args:
        input_ptr: 输入数据指针
        output_ptr: 输出数据指针
        n_elements: 元素数量
        BLOCK_SIZE: 每个 block 处理的元素数
    """
    # 获取当前 program ID
    pid = tl.program_id(0)
    
    # 计算偏移量
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 边界检查
    mask = offsets < n_elements
    
    # 加载、计算、存储
    data = tl.load(input_ptr + offsets, mask=mask)
    result = data * 2  # 示例计算
    tl.store(output_ptr + offsets, result, mask=mask)
```

## 测试要求

### 单元测试

每个新功能都需要对应的单元测试：

```python
# tests/test_new_feature.py
import pytest
import torch

class TestNewFeature:
    def test_basic_functionality(self):
        """测试基本功能"""
        # 准备输入
        # 调用函数
        # 验证输出
        pass
    
    def test_edge_cases(self):
        """测试边界情况"""
        pass
    
    def test_error_handling(self):
        """测试错误处理"""
        with pytest.raises(ValueError):
            # 触发错误的代码
            pass
```

### 属性测试

对于核心算法，添加属性测试：

```python
from hypothesis import given, strategies as st, settings

class TestNewFeatureProperty:
    @settings(max_examples=100, deadline=None)
    @given(
        size=st.integers(min_value=16, max_value=256),
    )
    def test_correctness_property(self, size):
        """
        Feature: new-feature, Property N: Description
        Validates: Requirements X.Y
        """
        # 生成随机输入
        # 计算结果
        # 验证属性
        pass
```

### 运行测试

```bash
# 运行所有测试
make test

# 运行特定测试
pytest tests/test_new_feature.py -v

# 运行测试并显示覆盖率
pytest tests/ --cov=kernels --cov=utils --cov-report=html
```

## 提交规范

### Commit 消息格式

使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Type 类型

- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `test`: 测试相关
- `refactor`: 代码重构
- `perf`: 性能优化
- `chore`: 构建/工具相关

### 示例

```
feat(kernels): 添加 FP8 矩阵乘法支持

- 实现 FP8 E4M3 和 E5M2 格式转换
- 添加 Hopper GPU 检测
- 包含 fallback 到 FP16

Closes #123
```

## Pull Request 流程

### 1. 创建分支

```bash
git checkout -b feature/your-feature-name
```

### 2. 开发和测试

```bash
# 编写代码
# 添加测试
make test
```

### 3. 提交更改

```bash
git add .
git commit -m "feat(scope): description"
```

### 4. 推送并创建 PR

```bash
git push origin feature/your-feature-name
# 在 GitHub 上创建 Pull Request
```

### PR 检查清单

- [ ] 代码遵循项目风格规范
- [ ] 添加了必要的测试
- [ ] 所有测试通过
- [ ] 更新了相关文档
- [ ] Commit 消息符合规范

## 报告问题

### Bug 报告

请包含以下信息：

1. **环境信息**
   - Python 版本
   - PyTorch 版本
   - Triton 版本
   - GPU 型号和 CUDA 版本

2. **问题描述**
   - 期望行为
   - 实际行为
   - 复现步骤

3. **错误信息**
   - 完整的错误堆栈
   - 相关日志

### 功能请求

请描述：

1. 你想要的功能
2. 使用场景
3. 可能的实现方式（可选）

## 文档贡献

### 文档结构

- `README.md` - 项目概述
- `docs/tutorial.md` - 教程
- `docs/api.md` - API 参考
- `CHANGELOG.md` - 版本历史
- `CONTRIBUTING.md` - 贡献指南

### 文档风格

- 使用中文编写
- 代码示例需要可运行
- 添加必要的注释和解释

## 联系方式

如有问题，可以通过以下方式联系：

- 提交 GitHub Issue
- 发送邮件到 [email]

感谢你的贡献！🎉
