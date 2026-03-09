#!/usr/bin/env python3
"""
运行所有 Benchmark 并生成报告

用法:
    python scripts/run_all_benchmarks.py
    python scripts/run_all_benchmarks.py --output report.md
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent


def run_command(cmd: list, capture: bool = True) -> str:
    """运行命令并返回输出"""
    print(f"运行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=capture, text=True, cwd=ROOT)
    if result.returncode != 0:
        print(f"错误: {result.stderr}")
        return f"命令失败: {result.stderr}"
    return result.stdout if capture else ""


def get_gpu_info() -> str:
    """获取 GPU 信息"""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "No GPU"
    except Exception:
        return "Unknown"


def main():
    parser = argparse.ArgumentParser(description="运行所有 Benchmark")
    parser.add_argument("--output", "-o", type=str, help="输出报告文件")
    args = parser.parse_args()

    report_lines = []

    # 标题
    report_lines.append("# DIY FlashAttention Benchmark 报告")
    report_lines.append("")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"GPU: {get_gpu_info()}")
    report_lines.append("")

    # 运行矩阵乘法 benchmark
    report_lines.append("## 矩阵乘法 Benchmark")
    report_lines.append("")
    report_lines.append("```")
    output = run_command([sys.executable, "benchmarks/bench_matmul.py"])
    report_lines.append(output)
    report_lines.append("```")
    report_lines.append("")

    # 运行 FlashAttention benchmark
    report_lines.append("## FlashAttention Benchmark")
    report_lines.append("")
    report_lines.append("```")
    output = run_command([sys.executable, "benchmarks/bench_flash.py"])
    report_lines.append(output)
    report_lines.append("```")
    report_lines.append("")

    # 运行测试
    report_lines.append("## 测试结果")
    report_lines.append("")
    report_lines.append("```")
    output = run_command([sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"])
    report_lines.append(output)
    report_lines.append("```")

    # 输出报告
    report = "\n".join(report_lines)

    if args.output:
        Path(args.output).write_text(report)
        print(f"\n报告已保存到: {args.output}")
    else:
        print("\n" + "=" * 60)
        print(report)


if __name__ == "__main__":
    main()
