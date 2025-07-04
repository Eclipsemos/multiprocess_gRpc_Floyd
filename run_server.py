#!/usr/bin/env python3
"""
Floyd算法gRPC服务器启动脚本
"""

import sys
import os
import asyncio

# 启用 gRPC 的 fork 支持，必须在导入 grpc 之前设置
os.environ["GRPC_FORK_SUPPORT_ENABLED"] = "true"

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from floyd_server import serve

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Floyd算法gRPC服务器")
    parser.add_argument("--port", type=int, default=50051, help="服务端口 (默认: 50051)")
    parser.add_argument("--max-workers", type=int, default=10, help="最大工作线程数 (默认: 10)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("        Floyd算法多进程gRPC服务器")
    print("=" * 60)
    print(f"端口: {args.port}")
    print(f"最大工作线程数: {args.max_workers}")
    print(f"系统CPU核心数: {os.cpu_count()}")
    print("按 Ctrl+C 停止服务器")
    print("=" * 60)
    
    try:
        asyncio.run(serve(port=args.port))
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"服务器启动失败: {e!r}")
        sys.exit(1) 