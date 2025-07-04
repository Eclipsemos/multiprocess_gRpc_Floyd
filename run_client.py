#!/usr/bin/env python3
"""
Floyd算法gRPC客户端启动脚本
"""

import sys
import os
import asyncio

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from floyd_client import demo_small_graph, demo_large_graph, benchmark_different_sizes

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Floyd算法gRPC客户端")
    parser.add_argument("--server", default="127.0.0.1:50051", help="服务器地址 (默认: 127.0.0.1:50051)")
    parser.add_argument("--demo", choices=["small", "large", "benchmark"], default="small", 
                       help="运行演示类型 (默认: small)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("        Floyd算法多进程gRPC客户端")
    print("=" * 60)
    print(f"连接服务器: {args.server}")
    print(f"演示类型: {args.demo}")
    print("=" * 60)
    
    try:
        # 在运行前设置 no_proxy 环境变量
        if "no_proxy" not in os.environ:
             os.environ["no_proxy"] = "localhost,127.0.0.1"

        if args.demo == "small":
            print("运行小图演示...")
            asyncio.run(demo_small_graph(args.server))
        elif args.demo == "large":
            print("运行大图演示...")
            asyncio.run(demo_large_graph(args.server))
        elif args.demo == "benchmark":
            print("运行性能基准测试...")
            asyncio.run(benchmark_different_sizes(args.server))
        
        print("\n演示完成!")
        
    except KeyboardInterrupt:
        print("\n客户端已停止")
    except Exception as e:
        print(f"客户端运行失败: {e!r}")
        sys.exit(1) 