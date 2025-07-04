#!/usr/bin/env python3
"""
Floyd算法多进程测试脚本
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from floyd_multiprocess import FloydMultiprocess, create_sample_graph, benchmark_floyd

def test_basic_floyd():
    """测试基本Floyd算法功能"""
    print("=== 基本Floyd算法测试 ===")
    
    # 创建一个简单的5节点图
    num_vertices = 5
    edges = [
        (0, 1, 2.0),
        (0, 2, 8.0),
        (1, 2, 1.0),
        (1, 3, 4.0),
        (2, 3, 2.0),
        (2, 4, 3.0),
        (3, 4, 1.0)
    ]
    
    print(f"图信息: {num_vertices} 个顶点, {len(edges)} 条边")
    
    # 测试不同进程数
    for num_processes in [1, 4, 8]:
        floyd_calc = FloydMultiprocess(num_processes)
        dist, comp_time, processes_used = floyd_calc.compute_shortest_paths(num_vertices, edges)
        
        print(f"进程数 {num_processes}: 耗时 {comp_time:.4f}s")
        
        # 打印距离矩阵
        if num_processes == 1:  # 只打印一次
            print("\n距离矩阵:")
            for i in range(num_vertices):
                for j in range(num_vertices):
                    if dist[i][j] == float('inf'):
                        print("  inf", end="")
                    else:
                        print(f"{dist[i][j]:6.2f}", end="")
                print()

def test_large_graph():
    """测试大图性能"""
    print("\n=== 大图性能测试 ===")
    
    sizes = [50, 100]
    
    for size in sizes:
        print(f"\n图大小: {size} 个顶点")
        edges = create_sample_graph(size)
        print(f"边数: {len(edges)}")
        
        # 测试不同进程数
        for num_processes in [1, 8, 16, 32]:
            floyd_calc = FloydMultiprocess(num_processes)
            dist, comp_time, processes_used = floyd_calc.compute_shortest_paths(size, edges)
            
            print(f"  进程数 {num_processes:2d}: {comp_time:.4f}s (实际使用 {processes_used} 进程)")

if __name__ == "__main__":
    print("Floyd算法多进程计算测试")
    print("=" * 50)
    
    try:
        test_basic_floyd()
        test_large_graph()
        
        print("\n运行完整基准测试...")
        benchmark_floyd([10, 50, 100])
        
    except KeyboardInterrupt:
        print("\n测试被中断")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n所有测试完成!") 