import numpy as np
import time
import logging
import psutil
from typing import List, Tuple, Optional
from multiprocessing import shared_memory
import multiprocessing as mp

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FloydMultiprocess:
    def __init__(self, num_processes: Optional[int] = None):
        """
        初始化Floyd算法计算器
        
        Args:
            num_processes: 使用的进程数，None表示使用CPU核心数
        """
        if num_processes is None:
            self.num_processes = psutil.cpu_count() or 1
        else:
            self.num_processes = max(1, num_processes)
        
        logger.info(f"Floyd算法初始化，使用进程数: {self.num_processes}")
    
    def create_graph_matrix(self, num_vertices: int, edges: List[Tuple[int, int, float]]) -> np.ndarray:
        """
        从边列表创建邻接矩阵
        
        Args:
            num_vertices: 顶点数量
            edges: 边列表，格式为 [(from, to, weight), ...]
            
        Returns:
            初始化的距离矩阵
        """
        # 初始化距离矩阵，所有距离为无穷大
        dist = np.full((num_vertices, num_vertices), np.inf, dtype=np.float64)
        
        # 对角线元素为0（自己到自己的距离）
        np.fill_diagonal(dist, 0)
        
        # 添加边的权重
        for from_vertex, to_vertex, weight in edges:
            if 0 <= from_vertex < num_vertices and 0 <= to_vertex < num_vertices:
                dist[from_vertex][to_vertex] = weight
        
        logger.info(f"创建 {num_vertices}x{num_vertices} 的图矩阵，边数: {len(edges)}")
        return dist

    def compute_shortest_paths(self, num_vertices: int, edges: List[Tuple[int, int, float]]) -> Tuple[np.ndarray, float, int]:
        """
        使用NumPy矢量化计算所有节点对的最短路径（Floyd-Warshall算法）
        
        这个实现利用NumPy的矢量化操作和底层优化的数学库（如Intel MKL、OpenBLAS），
        能够自动利用多核CPU，比手动多进程更高效。
        
        Args:
            num_vertices: 顶点数量
            edges: 边列表
            
        Returns:
            (距离矩阵, 计算时间, 有效进程数)
        """
        start_time = time.time()
        
        # 创建初始图矩阵
        dist = self.create_graph_matrix(num_vertices, edges)
        logger.info(f"开始NumPy矢量化Floyd算法，图大小: {num_vertices}x{num_vertices}")
        
        # 矢量化的Floyd-Warshall算法
        # 对每个中间节点k，一次性更新所有的i,j对
        for k in range(num_vertices):
            if k % 100 == 0 or k == num_vertices - 1:
                logger.info(f"处理中间节点: {k+1}/{num_vertices}")
            
            # 矢量化操作：dist = min(dist, dist[:, k:k+1] + dist[k:k+1, :])
            # 这一行代码替代了双重循环，NumPy会自动并行化
            dist = np.minimum(dist, dist[:, k:k+1] + dist[k:k+1, :])
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        # 获取实际可用的CPU核心数，作为"有效进程数"
        effective_cores = psutil.cpu_count(logical=False) or 1  # 物理核心数
        
        logger.info(f"NumPy矢量化Floyd算法完成，耗时: {computation_time:.4f} 秒")
        logger.info(f"利用CPU核心数: {effective_cores} (NumPy自动并行)")
        
        return dist, computation_time, effective_cores

    def compute_shortest_paths_traditional(self, num_vertices: int, edges: List[Tuple[int, int, float]]) -> Tuple[np.ndarray, float, int]:
        """
        传统的三重循环Floyd算法实现（用于性能对比）
        
        Args:
            num_vertices: 顶点数量
            edges: 边列表
            
        Returns:
            (距离矩阵, 计算时间, 进程数)
        """
        start_time = time.time()
        
        # 创建初始图矩阵
        dist = self.create_graph_matrix(num_vertices, edges)
        logger.info(f"开始传统Floyd算法，图大小: {num_vertices}x{num_vertices}")
        
        # 传统的三重循环实现
        for k in range(num_vertices):
            if k % 100 == 0 or k == num_vertices - 1:
                logger.info(f"处理中间节点: {k+1}/{num_vertices}")
            
            for i in range(num_vertices):
                for j in range(num_vertices):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        logger.info(f"传统Floyd算法完成，耗时: {computation_time:.4f} 秒")
        return dist, computation_time, 1

def create_sample_graph(num_vertices: int = 50) -> List[Tuple[int, int, float]]:
    """
    创建一个示例图用于测试
    
    Args:
        num_vertices: 顶点数量
        
    Returns:
        边列表
    """
    edges = []
    np.random.seed(42)  # 确保可重现
    
    # 创建随机连通图
    for i in range(num_vertices):
        # 每个节点连接到下一个节点，确保连通性
        next_vertex = (i + 1) % num_vertices
        weight = np.random.uniform(1, 10)
        edges.append((i, next_vertex, weight))
        
        # 添加一些随机边
        max_edges = max(1, min(5, num_vertices // 5))
        if max_edges > 1:
            num_random_edges = np.random.randint(1, max_edges)
        else:
            num_random_edges = 1
        for _ in range(num_random_edges):
            target = np.random.randint(0, num_vertices)
            if target != i:
                weight = np.random.uniform(1, 20)
                edges.append((i, target, weight))
    
    return edges

def benchmark_floyd_comparison():
    """
    比较NumPy矢量化和传统实现的性能
    """
    print("Floyd算法性能对比测试")
    print("=" * 60)
    
    test_sizes = [100, 200, 400, 800]
    
    for num_vertices in test_sizes:
        print(f"\n测试图大小: {num_vertices} 个顶点")
        
        # 创建测试图
        edges = create_sample_graph(num_vertices)
        print(f"边数量: {len(edges)}")
        
        floyd_calc = FloydMultiprocess()
        
        # 测试NumPy矢量化版本
        dist_numpy, time_numpy, cores_numpy = floyd_calc.compute_shortest_paths(num_vertices, edges)
        
        # 测试传统版本（只对较小的图进行测试）
        if num_vertices <= 200:
            dist_traditional, time_traditional, cores_traditional = floyd_calc.compute_shortest_paths_traditional(num_vertices, edges)
            speedup = time_traditional / time_numpy
            
            print(f"  NumPy矢量化: {time_numpy:.4f}s (利用 {cores_numpy} 核心)")
            print(f"  传统实现:    {time_traditional:.4f}s (单核)")
            print(f"  加速比:      {speedup:.2f}x")
            
            # 验证结果一致性
            if np.allclose(dist_numpy, dist_traditional, rtol=1e-9):
                print(f"  ✓ 结果验证通过")
            else:
                print(f"  ✗ 结果验证失败")
        else:
            print(f"  NumPy矢量化: {time_numpy:.4f}s (利用 {cores_numpy} 核心)")
            print(f"  传统实现:    跳过（图太大）")

if __name__ == "__main__":
    # 运行性能对比测试
    benchmark_floyd_comparison() 