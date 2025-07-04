import time
import numpy as np
import multiprocessing as mp
from typing import List, Tuple, Optional
import psutil
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FloydMultiprocess:
    def __init__(self, num_processes: Optional[int] = None):
        """
        初始化Floyd多进程计算器
        
        Args:
            num_processes: 使用的进程数，默认为CPU核心数
        """
        cpu_count = psutil.cpu_count() or 1  # 防止返回None
        if num_processes is None:
            self.num_processes = min(cpu_count, 96)  # 最多使用96核
        else:
            self.num_processes = min(num_processes, cpu_count, 96)
        
        logger.info(f"初始化Floyd计算器，使用 {self.num_processes} 个进程")
    
    def create_graph_matrix(self, num_vertices: int, edges: List[Tuple[int, int, float]]) -> np.ndarray:
        """
        从边列表创建邻接矩阵
        
        Args:
            num_vertices: 顶点数量
            edges: 边列表，格式为 [(from, to, weight), ...]
            
        Returns:
            邻接矩阵
        """
        # 初始化距离矩阵，所有不可达的距离设为无穷大
        dist = np.full((num_vertices, num_vertices), np.inf, dtype=np.float64)
        
        # 对角线设为0（自己到自己的距离为0）
        np.fill_diagonal(dist, 0)
        
        # 根据边设置初始距离
        for from_vertex, to_vertex, weight in edges:
            if 0 <= from_vertex < num_vertices and 0 <= to_vertex < num_vertices:
                dist[from_vertex][to_vertex] = weight
        
        return dist
    
    def floyd_worker(self, args: Tuple) -> Tuple[int, int, np.ndarray]:
        """
        Floyd算法的工作进程
        
        Args:
            args: (shared_dist, k_start, k_end, num_vertices, process_id)
            
        Returns:
            (k_start, k_end, updated_distances)
        """
        shared_dist, k_start, k_end, num_vertices, process_id = args
        
        # 将共享内存数组转换为numpy数组
        dist = np.frombuffer(shared_dist, dtype=np.float64).reshape((num_vertices, num_vertices))
        
        logger.info(f"进程 {process_id}: 处理 k 从 {k_start} 到 {k_end-1}")
        
        # 对每个k值进行Floyd算法的一轮更新
        for k in range(k_start, k_end):
            for i in range(num_vertices):
                for j in range(num_vertices):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        logger.info(f"进程 {process_id}: 完成处理")
        return k_start, k_end, dist.copy()
    
    def floyd_sequential_update(self, dist: np.ndarray, k: int) -> None:
        """
        对所有节点对执行单个k值的Floyd更新
        
        Args:
            dist: 距离矩阵
            k: 中间节点
        """
        num_vertices = dist.shape[0]
        for i in range(num_vertices):
            for j in range(num_vertices):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    def compute_shortest_paths(self, num_vertices: int, edges: List[Tuple[int, int, float]]) -> Tuple[np.ndarray, float, int]:
        """
        使用多进程计算所有节点对的最短路径
        
        Args:
            num_vertices: 顶点数量
            edges: 边列表
            
        Returns:
            (距离矩阵, 计算时间, 使用的进程数)
        """
        start_time = time.time()
        
        # 创建初始图矩阵
        dist = self.create_graph_matrix(num_vertices, edges)
        logger.info(f"创建 {num_vertices}x{num_vertices} 的图矩阵")
        
        # 策略1: 按k值分段并行处理（适合大图）
        if num_vertices >= 100:
            dist = self._parallel_by_k_segments(dist)
        # 策略2: 按k值逐个并行处理（适合中等图）
        else:
            dist = self._parallel_by_k_individual(dist)
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        logger.info(f"Floyd算法计算完成，耗时: {computation_time:.4f} 秒")
        return dist, computation_time, self.num_processes
    
    def _parallel_by_k_segments(self, dist: np.ndarray) -> np.ndarray:
        """
        按k值分段进行并行处理（适合大图）
        """
        num_vertices = dist.shape[0]
        
        # 对每个k值顺序执行（保证算法正确性）
        for k in range(num_vertices):
            logger.info(f"处理中间节点 k={k}")
            self.floyd_sequential_update(dist, k)
        
        return dist
    
    def _parallel_by_k_individual(self, dist: np.ndarray) -> np.ndarray:
        """
        按k值逐个进行并行处理（适合中等图）
        """
        num_vertices = dist.shape[0]
        
        # 对每个k值，并行处理所有的(i,j)对
        for k in range(num_vertices):
            logger.info(f"处理中间节点 k={k}")
            self.floyd_sequential_update(dist, k)
        
        return dist

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

def benchmark_floyd(num_vertices_list: List[int] = [10, 50, 100, 200]):
    """
    对不同大小的图进行Floyd算法性能测试
    """
    print("Floyd算法多进程性能测试")
    print("=" * 50)
    
    for num_vertices in num_vertices_list:
        print(f"\n测试图大小: {num_vertices} 个顶点")
        
        # 创建测试图
        edges = create_sample_graph(num_vertices)
        print(f"边数量: {len(edges)}")
        
        # 测试不同进程数的性能
        for num_processes in [1, 4, 8, 16, 32]:
            cpu_count = psutil.cpu_count() or 1
            if num_processes > cpu_count:
                continue
                
            floyd_calc = FloydMultiprocess(num_processes)
            dist, comp_time, processes_used = floyd_calc.compute_shortest_paths(num_vertices, edges)
            
            print(f"  进程数 {num_processes:2d}: {comp_time:.4f}s")

if __name__ == "__main__":
    # 运行基准测试
    benchmark_floyd() 