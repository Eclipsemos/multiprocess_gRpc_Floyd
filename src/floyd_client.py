import grpc.aio
import logging
import time
from typing import List, Tuple, Optional
import asyncio

# 导入生成的protobuf类
import floyd_pb2
import floyd_pb2_grpc
from floyd_multiprocess import create_sample_graph

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FloydClient:
    def __init__(self, server_address: str = "localhost:50051"):
        """
        初始化Floyd客户端
        
        Args:
            server_address: 服务器地址，格式为 "host:port"
        """
        self.server_address = server_address
        self._channel = None
        self._stub = None
        logger.info(f"Floyd客户端初始化，服务器地址: {server_address}")
    
    async def __aenter__(self):
        """支持异步上下文管理器 (async with)"""
        # 设置消息大小限制 (100MB)
        options = [
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ]
        
        self._channel = grpc.aio.insecure_channel(self.server_address, options=options)
        self._stub = floyd_pb2_grpc.FloydServiceStub(self._channel)
        # aio 客户端不需要显式 connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """在退出时关闭通道"""
        if self._channel:
            await self._channel.close()
            logger.info("已断开与服务器的连接")

    async def compute_shortest_paths(self, 
                             num_vertices: int, 
                             edges: List[Tuple[int, int, float]], 
                             num_processes: Optional[int] = None) -> Optional[floyd_pb2.GraphResponse]:
        """
        请求计算图的最短路径
        
        Args:
            num_vertices: 顶点数量
            edges: 边列表，格式为 [(from_node, to, weight), ...]
            num_processes: 使用的进程数，None表示使用默认值
            
        Returns:
            计算结果响应，失败时返回None
        """
        if not self._stub:
            logger.error("客户端未连接到服务器")
            return None
        
        try:
            # 创建边对象列表
            edge_objects = []
            for from_vertex, to_vertex, weight in edges:
                edge = floyd_pb2.Edge(
                    from_node=from_vertex,
                    to=to_vertex,
                    weight=weight
                )
                edge_objects.append(edge)
            
            # 创建请求
            request = floyd_pb2.GraphRequest(
                num_vertices=num_vertices,
                edges=edge_objects,
                num_processes=num_processes or 0  # 0表示使用默认进程数
            )
            
            logger.info(f"发送计算请求: 顶点数={num_vertices}, 边数={len(edges)}, 进程数={num_processes}")
            
            # 发送请求
            start_time = time.time()
            response = await self._stub.ComputeShortestPaths(request, timeout=300) # 增加超时时间
            request_time = time.time() - start_time
            
            if response.success:
                logger.info(f"计算成功完成")
                logger.info(f"  服务器计算时间: {response.computation_time:.4f} 秒")
                logger.info(f"  网络请求时间: {request_time:.4f} 秒")
                logger.info(f"  使用进程数: {response.processes_used}")
                logger.info(f"  服务器消息: {response.message}")
            else:
                logger.error(f"计算失败: {response.message}")
            
            return response
            
        except grpc.aio.AioRpcError as e:
            logger.error(f"gRPC调用失败: {e.code()} - {e.details()}")
            return None
        except Exception as e:
            logger.error(f"计算请求失败: {e!r}")
            return None
    
    async def get_compute_status(self, compute_id: str) -> Optional[floyd_pb2.StatusResponse]:
        """
        获取计算任务状态
        
        Args:
            compute_id: 计算任务ID
            
        Returns:
            状态响应，失败时返回None
        """
        if not self._stub:
            logger.error("客户端未连接到服务器")
            return None
        
        try:
            request = floyd_pb2.StatusRequest(compute_id=compute_id)
            response = await self._stub.GetComputeStatus(request)
            
            logger.info(f"任务状态: {response.status}")
            logger.info(f"进度: {response.progress:.2%}")
            logger.info(f"消息: {response.message}")
            
            return response
            
        except grpc.aio.AioRpcError as e:
            logger.error(f"获取状态失败: {e.code()} - {e.details()}")
            return None
        except Exception as e:
            logger.error(f"状态请求失败: {e!r}")
            return None

def print_distance_matrix(response: floyd_pb2.GraphResponse, max_display: int = 10):
    """
    打印距离矩阵
    
    Args:
        response: 服务器响应
        max_display: 最多显示的顶点数量
    """
    if not response.success or not response.distance_matrix:
        print("没有可显示的距离矩阵")
        return
    
    matrix_size = len(response.distance_matrix)
    display_size = min(matrix_size, max_display)
    
    print(f"\n距离矩阵 ({matrix_size}x{matrix_size}):")
    if matrix_size > max_display:
        print(f"(仅显示前 {display_size}x{display_size} 部分)")
    
    # 打印列标题
    print("     ", end="")
    for j in range(display_size):
        print(f"{j:8}", end="")
    print()
    
    # 打印矩阵内容
    for i in range(display_size):
        print(f"{i:3}: ", end="")
        row = response.distance_matrix[i]
        for j in range(min(display_size, len(row.distances))):
            dist = row.distances[j]
            if dist == -1.0:  # 无穷大
                print("     inf", end="")
            else:
                print(f"{dist:8.2f}", end="")
        print()

async def demo_small_graph(server_address: str):
    """演示小图计算"""
    print("\n=== 小图演示 ===")
    
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
    
    async with FloydClient(server_address) as client:
        response = await client.compute_shortest_paths(num_vertices, edges, num_processes=4)
        if response:
            print_distance_matrix(response)

async def demo_large_graph(server_address: str):
    """演示大图计算"""
    print("\n=== 大图演示 ===")
    
    # 创建一个较大的随机图
    num_vertices = 800
    edges = create_sample_graph(num_vertices)
    
    async with FloydClient(server_address) as client:
        # 测试不同进程数的性能
        for num_processes in [32, 64, 96]:
            print(f"\n测试 {num_processes} 个进程:")
            response = await client.compute_shortest_paths(num_vertices, edges, num_processes)
            if response and response.success:
                print(f"  计算时间: {response.computation_time:.4f} 秒")
                print(f"  使用进程: {response.processes_used}")
            else:
                print(f"  计算失败")
            await asyncio.sleep(1)  # 避免服务器过载

async def benchmark_different_sizes(server_address: str):
    """测试不同图大小的性能"""
    print("\n=== 不同图大小性能测试 ===")
    
    sizes = [10, 50, 100, 200]
    
    async with FloydClient(server_address) as client:
        for size in sizes:
            print(f"\n图大小: {size} 个顶点")
            edges = create_sample_graph(size)
            
            response = await client.compute_shortest_paths(size, edges, num_processes=32)
            if response and response.success:
                print(f"  边数: {len(edges)}")
                print(f"  计算时间: {response.computation_time:.4f} 秒")
                print(f"  使用进程: {response.processes_used}")
                
                # 显示部分结果
                if size <= 10:
                    print_distance_matrix(response)
            else:
                print(f"  计算失败")
            
            await asyncio.sleep(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Floyd算法gRPC客户端")
    parser.add_argument("--server", default="localhost:50051", help="服务器地址 (默认: localhost:50051)")
    parser.add_argument("--demo", choices=["small", "large", "benchmark"], default="small", 
                       help="运行演示类型 (默认: small)")
    
    args = parser.parse_args()
    
    print(f"Floyd算法gRPC客户端")
    print(f"连接服务器: {args.server}")
    print(f"演示类型: {args.demo}")
    
    # 更新服务器地址
    if args.demo == "small":
        asyncio.run(demo_small_graph(args.server))
    elif args.demo == "large":
        asyncio.run(demo_large_graph(args.server))
    elif args.demo == "benchmark":
        asyncio.run(demo_benchmark_different_sizes(args.server))
    
    print("\n演示完成!") 