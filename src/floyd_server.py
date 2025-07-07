import asyncio
import grpc.aio
import logging
import time
import threading
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from typing import Dict, Optional
import uuid
from functools import partial

# 导入生成的protobuf类
import floyd_pb2
import floyd_pb2_grpc
from floyd_multiprocess import FloydMultiprocess

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Explicitly create a 'spawn' context for clean child processes
spawn_context = mp.get_context('spawn')

# 创建一个全局的进程池来处理计算密集型任务
# 这可以避免在每个请求中都创建新进程的开销
process_pool = ProcessPoolExecutor(max_workers=4, mp_context=spawn_context) # 可以根据CPU核心数调整

class FloydServicer(floyd_pb2_grpc.FloydServiceServicer):
    def __init__(self):
        """初始化Floyd服务"""
        self.compute_status: Dict[str, Dict] = {}  # 存储计算状态
        self.lock = threading.Lock()
        logger.info("Floyd gRPC服务初始化完成")
    
    async def ComputeShortestPaths(self, request: floyd_pb2.GraphRequest, context) -> floyd_pb2.GraphResponse:
        """
        计算图的最短路径
        
        Args:
            request: 包含图数据的请求
            context: gRPC上下文
            
        Returns:
            包含计算结果的响应
        """
        compute_id = str(uuid.uuid4())
        
        try:
            logger.info(f"收到计算请求: 顶点数={request.num_vertices}, 边数={len(request.edges)}, 进程数={request.num_processes}")
            
            # 更新计算状态
            with self.lock:
                self.compute_status[compute_id] = {
                    "status": "running",
                    "progress": 0.0,
                    "message": "开始计算",
                    "start_time": time.time()
                }
            
            # 转换边数据
            edges = []
            for edge in request.edges:
                edges.append((edge.from_node, edge.to, edge.weight))
            
            # 验证输入数据
            if request.num_vertices <= 0:
                raise ValueError("顶点数必须大于0")
            
            if request.num_processes <= 0:
                num_processes = None  # 使用默认进程数
            else:
                num_processes = request.num_processes
            
            # 使用 ProcessPoolExecutor 在完全独立的进程中运行阻塞的计算任务
            loop = asyncio.get_running_loop()
            
            # 注意：传递给 run_in_executor 的函数不能是实例方法，
            # 并且其参数必须可以被 pickle。
            # 我们将使用一个辅助函数来完成这个任务。
            distance_matrix, computation_time, processes_used = await loop.run_in_executor(
                process_pool, run_floyd_computation, request.num_vertices, edges, num_processes
            )
            
            # 更新状态
            with self.lock:
                self.compute_status[compute_id]["status"] = "completed"
                self.compute_status[compute_id]["progress"] = 1.0
                self.compute_status[compute_id]["message"] = "计算完成"
                self.compute_status[compute_id]["computation_time"] = computation_time
            
            # 转换结果为protobuf格式
            response_matrix = []
            for row in distance_matrix:
                row_pb = floyd_pb2.Row()
                # 处理无穷大值
                distances = []
                for dist in row:
                    if dist == float('inf'):
                        distances.append(-1.0)  # 用-1表示无穷大
                    else:
                        distances.append(float(dist))
                row_pb.distances.extend(distances)
                response_matrix.append(row_pb)
            
            # 创建响应
            response = floyd_pb2.GraphResponse(
                success=True,
                message=f"计算成功完成。使用 {processes_used} 个进程，耗时 {computation_time:.4f} 秒",
                distance_matrix=response_matrix,
                computation_time=computation_time,
                processes_used=processes_used
            )
            
            logger.info(f"计算完成: ID={compute_id}, 耗时={computation_time:.4f}s, 进程数={processes_used}")
            return response
            
        except Exception as e:
            error_msg = f"计算失败: {str(e)}"
            logger.error(f"计算错误: {error_msg}")
            
            # 更新错误状态
            with self.lock:
                self.compute_status[compute_id] = {
                    "status": "failed",
                    "progress": 0.0,
                    "message": error_msg
                }
            
            # 返回错误响应
            return floyd_pb2.GraphResponse(
                success=False,
                message=error_msg,
                distance_matrix=[],
                computation_time=0.0,
                processes_used=0
            )
    
    async def GetComputeStatus(self, request: floyd_pb2.StatusRequest, context) -> floyd_pb2.StatusResponse:
        """
        获取计算状态
        
        Args:
            request: 状态请求
            context: gRPC上下文
            
        Returns:
            计算状态响应
        """
        compute_id = request.compute_id
        
        with self.lock:
            if compute_id in self.compute_status:
                status_info = self.compute_status[compute_id]
                return floyd_pb2.StatusResponse(
                    status=status_info["status"],
                    progress=status_info["progress"],
                    message=status_info["message"]
                )
            else:
                return floyd_pb2.StatusResponse(
                    status="not_found",
                    progress=0.0,
                    message=f"未找到计算任务: {compute_id}"
                )

def run_floyd_computation(num_vertices: int, edges: list, num_processes: Optional[int]):
    """
    一个独立的顶层函数，用于在 ProcessPoolExecutor 中执行。
    这样可以确保它能被正确地 pickle。
    """
    floyd_calc = FloydMultiprocess(num_processes)
    return floyd_calc.compute_shortest_paths(num_vertices, edges)

async def serve(port: int = 50051):
    """
    启动异步gRPC服务器
    """
    # 设置消息大小限制 (100MB)
    options = [
        ('grpc.max_send_message_length', 100 * 1024 * 1024),
        ('grpc.max_receive_message_length', 100 * 1024 * 1024),
    ]
    
    server = grpc.aio.server(options=options)
    floyd_pb2_grpc.add_FloydServiceServicer_to_server(FloydServicer(), server)
    
    listen_addr = f'0.0.0.0:{port}'
    server.add_insecure_port(listen_addr)
    
    logger.info("启动异步 Floyd gRPC 服务器...")
    await server.start()
    logger.info(f"异步服务器启动成功，监听端口: {port}")
    
    try:
        await server.wait_for_termination()
    finally:
        logger.info("正在关闭进程池...")
        process_pool.shutdown(wait=True)
        logger.info("进程池已关闭")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Floyd算法gRPC服务器")
    parser.add_argument("--port", type=int, default=50051, help="服务端口 (默认: 50051)")
    
    args = parser.parse_args()
    
    print(f"启动Floyd算法gRPC服务器...")
    print(f"端口: {args.port}")
    print("按 Ctrl+C 停止服务器")
    
    asyncio.run(serve(port=args.port)) 