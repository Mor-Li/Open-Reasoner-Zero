import socket

import ray
import torch
from vllm.worker.worker import Worker

from orz.exp_engine.parallels.orz_distributed_c10d import orz_init_process_group


class WorkerWrap(Worker):
    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend="nccl"):
        """初始化用于模型权重更新的torch进程组"""
        assert torch.distributed.is_initialized(), "默认的torch进程组必须已初始化"
        assert group_name != "", "组名不能为空"

        # 计算当前进程的rank
        rank = torch.distributed.get_rank() + rank_offset
        # 初始化进程组
        self._model_update_group = orz_init_process_group(
            backend=backend,
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=rank,
            group_name=group_name,
        )
        print(
            f"init_process_group: master_address={master_address}, master_port={master_port}, ",
            f"rank={rank}, world_size={world_size}, group_name={group_name}",
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        """从源rank 0（actor模型）广播权重到所有vllm工作者"""
        # 确保数据类型匹配
        assert dtype == self.model_config.dtype, f"数据类型不匹配: 源 {dtype}, 目标 {self.model_config.dtype}"
        # 创建一个空的权重张量
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        # 广播权重
        torch.distributed.broadcast(weight, 0, group=self._model_update_group)

        # 加载权重到模型
        self.model_runner.model.load_weights(weights=[(name, weight)])

        # 删除权重以释放内存
        del weight
        # TODO: 如果所有权重都已更新，是否应该清空缓存？
        # if empty_cache:
        #     torch.cuda.empty_cache()

    def update_weight_internal_with_cuda_ipc(self, name, dtype, shape, cudaipc_handler, empty_cache=False):
        """使用CUDA IPC更新权重"""
        # 确保数据类型匹配
        assert dtype == self.model_config.dtype, f"数据类型不匹配: 源 {dtype}, 目标 {self.model_config.dtype}"
        # 重建并克隆权重
        weight = cudaipc_handler.rebuild().clone()
        # 加载权重到模型
        self.model_runner.model.load_weights(weights=[(name, weight)])
        # 删除权重以释放内存
        del weight

    def get_ip_and_port(self):
        """获取当前节点的IP地址和一个可用端口"""
        master_address = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            master_port = sock.getsockname()[1]
        return master_address, master_port

    def free_weight(self):
        """将模型权重转移到meta设备以释放内存"""
        self.model_runner.model.to("meta")

    def free_cache_engine(self):
        """释放缓存引擎和GPU缓存"""
        self.cache_engine = None
        self.gpu_cache = None

    def init_cache_engine(self):
        """初始化缓存引擎"""
        if self.cache_engine is None and self.gpu_cache is None:
            super()._init_cache_engine()


class OffloadableVLLMWorker(WorkerWrap):
    """对vLLM工作者进行猴子补丁以操作模型参数。

    当VLLMAccelerated-InferenceModelWorker被导入时，这个类将替换原始的Worker类，灵感来自`OpenRLHF`。
    """

    def offload_cpu(self):
        """将模型参数卸载到CPU"""
        assert self.model_config.enforce_eager, "必须使用eager模式进行卸载！"
        for param in self.model_runner.model.parameters():
            # 将参数数据转移到meta设备
            param.meta_tensor = param.data.to("meta")
            param.data = torch.Tensor([])

        # 释放缓存引擎和GPU缓存
        self.cache_engine = None
        self.gpu_cache = None
        torch.cuda.empty_cache()

    def load_gpu(self):
        """将模型参数加载到GPU"""
        assert self.model_config.enforce_eager, "必须使用eager模式进行卸载！"
        for param in self.model_runner.model.parameters():
            # 创建一个与meta张量相同大小的空张量，并将其加载到GPU
            param.data = torch.empty_like(param.meta_tensor, device="cuda")
            param.meta_tensor = None
        if self.cache_engine is None and self.gpu_cache is None:
            super()._init_cache_engine()
