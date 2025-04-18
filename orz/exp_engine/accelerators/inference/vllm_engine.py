from vllm.core.scheduler import Scheduler


class LLMActor:
    def __init__(self, *args, **kwargs):
        import vllm

        # 获取vllm的版本号并进行版本检查，确保版本符合要求
        self.__version__ = vllm.__version__
        assert self.__version__ >= "0.4.1", "OpenRLHF 仅支持 vLLM >= 0.4.1"

        # 判断是否使用GPU执行器，依据tensor_parallel_size参数
        self.use_gpu_executor = kwargs["tensor_parallel_size"] == 1

        # 如果使用GPU执行器，导入并使用OffloadableVLLMWorker
        if self.use_gpu_executor:
            from .vllm_worker_wrap import OffloadableVLLMWorker

            # 将Worker类替换为OffloadableVLLMWorker
            vllm.worker.worker.Worker = OffloadableVLLMWorker
        else:
            # 如果不使用GPU执行器，设置使用Ray的标志
            kwargs["worker_use_ray"] = True

            # 根据vllm版本选择不同的worker类
            if vllm.__version__ > "0.6.4.post1":
                # 如果vllm版本大于0.6.4.post1，使用指定的worker类路径
                kwargs[
                    "worker_cls"
                ] = "orz.exp_engine.accelerators.inference.vllm_worker_wrap.OffloadableVLLMWorker"
            else:
                # 否则，定义一个RayWorkerWrapper类来包装原有的RayWorkerWrapper
                RayWorkerWrapperPath = vllm.executor.ray_utils

                class RayWorkerWrapper(RayWorkerWrapperPath.RayWorkerWrapper):
                    def __init__(self, *args, **kwargs) -> None:
                        # 设置worker模块名和类名
                        kwargs[
                            "worker_module_name"
                        ] = "orz.exp_engine.accelerators.inference.vllm_worker_wrap"
                        kwargs["worker_class_name"] = "OffloadableVLLMWorker"
                        # 调用父类的初始化方法
                        super().__init__(*args, **kwargs)

                # 替换原有的RayWorkerWrapper类
                RayWorkerWrapperPath.RayWorkerWrapper = RayWorkerWrapper

        # 强制使用eager模式
        kwargs["enforce_eager"] = True
        # 初始化LLM对象
        self.llm = vllm.LLM(*args, **kwargs)
        # 获取LLM引擎的各种配置
        self.scheduler_config = self.llm.llm_engine.scheduler_config
        self.model_config = self.llm.llm_engine.model_config
        self.cache_config = self.llm.llm_engine.cache_config
        self.lora_config = self.llm.llm_engine.lora_config
        self.parallel_config = self.llm.llm_engine.parallel_config

    def generate(self, *args, **kwargs):
        # 调用LLM的generate方法生成文本
        return self.llm.generate(*args, **kwargs)

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        # 初始化进程组，区分GPU执行器和非GPU执行器的处理方式
        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.init_process_group(
                master_address, master_port, rank_offset, world_size, group_name, backend
            )
        else:
            return self.llm.llm_engine.model_executor._run_workers(
                "init_process_group", master_address, master_port, rank_offset, world_size, group_name, backend
            )

    def get_ip_and_port(self):
        # 获取当前节点的IP地址和端口，区分GPU执行器和非GPU执行器的处理方式
        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.get_ip_and_port()
        else:
            return self.llm.llm_engine.model_executor._run_workers("get_ip_and_port")

    def offload_to_cpu(self):
        # 将模型从GPU卸载到CPU，区分GPU执行器和非GPU执行器的处理方式
        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.offload_cpu()
        else:
            return self.llm.llm_engine.model_executor._run_workers("offload_cpu")

    def backload_to_gpu(self):
        # 将模型从CPU加载回GPU，区分GPU执行器和非GPU执行器的处理方式
        if self.use_gpu_executor:
            self.llm.llm_engine.model_executor.driver_worker.load_gpu()
        else:
            self.llm.llm_engine.model_executor._run_workers("load_gpu")
        # 重建调度器以适应新的GPU环境
        self.llm.llm_engine.scheduler = [
            Scheduler(
                self.scheduler_config,
                self.cache_config,
                self.lora_config,
                self.parallel_config.pipeline_parallel_size,
                self.async_callbacks[v_id] if self.model_config.use_async_output_proc else None,
            )
            for v_id in range(self.parallel_config.pipeline_parallel_size)
        ]

    def update_weight(self, name, dtype, shape, empty_cache=False):
        # 更新模型权重前停止远程工作者的执行循环
        self.stop_remote_worker_execution_loop()

        # 更新模型权重，区分GPU执行器和非GPU执行器的处理方式
        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.update_weight(name, dtype, shape, empty_cache)
        else:
            return self.llm.llm_engine.model_executor._run_workers("update_weight", name, dtype, shape, empty_cache)

    def update_weight_internal_with_cuda_ipc(self, name, dtype, shape, cudaipc_handler, empty_cache=False):
        # 使用CUDA IPC更新模型权重，区分GPU执行器和非GPU执行器的处理方式
        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.update_weight_internal_with_cuda_ipc(
                name, dtype, shape, cudaipc_handler, empty_cache
            )
        else:
            return self.llm.llm_engine.model_executor._run_workers(
                "update_weight_internal_with_cuda_ipc", name, dtype, shape, cudaipc_handler, empty_cache
            )

    def stop_remote_worker_execution_loop(self):
        # 停止远程工作者的执行循环，修复使用两个通信组时的错误
        if self.__version__ > "0.4.2":
            self.llm.llm_engine.model_executor.stop_remote_worker_execution_loop()

    def get_gpu_memory(self):
        """获取当前Actor使用的GPU内存"""
        import torch

        # 清空CUDA缓存以获取准确的内存使用情况
        torch.cuda.empty_cache()
        return torch.cuda.memory_allocated() / 1024**2  # 转换为MB

    def get_weight_statistics(self):
        """计算模型权重的轻量级统计信息"""
        stats = {}
        model_runner = self.llm.llm_engine.model_executor.driver_worker.model_runner
        for name, param in model_runner.model.named_parameters():
            # 计算关键统计信息，包括均值、标准差、范数、形状以及最大最小值
            tensor_stats = {
                "mean": param.mean().item(),
                "std": param.std().item(),
                "norm": param.norm().item(),
                "shape": tuple(param.shape),
                "max": param.max().item(),
                "min": param.min().item(),
            }
            stats[name] = tensor_stats
        return stats
