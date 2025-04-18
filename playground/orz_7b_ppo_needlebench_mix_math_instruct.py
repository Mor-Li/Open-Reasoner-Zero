"""
Qwen2.5-7B base model + ppo

debug running command in single node:
DEBUG_MODE=True python -m playground.orz_7b_ppo

Multi-node Training:

on master node, first run `ray start --head`
then on other nodes, run `ray start --address='<master-node-ip>:<master-node-port>'`
then on master node, run `python -m playground.orz_7b_ppo`

"""

import asyncio
import copy
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from itertools import islice, zip_longest
from typing import Any, Awaitable, Callable, List, Optional, Tuple

import numpy as np
import ray
import torch
from loguru import logger
from omegaconf.listconfig import ListConfig
from typing_extensions import override

from orz.exps.examples.ppo.ppo_base_exp import BasePPOExp, BasePPOExpConfig
from orz.ppo import RayPPOTrainer
from orz.ppo.tools.math_utils import is_equal, solution2answer
from orz.ppo.utils import check_reflection_pattern
from playground.zero_setting_base import CustomDataset, EvalCustomDataset

# 全局调试标志，通过环境变量控制是否开启调试模式
DEBUG_MODE = False if os.environ.get("DEBUG_MODE", "False") == "False" else True  # Global debug flag

# 获取当前文件名，用于命名保存路径
file_name = f"{'debug_' if DEBUG_MODE else ''}{os.path.splitext(os.path.basename(__file__))[0]}"
# 创建线程池执行器，用于并行处理任务
executor = ThreadPoolExecutor(max_workers=64)


def repeatness(s: str):
    """
    计算字符串的重复度
    
    Args:
        s: 输入字符串
    
    Returns:
        重复度得分，范围在0到1之间
    """
    def ranks(l):
        """将列表元素映射为它们在排序后的索引"""
        index = {v: i for i, v in enumerate(sorted(set(l)))}
        return [index[v] for v in l]

    def suffixArray(s):
        """构建后缀数组和逆后缀数组"""
        line = ranks(s)
        n, k, ans, sa = len(s), 1, line, [0] * len(s)
        while k < n - 1:
            line = ranks(list(zip_longest(line, islice(line, k, None), fillvalue=-1)))
            ans, k = line, k << 1
        for i, k in enumerate(ans):
            sa[k] = i
        return ans, sa

    def lcp(arr, suffixArr, inv_suff):
        """计算最长公共前缀数组"""
        n, ans, k = len(arr), [0] * len(arr), 0

        for i in range(n):
            if inv_suff[i] == n - 1:
                k = 0
                continue

            j = suffixArr[inv_suff[i] + 1]
            while i + k < n and j + k < n and arr[i + k] == arr[j + k]:
                k += 1

            ans[inv_suff[i]] = k
            if k > 0:
                k -= 1

        return ans

    arr = [ord(i) for i in s]  # 将字符串转换为ASCII码数组
    n = len(arr)
    if n <= 1:
        return 0
    c, sa = suffixArray(arr)  # 构建后缀数组
    cnt = sum(lcp(arr, sa, c))  # 计算所有最长公共前缀的和

    # 返回重复度得分，使用公式 2*LCP总和/(n*(n+1))
    return cnt * 2 / (n * (n + 1))


@dataclass
class PPOExpConfig(BasePPOExpConfig):
    """PPO实验配置类，继承自BasePPOExpConfig"""
    use_compute_reward_fn: bool = True  # 是否使用自定义的奖励计算函数
    use_orm_score: bool = False  # 是否使用ORM评分

    # 条件设置，根据是否为调试模式设置不同的值
    total_num_nodes: int = 32 if not DEBUG_MODE else 8  # 总节点数

    # 资源相关设置
    ref_num_nodes: int = total_num_nodes  # 参考模型使用的节点数
    ref_num_gpus_per_node: int = 1  # 每个参考模型节点使用的GPU数
    actor_num_nodes: int = total_num_nodes  # Actor模型使用的节点数
    actor_num_gpus_per_node: int = 1  # 每个Actor节点使用的GPU数
    critic_num_nodes: int = total_num_nodes  # Critic模型使用的节点数
    critic_num_gpus_per_node: int = 1  # 每个Critic节点使用的GPU数
    colocate_all: bool = True  # 是否将所有组件放在同一节点上
    colocate_critic_reward: bool = True  # 是否将Critic和奖励计算放在同一节点上
    colocate_actor_ref: bool = True  # 是否将Actor和参考模型放在同一节点上
    vllm_num_engines: int = total_num_nodes  # vLLM引擎数量
    vllm_tensor_parallel_size: int = 1  # vLLM张量并行大小
    adam_offload: bool = False  # 是否将Adam优化器状态卸载到CPU
    zero_stage: int = 3  # ZeRO优化阶段

    # 路径相关设置
    pretrain: Optional[str] = "Qwen/Qwen2.5-7B-Instruct" # 预训练模型路径，可以替换为本地下载的模型路径
    reward_pretrain: Optional[str] = None  # 奖励模型预训练路径
    save_interval: int = 50  # 保存模型的间隔步数
    ckpt_path: str = f"orz_ckpt/{file_name}"  # 检查点保存路径
    save_path: str = f"orz_ckpt/{file_name}"  # 模型保存路径
    tensorboard_log_dir: str = f"orz_logs/{file_name}"  # TensorBoard日志目录

    # 数据集设置
    # MathTrain数据集和Math500评估数据集
    prompt_data: ListConfig = ListConfig(
        [
            "data/needlebench_atc_training.json",  # 训练数据集路径
            "data/orz_math_57k_collected.json"
        ]
    )
    eval_prompt_data: ListConfig = ListConfig(
        [
            "data/eval_data/math500.json",  # 评估数据集路径
            "data/eval_data/aime2024.json",  # AIME 2024评估数据集
            "data/eval_data/gpqa_diamond.json",  # GPQA Diamond评估数据集
            "data/eval_data/needlebench_atc.json",  # GPQA Diamond评估数据集
        ]
    )
    prompt_data_probs: ListConfig = ListConfig([1.0])  # 各数据集的采样概率
    # PPO相关设置
    actor_learning_rate: float = 1e-6  # Actor网络的学习率
    critic_learning_rate: float = 5e-6  # Critic网络的学习率
    num_warmup_steps: int = 50  # 预热步数，在此期间学习率会逐渐增加
    prompt_max_len: int = 2048  # 提示文本的最大长度
    enable_prefix_caching: bool = True  # 是否启用前缀缓存，可以加速训练
    update_ref_every_epoch: bool = True  # 是否在每个epoch后更新参考模型
    advantage_normalize: bool = True  # 是否对优势函数进行归一化处理

    num_episodes: int = 20  # 训练的总episode数
    rollout_batch_size: int = 128 if not DEBUG_MODE else 16  # 每次rollout的批量大小，调试模式下使用较小值
    n_samples_per_prompt: int = 64 if not DEBUG_MODE else 2  # 每个提示生成的样本数，调试模式下使用较小值
    micro_rollout_batch_size: int = 128 if not DEBUG_MODE else 128  # 微批量rollout的大小

    policy_update_steps: int = 1  # 每次迭代中策略网络的更新步数
    critic_update_steps: int = 12 if not DEBUG_MODE else 1  # 每次迭代中价值网络的更新步数，调试模式下使用较小值
    micro_train_batch_size: int = 1  # 训练时的微批量大小
    micro_forward_batch_size: int = 1  # 前向传播的微批量大小
    freezing_actor_steps: int = -1  # Actor冻结的步数，-1表示不冻结
    init_kl_coef: float = 0  # KL散度系数的初始值
    # KL损失相关设置
    kl_loss_coef: float = 0.0  # KL损失的系数
    use_kl_loss: bool = True  # 是否使用KL损失
    use_kl_estimator_k3: bool = True  # 是否使用K3估计器计算KL散度

    enable_eval: bool = True  # 是否启用评估
    eval_interval: int = 10  # 评估的间隔步数

    # 生成相关设置
    packing_max_len: int = 16384  # 打包的最大长度
    generate_max_len: int = 8000  # 生成文本的最大长度，后续可能会调整为更大的值
    max_len: int = 8192  # 最大序列长度，后续可能会调整为更大的值
    temperature: float = 1.0  # 采样温度，控制生成文本的随机性
    top_p: float = 1.0  # 核采样的概率阈值
    top_k: int = -1  # Top-k采样的k值，-1表示不使用top-k采样
    stop: ListConfig = ListConfig(["User:", "Human:", "Assistant:", "</answer>"])  # 生成停止的标记列表

    # GRPO（Generalized Reward-weighted Policy Optimization）相关设置
    use_grpo: bool = False  # 是否使用GRPO算法

    # GPU内存利用率设置，根据是否使用GRPO和调试模式调整
    gpu_memory_utilization: float = 0.75 if use_grpo else 0.7 if not DEBUG_MODE else 0.5
    critic_pretrain: Optional[str] = "" if use_grpo else pretrain  # Critic预训练模型路径

    # 强化学习参数
    gamma: float = 1.0  # 折扣因子，用于计算累积奖励
    lambd: float = 1.0  # GAE-Lambda参数，用于计算广义优势估计


class CustomRewardTrainer(RayPPOTrainer):
    @override
    async def custom_reward_fn(
        self,
        prompts: List[str],  # 输入的提示文本列表
        outputs: List[Any],  # 模型生成的输出列表
        extras: List[dict],  # 额外信息的字典列表
        reward_model_fn: Callable[[List[str], List[str]], Awaitable[torch.Tensor]],  # 奖励模型函数
    ) -> Tuple[List[str], List[str], List[torch.Tensor]]:
        # 创建日志指标
        scores = []  # 存储每个输出的得分
        responses = []  # 存储每个输出的响应文本
        avg_non_stop_count = 0  # 未正常停止的响应计数
        pass_at_n_dict = defaultdict(list)  # 用于计算pass@n指标的字典
        num_tokens: List[int] = []  # 存储每个响应的token数量

        @ray.remote(num_cpus=1)
        def get_repeat_score(res):
            # 远程函数：计算文本的重复度得分
            return repeatness(res)

        @ray.remote(num_cpus=1)
        def get_reflection_pattern_score(res):
            # 远程函数：检查文本中的反思模式并计算得分
            reflection_pattern_dict = check_reflection_pattern(res)
            reflection_pattern_num = sum(reflection_pattern_dict.values())
            return reflection_pattern_num

        rep_tasks = []  # 存储重复度和反思模式计算的远程任务
        for output in outputs:
            response = output["response"]
            # 为每个输出创建计算重复度和反思模式的远程任务
            rep_tasks.extend([get_repeat_score.remote(response), get_reflection_pattern_score.remote(response)])
        rep_task_results = ray.get(rep_tasks)  # 获取所有远程任务的结果

        repeat_scores = []  # 存储重复度得分
        reflection_pattern_scores = []  # 存储反思模式得分
        for idx in range(len(outputs)):
            repeat_scores.append(rep_task_results[idx * 2])  # 偶数索引存储重复度得分
            reflection_pattern_scores.append(rep_task_results[idx * 2 + 1])  # 奇数索引存储反思模式得分

        for output in outputs:
            responses.append(output["response"])  # 收集所有响应文本
        output_tokens = self._tokenize(responses, self.cfg.generate_max_len, padding=False)["input_ids"]  # 对响应进行分词
        # 将生成的样本信息添加到TensorBoard中，便于可视化和调试
        self.writer.add_text(
            "generated_raws",
            f"prompts: {prompts[0]}\n\noutputs: {outputs[0]['response']}\n\nfinal_answer: {outputs[0]['final_answer']}\n\nis_correct: {outputs[0]['iscorrect']}\n\nstop_reason: {outputs[0]['stop_reason']}\n\nresponse_token: {len(output_tokens[0])}",
            self.global_step,
        )
        
        # 遍历所有输出，计算每个样本的奖励
        for idx in range(len(outputs)):
            prompt, output, out_token = prompts[idx], outputs[idx], output_tokens[idx]
            rep_score, reflection_pattern_score = repeat_scores[idx], reflection_pattern_scores[idx]
            iscorrect = output["iscorrect"]  # 答案是否正确
            stop_reason = output["stop_reason"]  # 生成停止的原因
            response_token = len(out_token)  # 响应的token数量
            
            # 将重复度和反思模式得分添加到输出信息中
            output["repeat_score"] = rep_score
            output["reflection_pattern_score"] = reflection_pattern_score
            
            # 只有正确停止的响应才能获得奖励
            # 如果模型正常停止且答案正确，得分为1.0；如果答案错误，得分为0.0
            if stop_reason == "stop":
                score = 1.0 if iscorrect else 0.0
            else:
                # 如果模型没有正常停止，计数加1，得分为0.0
                avg_non_stop_count += 1
                score = 0.0
            scores.append(score)

            # 计算pass@n指标（用于评估模型在n次尝试中至少一次正确的概率）
            pass_at_n_dict[prompt].append(scores[-1])
            # 记录token数量
            num_tokens.append(response_token)

        # 必须在GRPO之前计算，因为GRPO会改变scores
        # 将token数量和得分转换为numpy数组，便于统计分析
        num_tokens_arr = np.array(num_tokens, dtype=np.float32)  # 必须是float类型才能计算均值和标准差
        scores_arr = np.array(scores)
        # 提取正确和错误响应的token数量
        correct_tokens_arr = np.array([]) if np.all(scores_arr == 0) else np.array(num_tokens_arr[scores_arr == 1])
        incorrect_tokens_arr = np.array([]) if np.all(scores_arr == 1) else np.array(num_tokens_arr[scores_arr == 0])

        # GRPO（Generalized Reward-Penalty Optimization）奖励归一化
        if self.cfg.use_grpo:
            # 记录原始奖励的均值
            self.writer.add_scalar("grpo_raw_reward", np.mean(scores), self.global_step)
            # 对每个提示的奖励进行归一化处理
            for i, prompt in enumerate(prompts):
                # 减去同一提示下所有响应的平均得分
                scores[i] -= np.mean(pass_at_n_dict[prompt])
                # 如果标准差大于0，则除以标准差进行归一化
                if std := np.std(pass_at_n_dict[prompt]) > 0:
                    scores[i] /= std

        # 定义函数：将结果保存到文件中
        def dump_results(prompts, outputs, scores):
            saved = []
            for prompt, output, score in zip(prompts, outputs, scores):
                saved.append(dict(prompt=prompt, score=score, outputs=output))
            json.dump(
                saved,
                open(os.path.join(self.cfg.save_path, f"iter{self.global_step}_generation_results.json"), "w"),
                ensure_ascii=False,
                indent=2,
            )

        # 异步执行结果保存，不阻塞主线程
        global executor
        asyncio.get_event_loop().run_in_executor(
            executor, dump_results, copy.deepcopy(prompts), copy.deepcopy(outputs), copy.deepcopy(scores)
        )

        # 计算并记录各种评估指标
        log_dict = {
            "avg_non_stop_count": avg_non_stop_count / len(prompts),  # 平均未正常停止的比例
            "avg_repeat_score": sum(repeat_scores) / len(prompts),  # 平均重复度得分
            "avg_reflection_pattern_score": sum(reflection_pattern_scores) / len(prompts),  # 平均反思模式得分
            "avg_pass_at_n": sum(1 for v in pass_at_n_dict.values() if np.sum(v) > 0) / len(pass_at_n_dict),  # 平均pass@n
            "avg_num_tokens": np.mean(num_tokens_arr).item(),  # 平均token数量
            "std_num_tokens": np.std(num_tokens_arr).item(),  # token数量的标准差
            "avg_correct_num_tokens": 0 if len(correct_tokens_arr) == 0 else np.mean(correct_tokens_arr).item(),  # 正确响应的平均token数量
            "std_correct_num_tokens": 0 if len(correct_tokens_arr) == 0 else np.std(correct_tokens_arr).item(),  # 正确响应token数量的标准差
            "avg_incorrect_num_tokens": 0 if len(incorrect_tokens_arr) == 0 else np.mean(incorrect_tokens_arr).item(),  # 错误响应的平均token数量
            "std_incorrect_num_tokens": 0 if len(incorrect_tokens_arr) == 0 else np.std(incorrect_tokens_arr).item(),  # 错误响应token数量的标准差
        }
        # 将指标添加到TensorBoard
        for k, v in log_dict.items():
            self.writer.add_scalar(k, v, self.global_step)
        # 将指标格式化为日志字符串并输出
        logging_str = ",".join([f"{k}: {v:.4f}" for k, v in log_dict.items()])
        logger.info(logging_str)

        # 为正确和错误响应的长度创建直方图，便于可视化分析
        if len(correct_tokens_arr) > 0:
            self.writer.add_histogram("correct_response_length", correct_tokens_arr, self.global_step)
        if len(incorrect_tokens_arr) > 0:
            self.writer.add_histogram("incorrect_response_length", incorrect_tokens_arr, self.global_step)

        # 为每个输出创建一个pre-token得分张量，例如：[0, 0, 0, 0, r]
        # 只有最后一个token获得奖励，这是PPO算法的常见做法
        score_tensors = []
        for score, output_token in zip(scores, output_tokens):
            score_tensor = torch.zeros(len(output_token))
            if len(output_token) > 0:
                score_tensor[-1] = score
            score_tensors.append(score_tensor)

        # 移除空响应，确保返回的数据有效
        res_prompts = []
        res_responses = []
        res_score_tensors = []
        for prompt, response, score_tensor in zip(prompts, responses, score_tensors):
            if len(response) > 0:
                res_prompts.append(prompt)
                res_responses.append(response)
                res_score_tensors.append(score_tensor)

        return res_prompts, res_responses, res_score_tensors

    @override
    @torch.no_grad()
    async def generate_vllm(
        self,
        gen_func: Callable[[List[str]], Awaitable[List[str | Any]]],
        prompts: List[str],
        extras: List[dict],
        **kwargs,
    ) -> List[str | Any]:
        from vllm import SamplingParams

        # 从配置中读取采样参数，设置生成文本的控制参数
        sampling_params = SamplingParams(
            temperature=self.cfg.temperature,  # 温度参数，控制生成的随机性
            top_p=self.cfg.top_p,  # 累积概率阈值，只考虑概率最高的一部分token
            top_k=self.cfg.top_k,  # 只考虑概率最高的k个token
            max_tokens=self.cfg.generate_max_len,  # 生成的最大token数
            skip_special_tokens=False,  # 是否跳过特殊token
            include_stop_str_in_output=True,  # 是否在输出中包含停止字符串
            stop=self.cfg.stop,  # 停止生成的标记
        )
        # 调用生成函数获取响应和停止原因
        responses, stop_reasons = await gen_func(
            prompts=prompts, sampling_params=sampling_params, use_tqdm=False, truncate_prompt=True
        )

        @ray.remote(num_cpus=1)
        def extract_final_answers_batch(responses: List[str]) -> List[str]:
            # 使用正则表达式提取答案中的boxed部分
            # pattern = re.compile(r"(\\boxed{.*})")
            pattern = re.compile(r"<answer>.*?(\\boxed{.*}).*?</answer>", re.DOTALL)
            results = []
            for response in responses:
                # 查找所有匹配项，如果有多个匹配，取最后一个；如果没有匹配，返回空字符串
                matches = re.findall(pattern, response)
                results.append(matches[-1] if matches else "")
            return results

        BATCH_SIZE = 16  # 设置批处理大小，提高并行效率
        num_batches = (len(responses) + BATCH_SIZE - 1) // BATCH_SIZE  # 计算需要的批次数量，向上取整

        # 使用Ray并行处理提取最终答案
        extract_tasks = []
        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, len(responses))
            batch = responses[start_idx:end_idx]  # 获取当前批次的响应
            extract_tasks.append(extract_final_answers_batch.remote(batch))  # 提交远程任务
        # 异步收集所有批次的结果
        batched_results = await asyncio.gather(*[asyncio.to_thread(ray.get, task) for task in extract_tasks])
        final_answers = [answer for batch in batched_results for answer in batch]  # 展平结果列表

        # 判断模型生成的答案是否正确
        global executor  # 使用全局线程池执行器
        equal_tasks = []
        for extra, final_answer in zip(extras, final_answers):
            # 提交判断答案是否相等的任务
            equal_tasks.append(is_equal(solution2answer(extra["answer"]), solution2answer(final_answer), executor))
        equal_results = await asyncio.gather(*equal_tasks)  # 异步等待所有判断结果

        # 整合所有结果
        results = []
        for extra, response, final_answer, stop_reason, iscorrect in zip(
            extras, responses, final_answers, stop_reasons, equal_results
        ):
            results.append(
                dict(
                    response=response,  # 模型的完整响应
                    iscorrect=iscorrect,  # 答案是否正确
                    stop_reason=stop_reason,  # 生成停止的原因
                    final_answer=final_answer,  # 提取的最终答案
                )
            )

        return results

    @override
    async def eval(self):
        """评估模型在验证集上的性能"""
        logger.info("Start evaluating on val set")
        from vllm import SamplingParams

        # 设置生成参数
        sampling_params = SamplingParams(
            temperature=self.cfg.temperature,  # 温度参数，控制随机性
            top_p=self.cfg.top_p,  # 累积概率阈值
            max_tokens=self.cfg.generate_max_len,  # 最大生成长度
            stop=self.cfg.stop,  # 停止标记
            skip_special_tokens=False,  # 不跳过特殊标记
            include_stop_str_in_output=True,  # 在输出中包含停止字符串
        )

        from torch.utils.data import DataLoader

        # 准备数据集和数据加载器
        dataset = self.eval_dataset
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, drop_last=False)
        # 计算每个vLLM引擎处理的提示数量
        prompt_pre_llm = (len(dataset) + self.cfg.vllm_num_engines - 1) // self.cfg.vllm_num_engines

        output_for_save = []  # 存储所有输出结果用于保存
        log_dict = defaultdict(float)  # 用于记录评估指标
        for batch in dataloader:
            prompts = list(batch[0])  # 提取提示
            answers = list(batch[1]["answer"])  # 提取标准答案
            file_names = list(batch[1]["file_name"])  # 提取文件名
            outputs = []
            # 将提示分配给不同的vLLM引擎并行处理
            for i, llm in enumerate(self.vllm_engines):
                outputs.append(
                    llm.generate.remote(
                        prompts=prompts[i * prompt_pre_llm : (i + 1) * prompt_pre_llm], sampling_params=sampling_params
                    )
                )
            outputs = await asyncio.gather(*outputs)  # 等待所有生成完成
            outputs = sum(outputs, [])  # 合并所有输出

            # 提取最终答案
            final_answers = []
            pattern = re.compile(r"<answer>.*?(\\boxed{.*}).*?</answer>", re.DOTALL)
            for output in outputs:
                matches = re.findall(pattern, output.outputs[0].text)
                if len(matches) > 0:
                    final_answers.append(matches[-1])  # 取最后一个匹配
                else:
                    final_answers.append("")  # 没有匹配则返回空字符串

            # 处理每个样本的结果
            for prompt, output, final_answer, answer, file_name in zip(
                prompts, outputs, final_answers, answers, file_names
            ):
                label = solution2answer(answer)  # 处理标准答案
                prefix_response = solution2answer(final_answer)  # 处理模型生成的答案
                iscorrect = await is_equal(label, prefix_response, executor)  # 判断答案是否正确
                # 保存详细结果
                output_for_save.append(
                    dict(
                        prompt=prompt,  # 提示
                        output=output.outputs[0].text,  # 完整输出
                        final_answer=final_answer,  # 提取的答案
                        answer=answer,  # 标准答案
                        iscorrect=iscorrect,  # 是否正确
                    )
                )
                # 累计统计数据
                log_dict[f"{file_name}/total_response_len_in_char"] += len(output.outputs[0].text)
                log_dict[f"{file_name}/correct"] += iscorrect
                log_dict[f"{file_name}/total"] += 1

        # 获取所有评估数据集的文件名
        all_file_names: List[str] = [
            os.path.splitext(os.path.basename(file_path))[0] for file_path in self.cfg.eval_prompt_data
        ]
        # 计算每个数据集的平均响应长度和准确率
        for file_name in all_file_names:
            log_dict[f"{file_name}/response_len_in_char"] = (
                log_dict[f"{file_name}/total_response_len_in_char"] / log_dict[f"{file_name}/total"]
            )
            log_dict[f"{file_name}/accuracy"] = log_dict[f"{file_name}/correct"] / log_dict[f"{file_name}/total"]
            # 移除中间统计数据
            log_dict.pop(f"{file_name}/total_response_len_in_char")
            log_dict.pop(f"{file_name}/correct")
            log_dict.pop(f"{file_name}/total")
        # 计算所有数据集的平均准确率
        log_dict["eval_accuracy"] = sum([log_dict[f"{file_name}/accuracy"] for file_name in all_file_names]) / len(
            all_file_names
        )

        # 构建输出文件名，包含迭代次数和各数据集的准确率
        dump_file_name = f"eval_output_iter{self.global_step}"
        for file_name in all_file_names:
            dump_file_name += f"_{file_name}{log_dict[f'{file_name}/accuracy']:.4f}"
        dump_file_name += ".jsonl"
        # 将结果保存为JSONL格式
        with open(
            os.path.join(
                self.cfg.save_path,
                dump_file_name,
            ),
            "w",
        ) as f:
            for item in output_for_save:
                f.write(
                    json.dumps(item, ensure_ascii=False) + "\n",
                )

        # 记录和输出评估结果
        logging_str = ",".join([f"{k}: {v:.4f}" for k, v in log_dict.items()])
        logger.info(logging_str)
        for k, v in log_dict.items():
            self.writer.add_scalar(f"evals/{k}", v, self.global_step)


class PPOExp(BasePPOExp):
    @cached_property
    def trainer(self):
        """创建并返回训练器实例"""
        vllm_engines = self.create_inference_engine()  # 创建推理引擎
        return CustomRewardTrainer(
            cfg=self.cfg,  # 配置
            strategy=self.strategy,  # 训练策略
            tokenizer=self.tokenizer,  # 分词器
            train_dataset=self.train_dataset,  # 训练数据集
            eval_dataset=self.eval_dataset,  # 评估数据集
            vllm_engines=vllm_engines,  # vLLM引擎
            colocate_pg=self.get_colocate_pg,  # 是否共置策略梯度
        )

    @override
    @cached_property
    def train_dataset(self):
        """加载和处理训练数据集"""
        dialogues = []
        # 从多个文件加载对话数据
        for file_path in self.cfg.prompt_data:
            with open(file_path, "r") as f:
                dialogues.extend(json.load(f))
        logger.info(f"Start processing {len(dialogues)} dialogues")
        # 创建自定义数据集
        prompts_dataset = CustomDataset(
            dialogues,
            self.tokenizer,
            self.cfg.prompt_max_len,
            self.strategy,
            pretrain_mode=False,
            num_processors=1,
        )
        logger.info(f"Finished processing {len(prompts_dataset)} dialogues")
        return prompts_dataset

    @override
    @cached_property
    def eval_dataset(self):
        """加载和处理评估数据集"""
        dialogues = []
        # 从多个文件加载评估数据
        for file_path in self.cfg.eval_prompt_data:
            with open(file_path, "r") as f:
                loaded_data = json.load(f)
                for loaded_data_item in loaded_data:
                    # 只保留文件名，不包括后缀
                    loaded_data_item["file_name"] = os.path.splitext(os.path.basename(file_path))[0]
                dialogues.extend(loaded_data)
        logger.info(f"Start processing {len(dialogues)} dialogues")
        # 创建评估用的自定义数据集
        prompts_dataset = EvalCustomDataset(
            dialogues,
            self.tokenizer,
            self.cfg.prompt_max_len,
            self.strategy,
            pretrain_mode=False,
            num_processors=1,
        )
        logger.info(f"Finished processing {len(prompts_dataset)} dialogues")
        return prompts_dataset


if __name__ == "__main__":
    # 创建实验实例并设置配置
    exp = PPOExp().set_cfg(PPOExpConfig())
    logger.info(exp.get_cfg_as_str(exp.cfg))
    # 确保所有必要的目录存在
    if not os.path.exists(exp.cfg.save_path):
        os.makedirs(exp.cfg.save_path, exist_ok=True)
    if not os.path.exists(exp.cfg.tensorboard_log_dir):
        os.makedirs(exp.cfg.tensorboard_log_dir, exist_ok=True)
    if not os.path.exists(exp.cfg.ckpt_path):
        os.makedirs(exp.cfg.ckpt_path, exist_ok=True)
    # 运行实验
    asyncio.run(exp.run())
