#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试脚本用于比较两种批处理方法：
1. 普通批处理 (_convert_prompts_outputs_to_batch_tensors)
2. 序列打包批处理 (_convert_prompts_outputs_to_batch_tensors_packing)
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

# 设置字体顺序：先尝试一个英文字体，再回退到中文字体
plt.rcParams['font.family'] = ['DejaVu Sans', 'Droid Sans Fallback']
# 确保负号正确显示
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@dataclass
class FakeConfig:
    """模拟配置类"""
    prompt_max_len: int = 256
    generate_max_len: int = 256
    packing_max_len: int = 1024


class SimpleTokenizer:
    """简单的tokenizer实现，不需要下载模型"""
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.vocab = {"<pad>": 0, "<eos>": 1}
        self.next_id = 2
        
    def tokenize(self, text):
        # 简单实现：按字符分割
        return list(text)
    
    def __call__(self, texts, return_tensors=None, add_special_tokens=False, 
                max_length=None, padding=False, truncation=False):
        if isinstance(texts, str):
            texts = [texts]
            
        all_input_ids = []
        for text in texts:
            tokens = self.tokenize(text)
            if max_length and truncation and len(tokens) > max_length:
                tokens = tokens[:max_length]
                
            # 将token转换为id
            token_ids = []
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = self.next_id
                    self.next_id += 1
                token_ids.append(self.vocab[token])
                
            all_input_ids.append(token_ids)
            
        # 处理padding
        if padding and return_tensors == "pt":
            max_len = max(len(ids) for ids in all_input_ids)
            padded_ids = []
            for ids in all_input_ids:
                padded_ids.append(ids + [self.pad_token_id] * (max_len - len(ids)))
            all_input_ids = padded_ids
            
        # 返回tensor
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(all_input_ids)}
        return {"input_ids": all_input_ids}


class BatchingTester:
    """
    用于测试两种批处理方法的类
    """
    def __init__(self):
        # 使用本地自定义Tokenizer以避免下载模型
        self.tokenizer = SimpleTokenizer()
        self.cfg = FakeConfig()
    
    def _tokenize(self, texts, max_length=99999999, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    def _process_sequences(self, sequences: torch.Tensor, input_len, eos_token_id, pad_token_id):
        """简化版的处理序列函数"""
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)

        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)

        # 对于中间可能存在eos的情况
        first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(device=sequences.device)
        attention_mask = (mask >= first_token_indices) & (mask <= eos_indices).to(dtype=torch.long)

        # RL中的状态动作
        state_seq = sequences[:, input_len - 1 : -1]
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        action_mask[:, 0] = 1

        return sequences, attention_mask, action_mask

    def _convert_prompts_outputs_to_batch_tensors(self, prompts: List[str], outputs: List[str]):
        """普通批处理方法"""
        max_input_len, max_output_len = 0, 0
        prompt_token_lens, response_token_lens = [], []
        inputs_token_ids, outputs_token_ids = [], []
        for prompt, output in zip(prompts, outputs):
            input_token_ids = self._tokenize(prompt, self.cfg.prompt_max_len, padding=False)["input_ids"]
            response_token_ids = self._tokenize(output, self.cfg.generate_max_len, padding=False)["input_ids"]

            inputs_token_ids.append(input_token_ids)
            outputs_token_ids.append(response_token_ids)

            prompt_token_len = len(input_token_ids)
            response_token_len = len(response_token_ids)
            prompt_token_lens.append(prompt_token_len)
            response_token_lens.append(response_token_len)

            max_input_len = max(max_input_len, prompt_token_len)
            max_output_len = max(max_output_len, response_token_len)

        pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
        sequences = []
        for i, _ in enumerate(prompts):
            # 左填充输入
            input_len = prompt_token_lens[i]
            input_ids = [pad_token_id] * (max_input_len - input_len) + list(inputs_token_ids[i])

            # 右填充输出
            output_len = response_token_lens[i]
            output_ids = list(outputs_token_ids[i]) + [pad_token_id] * (max_output_len - output_len)

            # 连接输入和输出
            sequences.append(input_ids + output_ids)

        sequences = torch.tensor(sequences)

        sequences, attention_mask, action_mask = self._process_sequences(
            sequences, max_input_len, eos_token_id, pad_token_id
        )
        
        # 计算填充率
        padding_ratio = (sequences == pad_token_id).float().mean().item()
        
        return sequences, attention_mask, action_mask, padding_ratio

    def _convert_prompts_outputs_to_batch_tensors_packing(
        self, prompts: List[str], outputs: List[str], custom_rewards: Optional[List[torch.Tensor]] = None
    ):
        """序列打包批处理方法"""
        packing_max_len = self.cfg.packing_max_len
        ret_sequences = []
        ret_attention_masks = []
        ret_num_actions = []
        ret_packed_seq_lens = []
        if custom_rewards is not None:
            ret_custom_rewards = []
        else:
            ret_custom_rewards = None

        assert (
            len(prompts) == len(outputs) and len(prompts) > 0
        ), "prompts and outputs must have the same length and length must be greater than 0"

        def _new_instance():
            out_sequence = torch.full((packing_max_len,), self.tokenizer.pad_token_id, dtype=torch.long)
            out_attention_mask = torch.zeros((packing_max_len,), dtype=torch.int)
            out_num_actions = []
            out_packed_seq_lens = []
            rewards = [] if custom_rewards else None
            seq_offset = 0
            seq_index = 0
            return (
                out_sequence,
                out_attention_mask,
                out_num_actions,
                out_packed_seq_lens,
                rewards,
                seq_offset,
                seq_index,
            )

        def _accumulate(
            out_sequence,
            out_attention_mask,
            out_num_actions,
            out_packed_seq_lens,
            rewards,
            seq_offset,
            seq_index,
            sequence,
            attention_mask,
            num_action,
            total_len,
            custom_rewards,
            i,
        ):
            out_sequence[seq_offset : seq_offset + total_len] = torch.tensor(sequence)
            out_attention_mask[seq_offset : seq_offset + total_len] = seq_index + 1
            out_num_actions.append(num_action)
            out_packed_seq_lens.append(total_len)
            if custom_rewards:
                rewards.append(custom_rewards[i])
            return seq_offset + total_len, seq_index + 1

        sequences = []
        attention_masks = []
        num_actions = []
        total_lens = []

        input_token_ids = self._tokenize(prompts, self.cfg.prompt_max_len, padding=False)["input_ids"]
        response_token_ids = self._tokenize(outputs, self.cfg.generate_max_len, padding=False)["input_ids"]

        for input_ids, response_ids in zip(input_token_ids, response_token_ids):
            sequences.append(input_ids + response_ids)
            attention_masks.append(torch.ones((len(input_ids) + len(response_ids),), dtype=torch.float32))
            num_actions.append(len(response_ids))
            total_lens.append(len(input_ids) + len(response_ids))

        # 创建打包序列
        (
            out_sequence,
            out_attention_mask,
            out_num_actions,
            out_packed_seq_lens,
            rewards,
            seq_offset,
            seq_index,
        ) = _new_instance()
        for i, (sequence, attention_mask, num_action, total_len) in enumerate(
            zip(sequences, attention_masks, num_actions, total_lens)
        ):
            if seq_offset + total_len < packing_max_len:
                seq_offset, seq_index = _accumulate(
                    out_sequence,
                    out_attention_mask,
                    out_num_actions,
                    out_packed_seq_lens,
                    rewards,
                    seq_offset,
                    seq_index,
                    sequence,
                    attention_mask,
                    num_action,
                    total_len,
                    custom_rewards,
                    i,
                )
            elif seq_offset + total_len == packing_max_len:
                seq_offset, seq_index = _accumulate(
                    out_sequence,
                    out_attention_mask,
                    out_num_actions,
                    out_packed_seq_lens,
                    rewards,
                    seq_offset,
                    seq_index,
                    sequence,
                    attention_mask,
                    num_action,
                    total_len,
                    custom_rewards,
                    i,
                )
                valid_size = out_attention_mask.nonzero().size(0)
                ret_sequences.append(out_sequence[:valid_size].unsqueeze(0))
                ret_attention_masks.append(out_attention_mask[:valid_size].unsqueeze(0))
                ret_num_actions.append(out_num_actions)
                ret_packed_seq_lens.append(out_packed_seq_lens)
                if custom_rewards:
                    ret_custom_rewards.append(rewards)
                (
                    out_sequence,
                    out_attention_mask,
                    out_num_actions,
                    out_packed_seq_lens,
                    rewards,
                    seq_offset,
                    seq_index,
                ) = _new_instance()
            elif seq_offset + total_len > packing_max_len:
                if seq_offset > 0:
                    valid_size = out_attention_mask.nonzero().size(0)
                    ret_sequences.append(out_sequence[:valid_size].unsqueeze(0))
                    ret_attention_masks.append(out_attention_mask[:valid_size].unsqueeze(0))
                    ret_num_actions.append(out_num_actions)
                    ret_packed_seq_lens.append(out_packed_seq_lens)
                    if custom_rewards:
                        ret_custom_rewards.append(rewards)
                    (
                        out_sequence,
                        out_attention_mask,
                        out_num_actions,
                        out_packed_seq_lens,
                        rewards,
                        seq_offset,
                        seq_index,
                    ) = _new_instance()
                    seq_offset, seq_index = _accumulate(
                        out_sequence,
                        out_attention_mask,
                        out_num_actions,
                        out_packed_seq_lens,
                        rewards,
                        seq_offset,
                        seq_index,
                        sequence,
                        attention_mask,
                        num_action,
                        total_len,
                        custom_rewards,
                        i,
                    )

        if seq_offset > 0:
            valid_size = out_attention_mask.nonzero().size(0)
            ret_sequences.append(out_sequence[:valid_size].unsqueeze(0))
            ret_attention_masks.append(out_attention_mask[:valid_size].unsqueeze(0))
            ret_num_actions.append(out_num_actions)
            ret_packed_seq_lens.append(out_packed_seq_lens)
            if custom_rewards:
                ret_custom_rewards.append(rewards)
        
        # 计算填充率
        total_tokens = 0
        total_pad_tokens = 0
        for seq in ret_sequences:
            total_tokens += seq.numel()
            total_pad_tokens += (seq == self.tokenizer.pad_token_id).sum().item()
        
        padding_ratio = total_pad_tokens / total_tokens if total_tokens > 0 else 0
        
        return ret_sequences, ret_attention_masks, ret_num_actions, ret_packed_seq_lens, ret_custom_rewards, padding_ratio

    def test_both_methods(self, prompts: List[str], outputs: List[str]):
        """
        测试两种方法并比较效率
        """
        print("\n=== 测试两种批处理方法 ===")
        print(f"样本数量: {len(prompts)}")
        
        # 测试普通批处理
        sequences, attention_mask, action_mask, padding_ratio_normal = self._convert_prompts_outputs_to_batch_tensors(prompts, outputs)
        
        total_tokens_normal = sequences.numel()
        valid_tokens_normal = (sequences != self.tokenizer.pad_token_id).sum().item()
        
        print(f"\n1. 普通批处理方法:")
        print(f"   - 批次形状: {sequences.shape}")
        print(f"   - 总token数: {total_tokens_normal}")
        print(f"   - 有效token数: {valid_tokens_normal}")
        print(f"   - 填充率: {padding_ratio_normal:.2%}")
        
        # 测试序列打包批处理
        (ret_sequences, ret_attention_masks, ret_num_actions, 
         ret_packed_seq_lens, _, padding_ratio_packing) = self._convert_prompts_outputs_to_batch_tensors_packing(prompts, outputs)
        
        total_tokens_packing = sum(seq.numel() for seq in ret_sequences)
        valid_tokens_packing = sum((seq != self.tokenizer.pad_token_id).sum().item() for seq in ret_sequences)
        
        print(f"\n2. 序列打包批处理方法:")
        print(f"   - 打包后批次数: {len(ret_sequences)}")
        for i, seq in enumerate(ret_sequences):
            print(f"   - 批次 {i+1} 形状: {seq.shape}")
        print(f"   - 总token数: {total_tokens_packing}")
        print(f"   - 有效token数: {valid_tokens_packing}")
        print(f"   - 填充率: {padding_ratio_packing:.2%}")
        
        # 计算节省
        token_reduction = 1 - (total_tokens_packing / total_tokens_normal)
        
        print(f"\n比较结果:")
        print(f"   - 填充率减少: {padding_ratio_normal - padding_ratio_packing:.2%}")
        print(f"   - 总token数减少: {token_reduction:.2%}")
        
        return {
            "normal": {
                "total_tokens": total_tokens_normal,
                "valid_tokens": valid_tokens_normal,
                "padding_ratio": padding_ratio_normal,
                "shapes": [sequences.shape],
            },
            "packing": {
                "total_tokens": total_tokens_packing,
                "valid_tokens": valid_tokens_packing,
                "padding_ratio": padding_ratio_packing,
                "shapes": [seq.shape for seq in ret_sequences],
            },
            "reduction": token_reduction,
        }
    
    def visualize_results(self, results):
        """
        可视化比较结果
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 第一张图：token数量比较
        labels = ['普通批处理', '序列打包']
        total_tokens = [results['normal']['total_tokens'], results['packing']['total_tokens']]
        valid_tokens = [results['normal']['valid_tokens'], results['packing']['valid_tokens']]
        padding_tokens = [total - valid for total, valid in zip(total_tokens, valid_tokens)]
        
        x = np.arange(len(labels))
        width = 0.35
        
        ax1.bar(x, valid_tokens, width, label='有效Token')
        ax1.bar(x, padding_tokens, width, bottom=valid_tokens, label='填充Token')
        
        ax1.set_ylabel('Token数量')
        ax1.set_title('Token使用比较')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.legend()
        
        # 为柱状图添加数值标签
        for i, v in enumerate(total_tokens):
            ax1.text(i, v + 0.05 * max(total_tokens), f'{v}', ha='center')
            ax1.text(i, valid_tokens[i] - 0.1 * max(valid_tokens), f'{valid_tokens[i]}', ha='center', color='white')
        
        # 第二张图：填充率比较
        padding_ratios = [results['normal']['padding_ratio'], results['packing']['padding_ratio']]
        
        ax2.bar(x, padding_ratios, width, color=['#ff9999', '#66b3ff'])
        ax2.set_ylabel('填充率')
        ax2.set_title('填充率比较')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.set_ylim(0, max(padding_ratios) * 1.2)
        
        # 为柱状图添加百分比标签
        for i, v in enumerate(padding_ratios):
            ax2.text(i, v + 0.01, f'{v:.2%}', ha='center')
        
        plt.tight_layout()
        # 确保文件夹存在
        os.makedirs('analysis', exist_ok=True)
        plt.savefig('analysis/batch_comparison.png')
        print(f"\n可视化结果保存到: analysis/batch_comparison.png")
        plt.close()


def generate_test_data(num_samples=10, prompt_len_range=(5, 50), response_len_range=(5, 100)):
    """
    生成测试数据
    """
    prompts = []
    outputs = []
    
    for i in range(num_samples):
        prompt_len = np.random.randint(*prompt_len_range)
        response_len = np.random.randint(*response_len_range)
        
        prompt = f"这是一个长度为{prompt_len}词的提示，用于测试批处理效率。" + " ".join([f"词{j}" for j in range(prompt_len)])
        output = f"这是一个长度为{response_len}词的回答。" + " ".join([f"词{j}" for j in range(response_len)])
        
        prompts.append(prompt)
        outputs.append(output)
    
    return prompts, outputs


def main():
    # 创建一些变长样本来测试
    prompts, outputs = generate_test_data(num_samples=15, prompt_len_range=(10, 100), response_len_range=(30, 200))
    
    # 初始化测试器
    tester = BatchingTester()
    
    # 测试两种方法
    results = tester.test_both_methods(prompts, outputs)
    
    # 可视化结果
    tester.visualize_results(results)
    
    print("\n=== 结论 ===")
    print(f"序列打包方法相比普通批处理方法节省了 {results['reduction']:.2%} 的计算资源。")
    print("通过减少填充token数量，打包方法能更有效地利用GPU资源，减少计算浪费。")


if __name__ == "__main__":
    main() 