#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化版演示脚本：比较普通批处理和序列打包批处理的效率差异
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List

# 设置字体顺序：先尝试一个英文字体，再回退到中文字体
plt.rcParams['font.family'] = ['DejaVu Sans', 'Droid Sans Fallback']
# 确保负号正确显示
plt.rcParams['axes.unicode_minus'] = False


class BatchingDemo:
    """演示批处理和序列打包的区别"""
    
    def __init__(self):
        # 设置基本参数
        self.pad_token_id = 0
        self.eos_token_id = 1
        
    def normal_batching(self, sequences: List[List[int]]):
        """普通批处理：找出最长序列长度，所有序列都填充到该长度"""
        max_length = max(len(seq) for seq in sequences)
        
        # 填充所有序列到最大长度
        padded_sequences = []
        for seq in sequences:
            padded_seq = seq + [self.pad_token_id] * (max_length - len(seq))
            padded_sequences.append(padded_seq)
        
        # 转换为tensor
        batch = torch.tensor(padded_sequences)
        
        # 计算填充率
        total_tokens = batch.numel()
        pad_tokens = (batch == self.pad_token_id).sum().item()
        padding_ratio = pad_tokens / total_tokens
        
        return batch, padding_ratio
    
    def sequence_packing(self, sequences: List[List[int]], max_packed_length: int = 1024):
        """序列打包：多个序列打包到一个长批次中，减少填充"""
        batches = []
        
        current_batch = []
        current_length = 0
        
        for seq in sequences:
            seq_len = len(seq)
            
            # 如果添加当前序列会超过最大长度，创建新批次
            if current_length + seq_len > max_packed_length and current_length > 0:
                # 填充当前批次到最大长度
                flat_batch = []
                for s in current_batch:
                    flat_batch.extend(s)
                
                # 添加填充
                padded_batch = flat_batch + [self.pad_token_id] * (max_packed_length - len(flat_batch))
                batches.append(torch.tensor([padded_batch]))
                
                # 重置
                current_batch = [seq]
                current_length = seq_len
            else:
                # 添加到当前批次
                current_batch.append(seq)
                current_length += seq_len
        
        # 处理最后一个批次
        if current_batch:
            flat_batch = []
            for s in current_batch:
                flat_batch.extend(s)
            
            # 添加填充
            padded_batch = flat_batch + [self.pad_token_id] * (max_packed_length - len(flat_batch))
            batches.append(torch.tensor([padded_batch[:max_packed_length]]))
        
        # 计算填充率
        total_tokens = sum(batch.numel() for batch in batches)
        pad_tokens = sum((batch == self.pad_token_id).sum().item() for batch in batches)
        padding_ratio = pad_tokens / total_tokens
        
        return batches, padding_ratio

    @staticmethod
    def generate_random_sequences(num_sequences: int = 10, 
                                 min_length: int = 50, 
                                 max_length: int = 500,
                                 vocab_size: int = 1000):
        """生成随机长度的序列进行测试"""
        sequences = []
        for _ in range(num_sequences):
            length = np.random.randint(min_length, max_length)
            # 生成随机token ID (2-vocab_size)，避开pad(0)和eos(1)
            seq = np.random.randint(2, vocab_size, length).tolist()
            sequences.append(seq)
        return sequences
    
    def compare_methods(self, num_sequences: int = 15, 
                       min_length: int = 50, 
                       max_length: int = 300):
        """比较两种方法的效率"""
        # 生成随机序列
        sequences = self.generate_random_sequences(
            num_sequences=num_sequences,
            min_length=min_length,
            max_length=max_length
        )
        
        # 打印序列长度分布
        lengths = [len(seq) for seq in sequences]
        print(f"序列数量: {len(sequences)}")
        print(f"序列长度: 最小={min(lengths)}, 最大={max(lengths)}, 平均={sum(lengths)/len(lengths):.1f}")
        
        # 普通批处理
        normal_batch, normal_padding_ratio = self.normal_batching(sequences)
        
        print("\n1. 普通批处理方法:")
        print(f"   - 批次数: 1")
        print(f"   - 批次形状: {normal_batch.shape}")
        print(f"   - 总token数: {normal_batch.numel()}")
        print(f"   - 有效token数: {sum(lengths)}")
        print(f"   - 填充token数: {normal_batch.numel() - sum(lengths)}")
        print(f"   - 填充率: {normal_padding_ratio:.2%}")
        
        # 序列打包批处理
        max_packed_length = max(lengths) * 2  # 设置一个合理的打包长度
        packed_batches, packed_padding_ratio = self.sequence_packing(sequences, max_packed_length)
        
        total_packed_tokens = sum(batch.numel() for batch in packed_batches)
        
        print("\n2. 序列打包批处理方法:")
        print(f"   - 批次数: {len(packed_batches)}")
        for i, batch in enumerate(packed_batches):
            print(f"   - 批次 {i+1} 形状: {batch.shape}")
        print(f"   - 总token数: {total_packed_tokens}")
        print(f"   - 有效token数: {sum(lengths)}")
        print(f"   - 填充token数: {total_packed_tokens - sum(lengths)}")
        print(f"   - 填充率: {packed_padding_ratio:.2%}")
        
        # 计算节省比例
        token_reduction = 1 - (total_packed_tokens / normal_batch.numel())
        
        print("\n比较结果:")
        print(f"   - 填充率减少: {normal_padding_ratio - packed_padding_ratio:.2%}")
        print(f"   - 总token数减少: {token_reduction:.2%}")
        
        return {
            "normal": {
                "batch": normal_batch,
                "total_tokens": normal_batch.numel(),
                "valid_tokens": sum(lengths),
                "padding_ratio": normal_padding_ratio,
            },
            "packing": {
                "batches": packed_batches,
                "total_tokens": total_packed_tokens,
                "valid_tokens": sum(lengths),
                "padding_ratio": packed_padding_ratio,
            },
            "reduction": token_reduction,
            "sequences": sequences
        }
    
    def visualize_results(self, results):
        """可视化结果比较"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. 序列长度分布
        lengths = [len(seq) for seq in results['sequences']]
        axes[0].hist(lengths, bins=10, color='skyblue', edgecolor='black')
        axes[0].set_title('序列长度分布')
        axes[0].set_xlabel('序列长度')
        axes[0].set_ylabel('频率')
        
        # 2. Token数量对比
        labels = ['普通批处理', '序列打包']
        total_tokens = [results['normal']['total_tokens'], results['packing']['total_tokens']]
        valid_tokens = [results['normal']['valid_tokens'], results['packing']['valid_tokens']]
        padding_tokens = [total - valid for total, valid in zip(total_tokens, valid_tokens)]
        
        x = np.arange(len(labels))
        width = 0.35
        
        axes[1].bar(x, valid_tokens, width, label='有效Token')
        axes[1].bar(x, padding_tokens, width, bottom=valid_tokens, label='填充Token')
        
        axes[1].set_ylabel('Token数量')
        axes[1].set_title('Token使用比较')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels)
        axes[1].legend()
        
        # 为柱状图添加数值标签
        for i, v in enumerate(total_tokens):
            axes[1].text(i, v + 0.05 * max(total_tokens), f'{v}', ha='center')
            axes[1].text(i, valid_tokens[i] - 0.1 * max(valid_tokens), f'{valid_tokens[i]}', ha='center', color='white')
        
        # 3. 填充率比较
        padding_ratios = [results['normal']['padding_ratio'], results['packing']['padding_ratio']]
        
        axes[2].bar(x, padding_ratios, width, color=['#ff9999', '#66b3ff'])
        axes[2].set_ylabel('填充率')
        axes[2].set_title('填充率比较')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(labels)
        axes[2].set_ylim(0, max(padding_ratios) * 1.2)
        
        # 为柱状图添加百分比标签
        for i, v in enumerate(padding_ratios):
            axes[2].text(i, v + 0.01, f'{v:.2%}', ha='center')
        
        plt.tight_layout()
        
        # 确保目录存在
        os.makedirs('analysis', exist_ok=True)
        plt.savefig('analysis/batch_comparison.png')
        print(f"\n可视化结果保存到: analysis/batch_comparison.png")
        plt.close()


def main():
    # 创建演示对象
    demo = BatchingDemo()
    
    # 运行比较
    results = demo.compare_methods(
        num_sequences=15,
        min_length=50,
        max_length=300
    )
    
    # 可视化结果
    demo.visualize_results(results)
    
    print("\n=== 结论 ===")
    print(f"序列打包方法相比普通批处理方法节省了 {results['reduction']:.2%} 的计算资源。")
    print("通过减少填充token数量，打包方法能更有效地利用GPU资源，减少计算浪费。")
    print("在处理变长序列时，序列打包的优势更为明显。")


if __name__ == "__main__":
    main() 