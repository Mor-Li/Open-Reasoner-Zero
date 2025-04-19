#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
序列打包解码工具：将打包的Experience解码回原始序列
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Union
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent))

# 导入Experience类和可视化工具
from experience_visualizer import Experience, ExperienceVisualizer

class SequenceDecoder:
    """序列打包解码器"""
    
    def __init__(self):
        """初始化解码器"""
        self.visualizer = ExperienceVisualizer()
    
    def load_experience(self, filepath):
        """加载Experience数据"""
        return self.visualizer.load_experiences_from_jsonl(filepath)
    
    def split_input_output(self, experience):
        """拆分序列为输入部分和输出部分"""
        # 提取序列数据
        if not hasattr(experience, 'sequences') or experience.sequences is None:
            return None, None
        
        sequences = experience.sequences
        if not isinstance(sequences, torch.Tensor):
            return None, None
        
        # 确保是一维数据
        if len(sequences.shape) > 1:
            sequences = sequences.flatten()
        
        # 获取动作数量
        num_actions = None
        if hasattr(experience, 'num_actions') and experience.num_actions is not None:
            if isinstance(experience.num_actions, torch.Tensor):
                if experience.num_actions.numel() > 0:
                    num_actions = experience.num_actions.item() if experience.num_actions.numel() == 1 else experience.num_actions[0].item()
            else:
                num_actions = experience.num_actions
        
        # 如果没有动作数量信息，尝试从值函数推断
        if num_actions is None and hasattr(experience, 'values') and experience.values is not None:
            if isinstance(experience.values, torch.Tensor):
                num_actions = experience.values.numel()
        
        # 如果还是没有动作数量信息，尝试使用启发式方法
        if num_actions is None:
            # 查看值函数形状
            values_shape = experience.values.shape if hasattr(experience, 'values') and experience.values is not None else None
            if values_shape:
                print(f"从值函数形状推断动作数量: {values_shape}")
                num_actions = values_shape[0] if len(values_shape) == 1 else values_shape[-1]
            else:
                # 假设输入部分占比为90%左右(根据之前打印的数据)
                num_actions = int(len(sequences) * 0.1)
                print(f"使用启发式方法估计动作数量: {num_actions}")
        
        # 确保num_actions是整数
        num_actions = int(num_actions)
        
        # 拆分序列
        input_len = len(sequences) - num_actions
        input_part = sequences[:input_len]
        output_part = sequences[input_len:]
        
        return input_part, output_part
    
    def analyze_sequence_structure(self, experience):
        """分析序列结构并可视化输入输出部分"""
        # 拆分序列
        input_part, output_part = self.split_input_output(experience)
        
        if input_part is None or output_part is None:
            print("无法拆分序列")
            return
        
        # 打印统计信息
        print(f"\n序列总长度: {len(input_part) + len(output_part)}")
        print(f"输入部分长度: {len(input_part)} ({len(input_part)/(len(input_part) + len(output_part)):.2%})")
        print(f"输出部分长度: {len(output_part)} ({len(output_part)/(len(input_part) + len(output_part)):.2%})")
        
        # 分析输入部分和输出部分的令牌分布
        print("\n输入部分令牌统计:")
        print(f"令牌值范围: {input_part.min().item()} - {input_part.max().item()}")
        print(f"前10个令牌: {input_part[:10].tolist()}")
        print(f"后10个令牌: {input_part[-10:].tolist()}")
        
        print("\n输出部分令牌统计:")
        print(f"令牌值范围: {output_part.min().item()} - {output_part.max().item()}")
        print(f"前10个令牌: {output_part[:10].tolist()}")
        print(f"后10个令牌: {output_part[-10:].tolist()}")
        
        # 可视化
        fig, axes = plt.subplots(2, 1, figsize=(15, 6))
        
        # 输入部分可视化
        axes[0].plot(input_part.numpy(), 'b-', alpha=0.5)
        axes[0].set_title('输入部分 (Context/Prompt)')
        axes[0].set_ylabel('令牌ID')
        axes[0].set_xlabel('序列位置')
        axes[0].grid(True, alpha=0.3)
        
        # 输出部分可视化
        axes[1].plot(output_part.numpy(), 'g-', alpha=0.5)
        axes[1].set_title('输出部分 (Generation/Action)')
        axes[1].set_ylabel('令牌ID')
        axes[1].set_xlabel('序列位置')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs('analysis/sequence_packing/visualizations', exist_ok=True)
        plt.savefig('analysis/sequence_packing/visualizations/input_output_split.png')
        print("\n序列结构可视化已保存至: analysis/sequence_packing/visualizations/input_output_split.png")
        plt.close()
        
        return input_part, output_part
    
    def analyze_ppo_components(self, experience):
        """分析PPO组件(值函数、回报、优势等)"""
        # 检查值函数
        if not hasattr(experience, 'values') or experience.values is None:
            print("无值函数数据")
            return
        
        # 提取值函数数据
        values = experience.values
        if isinstance(values, torch.Tensor):
            if len(values.shape) > 1:
                values = values[0] if values.shape[0] == 1 else values.flatten()
        else:
            print(f"不支持的值函数数据类型: {type(values)}")
            return
        
        # 提取回报数据
        returns = None
        if hasattr(experience, 'returns') and experience.returns is not None:
            returns = experience.returns
            if isinstance(returns, torch.Tensor):
                if len(returns.shape) > 1:
                    returns = returns[0] if returns.shape[0] == 1 else returns.flatten()
        
        # 获取输入、输出部分
        input_part, output_part = self.split_input_output(experience)
        
        # 打印值函数与序列长度的关系
        print("\n===== PPO组件分析 =====")
        print(f"值函数长度: {len(values)}")
        if output_part is not None:
            print(f"输出部分长度: {len(output_part)}")
            
            # 比较这两个长度
            ratio = len(values) / len(output_part) if len(output_part) > 0 else 0
            print(f"值函数长度/输出部分长度的比率: {ratio:.2f}")
            
            if len(values) > len(output_part):
                print(f"值函数比输出部分长 {len(values) - len(output_part)} 个令牌")
                print("这可能表示值函数中包含了一些填充或未使用的位置")
            elif len(values) < len(output_part):
                print(f"值函数比输出部分短 {len(output_part) - len(values)} 个令牌")
                print("这可能表示并非所有输出令牌都有对应的值函数评估")
            else:
                print("值函数长度与输出部分完全匹配")
        
        # 可视化PPO组件
        fig, ax = plt.subplots(figsize=(15, 5))
        
        x = np.arange(len(values))
        ax.plot(x, values.numpy(), 'b-', label='值函数')
        
        if returns is not None:
            if len(returns) == len(values):
                ax.plot(x, returns.numpy(), 'g-', label='回报')
            else:
                print(f"回报长度({len(returns)})与值函数长度({len(values)})不匹配，无法在同一图表上显示")
        
        ax.set_title('PPO值函数与回报')
        ax.set_xlabel('动作索引')
        ax.set_ylabel('值')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs('analysis/sequence_packing/visualizations', exist_ok=True)
        plt.savefig('analysis/sequence_packing/visualizations/ppo_values_analysis.png')
        print("\nPPO组件可视化已保存至: analysis/sequence_packing/visualizations/ppo_values_analysis.png")
        plt.close()
    
    def analyze_attention_mask(self, experience):
        """详细分析注意力掩码的结构"""
        # 提取掩码数据
        if not hasattr(experience, 'attention_mask') or experience.attention_mask is None:
            print("无注意力掩码数据")
            return
        
        attention_mask = experience.attention_mask
        if not isinstance(attention_mask, torch.Tensor):
            print(f"不支持的掩码数据类型: {type(attention_mask)}")
            return
        
        # 确保是一维数据
        if len(attention_mask.shape) > 1:
            attention_mask = attention_mask.flatten()
        
        # 统计掩码数据
        mask_values, counts = torch.unique(attention_mask, return_counts=True)
        print("\n===== 注意力掩码分析 =====")
        print(f"掩码长度: {len(attention_mask)}")
        print(f"掩码中的唯一值: {mask_values.tolist()}")
        
        value_counts = {v.item(): c.item() for v, c in zip(mask_values, counts)}
        for value, count in value_counts.items():
            print(f"值 {value}: {count} 个 ({count/len(attention_mask):.2%})")
        
        # 查找连续的掩码段
        segments = []
        current_value = attention_mask[0].item()
        start_idx = 0
        
        for i in range(1, len(attention_mask)):
            if attention_mask[i].item() != current_value:
                segments.append({
                    'value': current_value,
                    'start': start_idx,
                    'end': i-1,
                    'length': i - start_idx
                })
                current_value = attention_mask[i].item()
                start_idx = i
        
        # 添加最后一个段
        segments.append({
            'value': current_value,
            'start': start_idx,
            'end': len(attention_mask)-1,
            'length': len(attention_mask) - start_idx
        })
        
        print(f"\n掩码中识别出 {len(segments)} 个连续的段:")
        for i, segment in enumerate(segments[:10]):  # 只打印前10个段
            print(f"段 {i+1}: 值={segment['value']}, 位置={segment['start']}-{segment['end']}, 长度={segment['length']}")
        
        if len(segments) > 10:
            print(f"... 还有 {len(segments) - 10} 个段未显示")
        
        # 可视化掩码
        fig, ax = plt.subplots(figsize=(15, 3))
        ax.imshow(attention_mask.numpy().reshape(1, -1), cmap='viridis', aspect='auto')
        ax.set_title('注意力掩码')
        ax.set_yticks([])
        ax.set_xlabel('序列位置')
        
        # 标记段边界
        for segment in segments:
            ax.axvline(x=segment['start'], color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        os.makedirs('analysis/sequence_packing/visualizations', exist_ok=True)
        plt.savefig('analysis/sequence_packing/visualizations/attention_mask_analysis.png')
        print("\n注意力掩码可视化已保存至: analysis/sequence_packing/visualizations/attention_mask_analysis.png")
        plt.close()
    
    def analyze_experience(self, experience_file):
        """全面分析Experience"""
        # 加载Experience
        experiences = self.load_experience(experience_file)
        
        if not experiences:
            print("没有找到Experience数据")
            return
        
        # 分析第一条Experience
        exp = experiences[0]
        print("\n===== 分析第一条Experience结构 =====")
        
        # 1. 分析序列和注意力掩码的基本信息
        if hasattr(exp, 'sequences') and exp.sequences is not None:
            if isinstance(exp.sequences, torch.Tensor):
                print(f"序列形状: {exp.sequences.shape}")
            else:
                print(f"序列类型: {type(exp.sequences)}")
        
        if hasattr(exp, 'values') and exp.values is not None:
            if isinstance(exp.values, torch.Tensor):
                print(f"值函数形状: {exp.values.shape}")
            else:
                print(f"值函数类型: {type(exp.values)}")
        
        # 2. 分析序列结构，拆分输入和输出部分
        input_part, output_part = self.analyze_sequence_structure(exp)
        
        # 3. 分析注意力掩码
        self.analyze_attention_mask(exp)
        
        # 4. 分析PPO组件
        self.analyze_ppo_components(exp)
        
        return experiences


def main():
    """主函数"""
    # 创建解码器
    decoder = SequenceDecoder()
    
    # 指定Experience文件路径
    experience_file = '/fs-computility/mllm1/limo/workspace/stepfun/Open-Reasoner-Zero/orz_ckpt/debug_orz_7b_ppo_needlebench/dumped_replay_buffer/iter0_replay_buffer.jsonl'
    
    # 分析Experience
    decoder.analyze_experience(experience_file)
    
    print("\n分析完成！可视化结果已保存到 analysis/sequence_packing/visualizations/ 目录")


if __name__ == "__main__":
    main() 