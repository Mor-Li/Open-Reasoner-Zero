#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experience可视化工具：展示PPO训练中的Experience类数据结构与序列打包
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Union
from matplotlib.colors import ListedColormap

# 设置字体顺序：先尝试一个英文字体，再回退到中文字体
plt.rcParams['font.family'] = ['DejaVu Sans', 'Droid Sans Fallback']
# 确保负号正确显示
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class Experience:
    """
    模拟PPO训练中的Experience类，用于存储序列打包数据
    
    Shapes of each tensor:
    sequences: (B, S) - B是批次大小，S是序列长度
    action_log_probs: (B, A) - A是动作数量
    base_action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advantages: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor
    values: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    action_mask: Optional[torch.Tensor] = None
    num_actions: Optional[torch.Tensor] = None
    packed_seq_lens: Optional[torch.Tensor] = None
    info: Dict = field(default_factory=dict)
    kl: Optional[torch.Tensor] = None
    
    def to_json(self):
        """将Experience转换为JSON格式"""
        data = {}
        for key, value in asdict(self).items():
            if isinstance(value, torch.Tensor):
                data[key] = value.tolist()
            elif value is None:
                data[key] = None
            else:
                data[key] = value
        return data
    
    @classmethod
    def from_json(cls, data):
        """从JSON数据创建Experience实例"""
        tensors = {}
        for key, value in data.items():
            if value is not None and key not in ["info"]:
                if isinstance(value, list):
                    tensors[key] = torch.tensor(value)
                else:
                    tensors[key] = value
            else:
                tensors[key] = value
        return cls(**tensors)


class ExperienceVisualizer:
    """Experience可视化工具"""
    
    def __init__(self):
        """初始化可视化工具"""
        self.colors = ['#FF9999', '#99FF99', '#9999FF', '#FFFF99', '#FF99FF', '#99FFFF']
        
    def generate_sample_experiences(self, num_samples=4, vocab_size=1000):
        """生成示例Experience数据"""
        experiences = []
        
        # 模拟不同长度的序列
        sequence_lengths = [150, 200, 120, 180]
        action_lengths = [50, 70, 40, 60]
        
        for i in range(num_samples):
            seq_len = sequence_lengths[i % len(sequence_lengths)]
            act_len = action_lengths[i % len(action_lengths)]
            
            # 生成序列数据
            sequences = torch.randint(1, vocab_size, (1, seq_len))
            
            # 生成各种概率和值
            action_log_probs = torch.randn(1, act_len) * 0.1
            base_action_log_probs = action_log_probs.clone()
            
            # 随机值和回报
            values = torch.randn(1, act_len) * 0.1
            returns = torch.rand(1, act_len)
            advantages = returns - values
            
            # 注意力掩码
            attention_mask = torch.ones(1, seq_len)
            
            # 创建Experience实例
            exp = Experience(
                sequences=sequences,
                action_log_probs=action_log_probs,
                base_action_log_probs=base_action_log_probs,
                values=values,
                returns=returns,
                advantages=advantages,
                attention_mask=attention_mask,
                action_mask=None,
                num_actions=torch.tensor([act_len]),
                packed_seq_lens=torch.tensor([seq_len]),
                info={"sample_id": i}
            )
            experiences.append(exp)
            
        return experiences
    
    def load_experiences_from_jsonl(self, filepath):
        """从JSONL文件加载Experience数据"""
        experiences = []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    experiences.append(Experience.from_json(data))
            print(f"成功从 {filepath} 加载了 {len(experiences)} 条经验数据")
        except Exception as e:
            print(f"加载数据失败: {e}")
            experiences = self.generate_sample_experiences()
            print(f"已生成 {len(experiences)} 条模拟经验数据代替")
        return experiences
    
    def visualize_sequence_packing(self, experiences, max_display=3):
        """可视化序列打包的结构"""
        n_experiences = min(len(experiences), max_display)
        fig, axes = plt.subplots(n_experiences, 1, figsize=(12, 3*n_experiences))
        if n_experiences == 1:
            axes = [axes]
        
        for i, exp in enumerate(experiences[:n_experiences]):
            ax = axes[i]
            
            # 提取序列数据并适应不同的数据结构
            seq_data = exp.sequences
            if isinstance(seq_data, torch.Tensor):
                # 如果是多维tensor，转换为一维
                if len(seq_data.shape) > 1:
                    sequence = seq_data[0].numpy() if seq_data.shape[0] == 1 else seq_data.flatten().numpy()
                else:
                    sequence = seq_data.numpy()
            elif isinstance(seq_data, list):
                sequence = np.array(seq_data)
            else:
                # 不支持的类型，使用占位数据
                sequence = np.arange(100)
                print(f"Warning: 不支持的序列数据类型: {type(seq_data)}")
            
            # 处理注意力掩码
            if exp.attention_mask is not None:
                if isinstance(exp.attention_mask, torch.Tensor):
                    if len(exp.attention_mask.shape) > 1:
                        attention_mask = exp.attention_mask[0].numpy() if exp.attention_mask.shape[0] == 1 else exp.attention_mask.flatten().numpy()
                    else:
                        attention_mask = exp.attention_mask.numpy()
                elif isinstance(exp.attention_mask, list):
                    attention_mask = np.array(exp.attention_mask)
                else:
                    attention_mask = None
            else:
                attention_mask = None
            
            # 确定序列长度
            seq_len = len(sequence)
            
            # 创建可视化矩阵
            matrix = np.zeros((1, seq_len))
            
            # 使用注意力掩码或全1填充
            if attention_mask is not None and len(attention_mask) == seq_len:
                for j in range(seq_len):
                    matrix[0, j] = attention_mask[j]
            else:
                matrix[0, :] = 1
            
            # 获取序列长度信息
            if exp.packed_seq_lens is not None:
                if isinstance(exp.packed_seq_lens, torch.Tensor):
                    if exp.packed_seq_lens.numel() > 0:
                        packed_seq_len = exp.packed_seq_lens.item() if exp.packed_seq_lens.numel() == 1 else exp.packed_seq_lens[0].item()
                    else:
                        packed_seq_len = seq_len
                else:
                    packed_seq_len = exp.packed_seq_lens
            else:
                packed_seq_len = seq_len
            
            # 获取动作数量
            if exp.num_actions is not None:
                if isinstance(exp.num_actions, torch.Tensor):
                    if exp.num_actions.numel() > 0:
                        num_action = exp.num_actions.item() if exp.num_actions.numel() == 1 else exp.num_actions[0].item()
                    else:
                        num_action = None
                else:
                    num_action = exp.num_actions
            else:
                num_action = None
            
            # 可视化
            cmap = ListedColormap(['#EEEEEE', '#99FF99'])
            ax.imshow(matrix, cmap=cmap, aspect='auto')
            
            # 添加标签
            ax.set_yticks([0])
            ax.set_yticklabels([f'Experience {i+1}'])
            ax.set_xlabel('序列位置索引')
            
            # 标记输入和输出部分
            if num_action is not None and num_action < seq_len:
                input_length = seq_len - num_action
                ax.axvline(x=input_length-0.5, color='black', linestyle='-', linewidth=2)
                ax.text(input_length/2, -0.3, '输入部分', ha='center', va='top', fontsize=10)
                ax.text(input_length + num_action/2, -0.3, '输出部分', ha='center', va='top', fontsize=10)
            
            ax.set_title(f'Experience {i+1} - 序列长度: {seq_len}' + 
                         (f', 动作数: {num_action}' if num_action is not None else ''))
        
        plt.tight_layout()
        # 确保目录存在
        os.makedirs('analysis/sequence_packing/visualizations', exist_ok=True)
        plt.savefig('analysis/sequence_packing/visualizations/sequence_structure.png')
        print(f"序列结构可视化已保存至: analysis/sequence_packing/visualizations/sequence_structure.png")
        plt.close()
        
    def visualize_experience_values(self, experiences, max_display=3):
        """可视化Experience中的值函数和回报"""
        n_experiences = min(len(experiences), max_display)
        fig, axes = plt.subplots(n_experiences, 2, figsize=(15, 4*n_experiences))
        if n_experiences == 1:
            axes = axes.reshape(1, 2)
        
        for i, exp in enumerate(experiences[:n_experiences]):
            # 提取值和回报
            if exp.values is not None:
                if isinstance(exp.values, torch.Tensor):
                    if len(exp.values.shape) > 1:
                        values = exp.values[0].numpy() if exp.values.shape[0] == 1 else exp.values.flatten().numpy()
                    else:
                        values = exp.values.numpy()
                elif isinstance(exp.values, list):
                    values = np.array(exp.values)
                else:
                    values = None
            else:
                values = None
                
            if exp.returns is not None:
                if isinstance(exp.returns, torch.Tensor):
                    if len(exp.returns.shape) > 1:
                        returns = exp.returns[0].numpy() if exp.returns.shape[0] == 1 else exp.returns.flatten().numpy()
                    else:
                        returns = exp.returns.numpy()
                elif isinstance(exp.returns, list):
                    returns = np.array(exp.returns)
                else:
                    returns = None
            else:
                returns = None
            
            # 如果有值函数数据
            if values is not None and len(values) > 0:
                ax = axes[i, 0]
                ax.plot(values, color='blue', marker='o', ms=3, label='值函数')
                ax.set_title(f'Experience {i+1} - 值函数')
                ax.set_xlabel('动作索引')
                ax.set_ylabel('值函数估计')
                
                # 添加统计信息
                stats = f"平均值: {values.mean():.4f}\n最大值: {values.max():.4f}\n最小值: {values.min():.4f}"
                ax.text(0.05, 0.95, stats, transform=ax.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                ax.grid(linestyle='--', alpha=0.7)
            
            # 如果有回报数据
            if returns is not None and len(returns) > 0:
                ax = axes[i, 1]
                ax.plot(returns, color='green', marker='o', ms=3, label='回报')
                ax.set_title(f'Experience {i+1} - 回报')
                ax.set_xlabel('动作索引')
                ax.set_ylabel('回报值')
                
                # 添加统计信息
                stats = f"平均值: {returns.mean():.4f}\n最大值: {returns.max():.4f}\n最小值: {returns.min():.4f}"
                ax.text(0.05, 0.95, stats, transform=ax.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                ax.grid(linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        os.makedirs('analysis/sequence_packing/visualizations', exist_ok=True)
        plt.savefig('analysis/sequence_packing/visualizations/values_and_returns.png')
        print(f"值函数和回报可视化已保存至: analysis/sequence_packing/visualizations/values_and_returns.png")
        plt.close()
    
    def visualize_kl_divergence(self, experiences, max_display=3):
        """可视化KL散度"""
        n_experiences = min(len(experiences), max_display)
        fig, axes = plt.subplots(n_experiences, 1, figsize=(12, 3*n_experiences))
        if n_experiences == 1:
            axes = [axes]
        
        for i, exp in enumerate(experiences[:n_experiences]):
            ax = axes[i]
            
            # 提取或计算KL散度
            if exp.kl is not None:
                if isinstance(exp.kl, torch.Tensor):
                    if len(exp.kl.shape) > 1:
                        kl = exp.kl[0].numpy() if exp.kl.shape[0] == 1 else exp.kl.flatten().numpy()
                    else:
                        kl = exp.kl.numpy()
                elif isinstance(exp.kl, list):
                    kl = np.array(exp.kl)
                else:
                    kl = None
            else:
                # 从动作概率计算KL
                if (hasattr(exp, 'action_log_probs') and exp.action_log_probs is not None and 
                    hasattr(exp, 'base_action_log_probs') and exp.base_action_log_probs is not None):
                    
                    # 提取log概率并处理不同的数据结构
                    if isinstance(exp.action_log_probs, torch.Tensor) and isinstance(exp.base_action_log_probs, torch.Tensor):
                        if len(exp.action_log_probs.shape) > 1:
                            action_log_probs = exp.action_log_probs[0] if exp.action_log_probs.shape[0] == 1 else exp.action_log_probs.flatten()
                        else:
                            action_log_probs = exp.action_log_probs
                            
                        if len(exp.base_action_log_probs.shape) > 1:
                            base_action_log_probs = exp.base_action_log_probs[0] if exp.base_action_log_probs.shape[0] == 1 else exp.base_action_log_probs.flatten()
                        else:
                            base_action_log_probs = exp.base_action_log_probs
                        
                        kl = (action_log_probs - base_action_log_probs).numpy()
                    else:
                        # 使用样例数据以防不兼容
                        kl = np.zeros(100)
                else:
                    kl = np.zeros(100)
            
            # 检查KL散度是否有效
            if kl is not None and len(kl) > 0:
                ax.plot(kl, color='red', marker='o', ms=3)
                ax.set_title(f'Experience {i+1} - KL散度')
                ax.set_xlabel('动作索引')
                ax.set_ylabel('KL散度')
                
                # 添加统计信息
                stats = f"平均值: {kl.mean():.4f}\n最大值: {kl.max():.4f}\n最小值: {kl.min():.4f}"
                ax.text(0.05, 0.95, stats, transform=ax.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                ax.grid(linestyle='--', alpha=0.7)
            else:
                ax.text(0.5, 0.5, "无可用KL散度数据", ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        os.makedirs('analysis/sequence_packing/visualizations', exist_ok=True)
        plt.savefig('analysis/sequence_packing/visualizations/kl_divergence.png')
        print(f"KL散度可视化已保存至: analysis/sequence_packing/visualizations/kl_divergence.png")
        plt.close()
    
    def print_experience_summary(self, experiences):
        """打印Experience数据摘要"""
        print("\n=== Experience数据摘要 ===")
        print(f"共计 {len(experiences)} 条Experience数据")
        
        if not experiences:
            return
        
        # 获取序列长度信息
        seq_lens = []
        for exp in experiences:
            if exp.packed_seq_lens is not None:
                if isinstance(exp.packed_seq_lens, torch.Tensor):
                    if exp.packed_seq_lens.numel() > 0:
                        seq_lens.append(float(exp.packed_seq_lens.item() if exp.packed_seq_lens.numel() == 1 else exp.packed_seq_lens[0].item()))
                else:
                    seq_lens.append(float(exp.packed_seq_lens))
            elif hasattr(exp, 'sequences') and exp.sequences is not None:
                if isinstance(exp.sequences, torch.Tensor):
                    seq_lens.append(float(exp.sequences.shape[-1]))
                elif isinstance(exp.sequences, list):
                    seq_lens.append(float(len(exp.sequences)))
        
        # 获取动作数量信息
        act_lens = []
        for exp in experiences:
            if exp.num_actions is not None:
                if isinstance(exp.num_actions, torch.Tensor):
                    if exp.num_actions.numel() > 0:
                        act_lens.append(float(exp.num_actions.item() if exp.num_actions.numel() == 1 else exp.num_actions[0].item()))
                else:
                    act_lens.append(float(exp.num_actions))
        
        if seq_lens:
            print("\n序列长度统计:")
            print(f"平均长度: {np.mean(seq_lens):.1f}")
            print(f"最大长度: {max(seq_lens)}")
            print(f"最小长度: {min(seq_lens)}")
        
        if act_lens:
            print("\n动作数量统计:")
            print(f"平均动作数: {np.mean(act_lens):.1f}")
            print(f"最大动作数: {max(act_lens)}")
            print(f"最小动作数: {min(act_lens)}")
            
            # 计算输入比例
            if seq_lens and len(seq_lens) == len(act_lens):
                input_ratios = [(seq_len - act_len) / seq_len for seq_len, act_len in zip(seq_lens, act_lens)]
                print(f"\n输入部分占比: {np.mean(input_ratios):.2%}")
        
        # 打印第一条Experience的更多详情
        exp = experiences[0]
        print("\n第一条Experience详情:")
        
        if hasattr(exp, 'sequences') and exp.sequences is not None:
            if isinstance(exp.sequences, torch.Tensor):
                print(f"序列形状: {exp.sequences.shape}")
            else:
                print(f"序列类型: {type(exp.sequences)}")
        
        # 打印值函数统计
        if exp.values is not None:
            if isinstance(exp.values, torch.Tensor):
                print(f"值函数形状: {exp.values.shape}")
                if exp.values.numel() > 0:
                    # 处理多维tensor
                    if len(exp.values.shape) > 1:
                        values = exp.values[0] if exp.values.shape[0] == 1 else exp.values.flatten()
                    else:
                        values = exp.values
                    
                    print(f"值函数统计: 平均值={values.mean().item():.4f}, "
                        f"最大值={values.max().item():.4f}, "
                        f"最小值={values.min().item():.4f}")
            else:
                print(f"值函数类型: {type(exp.values)}")
        
        # 打印回报统计
        if exp.returns is not None:
            if isinstance(exp.returns, torch.Tensor):
                print(f"回报形状: {exp.returns.shape}")
                if exp.returns.numel() > 0:
                    # 处理多维tensor
                    if len(exp.returns.shape) > 1:
                        returns = exp.returns[0] if exp.returns.shape[0] == 1 else exp.returns.flatten()
                    else:
                        returns = exp.returns
                    
                    print(f"回报统计: 平均值={returns.mean().item():.4f}, "
                        f"最大值={returns.max().item():.4f}, "
                        f"最小值={returns.min().item():.4f}")
            else:
                print(f"回报类型: {type(exp.returns)}")
        
        # 打印KL散度统计
        if exp.kl is not None:
            kl = exp.kl
        elif hasattr(exp, 'action_log_probs') and exp.action_log_probs is not None and hasattr(exp, 'base_action_log_probs') and exp.base_action_log_probs is not None:
            kl = exp.action_log_probs - exp.base_action_log_probs
        else:
            kl = None
        
        if kl is not None and isinstance(kl, torch.Tensor) and kl.numel() > 0:
            # 处理多维tensor
            if len(kl.shape) > 1:
                kl_tensor = kl[0] if kl.shape[0] == 1 else kl.flatten()
            else:
                kl_tensor = kl
                
            print(f"KL散度统计: 平均值={kl_tensor.mean().item():.4f}, "
                f"最大值={kl_tensor.max().item():.4f}, "
                f"最小值={kl_tensor.min().item():.4f}")
    
    def explain_sequence_packing(self):
        """解释序列打包的概念和工作原理"""
        print("\n===== 序列打包 (Sequence Packing) 工作原理 =====")
        print("序列打包是一种优化技术，可以减少计算资源浪费，提高训练效率。")
        
        print("\n1. 传统批处理方法的问题:")
        print("   - 传统方法将所有序列填充到最长序列的长度")
        print("   - 当序列长度差异较大时，产生大量无效填充计算")
        print("   - 对于大型语言模型而言，填充率可能高达30-50%")
        
        print("\n2. 序列打包的解决方案:")
        print("   - 将多个较短序列打包成一个长序列")
        print("   - 使用注意力掩码(attention_mask)区分不同序列")
        print("   - 追踪每个序列的实际长度和动作数量")
        
        print("\n3. Experience类中的相关字段:")
        print("   - sequences: 打包后的序列数据")
        print("   - attention_mask: 区分不同序列的掩码")
        print("   - num_actions: 每个序列的动作数量")
        print("   - packed_seq_lens: 每个打包序列的总长度")
        
        print("\n4. 工作流程举例:")
        print("   假设有以下三个序列(数字表示长度):")
        print("   - 序列A: 50 tokens")
        print("   - 序列B: 80 tokens")
        print("   - 序列C: 40 tokens")
        print("\n   传统批处理需要: 3 * 80 = 240 tokens(填充率 46%)")
        print("   序列打包可能为: 50+80 = 130, 40 = 40, 总共170 tokens(填充率 0%)")
        print("   减少了约29%的计算资源")
        
        print("\n5. 关键优势:")
        print("   - 减少填充，提高GPU利用率")
        print("   - 节约计算资源和内存")
        print("   - 加速训练和推理")


def main():
    """主函数"""
    # 创建可视化器
    visualizer = ExperienceVisualizer()
    
    # 尝试加载真实数据，失败则生成示例数据
    try:
        # 先尝试读取用户目录下的数据
        experiences = visualizer.load_experiences_from_jsonl(
            '/fs-computility/mllm1/limo/workspace/stepfun/Open-Reasoner-Zero/orz_ckpt/debug_orz_7b_ppo_needlebench/dumped_replay_buffer/iter0_replay_buffer.jsonl'
        )
    except Exception as e:
        print(f"无法加载真实数据: {e}")
        # 失败则生成示例数据
        experiences = visualizer.generate_sample_experiences()
    
    # 确保可视化目录存在
    os.makedirs('analysis/sequence_packing/visualizations', exist_ok=True)
    
    # 打印Experience结构摘要
    visualizer.print_experience_summary(experiences)
    
    # 可视化序列打包结构
    visualizer.visualize_sequence_packing(experiences)
    
    # 可视化值函数和回报
    visualizer.visualize_experience_values(experiences)
    
    # 可视化KL散度
    visualizer.visualize_kl_divergence(experiences)
    
    # 解释序列打包原理
    visualizer.explain_sequence_packing()
    
    print("\n可视化结果已保存到 analysis/sequence_packing/visualizations/ 目录")
    print("运行该脚本可以帮助理解Experience类的结构和序列打包的工作原理")
    print("\n优化建议：您可以使用 analysis/organize_directory.py 脚本来整理analysis目录")


if __name__ == "__main__":
    main() 