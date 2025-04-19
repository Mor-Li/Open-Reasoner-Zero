#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
序列打包(Sequence Packing)过程可视化解释
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import torch
from typing import List, Dict

# 设置字体顺序：先尝试一个英文字体，再回退到中文字体
plt.rcParams['font.family'] = ['DejaVu Sans', 'Droid Sans Fallback']
# 确保负号正确显示
plt.rcParams['axes.unicode_minus'] = False


class SequencePackingVisualizer:
    """序列打包过程可视化"""
    
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.colors = ['#FF9999', '#99FF99', '#9999FF', '#FFFF99', '#FF99FF', '#99FFFF']
        
    def generate_sequences(self, num_sequences=5, lengths=None, min_len=20, max_len=100):
        """生成测试序列"""
        if lengths is None:
            lengths = [np.random.randint(min_len, max_len) for _ in range(num_sequences)]
        
        sequences = []
        for length in lengths:
            # 生成随机序列，避开pad和eos token
            seq = np.random.randint(2, 1000, length).tolist()
            sequences.append(seq)
            
        return sequences
    
    def visualize_normal_batching(self, sequences, ax=None):
        """可视化普通批处理"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, len(sequences)*0.5 + 1))
        
        max_length = max(len(seq) for seq in sequences)
        
        # 创建矩阵表示
        matrix = np.zeros((len(sequences), max_length))
        
        # 有效token设为1，填充token保持为0
        for i, seq in enumerate(sequences):
            matrix[i, :len(seq)] = 1
            
        # 使用不同颜色表示不同序列
        cmap = ListedColormap(['#DDDDDD'] + self.colors[:len(sequences)])
        colored_matrix = np.zeros((len(sequences), max_length))
        
        for i in range(len(sequences)):
            colored_matrix[i, :] = i + 1
            # 填充部分设为0
            colored_matrix[i, len(sequences[i]):] = 0
        
        ax.imshow(colored_matrix, cmap=cmap, aspect='auto')
        
        # 添加标签和网格
        ax.set_yticks(np.arange(len(sequences)))
        ax.set_yticklabels([f'序列 {i+1} ({len(seq)}个token)' for i, seq in enumerate(sequences)])
        ax.set_xticks([0, max_length//4, max_length//2, 3*max_length//4, max_length-1])
        ax.set_xticklabels([1, max_length//4+1, max_length//2+1, 3*max_length//4+1, max_length])
        ax.set_xlabel('位置索引')
        
        # 计算填充统计
        total_tokens = len(sequences) * max_length
        valid_tokens = sum(len(seq) for seq in sequences)
        padding_tokens = total_tokens - valid_tokens
        padding_ratio = padding_tokens / total_tokens
        
        ax.set_title(f'普通批处理 (填充率: {padding_ratio:.1%})')
        
        # 标记填充区域
        for i, seq in enumerate(sequences):
            rect = patches.Rectangle((len(seq)-0.5, i-0.5), max_length-len(seq), 1, 
                                    linewidth=1, edgecolor='red', facecolor='none',
                                    linestyle='--')
            ax.add_patch(rect)
            if max_length - len(seq) > 10:  # 只在填充区域较大时添加文本
                ax.text(len(seq) + (max_length-len(seq))//2, i, '填充区域', 
                        ha='center', va='center', color='red', fontsize=9)
        
        return {
            'total_tokens': total_tokens,
            'valid_tokens': valid_tokens,
            'padding_tokens': padding_tokens,
            'padding_ratio': padding_ratio
        }
    
    def visualize_sequence_packing(self, sequences, max_packed_length=None, ax=None):
        """可视化序列打包过程"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 3))
            
        # 如果未指定最大打包长度，则设为所有序列总长度的一半
        if max_packed_length is None:
            max_packed_length = sum(len(seq) for seq in sequences) // 2
            
        # 排序序列以便更好地打包（可选）
        sorted_seqs = sorted(sequences, key=len, reverse=True)
        
        packed_sequences = []
        current_pack = []
        current_length = 0
        
        for seq in sorted_seqs:
            if current_length + len(seq) <= max_packed_length:
                current_pack.append(seq)
                current_length += len(seq)
            else:
                packed_sequences.append(current_pack)
                current_pack = [seq]
                current_length = len(seq)
                
        if current_pack:
            packed_sequences.append(current_pack)
            
        # 计算每个打包序列的总长度
        pack_lengths = [sum(len(seq) for seq in pack) for pack in packed_sequences]
        max_length = max(max_packed_length, max(pack_lengths))
        
        # 构建可视化矩阵
        num_packs = len(packed_sequences)
        matrix = np.zeros((num_packs, max_length))
        color_matrix = np.zeros((num_packs, max_length))
        
        # 为每个原始序列分配颜色
        seq_colors = {}
        for i, seq in enumerate(sorted_seqs):
            seq_id = id(seq)
            seq_colors[seq_id] = (i % len(self.colors)) + 1
        
        # 填充矩阵
        for pack_idx, pack in enumerate(packed_sequences):
            position = 0
            for seq in pack:
                seq_length = len(seq)
                matrix[pack_idx, position:position+seq_length] = 1
                
                # 使用一致的颜色
                color_matrix[pack_idx, position:position+seq_length] = seq_colors[id(seq)]
                
                position += seq_length
        
        # 可视化
        cmap = ListedColormap(['#DDDDDD'] + self.colors)
        ax.imshow(color_matrix, cmap=cmap, aspect='auto')
        
        # 添加标签和网格
        ax.set_yticks(np.arange(num_packs))
        ax.set_yticklabels([f'打包序列 {i+1} ({length}个token)' for i, length in enumerate(pack_lengths)])
        
        ticks = [0, max_length//4, max_length//2, 3*max_length//4, max_length-1]
        ticks = [t for t in ticks if t < max_length]
        ax.set_xticks(ticks)
        ax.set_xticklabels([t+1 for t in ticks])
        ax.set_xlabel('位置索引')
        
        # 计算填充统计
        total_tokens = num_packs * max_length
        valid_tokens = sum(len(seq) for seq in sorted_seqs)
        padding_tokens = total_tokens - valid_tokens
        padding_ratio = padding_tokens / total_tokens
        
        ax.set_title(f'序列打包 (填充率: {padding_ratio:.1%})')
        
        # 标记填充区域
        for i, pack in enumerate(packed_sequences):
            pack_length = pack_lengths[i]
            if max_length - pack_length > 5:  # 只在填充区域较大时添加标记
                rect = patches.Rectangle((pack_length-0.5, i-0.5), max_length-pack_length, 1, 
                                      linewidth=1, edgecolor='red', facecolor='none',
                                      linestyle='--')
                ax.add_patch(rect)
                if max_length - pack_length > 10:  # 只在填充区域较大时添加文本
                    ax.text(pack_length + (max_length-pack_length)//2, i, '填充区域', 
                         ha='center', va='center', color='red', fontsize=9)
        
        # 标记序列边界
        for i, pack in enumerate(packed_sequences):
            position = 0
            for seq in pack:
                position += len(seq)
                if position < max_length:
                    ax.axvline(x=position-0.5, ymin=(i/num_packs), ymax=((i+1)/num_packs), 
                             color='black', linestyle='-', linewidth=1)
        
        return {
            'packed_sequences': packed_sequences,
            'total_tokens': total_tokens,
            'valid_tokens': valid_tokens,
            'padding_tokens': padding_tokens,
            'padding_ratio': padding_ratio,
        }
    
    def visualize_comparison(self, sequences, max_packed_length=None):
        """比较并可视化两种方法"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), 
                               gridspec_kw={'height_ratios': [len(sequences)*0.5 + 1, 3, 2]})
        
        # 普通批处理
        normal_stats = self.visualize_normal_batching(sequences, ax=axes[0])
        
        # 序列打包
        packing_stats = self.visualize_sequence_packing(sequences, max_packed_length, ax=axes[1])
        
        # 比较条形图
        self.plot_comparison_bars(normal_stats, packing_stats, ax=axes[2])
        
        plt.tight_layout()
        
        # 确保目录存在
        os.makedirs('analysis', exist_ok=True)
        plt.savefig('analysis/sequence_packing_explanation.png', dpi=150)
        print(f"\n可视化结果保存到: analysis/sequence_packing_explanation.png")
        
        return {
            'normal': normal_stats,
            'packing': packing_stats,
            'reduction': 1 - (packing_stats['total_tokens'] / normal_stats['total_tokens'])
        }
        
    def plot_comparison_bars(self, normal_stats, packing_stats, ax=None):
        """绘制比较条形图"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 3))
            
        labels = ['普通批处理', '序列打包']
        valid_tokens = [normal_stats['valid_tokens'], packing_stats['valid_tokens']]
        padding_tokens = [normal_stats['padding_tokens'], packing_stats['padding_tokens']]
        total_tokens = [normal_stats['total_tokens'], packing_stats['total_tokens']]
        
        x = np.arange(len(labels))
        width = 0.35
        
        # 绘制有效token
        ax.bar(x, valid_tokens, width, label='有效Token')
        # 绘制填充token
        ax.bar(x, padding_tokens, width, bottom=valid_tokens, label='填充Token', color='#DDDDDD', hatch='//')
        
        # 添加标签和图例
        ax.set_ylabel('Token数量')
        ax.set_title('Token使用比较')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        # 添加数值标签
        for i, v in enumerate(total_tokens):
            ax.text(i, v + 5, f'总计: {v}', ha='center')
            ax.text(i, valid_tokens[i] - 20, f'有效: {valid_tokens[i]}', ha='center', color='white')
            
        # 添加填充率标签
        for i, (n, p) in enumerate(zip([normal_stats, packing_stats], x)):
            ax.text(i, total_tokens[i] - padding_tokens[i]//2, f'填充率: {n["padding_ratio"]:.1%}', 
                   ha='center', color='black')
            
        # 添加节省比例
        reduction = 1 - (packing_stats['total_tokens'] / normal_stats['total_tokens'])
        ax.text(0.5, max(total_tokens) * 1.1, 
               f'序列打包节省计算量: {reduction:.1%}', 
               ha='center', fontsize=12, fontweight='bold')
            
        return ax


def main():
    # 创建可视化器
    visualizer = SequencePackingVisualizer()
    
    # 创建不同长度的序列进行演示
    # 使用固定的长度以便展示更明显的效果
    lengths = [30, 80, 60, 45, 70]
    sequences = visualizer.generate_sequences(lengths=lengths)
    
    # 生成可视化结果
    results = visualizer.visualize_comparison(sequences, max_packed_length=150)
    
    # 打印统计结果
    print("=== 普通批处理 vs 序列打包比较 ===")
    print(f"原始序列数量: {len(sequences)}")
    print(f"序列长度: {lengths}")
    
    print("\n普通批处理:")
    print(f"总token数: {results['normal']['total_tokens']}")
    print(f"有效token数: {results['normal']['valid_tokens']}")
    print(f"填充token数: {results['normal']['padding_tokens']}")
    print(f"填充率: {results['normal']['padding_ratio']:.2%}")
    
    print("\n序列打包:")
    print(f"打包后序列数量: {len(results['packing']['packed_sequences'])}")
    print(f"总token数: {results['packing']['total_tokens']}")
    print(f"有效token数: {results['packing']['valid_tokens']}")
    print(f"填充token数: {results['packing']['padding_tokens']}")
    print(f"填充率: {results['packing']['padding_ratio']:.2%}")
    
    print(f"\n计算量减少: {results['reduction']:.2%}")
    
    print("\n=== 打包过程 ===")
    for i, pack in enumerate(results['packing']['packed_sequences']):
        seq_lengths = [len(seq) for seq in pack]
        print(f"打包序列 {i+1}: 包含 {len(pack)} 个原始序列, 长度分别为 {seq_lengths}, 总长度 {sum(seq_lengths)}")


if __name__ == "__main__":
    main() 