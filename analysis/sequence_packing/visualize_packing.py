#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
序列打包可视化工具：直观展示序列打包是如何工作的
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# 添加当前目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入测试打包的代码
from test_batch_packing import BatchingTester, generate_test_data

# 设置字体顺序：先尝试一个英文字体，再回退到中文字体
plt.rcParams['font.family'] = ['DejaVu Sans', 'Droid Sans Fallback']
plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示


class PackingVisualizer:
    """序列打包可视化器"""
    
    def __init__(self):
        """初始化可视化工具"""
        self.tester = BatchingTester()
        # 生成一组不同的颜色
        self.colors = [
            '#FF9999', '#99FF99', '#9999FF', '#FFFF99', '#FF99FF', '#99FFFF',
            '#FFC299', '#99FFC2', '#C299FF', '#FFDD99', '#FF99DD', '#99DDFF'
        ]
        self.num_colors = len(self.colors)
        
    def visualize_packing(self, prompts, outputs):
        """可视化序列打包过程"""
        # 获取序列打包的数据
        ret_sequences, ret_attention_masks, ret_num_actions, ret_packed_seq_lens, _, _ = \
            self.tester._convert_prompts_outputs_to_batch_tensors_packing(prompts, outputs)
        
        # 获取tokens长度信息
        input_token_ids = self.tester._tokenize(prompts, self.tester.cfg.prompt_max_len, padding=False)["input_ids"]
        output_token_ids = self.tester._tokenize(outputs, self.tester.cfg.generate_max_len, padding=False)["input_ids"]
        prompt_lens = [len(ids) for ids in input_token_ids]
        output_lens = [len(ids) for ids in output_token_ids]
        
        # 创建图形
        num_batches = len(ret_sequences)
        fig, axes = plt.subplots(num_batches, 1, figsize=(15, 5*num_batches))
        if num_batches == 1:
            axes = [axes]
        
        # 为每个样本分配一个颜色
        sample_colors = [self.colors[i % self.num_colors] for i in range(len(prompts))]
        
        # 处理每个打包批次
        sample_idx = 0
        for batch_idx, (sequence, mask, actions, lengths) in enumerate(
            zip(ret_sequences, ret_attention_masks, ret_num_actions, ret_packed_seq_lens)
        ):
            ax = axes[batch_idx]
            seq_flat = sequence.squeeze(0).numpy()
            mask_flat = mask.squeeze(0).numpy()
            
            # 创建一个矩阵来可视化打包序列
            height = len(actions)
            width = len(seq_flat)
            matrix = np.zeros((height, width))
            prompt_mask = np.zeros((height, width))
            
            # 当前位置
            pos = 0
            for i, (action, length) in enumerate(zip(actions, lengths)):
                input_len = length - action
                
                # 输入部分
                matrix[i, pos:pos+input_len] = 1
                prompt_mask[i, pos:pos+input_len] = 1
                
                # 输出部分
                matrix[i, pos+input_len:pos+length] = 1
                
                # 更新位置
                pos += length
                
            # 可视化
            cmap = ListedColormap(['#EEEEEE'] + sample_colors[:height])
            ax.imshow(matrix, cmap=cmap, aspect='auto')
            
            # 添加分界线
            pos = 0
            for i, (action, length) in enumerate(zip(actions, lengths)):
                input_len = length - action
                
                # 输入/输出分界线
                ax.axvline(x=pos+input_len-0.5, color='black', linestyle='-', linewidth=2)
                
                # 样本分界线
                if i < len(actions) - 1:
                    ax.axvline(x=pos+length-0.5, color='red', linestyle='--', linewidth=1)
                
                # 标签
                ax.text(pos + input_len/2, i, 'Prompt', ha='center', va='center', fontsize=10)
                ax.text(pos + input_len + action/2, i, 'Output', ha='center', va='center', fontsize=10)
                
                pos += length
            
            # 设置轴标签
            ax.set_title(f'打包批次 {batch_idx+1}')
            ax.set_yticks(range(height))
            ax.set_yticklabels([f'样本区域 {i+1}' for i in range(height)])
            ax.set_xlabel('序列位置')
        
        # 添加图例
        legend_elements = []
        for i in range(min(len(prompts), len(sample_colors))):
            legend_elements.append(mpatches.Patch(facecolor=sample_colors[i], label=f'样本 {i+1}'))
        
        legend_elements.extend([
            plt.Line2D([0], [0], color='black', linestyle='-', label='输入/输出边界'),
            plt.Line2D([0], [0], color='red', linestyle='--', label='样本边界'),
        ])
        
        fig.legend(handles=legend_elements, loc='lower center', ncol=min(6, len(legend_elements)), 
                  bbox_to_anchor=(0.5, 0.01))
        
        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1 + 0.02*min(len(legend_elements)//6 + 1, 3))
        
        # 保存图像
        save_path = 'analysis/sequence_packing/visualizations/packed_sequences.png'
        plt.savefig(save_path)
        print(f"序列打包可视化已保存至: {save_path}")
        plt.close()
        
        return save_path


def main():
    """主函数"""
    print("序列打包可视化工具启动...")
    
    # 生成测试数据
    prompts, outputs = generate_test_data(
        num_samples=8, 
        prompt_len_range=(10, 80), 
        response_len_range=(20, 120)
    )
    
    # 创建可视化器并生成图像
    visualizer = PackingVisualizer()
    save_path = visualizer.visualize_packing(prompts, outputs)
    
    print(f"可视化完成！请查看: {save_path}")


if __name__ == "__main__":
    main()
