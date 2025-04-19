#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
整理analysis目录结构，将类似功能的脚本组织在一起
"""

import os
import shutil
from pathlib import Path
import sys


# 定义分类规则
CATEGORIZATION = {
    'sequence_packing': [  # 序列打包相关
        'demo_batch_comparison.py',
        'test_batch_packing.py',
        'sequence_packing_explain.py',
        'sequence_packing_explanation.md',
        'experience_visualizer.py',
    ],
    'eval_analysis': [  # 评估分析相关
        'analyze_eval_outputs.ipynb',
        'analyze_experience_buffer.ipynb',
    ],
    'model_training': [  # 模型训练相关
        'train_transformer.py',
    ],
    'utils': [  # 工具类脚本
        'cached_property.py',
        'get_results.py',
        'remove_ckpt.py',
        'orz_step.py',
    ],
    'model_inference': [  # 模型推理相关
        'qwen_base_chat.py',
    ],
}

# 定义分类说明
CATEGORY_DESCRIPTIONS = {
    'sequence_packing': '序列打包相关的脚本和文档，用于分析和展示序列打包技术',
    'eval_analysis': '评估和分析相关的脚本，用于分析模型输出和经验数据',
    'model_training': '模型训练相关的脚本，用于训练和微调模型',
    'utils': '实用工具脚本，提供各种辅助功能',
    'model_inference': '模型推理相关的脚本，用于使用训练好的模型生成输出',
    'visualizations': '可视化结果存放目录',
}


def organize_directory(base_dir='analysis', create_readme=True, move_files=False, backup=True):
    """
    整理analysis目录结构
    
    参数:
    - base_dir: 基础目录
    - create_readme: 是否创建README文件
    - move_files: 是否移动文件到对应目录
    - backup: 是否备份原目录
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"目录 {base_dir} 不存在!")
        return
    
    # 如果需要备份，创建备份
    if backup and move_files:
        backup_dir = f"{base_dir}_backup"
        if not os.path.exists(backup_dir):
            shutil.copytree(base_dir, backup_dir)
            print(f"已创建备份目录: {backup_dir}")
    
    # 获取当前目录下所有文件
    existing_files = [f.name for f in base_path.glob('*') if f.is_file()]
    
    # 创建分类目录
    for category, description in CATEGORY_DESCRIPTIONS.items():
        category_dir = base_path / category
        if not category_dir.exists() and (move_files or create_readme):
            category_dir.mkdir(exist_ok=True)
            print(f"创建目录: {category_dir}")
    
    # 创建README文件
    if create_readme:
        create_readme_files(base_path)
    
    # 移动文件到对应目录
    if move_files:
        for category, files in CATEGORIZATION.items():
            category_dir = base_path / category
            for file in files:
                if file in existing_files:
                    source = base_path / file
                    target = category_dir / file
                    if source.exists() and not target.exists():
                        shutil.move(str(source), str(target))
                        print(f"移动文件: {file} -> {category}/{file}")
    
    print("\n目录整理完成!")


def create_readme_files(base_path):
    """创建README文件"""
    # 创建主README文件
    main_readme = base_path / "README.md"
    with open(main_readme, "w", encoding="utf-8") as f:
        f.write("# Analysis 工具目录\n\n")
        f.write("该目录包含用于分析Open-Reasoner-Zero项目中各种组件的工具和脚本。\n\n")
        
        f.write("## 目录结构\n\n")
        for category, description in CATEGORY_DESCRIPTIONS.items():
            f.write(f"- **{category}**: {description}\n")
        
        f.write("\n## 使用指南\n\n")
        f.write("1. 序列打包分析: 查看 `sequence_packing/` 目录下的脚本和文档\n")
        f.write("2. 评估结果分析: 查看 `eval_analysis/` 目录下的分析工具\n")
        f.write("3. 可视化结果: 查看 `visualizations/` 目录下的图片和可视化结果\n\n")
        
        f.write("## 快速入门\n\n")
        f.write("如果您想理解序列打包原理:\n")
        f.write("```bash\n")
        f.write("python analysis/sequence_packing/experience_visualizer.py\n")
        f.write("```\n\n")
        
        f.write("如果您想分析现有的经验数据:\n")
        f.write("```bash\n")
        f.write("jupyter notebook analysis/eval_analysis/analyze_experience_buffer.ipynb\n")
        f.write("```\n")
    
    print(f"创建主README文件: {main_readme}")
    
    # 为每个分类创建README文件
    for category, files in CATEGORIZATION.items():
        category_path = base_path / category
        category_readme = category_path / "README.md"
        
        with open(category_readme, "w", encoding="utf-8") as f:
            f.write(f"# {category.capitalize()} 目录\n\n")
            f.write(f"{CATEGORY_DESCRIPTIONS.get(category, '该目录包含相关脚本和工具')}\n\n")
            
            f.write("## 文件列表\n\n")
            for file in files:
                f.write(f"- **{file}**\n")
            
            f.write("\n## 使用示例\n\n")
            if category == "sequence_packing":
                f.write("了解序列打包原理:\n")
                f.write("```bash\n")
                f.write("python experience_visualizer.py\n")
                f.write("```\n")
            elif category == "eval_analysis":
                f.write("分析评估结果:\n")
                f.write("```bash\n")
                f.write("jupyter notebook analyze_experience_buffer.ipynb\n")
                f.write("```\n")
        
        print(f"创建分类README文件: {category_readme}")


def main():
    """主函数"""
    print("==== Analysis 目录整理工具 ====")
    print("该工具将帮助您整理analysis目录的结构，使其更加有组织")
    
    # 解析命令行参数
    move_files = "--move" in sys.argv
    no_backup = "--no-backup" in sys.argv
    
    organize_directory(
        base_dir='analysis',
        create_readme=True,
        move_files=move_files,
        backup=not no_backup
    )
    
    print("\n使用说明:")
    print("1. 默认情况下，该脚本只会创建目录和README文件，不会移动任何文件")
    print("2. 使用 --move 参数可以移动文件到对应的目录")
    print("3. 使用 --no-backup 参数可以在移动文件时不创建备份")
    print("\n示例命令:")
    print("  python organize_directory.py                  # 只创建目录和README")
    print("  python organize_directory.py --move           # 移动文件并创建备份")
    print("  python organize_directory.py --move --no-backup  # 移动文件不创建备份")


if __name__ == "__main__":
    main() 